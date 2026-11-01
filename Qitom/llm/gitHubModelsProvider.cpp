/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "gitHubModelsProvider.h"

#include <QEventLoop>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QTimer>
#include <QUrl>

namespace ito
{
namespace llm
{

namespace
{
QString mapNetworkError(QNetworkReply::NetworkError error)
{
    switch (error)
    {
    case QNetworkReply::AuthenticationRequiredError:
    case QNetworkReply::ContentAccessDenied:
        return QString("Authentication failed while calling GitHub Models.");
    case QNetworkReply::TimeoutError:
        return QString("GitHub Models request timed out.");
    case QNetworkReply::HostNotFoundError:
    case QNetworkReply::ConnectionRefusedError:
    case QNetworkReply::RemoteHostClosedError:
        return QString("Could not connect to GitHub Models endpoint.");
    default:
        return QString("GitHub Models request failed.");
    }
}

LlmErrorCategory mapErrorCategory(QNetworkReply::NetworkError error)
{
    switch (error)
    {
    case QNetworkReply::AuthenticationRequiredError:
    case QNetworkReply::ContentAccessDenied:
        return LlmErrorCategory::Auth;
    case QNetworkReply::TimeoutError:
        return LlmErrorCategory::Timeout;
    case QNetworkReply::HostNotFoundError:
    case QNetworkReply::ConnectionRefusedError:
    case QNetworkReply::RemoteHostClosedError:
        return LlmErrorCategory::Network;
    default:
        return LlmErrorCategory::ProviderError;
    }
}
}

GitHubModelsProvider::GitHubModelsProvider() :
    m_nextStreamHandle(1)
{
}

ProviderCapabilities GitHubModelsProvider::capabilities() const
{
    ProviderCapabilities caps;
    caps.supportsStreaming = true;
    caps.supportsCancellation = true;
    return caps;
}

LlmResponse GitHubModelsProvider::chat(const LlmRequest &request, LlmError *errorOut)
{
    LlmResponse response;

    if (request.endpoint.trimmed().isEmpty() || request.model.trimmed().isEmpty() ||
        request.apiKey.trimmed().isEmpty())
    {
        if (errorOut)
        {
            errorOut->category = LlmErrorCategory::Configuration;
            errorOut->userMessage = QString("GitHub Models configuration is incomplete.");
            errorOut->technicalMessage =
                QString("endpoint/model/apiKey must be configured for GitHub Models provider.");
        }
        return response;
    }

    QNetworkAccessManager manager;
    QNetworkRequest netRequest(QUrl(request.endpoint));
    netRequest.setHeader(QNetworkRequest::ContentTypeHeader, QString("application/json"));
    netRequest.setRawHeader("Authorization", QString("Bearer %1").arg(request.apiKey).toUtf8());

    QJsonArray messages;
    for (int i = 0; i < request.messages.size(); ++i)
    {
        const LlmMessage &msg = request.messages[i];
        QJsonObject messageObj;
        messageObj.insert("role", msg.role);
        messageObj.insert("content", msg.content);
        messages.append(messageObj);
    }

    QJsonObject root;
    root.insert("model", request.model);
    root.insert("messages", messages);
    root.insert("stream", false);

    const QByteArray payload = QJsonDocument(root).toJson(QJsonDocument::Compact);
    QNetworkReply *reply = manager.post(netRequest, payload);

    QEventLoop loop;
    QTimer timer;
    timer.setSingleShot(true);
    QObject::connect(&timer, SIGNAL(timeout()), &loop, SLOT(quit()));
    QObject::connect(reply, SIGNAL(finished()), &loop, SLOT(quit()));

    timer.start(qMax(1000, request.timeoutMs));
    loop.exec();

    if (reply->isRunning())
    {
        reply->abort();
    }

    if (reply->error() != QNetworkReply::NoError)
    {
        if (errorOut)
        {
            errorOut->category = mapErrorCategory(reply->error());
            errorOut->userMessage = mapNetworkError(reply->error());
            errorOut->technicalMessage = reply->errorString();
        }
        reply->deleteLater();
        return response;
    }

    const QByteArray body = reply->readAll();
    reply->deleteLater();

    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(body, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject())
    {
        if (errorOut)
        {
            errorOut->category = LlmErrorCategory::ProviderError;
            errorOut->userMessage = QString("GitHub Models returned invalid JSON.");
            errorOut->technicalMessage = parseError.errorString();
        }
        return response;
    }

    const QJsonObject rootObj = doc.object();
    const QJsonArray choices = rootObj.value("choices").toArray();
    if (choices.isEmpty())
    {
        if (errorOut)
        {
            errorOut->category = LlmErrorCategory::ProviderError;
            errorOut->userMessage = QString("GitHub Models response did not contain any choices.");
            errorOut->technicalMessage = QString::fromUtf8(body);
        }
        return response;
    }

    const QJsonObject first = choices.first().toObject();
    const QJsonObject message = first.value("message").toObject();

    response.content = message.value("content").toString();
    response.finishReason = first.value("finish_reason").toString();

    return response;
}

StreamHandle GitHubModelsProvider::streamChat(
    const LlmRequest &request,
    IStreamObserver *observer,
    LlmError *errorOut)
{
    StreamHandle handle;
    handle.value = m_nextStreamHandle++;

    if (!observer)
    {
        if (errorOut)
        {
            errorOut->category = LlmErrorCategory::ProviderError;
            errorOut->userMessage = QString("No stream observer provided.");
            errorOut->technicalMessage = QString("GitHubModelsProvider::streamChat observer is null.");
        }
        handle.value = 0;
        return handle;
    }

    LlmError localError;
    LlmResponse response = chat(request, &localError);

    if (!localError.userMessage.isEmpty())
    {
        observer->onError(localError);
        if (errorOut)
        {
            *errorOut = localError;
        }
        handle.value = 0;
        return handle;
    }

    if (m_cancelledHandles.contains(handle.value))
    {
        m_cancelledHandles.remove(handle.value);
        handle.value = 0;
        return handle;
    }

    LlmChunk chunk;
    chunk.sequenceNo = 0;
    chunk.textDelta = response.content;
    chunk.isFinal = true;
    chunk.finishReason = response.finishReason;
    observer->onChunk(chunk);
    observer->onCompleted(response);

    return handle;
}

void GitHubModelsProvider::cancel(const StreamHandle &handle)
{
    if (handle.isValid())
    {
        m_cancelledHandles.insert(handle.value);
    }
}

ProviderHealth GitHubModelsProvider::checkHealth()
{
    ProviderHealth health;
    health.available = true;
    health.statusMessage = QString("Provider available.");
    return health;
}

} // namespace llm
} // namespace ito
