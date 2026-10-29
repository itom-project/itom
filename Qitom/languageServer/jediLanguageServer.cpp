/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2025, Institut für Technische Optik (ITO),
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

#include "jediLanguageServer.h"

namespace ito {

//-------------------------------------------------------------------------------------
JediLanguageServer::JediLanguageServer(
    const QSharedPointer<PythonJediRunner>& jediRunner, QObject* parent) :
    ILanguageServerBackend(parent),
    m_jediRunner(jediRunner), m_initialized(false), m_nextRequestId(1)
{
}

//-------------------------------------------------------------------------------------
JediLanguageServer::~JediLanguageServer()
{
    // the jedi runner is owned by the python engine, hence it must not be deleted here.
}

//-------------------------------------------------------------------------------------
bool JediLanguageServer::isAvailable() const
{
    return !m_jediRunner.isNull();
}

//-------------------------------------------------------------------------------------
bool JediLanguageServer::initialize(const QString& includeItomImportString)
{
    if (m_jediRunner.isNull())
    {
        emit errorOccurred(tr("The jedi runner is not available."));
        return false;
    }

    if (!m_jediRunner->tryToLoadJediIfNotYetDone())
    {
        emit errorOccurred(tr("The python package jedi could not be loaded."));
        return false;
    }

    m_initialized = true;

    // this backend is initialized synchronously, hence the signal can be emitted here.
    emit initialized();

    return true;
}

//-------------------------------------------------------------------------------------
int JediLanguageServer::requestCalltip(const CalltipRequest& request)
{
    if (!m_initialized || m_jediRunner.isNull())
    {
        return -1;
    }

    // the jedi runner does not emit any signal, but calls the slot m_callbackFctName
    // of m_sender. Therefore this language server registers itself as receiver and
    // assigns an id, that is passed to the caller and repeated by calltipReady.
    int requestId = m_nextRequestId++;
    m_pendingCalltipIds.append(requestId);

    CalltipRequest internalRequest = request;
    internalRequest.m_sender = this;
    internalRequest.m_callbackFctName = "onCalltipResultAvailable";

    m_jediRunner->addCalltipRequest(internalRequest);

    return requestId;
}

//-------------------------------------------------------------------------------------
void JediLanguageServer::onCalltipResultAvailable(QVector<ito::CalltipItem> calltips)
{
    int requestId = -1;

    if (!m_pendingCalltipIds.isEmpty())
    {
        requestId = m_pendingCalltipIds.takeFirst();
    }

    emit calltipReady(requestId, calltips);
}

//-------------------------------------------------------------------------------------
int JediLanguageServer::requestCompletion(const CompletionRequest& request)
{
    if (!m_initialized || m_jediRunner.isNull())
    {
        return -1;
    }

    int requestId = m_nextRequestId++;

    CompletionRequest internalRequest = request;
    internalRequest.m_requestId = requestId;
    internalRequest.m_sender = this;
    internalRequest.m_callbackFctName = "onCompletionResultAvailable";

    m_jediRunner->addCompletionRequest(internalRequest);

    return requestId;
}

//-------------------------------------------------------------------------------------
void JediLanguageServer::onCompletionResultAvailable(
    int line, int col, int requestId, QVector<ito::CompletionItem> completions)
{
    Q_UNUSED(line);
    Q_UNUSED(col);
    emit completionReady(requestId, completions);
}

//-------------------------------------------------------------------------------------
int JediLanguageServer::requestGoToAssignment(const AssignmentRequest& request)
{
    if (!m_initialized || m_jediRunner.isNull())
    {
        return -1;
    }

    int requestId = m_nextRequestId++;
    m_pendingAssignmentIds.append(requestId);

    AssignmentRequest internalRequest = request;
    internalRequest.m_sender = this;
    internalRequest.m_callbackFctName = "onGoToAssignmentResultAvailable";

    m_jediRunner->addGoToAssignmentRequest(internalRequest);

    return requestId;
}

//-------------------------------------------------------------------------------------
void JediLanguageServer::onGoToAssignmentResultAvailable(QVector<ito::AssignmentItem> assignments)
{
    int requestId = -1;

    if (!m_pendingAssignmentIds.isEmpty())
    {
        requestId = m_pendingAssignmentIds.takeFirst();
    }

    emit goToAssignmentReady(requestId, assignments);
}

//-------------------------------------------------------------------------------------
int JediLanguageServer::requestGetHelp(const GetHelpRequest& request)
{
    if (!m_initialized || m_jediRunner.isNull())
    {
        return -1;
    }

    int requestId = m_nextRequestId++;
    m_pendingGetHelpIds.append(requestId);

    GetHelpRequest internalRequest = request;
    internalRequest.m_sender = this;
    internalRequest.m_callbackFctName = "onGetHelpResultAvailable";

    m_jediRunner->addGetHelpRequest(internalRequest);

    return requestId;
}

//-------------------------------------------------------------------------------------
void JediLanguageServer::onGetHelpResultAvailable(QVector<ito::HelpItem> helps)
{
    int requestId = -1;

    if (!m_pendingGetHelpIds.isEmpty())
    {
        requestId = m_pendingGetHelpIds.takeFirst();
    }

    // the backend interface provides one help object per request
    emit getHelpReady(requestId, helps.isEmpty() ? HelpItem() : helps.first());
}

//-------------------------------------------------------------------------------------
int JediLanguageServer::requestRename(const RenameRequest& request)
{
    if (!m_initialized || m_jediRunner.isNull())
    {
        return -1;
    }

    int requestId = m_nextRequestId++;
    m_pendingRenameIds.append(requestId);

    RenameRequest internalRequest = request;
    internalRequest.m_sender = this;
    internalRequest.m_callbackFctName = "onRenameResultAvailable";

    m_jediRunner->addRenameRequest(internalRequest);

    return requestId;
}

//-------------------------------------------------------------------------------------
void JediLanguageServer::onRenameResultAvailable(
    QVector<ito::RenameFileChanges> renames, QString oldValue, bool success, QString errorText)
{
    int requestId = -1;

    if (!m_pendingRenameIds.isEmpty())
    {
        requestId = m_pendingRenameIds.takeFirst();
    }

    emit renameReady(requestId, renames, oldValue, success, errorText);
}

} // end namespace ito
