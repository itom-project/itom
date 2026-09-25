/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2024, Institut für Technische Optik (ITO),
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

#include "languageServerManager.h"
#include "zubanLspBackend.h"
#include "jediLanguageServer.h"
#include "../python/pythonEngine.h"

#include <qsettings.h>
#include <qdebug.h>
#include "../AppManagement.h"

namespace ito {

LanguageServerManager* LanguageServerManager::m_instance = nullptr;

//!< maximum time, the asynchronous initialization of a backend may take, before
//!< the manager falls back to the legacy jedi implementation.
namespace {
    const int timeoutInitializationMs = 10000;
}

//-------------------------------------------------------------------------------------
LanguageServerManager* LanguageServerManager::getInstance()
{
    if (!m_instance)
    {
        m_instance = new LanguageServerManager();
    }
    return m_instance;
}

//-------------------------------------------------------------------------------------
LanguageServerManager::LanguageServerManager(QObject* parent)
    : QObject(parent),
      m_backendCreationStarted(false),
      m_pythonEngine(nullptr)
{
    m_pendingBackendTimer.setSingleShot(true);

    connect(&m_pendingBackendTimer, &QTimer::timeout,
        this, &LanguageServerManager::onPendingBackendTimeout);
}

//-------------------------------------------------------------------------------------
LanguageServerManager::~LanguageServerManager()
{
    shutdown();
}

//-------------------------------------------------------------------------------------
void LanguageServerManager::initialize(PythonEngine* pythonEngine)
{
    // optional: the python engine is also obtained on demand from AppManagement,
    // therefore this method does not need to be called at startup.
    // The backend itself is created lazily at the first call of activeBackend().
    m_pythonEngine = pythonEngine;
}

//-------------------------------------------------------------------------------------
void LanguageServerManager::shutdown()
{
    resetPendingBackend();
    m_backendCreationStarted = false;
    m_backend.clear();
    m_pythonEngine = nullptr;
}

//-------------------------------------------------------------------------------------
void LanguageServerManager::createBackendFromSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    // the default is the zuban language server, the jedi based implementation is legacy.
    bool useZuban = settings.value("useZubanLanguageServer", true).toBool();
    QString zubanPath = settings.value("zubanLsPath", "").toString();
    settings.endGroup();

    m_backendCreationStarted = true;

    if (useZuban)
    {
        // the zuban language server is an external process and does not require
        // the python engine of itom.
        QSharedPointer<ZubanLspBackend> zubanBackend(new ZubanLspBackend(zubanPath));

        if (zubanBackend->isAvailable())
        {
            // The zuban backend is initialized asynchronously: initialize() only starts
            // the server process and sends the 'initialize' request. The backend is only
            // ready to accept requests, once it emitted its initialized() signal.
            m_pendingBackend = zubanBackend;

            connect(zubanBackend.data(), &ILanguageServerBackend::initialized,
                this, &LanguageServerManager::onPendingBackendInitialized);
            connect(zubanBackend.data(), &ILanguageServerBackend::errorOccurred,
                this, &LanguageServerManager::onPendingBackendError);

            if (zubanBackend->initialize(""))
            {
                m_pendingBackendTimer.start(timeoutInitializationMs);
                return;
            }
        }

        resetPendingBackend();

        qWarning() << "LanguageServerManager: the zuban language server is not "
                      "available, falling back to the legacy jedi implementation.";
    }

    createJediFallbackBackend();
}

//-------------------------------------------------------------------------------------
void LanguageServerManager::createJediFallbackBackend()
{
    // fallback and legacy implementation: jedi, wrapped into a pseudo language server.
    // Only this backend is executed within the python engine of itom.
    if (!m_pythonEngine)
    {
        // the manager is usually not explicitly initialized, hence obtain the
        // python engine on demand.
        m_pythonEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    }

    if (!m_pythonEngine)
    {
        qWarning() << "LanguageServerManager: the jedi language server cannot be "
                      "created without a python engine.";
        return;
    }

    QSharedPointer<JediLanguageServer> jediBackend(
        new JediLanguageServer(m_pythonEngine->getJediRunner()));

    // this backend is initialized synchronously, hence its initialized() signal is
    // already emitted within initialize().
    if (jediBackend->isAvailable() && jediBackend->initialize(""))
    {
        m_backend = jediBackend;
        emit backendChanged();
    }
    else
    {
        qWarning() << "LanguageServerManager: no language server backend could be "
                      "initialized.";
    }
}

//-------------------------------------------------------------------------------------
void LanguageServerManager::onPendingBackendInitialized()
{
    if (m_pendingBackend.isNull())
    {
        return;
    }

    QSharedPointer<ILanguageServerBackend> backend = m_pendingBackend;
    resetPendingBackend();

    m_backend = backend;
    emit backendChanged();
}

//-------------------------------------------------------------------------------------
void LanguageServerManager::onPendingBackendError(const QString& errorMessage)
{
    if (m_pendingBackend.isNull())
    {
        return;
    }

    qWarning() << "LanguageServerManager: the zuban language server could not be "
                  "initialized, falling back to the legacy jedi implementation. Reason:"
               << errorMessage;

    resetPendingBackend();
    createJediFallbackBackend();
}

//-------------------------------------------------------------------------------------
void LanguageServerManager::onPendingBackendTimeout()
{
    if (m_pendingBackend.isNull())
    {
        return;
    }

    qWarning() << "LanguageServerManager: the zuban language server did not finish its "
                  "initialization within" << timeoutInitializationMs
               << "ms, falling back to the legacy jedi implementation.";

    resetPendingBackend();
    createJediFallbackBackend();
}

//-------------------------------------------------------------------------------------
void LanguageServerManager::resetPendingBackend()
{
    m_pendingBackendTimer.stop();

    if (!m_pendingBackend.isNull())
    {
        disconnect(m_pendingBackend.data(), nullptr, this, nullptr);
        m_pendingBackend.clear();
    }
}

//-------------------------------------------------------------------------------------
ILanguageServerBackend* LanguageServerManager::activeBackend()
{
    if (m_backend.isNull() && !m_backendCreationStarted)
    {
        // lazy initialization at the first usage
        createBackendFromSettings();
    }

    // this can still be nullptr, if the initialization of the backend
    // is asynchronous and not yet finished.
    return m_backend.data();
}

//-------------------------------------------------------------------------------------
bool LanguageServerManager::isAvailable() const
{
    return !m_backend.isNull() && m_backend->isAvailable();
}

//-------------------------------------------------------------------------------------
void LanguageServerManager::reloadSettings()
{
    resetPendingBackend();
    m_backend.clear();
    createBackendFromSettings();
}

} // namespace ito
