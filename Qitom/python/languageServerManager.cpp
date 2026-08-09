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
#include "jediBackend.h"
#include "zubanLspBackend.h"
#include "pythonEngine.h"

#include <qsettings.h>
#include <qdebug.h>
#include "../AppManagement.h"

namespace ito {

LanguageServerManager* LanguageServerManager::m_instance = nullptr;

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
      m_pythonEngine(nullptr)
{
}

//-------------------------------------------------------------------------------------
LanguageServerManager::~LanguageServerManager()
{
    shutdown();
}

//-------------------------------------------------------------------------------------
void LanguageServerManager::initialize(PythonEngine* pythonEngine)
{
    m_pythonEngine = pythonEngine;
    createBackendFromSettings(pythonEngine);
}

//-------------------------------------------------------------------------------------
void LanguageServerManager::shutdown()
{
    m_backend.clear();
    m_pythonEngine = nullptr;
}

//-------------------------------------------------------------------------------------
void LanguageServerManager::createBackendFromSettings(PythonEngine* pythonEngine)
{
    if (!pythonEngine)
    {
        qWarning() << "LanguageServerManager: Cannot create backend without PythonEngine";
        return;
    }

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");
    QString backendType = settings.value("languageServerBackend", "Jedi").toString();
    QString zubanPath = settings.value("zubanLsPath", "").toString();
    settings.endGroup();

    qDebug() << "LanguageServerManager: Creating backend:" << backendType;

    if (backendType == "ZubanLS" && !zubanPath.isEmpty())
    {
        // Try to use ZubanLS
        ZubanLspBackend* zubanBackend = new ZubanLspBackend(zubanPath, this);

        if (zubanBackend->isAvailable())
        {
            // ZubanLS doesn't need the itom import string since it's external
            if (zubanBackend->initialize(""))
            {
                m_backend = QSharedPointer<ILanguageServerBackend>(zubanBackend);
                qDebug() << "LanguageServerManager: ZubanLS backend initialized successfully";
                emit backendChanged();
                return;
            }
            else
            {
                qWarning() << "LanguageServerManager: ZubanLS initialization failed, falling back to Jedi";
                delete zubanBackend;
            }
        }
        else
        {
            qWarning() << "LanguageServerManager: ZubanLS not available, falling back to Jedi";
            delete zubanBackend;
        }
    }

    // Default or fallback: use Jedi through PythonEngine
    // For Jedi backend, we'll need to access it through PythonEngine's enqueue methods
    // Since we can't directly access m_jediRunner, we'll create a lightweight wrapper
    // that forwards to PythonEngine's public enqueue methods
    qDebug() << "LanguageServerManager: Using Jedi backend (via PythonEngine)";
    // Note: Jedi requests will continue to go through PythonEngine's enqueue methods
    // The backend interface is primarily for ZubanLS; Jedi remains in PythonEngine
}

//-------------------------------------------------------------------------------------
ILanguageServerBackend* LanguageServerManager::activeBackend() const
{
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
    if (m_pythonEngine)
    {
        m_backend.clear();
        createBackendFromSettings(m_pythonEngine);
    }
}

} // namespace ito
