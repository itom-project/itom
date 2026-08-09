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

#ifndef LANGUAGESERVERMANAGER_H
#define LANGUAGESERVERMANAGER_H

#include <qobject.h>
#include <qsharedpointer.h>
#include <qstring.h>

#include "languageServerBackend.h"

namespace ito {

class PythonEngine;

/**
 * @brief Manager for language server backends
 * 
 * This singleton manages the currently active language server backend
 * (Jedi or ZubanLS) based on user settings. Code editor modes query
 * this manager to get the active backend for completion, calltips, etc.
 */
class LanguageServerManager : public QObject
{
    Q_OBJECT

public:
    static LanguageServerManager* getInstance();

    void initialize(PythonEngine* pythonEngine);
    void shutdown();

    /**
     * @brief Get the currently active language server backend
     * @return The active backend, or nullptr if none is available
     */
    ILanguageServerBackend* activeBackend() const;

    /**
     * @brief Check if a language server backend is available
     */
    bool isAvailable() const;

    /**
     * @brief Reload settings and recreate backend if necessary
     */
    void reloadSettings();

signals:
    void backendChanged();

private:
    explicit LanguageServerManager(QObject* parent = nullptr);
    ~LanguageServerManager();

    void createBackendFromSettings(PythonEngine* pythonEngine);

    static LanguageServerManager* m_instance;
    QSharedPointer<ILanguageServerBackend> m_backend;
    PythonEngine* m_pythonEngine;
};

} // namespace ito

#endif // LANGUAGESERVERMANAGER_H
