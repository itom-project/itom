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
#include <qtimer.h>

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
     * @brief Get the currently active language server backend.
     *
     * The backend is created lazily at the first call, based on the setting
     * CodeEditor/useZubanLanguageServer.
     *
     * @return The active backend, or nullptr if none is available
     */
    ILanguageServerBackend* activeBackend();

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

private slots:
    //!< the pending backend reported, that it is ready to accept requests.
    void onPendingBackendInitialized();

    //!< the pending backend reported an error during its initialization.
    void onPendingBackendError(const QString& errorMessage);

    //!< the pending backend did not report its successful initialization in time.
    void onPendingBackendTimeout();

private:
    explicit LanguageServerManager(QObject* parent = nullptr);
    ~LanguageServerManager();

    //!< creates the backend, according to the setting CodeEditor/useZubanLanguageServer.
    //!< The python engine is only required and obtained for the legacy jedi backend.
    void createBackendFromSettings();

    //!< creates the legacy jedi backend, that is used if the zuban language server
    //!< is either disabled or could not be initialized.
    void createJediFallbackBackend();

    //!< disconnects the pending backend and stops the timeout timer.
    void resetPendingBackend();

    static LanguageServerManager* m_instance;
    QSharedPointer<ILanguageServerBackend> m_backend;

    //!< backend, whose asynchronous initialization is currently awaited.
    //!< It only becomes the active backend after its initialized() signal.
    QSharedPointer<ILanguageServerBackend> m_pendingBackend;

    //!< guards the lazy creation, such that the backend is not created again
    //!< while the initialization of the pending backend is still in progress.
    bool m_backendCreationStarted;

    //!< limits the time, the asynchronous initialization of a backend may take.
    QTimer m_pendingBackendTimer;

    //!< only required for the legacy jedi language server (nullptr, if not yet needed).
    PythonEngine* m_pythonEngine;
};

} // namespace ito

#endif // LANGUAGESERVERMANAGER_H
