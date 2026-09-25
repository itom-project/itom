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

#ifndef LANGUAGESERVERBACKEND_H
#define LANGUAGESERVERBACKEND_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QPointer>

#include "../python/pythonJedi.h"

namespace ito {

/**
 * @brief Abstract base class for language server backends (Jedi, ZubanLS, etc.)
 * 
 * This interface defines the common API for different language server implementations.
 * All backends must implement these methods to provide code completion, calltips,
 * go-to-definition, and other language intelligence features.
 * 
 * Compatible with Qt 5.6+ and Python 3.6+
 */
class ILanguageServerBackend : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief Backend types supported by itom
     */
    enum BackendType
    {
        Jedi,      //!< Python Jedi library (in-process)
        ZubanLS,   //!< ZubanLS Language Server (out-of-process)
        Unknown
    };

    explicit ILanguageServerBackend(QObject* parent = nullptr) : QObject(parent) {}
    virtual ~ILanguageServerBackend() {}

    /**
     * @brief Get the backend type
     * @return Type of this backend
     */
    virtual BackendType backendType() const = 0;

    /**
     * @brief Check if backend is available and ready to use
     * @return true if backend can be initialized, false otherwise
     */
    virtual bool isAvailable() const = 0;

    /**
     * @brief Initialize the backend
     * @param includeItomImportString Additional import string for itom modules
     * @return true if initialization succeeded, false otherwise
     */
    virtual bool initialize(const QString& includeItomImportString) = 0;

    /**
     * @brief Request code completion
     * @param request Completion request with source code, position, etc.
     * @return unique id of this request, or -1 if the request could not be started.
     *    The id is repeated by the corresponding completionReady signal.
     */
    virtual int requestCompletion(const JediCompletionRequest& request) = 0;

    /**
     * @brief Request calltip (function signature help)
     * @param request Calltip request with source code, position, etc.
     * @return unique id of this request, or -1 if the request could not be started.
     *    The id is repeated by the corresponding calltipReady signal.
     */
    virtual int requestCalltip(const JediCalltipRequest& request) = 0;

    /**
     * @brief Request go-to-definition/assignment
     * @param request Assignment request with source code, position, etc.
     * @return unique id of this request, or -1 if the request could not be started.
     */
    virtual int requestGoToAssignment(const JediAssignmentRequest& request) = 0;

    /**
     * @brief Request help/documentation
     * @param request Help request with source code, position, etc.
     * @return unique id of this request, or -1 if the request could not be started.
     */
    virtual int requestGetHelp(const JediGetHelpRequest& request) = 0;

    /**
     * @brief Request rename/refactoring
     * @param request Rename request with source code, position, new name, etc.
     * @return unique id of this request, or -1 if the request could not be started.
     */
    virtual int requestRename(const JediRenameRequest& request) = 0;

signals:
    /**
     * @brief Emitted when completion results are ready
     * @param requestId id, returned by requestCompletion
     * @param completions List of completion items
     */
    void completionReady(int requestId, QList<ito::JediCompletion> completions);

    /**
     * @brief Emitted when calltip results are ready
     * @param requestId id, returned by requestCalltip
     * @param calltips List of calltip information (may be empty)
     */
    void calltipReady(int requestId, QVector<ito::JediCalltip> calltips);

    /**
     * @brief Emitted when go-to-assignment results are ready
     * @param requestId id, returned by requestGoToAssignment
     * @param assignment Assignment location information
     */
    void goToAssignmentReady(int requestId, ito::JediAssignment assignment);

    /**
     * @brief Emitted when help/documentation is ready
     * @param requestId id, returned by requestGetHelp
     * @param help Help information
     */
    void getHelpReady(int requestId, ito::JediGetHelp help);

    /**
     * @brief Emitted when rename results are ready
     * @param requestId id, returned by requestRename
     * @param renames List of rename operations per file
     */
    void renameReady(int requestId, QList<ito::JediRename> renames);

    /**
     * @brief Emitted when an error occurs in the backend
     * @param errorMessage Human-readable error message
     */
    void errorOccurred(const QString& errorMessage);

    /**
     * @brief Emitted once the backend is really ready to accept requests.
     *
     * A successful return value of initialize() is not sufficient, since backends,
     * that are based on an external server process (e.g. ZubanLS), are initialized
     * asynchronously. Only after this signal has been emitted, the request methods
     * of this backend will return a valid request id.
     */
    void initialized();
};

} // namespace ito

Q_DECLARE_METATYPE(ito::ILanguageServerBackend::BackendType)

#endif // LANGUAGESERVERBACKEND_H
