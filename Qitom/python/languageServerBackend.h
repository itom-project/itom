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

#include "pythonJedi.h"

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
     */
    virtual void requestCompletion(const JediCompletionRequest& request) = 0;

    /**
     * @brief Request calltip (function signature help)
     * @param request Calltip request with source code, position, etc.
     */
    virtual void requestCalltip(const JediCalltipRequest& request) = 0;

    /**
     * @brief Request go-to-definition/assignment
     * @param request Assignment request with source code, position, etc.
     */
    virtual void requestGoToAssignment(const JediAssignmentRequest& request) = 0;

    /**
     * @brief Request help/documentation
     * @param request Help request with source code, position, etc.
     */
    virtual void requestGetHelp(const JediGetHelpRequest& request) = 0;

    /**
     * @brief Request rename/refactoring
     * @param request Rename request with source code, position, new name, etc.
     */
    virtual void requestRename(const JediRenameRequest& request) = 0;

signals:
    /**
     * @brief Emitted when completion results are ready
     * @param requestId Request ID from JediCompletionRequest
     * @param completions List of completion items
     * @param sender Original sender object
     */
    void completionReady(int requestId, QList<ito::JediCompletion> completions, QPointer<QObject> sender);

    /**
     * @brief Emitted when calltip results are ready
     * @param calltip Calltip information
     * @param sender Original sender object
     */
    void calltipReady(ito::JediCalltip calltip, QPointer<QObject> sender);

    /**
     * @brief Emitted when go-to-assignment results are ready
     * @param assignment Assignment location information
     * @param sender Original sender object
     */
    void goToAssignmentReady(ito::JediAssignment assignment, QPointer<QObject> sender);

    /**
     * @brief Emitted when help/documentation is ready
     * @param help Help information
     * @param sender Original sender object
     */
    void getHelpReady(ito::JediGetHelp help, QPointer<QObject> sender);

    /**
     * @brief Emitted when rename results are ready
     * @param renames List of rename operations per file
     * @param sender Original sender object
     */
    void renameReady(QList<ito::JediRename> renames, QPointer<QObject> sender);

    /**
     * @brief Emitted when an error occurs in the backend
     * @param errorMessage Human-readable error message
     */
    void errorOccurred(const QString& errorMessage);
};

} // namespace ito

Q_DECLARE_METATYPE(ito::ILanguageServerBackend::BackendType)

#endif // LANGUAGESERVERBACKEND_H
