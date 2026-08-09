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

#ifndef JEDIBACKEND_H
#define JEDIBACKEND_H

#include "languageServerBackend.h"
#include "pythonJediRunner.h"

#include <QObject>
#include <QPointer>

namespace ito {

/**
 * @brief Jedi language server backend - wraps existing PythonJediRunner
 * 
 * This backend maintains backward compatibility with the existing Jedi integration
 * by wrapping PythonJediRunner and forwarding requests/responses.
 */
class JediBackend : public ILanguageServerBackend
{
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param jediRunner Existing Jedi runner instance (takes ownership)
     * @param parent Parent QObject
     */
    explicit JediBackend(PythonJediRunner* jediRunner, QObject* parent = nullptr);

    /**
     * @brief Destructor
     */
    virtual ~JediBackend();

    // ILanguageServerBackend interface
    virtual BackendType backendType() const override { return Jedi; }
    virtual bool isAvailable() const override;
    virtual bool initialize(const QString& includeItomImportString) override;
    virtual void requestCompletion(const JediCompletionRequest& request) override;
    virtual void requestCalltip(const JediCalltipRequest& request) override;
    virtual void requestGoToAssignment(const JediAssignmentRequest& request) override;
    virtual void requestGetHelp(const JediGetHelpRequest& request) override;
    virtual void requestRename(const JediRenameRequest& request) override;

private slots:
    // Forward signals from PythonJediRunner
    void onJediCompletionReady(int requestId, QList<ito::JediCompletion> completions, QPointer<QObject> sender);
    void onJediCalltipReady(ito::JediCalltip calltip, QPointer<QObject> sender);
    void onJediGoToAssignmentReady(ito::JediAssignment assignment, QPointer<QObject> sender);
    void onJediGetHelpReady(ito::JediGetHelp help, QPointer<QObject> sender);
    void onJediRenameReady(QList<ito::JediRename> renames, QPointer<QObject> sender);

private:
    PythonJediRunner* m_jediRunner;
    bool m_initialized;
};

} // namespace ito

#endif // JEDIBACKEND_H
