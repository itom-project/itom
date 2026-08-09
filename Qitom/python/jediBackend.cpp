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

#include "jediBackend.h"

namespace ito {

//--------------------------------------------------------------------------------------
JediBackend::JediBackend(PythonJediRunner* jediRunner, QObject* parent)
    : ILanguageServerBackend(parent),
      m_jediRunner(jediRunner),
      m_initialized(false)
{
    if (m_jediRunner) {
        m_jediRunner->setParent(this);

        // Connect PythonJediRunner signals to our slots
        connect(m_jediRunner, &PythonJediRunner::completionResultsReady,
                this, &JediBackend::onJediCompletionReady);
        connect(m_jediRunner, &PythonJediRunner::calltipResultReady,
                this, &JediBackend::onJediCalltipReady);
        connect(m_jediRunner, &PythonJediRunner::goToAssignmentResultReady,
                this, &JediBackend::onJediGoToAssignmentReady);
        connect(m_jediRunner, &PythonJediRunner::getHelpResultReady,
                this, &JediBackend::onJediGetHelpReady);
        connect(m_jediRunner, &PythonJediRunner::renameResultReady,
                this, &JediBackend::onJediRenameReady);
    }
}

//--------------------------------------------------------------------------------------
JediBackend::~JediBackend()
{
    // m_jediRunner is deleted automatically as a child
}

//--------------------------------------------------------------------------------------
bool JediBackend::isAvailable() const
{
    // Jedi is always available if PythonJediRunner is valid
    return m_jediRunner != nullptr;
}

//--------------------------------------------------------------------------------------
bool JediBackend::initialize(const QString& includeItomImportString)
{
    if (!m_jediRunner) {
        return false;
    }

    // Try to load Jedi if not yet done
    bool jediLoaded = m_jediRunner->tryToLoadJediIfNotYetDone();

    if (!jediLoaded) {
        emit errorOccurred("Failed to load Jedi library");
        return false;
    }

    // Set the include import string
    m_jediRunner->setIncludeItomImportBeforeCodeAnalysis(!includeItomImportString.isEmpty());

    m_initialized = true;
    return true;
}

//--------------------------------------------------------------------------------------
void JediBackend::requestCompletion(const JediCompletionRequest& request)
{
    if (m_jediRunner && m_initialized) {
        m_jediRunner->addCompletionRequest(request);
    }
}

//--------------------------------------------------------------------------------------
void JediBackend::requestCalltip(const JediCalltipRequest& request)
{
    if (m_jediRunner && m_initialized) {
        m_jediRunner->addCalltipRequest(request);
    }
}

//--------------------------------------------------------------------------------------
void JediBackend::requestGoToAssignment(const JediAssignmentRequest& request)
{
    if (m_jediRunner && m_initialized) {
        m_jediRunner->addGoToAssignmentRequest(request);
    }
}

//--------------------------------------------------------------------------------------
void JediBackend::requestGetHelp(const JediGetHelpRequest& request)
{
    if (m_jediRunner && m_initialized) {
        m_jediRunner->addGetHelpRequest(request);
    }
}

//--------------------------------------------------------------------------------------
void JediBackend::requestRename(const JediRenameRequest& request)
{
    if (m_jediRunner && m_initialized) {
        m_jediRunner->addRenameRequest(request);
    }
}

//--------------------------------------------------------------------------------------
void JediBackend::onJediCompletionReady(int requestId, QList<ito::JediCompletion> completions, QPointer<QObject> sender)
{
    // Forward to our completionReady signal
    emit completionReady(requestId, completions, sender);
}

//--------------------------------------------------------------------------------------
void JediBackend::onJediCalltipReady(ito::JediCalltip calltip, QPointer<QObject> sender)
{
    // Forward to our calltipReady signal
    emit calltipReady(calltip, sender);
}

//--------------------------------------------------------------------------------------
void JediBackend::onJediGoToAssignmentReady(ito::JediAssignment assignment, QPointer<QObject> sender)
{
    // Forward to our goToAssignmentReady signal
    emit goToAssignmentReady(assignment, sender);
}

//--------------------------------------------------------------------------------------
void JediBackend::onJediGetHelpReady(ito::JediGetHelp help, QPointer<QObject> sender)
{
    // Forward to our getHelpReady signal
    emit getHelpReady(help, sender);
}

//--------------------------------------------------------------------------------------
void JediBackend::onJediRenameReady(QList<ito::JediRename> renames, QPointer<QObject> sender)
{
    // Forward to our renameReady signal
    emit renameReady(renames, sender);
}

} // namespace ito
