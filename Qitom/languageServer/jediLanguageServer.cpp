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
int JediLanguageServer::requestCalltip(const JediCalltipRequest& request)
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

    JediCalltipRequest internalRequest = request;
    internalRequest.m_sender = this;
    internalRequest.m_callbackFctName = "onCalltipResultAvailable";

    m_jediRunner->addCalltipRequest(internalRequest);

    return requestId;
}

//-------------------------------------------------------------------------------------
void JediLanguageServer::onCalltipResultAvailable(QVector<ito::JediCalltip> calltips)
{
    int requestId = -1;

    if (!m_pendingCalltipIds.isEmpty())
    {
        requestId = m_pendingCalltipIds.takeFirst();
    }

    emit calltipReady(requestId, calltips);
}

//-------------------------------------------------------------------------------------
int JediLanguageServer::requestCompletion(const JediCompletionRequest& request)
{
    // not yet routed via this language server: the code completion mode still uses
    // the legacy methods of the python engine.
    int requestId = -1;

    if (m_initialized && !m_jediRunner.isNull())
    {
        requestId = m_nextRequestId++;
        m_jediRunner->addCompletionRequest(request);
    }

    return requestId;
}

//-------------------------------------------------------------------------------------
int JediLanguageServer::requestGoToAssignment(const JediAssignmentRequest& request)
{
    // not yet routed via this language server, see requestCompletion.
    int requestId = -1;

    if (m_initialized && !m_jediRunner.isNull())
    {
        requestId = m_nextRequestId++;
        m_jediRunner->addGoToAssignmentRequest(request);
    }

    return requestId;
}

//-------------------------------------------------------------------------------------
int JediLanguageServer::requestGetHelp(const JediGetHelpRequest& request)
{
    // not yet routed via this language server, see requestCompletion.
    int requestId = -1;

    if (m_initialized && !m_jediRunner.isNull())
    {
        requestId = m_nextRequestId++;
        m_jediRunner->addGetHelpRequest(request);
    }

    return requestId;
}

//-------------------------------------------------------------------------------------
int JediLanguageServer::requestRename(const JediRenameRequest& request)
{
    // not yet routed via this language server, see requestCompletion.
    int requestId = -1;

    if (m_initialized && !m_jediRunner.isNull())
    {
        requestId = m_nextRequestId++;
        m_jediRunner->addRenameRequest(request);
    }

    return requestId;
}

} // end namespace ito
