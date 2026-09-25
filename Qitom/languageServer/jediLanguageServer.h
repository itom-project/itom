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

#ifndef JEDILANGUAGESERVER_H
#define JEDILANGUAGESERVER_H

#include "languageServerBackend.h"

#include "../python/pythonJediRunner.h"

#include <qbytearray.h>
#include <qhash.h>
#include <qpointer.h>
#include <qsharedpointer.h>
#include <qvector.h>

namespace ito {

/*!
    \class JediLanguageServer
    \brief legacy (pseudo) language server, based on the python package jedi.

    This backend does not implement the language server protocol. It is a thin
    wrapper around the existing PythonJediRunner, that maps the itom specific
    request / callback mechanism to the signals of ILanguageServerBackend.

    The PythonJediRunner does not provide any signals, but calls the slot
    m_callbackFctName of the object m_sender of every request. Therefore, this
    class registers itself as sender and assigns a unique id to every request.
    This id is repeated by the corresponding signal of the base class, such that
    the caller can identify its own results.
*/
class JediLanguageServer : public ILanguageServerBackend
{
    Q_OBJECT

public:
    explicit JediLanguageServer(
        const QSharedPointer<PythonJediRunner>& jediRunner, QObject* parent = nullptr);
    virtual ~JediLanguageServer();

    virtual BackendType backendType() const override { return Jedi; }
    virtual bool isAvailable() const override;
    virtual bool initialize(const QString& includeItomImportString) override;

    virtual int requestCompletion(const JediCompletionRequest& request) override;
    virtual int requestCalltip(const JediCalltipRequest& request) override;
    virtual int requestGoToAssignment(const JediAssignmentRequest& request) override;
    virtual int requestGetHelp(const JediGetHelpRequest& request) override;
    virtual int requestRename(const JediRenameRequest& request) override;

private slots:
    //!< called by the CalltipRunnable of the jedi runner, once the calltips are available.
    void onCalltipResultAvailable(QVector<ito::JediCalltip> calltips);

private:
    QSharedPointer<PythonJediRunner> m_jediRunner;
    bool m_initialized;

    //!< id, assigned to the next request.
    int m_nextRequestId;

    //!< ids of all pending calltip requests (FIFO, since the jedi runner processes
    //!< the calltip requests in the order of their arrival).
    QVector<int> m_pendingCalltipIds;
};

} // end namespace ito

#endif // JEDILANGUAGESERVER_H
