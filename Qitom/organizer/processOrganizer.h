/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2021, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

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

#pragma once

#include "../../common/sharedStructures.h"
#include "../global.h"

#include <qhash.h>
#include <qobject.h>
#include <qpair.h>
#include <qprocess.h>
#include <qsignalmapper.h>
#include <qstring.h>

namespace ito {

class ProcessOrganizer : public QObject
{
    Q_OBJECT
public:
    ProcessOrganizer();
    ~ProcessOrganizer();

    inline QMultiHash<QString, QPair<QProcess*, bool>> getProcesses()
    {
        return m_processes;
    }

    QProcess* getFirstExistingProcess(const QString& name);
    QProcess* getProcess(
        const QString& name,
        bool tryToUseExistingProcess,
        bool& existingProcess,
        bool closeOnFinalize = false);

    QByteArray getStandardOutputBuffer(const QString& processKey) const
    {
        return m_processStdOut[processKey];
    }

    void clearStandardOutputBuffer(const QString& processKey)
    {
        if (m_processStdOut.contains(processKey))
        {
            m_processStdOut[processKey].clear();
        }
    }

    bool bringWindowsOnTop(const QString& windowName);

    static QString getAbsQtToolPath(const QString& binaryName, bool* found = NULL);

protected:
    RetVal collectGarbage(bool forceToCloseAll = false);

private:
    // keyName (assistant, designer...) -> (Process-Pointer, boolean deciding
    // whether application should be closed on shutdown of itom or not)
    QMultiHash<QString, QPair<QProcess*, bool>> m_processes;
    QMap<QString, QByteArray> m_processStdOut;


public slots:
    void processFinished(int /*exitCode*/, QProcess::ExitStatus /*exitStatus*/);
    void processError(QProcess::ProcessError /*error*/);
    void readyReadStandardOutput();
};

} // end namespace ito
