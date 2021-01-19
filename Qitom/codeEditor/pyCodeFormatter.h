/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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

#include <qobject.h>
#include <qprocess.h>
#include <qstring.h>
#include <qsharedpointer.h>
#include <qprogressdialog.h>

#include "common/retVal.h"

namespace ito
{

    class PyCodeFormatter : public QObject
    {
        Q_OBJECT
    public:
        PyCodeFormatter(QObject *parent = nullptr);
        ~PyCodeFormatter();

        ito::RetVal startFormatting(const QString &cmd, const QString &code, QWidget *dialogParent = nullptr);

    private:
        QProcess m_process;
        QString m_currentCode;
        QString m_currentError;
        QSharedPointer<QProgressDialog> m_progressDialog;
        bool m_isCancelling;


    private slots:
        void errorOccurred(QProcess::ProcessError error);
        void finished(int exitCode, QProcess::ExitStatus exitStatus);
        void readyReadStandardError();
        void readyReadStandardOutput();
        void started();
        void cancelRequested();

    signals:
        void formattingDone(bool success, QString code);

    };

}; // end namepsace ito
