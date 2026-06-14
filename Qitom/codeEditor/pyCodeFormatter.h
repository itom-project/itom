/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut fuer Technische Optik (ITO),
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
#include <qtemporarydir.h>

#include "common/retVal.h"

namespace ito
{

    class PyCodeFormatter : public QObject
    {
        Q_OBJECT
    public:
        PyCodeFormatter(QObject *parent = nullptr);
        ~PyCodeFormatter();

        ito::RetVal startSortingAndFormatting(const QString &importSortingCmd, const QString &formattingCmd, const QString &code, QWidget *dialogParent = nullptr);

    private:
        QProcess m_processFormatter;
        QProcess m_processImportSort;
        QString m_currentCode;
        QString m_currentError;
        QSharedPointer<QProgressDialog> m_progressDialog;
        bool m_isCancelling;
        const QString m_importSortTempFileName;
        QTemporaryDir m_importSortTempDir;
        QString m_pythonExePath;
        QString m_formattingCmd;

        ito::RetVal getPythonPath(QString &path) const;
        ito::RetVal startImportsSorting(const QString& importSortingCmd, const QString& code);
        ito::RetVal startCodeFormatting(const QString &formattingCmd, const QString &code);


    private slots:
        void formatterErrorOccurred(QProcess::ProcessError error);
        void formatterFinished(int exitCode, QProcess::ExitStatus exitStatus);
        void formatterReadyReadStandardError();
        void formatterReadyReadStandardOutput();
        void formatterStarted();

        void importSortErrorOccurred(QProcess::ProcessError error);
        void importSortFinished(int exitCode, QProcess::ExitStatus exitStatus);
        void importSortReadyReadStandardError();
        void importSortStarted();

        void cancelRequested();

    signals:
        void formattingDone(bool success, QString code);

    };

}; // end namepsace ito
