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

    --------------------------------
    This class is a modified version of the class QToolTip of the
    Qt framework (licensed under LGPL):
    https://code.woboq.org/qt5/qtbase/src/widgets/kernel/qtooltip.cpp.html
*********************************************************************** */

#include "pyCodeFormatter.h"
#include "../python/pythonEngine.h"
#include "../AppManagement.h"

#include <qdebug.h>
#include <iostream>
#include <qdir.h>
#include <qfileinfo.h>
#include <qtemporarydir.h>

namespace ito {

//-------------------------------------------------------------------------------------
PyCodeFormatter::PyCodeFormatter(QObject *parent /*= nullptr*/) :
    QObject(parent),
    m_isCancelling(false),
    m_importSortTempFileName("isort_temp.py")
{
    const PythonEngine* pyeng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (pyeng)
    {
        m_pythonExePath = pyeng->getPythonExecutable();
    }
    else
    {
        m_pythonExePath = "";
    }

    connect(&m_processFormatter, &QProcess::errorOccurred,
        this, &PyCodeFormatter::formatterErrorOccurred);

    connect(&m_processFormatter, &QProcess::readyReadStandardOutput,
        this, &PyCodeFormatter::formatterReadyReadStandardOutput);

    connect(&m_processFormatter, &QProcess::readyReadStandardError,
        this, &PyCodeFormatter::formatterReadyReadStandardError);

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    connect(
        &m_processFormatter,
        SIGNAL(finished(int, QProcess::ExitStatus)),
        this,
        SLOT(formatterFinished(int, QProcess::ExitStatus)));
#else
    connect(&m_processFormatter, &QProcess::finished, this, &PyCodeFormatter::formatterFinished);
#endif


    connect(&m_processFormatter, &QProcess::started,
        this, &PyCodeFormatter::formatterStarted);

    connect(&m_processImportSort, &QProcess::errorOccurred,
        this, &PyCodeFormatter::importSortErrorOccurred);

    connect(&m_processImportSort, &QProcess::readyReadStandardError,
        this, &PyCodeFormatter::importSortReadyReadStandardError);

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    connect(
        &m_processImportSort,
        SIGNAL(finished(int, QProcess::ExitStatus)),
        this,
        SLOT(importSortFinished(int, QProcess::ExitStatus)));
#else
    connect(&m_processImportSort, &QProcess::finished, this, &PyCodeFormatter::importSortFinished);
#endif



    connect(&m_processImportSort, &QProcess::started,
        this, &PyCodeFormatter::importSortStarted);
}

//-------------------------------------------------------------------------------------
PyCodeFormatter::~PyCodeFormatter()
{
    if (!m_isCancelling)
    {
        cancelRequested();
    }
}

//-------------------------------------------------------------------------------------
ito::RetVal PyCodeFormatter::getPythonPath(QString &path) const
{
    return ito::retOk;
}

//-------------------------------------------------------------------------------------
ito::RetVal PyCodeFormatter::startImportsSorting(const QString& importSortingCmd, const QString& code)
{
    if (m_progressDialog)
    {
        m_progressDialog->setLabelText(tr("Run isort to sort import statements..."));
    }

    auto filePath = m_importSortTempDir.filePath(m_importSortTempFileName);
    QFile tempFile(filePath);

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    if (m_importSortTempDir.isValid() && tempFile.open(QIODevice::ReadWrite))
#else
    if (m_importSortTempDir.isValid() && tempFile.open(QIODeviceBase::ReadWrite))
#endif
    {
        tempFile.write(code.toUtf8());
        tempFile.close();
    }
    else
    {
        return ito::RetVal(ito::retError, 0, tr("Cannot execute isort, since a temporary file could not be created.").toLatin1().data());
    }

    if (m_pythonExePath == "")
    {
        return ito::RetVal(ito::retError, 0, tr("Path to Python executable not available").toLatin1().data());
    }

    QStringList args;
    // "isort" << "--py" << "3" << "--profile" << "black"
    args << "-m" << importSortingCmd.split(" ") << filePath; // QString("\"%1\"").arg(filePath);

    m_processImportSort.start(m_pythonExePath, args);

    return ito::retOk;
}

//-------------------------------------------------------------------------------------
/*
\param importSortingCmd is the (optional) command in the call python -m <cmd> that
    is used to sort imports and get the code passed in form of a temporary file path,
    where the code is temporarily saved before. If empty, this step is omitted.
\param formattingCmd is the command <cmd> in the call python -m <cmd> and must allow passing
    the code string via stdin.
*/
ito::RetVal PyCodeFormatter::startSortingAndFormatting(const QString& importSortingCmd, const QString& formattingCmd, const QString &code, QWidget *dialogParent /*= nullptr*/)
{
    m_isCancelling = false;

    if (m_processFormatter.state() != QProcess::NotRunning)
    {
        return ito::RetVal(ito::retError, 0, "process already started.");
    }

    if (dialogParent && !m_progressDialog)
    {
        m_progressDialog = QSharedPointer<QProgressDialog>(
            new QProgressDialog(
                "",
                tr("Cancel"),
                0, 100, dialogParent)
            );
        m_progressDialog->setModal(true);
        m_progressDialog->setValue(0);
        connect(m_progressDialog.data(), &QProgressDialog::canceled,
            this, &PyCodeFormatter::cancelRequested);
        m_progressDialog->show();
    }

    m_currentCode = code;
    m_currentError = "";
    m_formattingCmd = "";

    if (importSortingCmd != "")
    {
        m_formattingCmd = formattingCmd;
        return startImportsSorting(importSortingCmd, code);
    }
    else
    {
        return startCodeFormatting(formattingCmd, code);
    }
}

//-------------------------------------------------------------------------------------
ito::RetVal  PyCodeFormatter::startCodeFormatting(const QString& formattingCmd, const QString& code)
{
    if (m_progressDialog)
    {
        m_progressDialog->setLabelText(tr("The code formatter is running..."));
    }

    m_currentError = "";

    /* Under Windows, if itom is directly started from C:/Program Files,
    it seems, that the arguments must be passed to m_process as QStringList.
    Else a ProcessError::FailedToStart occurs. Therefore try to split the arguments...*/
    QStringList args = formattingCmd.split(" ");

    // now check if items start with a leading " and search for the
    // component that ends with the corresponding ", but not \" and
    // join them again.
    QChar sign = '"';
    int startIdx = -1;

    for (int idx = 0; idx < args.size(); ++idx)
    {
        if (startIdx == -1)
        {
            if (args[idx].startsWith(sign))
            {
                startIdx = idx;
            }
        }
        else
        {
            args[startIdx].append(" " + args[idx]);

            if (args[idx].endsWith(sign) && !args[idx].endsWith('\\' + sign))
            {
                startIdx = -1;
            }

            args[idx] = "";
        }
    }

    m_currentError = "";
    args.removeAll("");
    args.prepend("-m");
    m_processFormatter.start(m_pythonExePath, args);

    return ito::retOk;
}

//-------------------------------------------------------------------------------------
void PyCodeFormatter::cancelRequested()
{
    if (m_processImportSort.state() == QProcess::Running || m_processImportSort.state() == QProcess::Starting)
    {
        m_isCancelling = true;
        m_processImportSort.kill();
        m_processImportSort.waitForFinished(2000);
    }

    if (m_processFormatter.state() == QProcess::Running || m_processFormatter.state() == QProcess::Starting)
    {
        m_isCancelling = true;
        m_processFormatter.kill();
        m_processFormatter.waitForFinished(2000);
    }

    if (m_progressDialog)
    {
        m_progressDialog->accept();
        m_progressDialog.clear();
    }
}

//-------------------------------------------------------------------------------------
void PyCodeFormatter::formatterErrorOccurred(QProcess::ProcessError error)
{
    if (m_progressDialog)
    {
        m_progressDialog->accept();
        m_progressDialog.clear();
    }

    if (!m_isCancelling)
    {
        switch (error)
        {
        case QProcess::FailedToStart:
            emit formattingDone(false, tr("The code formatter could not be started. Maybe you do not have enough user rights."));
            break;
        case QProcess::ProcessError::Crashed:
            emit formattingDone(false, tr("The started code formatter process crashed."));
            break;
        default:
            emit formattingDone(false, tr("The started code formatter process finished with an error (code: %1).").arg(error));
            break;
        }
    }
    else
    {
        emit formattingDone(false, "");
    }
}

//-------------------------------------------------------------------------------------
void PyCodeFormatter::formatterFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (m_progressDialog)
    {
        m_progressDialog->setValue(100);
        m_progressDialog->accept();
        m_progressDialog.clear();
    }

    if (exitCode == 0 && !m_isCancelling)
    {
        emit formattingDone(true, m_currentCode);
    }
    else
    {
        emit formattingDone(false, m_currentError);
    }
}

//-------------------------------------------------------------------------------------
void PyCodeFormatter::formatterReadyReadStandardError()
{
    QByteArray output = m_processFormatter.readAllStandardError();
    m_currentError += QString::fromUtf8(output);
}

//-------------------------------------------------------------------------------------
void PyCodeFormatter::formatterReadyReadStandardOutput()
{
    QByteArray output = m_processFormatter.readAllStandardOutput();
    m_currentCode += QString::fromUtf8(output);
}

//-------------------------------------------------------------------------------------
void PyCodeFormatter::formatterStarted()
{
    if (m_progressDialog)
    {
        m_progressDialog->setValue(m_progressDialog->value() + 10);
    }

    QByteArray ba = m_currentCode.toUtf8();
    m_processFormatter.write(ba);
    m_currentCode = "";
    m_processFormatter.closeWriteChannel();

    if (m_progressDialog)
    {
        m_progressDialog->setValue(m_progressDialog->value() + 10);
    }
}

//-------------------------------------------------------------------------------------
void PyCodeFormatter::importSortErrorOccurred(QProcess::ProcessError error)
{
    if (m_progressDialog)
    {
        m_progressDialog->accept();
        m_progressDialog.clear();
    }

    if (!m_isCancelling)
    {
        switch (error)
        {
        case QProcess::FailedToStart:
            emit formattingDone(false, tr("The imports sorting could not be started. Maybe you do not have enough user rights."));
            break;
        case QProcess::ProcessError::Crashed:
            emit formattingDone(false, tr("The started imports sorting process crashed."));
            break;
        default:
            emit formattingDone(false, tr("The started imports sorting process finished with an error (code: %1).").arg(error));
            break;
        }
    }
    else
    {
        emit formattingDone(false, "");
    }
}

//-------------------------------------------------------------------------------------
void PyCodeFormatter::importSortFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (m_progressDialog)
    {
        m_progressDialog->setValue(10);
    }

    if (exitCode == 0 && !m_isCancelling)
    {
        if (m_importSortTempDir.isValid())
        {
            QFile tempFile(m_importSortTempDir.filePath(m_importSortTempFileName));

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
            if (tempFile.open(QIODevice::ReadOnly))
#else
            if (tempFile.open(QIODeviceBase::ReadOnly))
#endif
            {
                m_currentCode = QString::fromUtf8(tempFile.readAll());
                tempFile.close();
            }
            else
            {
                m_currentError = tr("Cannot execute isort, since a temporary file could not be re-opened.");
            }
        }
        else
        {
            m_currentError = tr("Invalid temporary directory for imports sorting result.");
        }

        if (m_currentError == "")
        {
            // continue with code formatting:
            ito::RetVal retval = startCodeFormatting(m_formattingCmd, m_currentCode);

            if (retval.containsError())
            {
                m_currentError = QLatin1String(retval.errorMessage());
            }
        }
    }

    if (m_currentError != "")
    {
        if (m_progressDialog)
        {
            m_progressDialog->accept();
            m_progressDialog.clear();
        }

        emit formattingDone(false, m_currentError);
    }
}

//-------------------------------------------------------------------------------------
void PyCodeFormatter::importSortReadyReadStandardError()
{
    QByteArray output = m_processImportSort.readAllStandardError();
    m_currentError += QString::fromUtf8(output);
}

//-------------------------------------------------------------------------------------
void PyCodeFormatter::importSortStarted()
{
    if (m_progressDialog)
    {
        m_progressDialog->setValue(20);
    }
}
} //end namespace ito
