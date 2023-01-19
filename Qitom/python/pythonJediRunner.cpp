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

#include "pythonJediRunner.h"

#include "../AppManagement.h"
#include "../global.h"
#include "pythonQtConversion.h"

#include <iostream>
#include <qdatetime.h>
#include <qdebug.h>
#include <qfileinfo.h>
#include <qmetaobject.h>


//-------------------------------------------------------------------------------------
namespace ito {

// static member initialization

/*static*/ unsigned char CompletionRunnable::mostRecentId = 0;
/*static*/ unsigned char GoToAssignmentRunnable::mostRecentId = 0;
/*static*/ unsigned char CalltipRunnable::mostRecentId = 0;
/*static*/ unsigned char GetHelpRunnable::mostRecentId = 0;
/*static*/ QMutex JediRunnable::m_mutex;

//-------------------------------------------------------------------------------------
PythonJediRunner::PythonJediRunner(const QString& includeItomImportString) :
    QObject(), m_threadPool(new QThreadPool(this)), m_pyModJediChecked(false), m_pyModJedi(NULL),
    m_includeItomImportString(includeItomImportString), m_includeItomImportBeforeCodeAnalysis(false)
{
#if QT_VERSION >= 0x050a00
    // increase the stack size for the jedi worker thread to allow a deeper recursion depth.
    // For Qt < 5.10, the stack size cannot be changed for Jedi / Python relevant
    // threads only (especially this thread pool). Therefore it is globally increased
    // for all threads for MSVC via CMake (Linker flag /STACK).
    m_threadPool->setStackSize(5000000);
#endif
    m_threadPool->setMaxThreadCount(1);
}

//-------------------------------------------------------------------------------------
PythonJediRunner::~PythonJediRunner()
{
    QThreadPool* pool = m_threadPool.take();

    pool->clear();

    // waits for all runners to be terminated
    DELETE_AND_SET_NULL(pool);

    if (m_pyModJedi)
    {
        PyGILState_STATE gstate = PyGILState_Ensure();

        Py_XDECREF(m_pyModJedi);
        m_pyModJedi = nullptr;

        PyGILState_Release(gstate);
    }
}

//-------------------------------------------------------------------------------------
bool PythonJediRunner::tryToLoadJediIfNotYetDone()
{
    if (m_pyModJediChecked)
    {
        return m_pyModJedi != NULL;
    }
    else
    {
        if (PyGILState_Check() > 0)
        {
            // this should usually never happen, however rare cases have been reported
            // where it was the case. Therefore this small workaround. If the GIL is
            // not released, Jedi will not be loaded and the check will not be finished, yet.
            QThread::msleep(100);

            if (PyGILState_Check() > 0)
            {
                // still not working
                return false;
            }
        }

        PyGILState_STATE gstate = PyGILState_Ensure();

        m_pyModJediChecked = true;
        m_pyModJedi = PyImport_ImportModule("itomJediLib"); // new reference

        if (m_pyModJedi == NULL)
        {
            QObject* mainWin = AppManagement::getMainWindow();

            if (mainWin)
            {
                QString text = tr("Auto completion, calltips, goto definition... "
                                  "not possible, since the package 'jedi' could not be "
                                  "loaded (Python packages 'jedi' and 'parso' are "
                                  "required for this feature).");

                QMetaObject::invokeMethod(
                    mainWin,
                    "showInfoMessageLine",
                    Q_ARG(QString, text),
                    Q_ARG(QString, "PythonEngine"));
            }

            PyErr_Clear();
            PyGILState_Release(gstate);
            return false;
        }

        PyGILState_Release(gstate);
        return true;
    }
}

//-------------------------------------------------------------------------------------
void PythonJediRunner::addCalltipRequest(const JediCalltipRequest& request)
{
    if (!m_threadPool.isNull())
    {
        CalltipRunnable* runnable =
            new CalltipRunnable(additionalImportString(), m_pyModJedi, request);

        /*qDebug()
            << QDateTime::currentDateTime().toString("hh:mm:ss.zzz")
            << "Calltip request enqueued. ID:"
            << runnable->getCurrentId();*/

        m_threadPool->start(runnable);
    }
}

//-------------------------------------------------------------------------------------
void PythonJediRunner::addCompletionRequest(const JediCompletionRequest& request)
{
    if (!m_threadPool.isNull())
    {
        CompletionRunnable* runnable =
            new CompletionRunnable(additionalImportString(), m_pyModJedi, request);

        /*qDebug()
            << QDateTime::currentDateTime().toString("hh:mm:ss.zzz")
            << "Completion request enqueued. ID:"
            << runnable->getCurrentId();*/

        m_threadPool->start(runnable);
    }
}

//-------------------------------------------------------------------------------------
void PythonJediRunner::addGoToAssignmentRequest(const JediAssignmentRequest& request)
{
    if (!m_threadPool.isNull())
    {
        GoToAssignmentRunnable* runnable =
            new GoToAssignmentRunnable(additionalImportString(), m_pyModJedi, request);

        /*qDebug()
            << QDateTime::currentDateTime().toString("hh:mm:ss.zzz")
            << "Assignment request enqueued. ID:"
            << runnable->getCurrentId();*/

        m_threadPool->start(runnable);
    }
}

//-------------------------------------------------------------------------------------
void PythonJediRunner::addGetHelpRequest(const JediGetHelpRequest& request)
{
    if (!m_threadPool.isNull())
    {
        GetHelpRunnable* runnable =
            new GetHelpRunnable(additionalImportString(), m_pyModJedi, request);

        /*qDebug()
            << QDateTime::currentDateTime().toString("hh:mm:ss.zzz")
            << "Calltip request enqueued. ID:"
            << runnable->getCurrentId();*/

        m_threadPool->start(runnable);
    }
}


//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
void CompletionRunnable::run()
{
    startRun();

    if (isOutdated())
    {
        return;
    }

    QVector<ito::JediCompletion> completions;

    PyGILState_STATE gstate = PyGILState_Ensure();

    try
    {
        PyObject* result = NULL;

        if (m_additionalImportString != "")
        {
            // add from itom import * as first line (this is afterwards removed from results)
            result = PyObject_CallMethod(
                m_pPyModJedi,
                "completions",
                "siiss",
                (m_additionalImportString + "\n" + m_request.m_source).toUtf8().constData(),
                m_request.m_line + 1,
                m_request.m_col,
                m_request.m_path.toUtf8().constData(),
                m_request.m_prefix.toUtf8().constData()); // new ref
        }
        else
        {
            result = PyObject_CallMethod(
                m_pPyModJedi,
                "completions",
                "siiss",
                m_request.m_source.toUtf8().constData(),
                m_request.m_line,
                m_request.m_col,
                m_request.m_path.toUtf8().constData(),
                m_request.m_prefix.toUtf8().constData()); // new ref
        }

        if (result && PyList_Check(result))
        {
            PyObject* pycompletion = NULL;
            const char* calltip;
            const char* description;
            const char* icon;
            PyObject* tooltips = NULL;

            for (Py_ssize_t idx = 0; idx < PyList_Size(result); ++idx)
            {
                pycompletion = PyList_GetItem(result, idx); // borrowed ref

                if (PyTuple_Check(pycompletion))
                {
                    if (PyArg_ParseTuple(
                            pycompletion,
                            "sssO!",
                            &calltip,
                            &description,
                            &icon,
                            &PyList_Type,
                            &tooltips))
                    {
                        bool ok;
                        QStringList tooltipList =
                            PythonQtConversion::PyObjToStringList(tooltips, false, ok);

                        completions.append(ito::JediCompletion(
                            QLatin1String(calltip),
                            tooltipList,
                            QLatin1String(icon),
                            QLatin1String(description)));
                    }
                    else
                    {
                        PyErr_Clear();
                        std::cerr << "Error in completion: invalid format of tuple\n" << std::endl;
                    }
                }
                else
                {
                    std::cerr << "Error in completion: list of tuples required\n" << std::endl;
                }
            }

            Py_DECREF(result);
        }
        else
        {
            Py_XDECREF(result);
#ifdef _DEBUG
            std::cerr << "Error when getting completions from jedi\n" << std::endl;
            PyErr_PrintEx(0);
#endif
        }
    }
    catch (...)
    {
        qDebug() << "jediCompletionRequestEnqueued4: exception";
        std::cerr << "Unknown exception in jediCompletionRequestEnqueued. Please report this bug.\n"
                  << std::endl;
    }

    PyGILState_Release(gstate);

    QObject* s = m_request.m_sender.data();
    if (s && m_request.m_callbackFctName != "")
    {
        QMetaObject::invokeMethod(
            s,
            m_request.m_callbackFctName.constData(),
            Q_ARG(int, m_request.m_line),
            Q_ARG(int, m_request.m_col),
            Q_ARG(int, m_request.m_requestId),
            Q_ARG(QVector<ito::JediCompletion>, completions));
    }

    endRun();
};

//-------------------------------------------------------------------------------------
void GoToAssignmentRunnable::run()
{
    startRun();

    if (isOutdated())
    {
        return;
    }

    QVector<ito::JediAssignment> assignments;

    PyGILState_STATE gstate = PyGILState_Ensure();

    int lineOffset = 0;
    PyObject* result = NULL;

    if (m_additionalImportString != "")
    {
        lineOffset = 1;
        // add from itom import * as first line (this is afterwards removed from results)
        result = PyObject_CallMethod(
            m_pPyModJedi,
            "goto_assignments",
            "siisi",
            (m_additionalImportString + "\n" + m_request.m_source).toUtf8().constData(),
            m_request.m_line + 1,
            m_request.m_col,
            m_request.m_path.toUtf8().constData(),
            m_request.m_mode); // new ref
    }
    else
    {
        result = PyObject_CallMethod(
            m_pPyModJedi,
            "goto_assignments",
            "siisi",
            m_request.m_source.toUtf8().constData(),
            m_request.m_line,
            m_request.m_col,
            m_request.m_path.toUtf8().constData(),
            m_request.m_mode); // new ref
    }

    if (result && PyList_Check(result))
    {
        PyObject* pydefinition = NULL;
        const char* path2;
        const char* fullName;
        int column;
        int line;

        for (Py_ssize_t idx = 0; idx < PyList_Size(result); ++idx)
        {
            pydefinition = PyList_GetItem(result, idx); // borrowed ref

            if (PyTuple_Check(pydefinition))
            {
                if (PyArg_ParseTuple(pydefinition, "siis", &path2, &line, &column, &fullName))
                {
                    if (line >= 0)
                    {
                        QFileInfo filepath2;
                        filepath2.setFile(QLatin1String(path2));

                        if (lineOffset == 1)
                        {
                            QFileInfo filepath(m_request.m_path);

                            if (filepath != filepath2)
                            {
                                lineOffset = 0;
                            }
                        }
                        assignments.append(ito::JediAssignment(
                            filepath2.canonicalFilePath(),
                            line - lineOffset,
                            column,
                            QLatin1String(fullName)));
                    }
                }
                else
                {
                    PyErr_Clear();
                    std::cerr << "Error in assignment / definition: invalid format of tuple\n"
                              << std::endl;
                }
            }
            else
            {
                std::cerr << "Error in assignment / definition: list of tuples required\n"
                          << std::endl;
            }
        }

        Py_DECREF(result);
    }

    else
    {
        Py_XDECREF(result);
#ifdef _DEBUG
        std::cerr << "Error when getting assignments or definitions from jedi\n" << std::endl;
        PyErr_PrintEx(0);
#endif
    }


    PyGILState_Release(gstate);

    QObject* s = m_request.m_sender.data();
    if (s && m_request.m_callbackFctName != "")
    {
        QMetaObject::invokeMethod(
            s,
            m_request.m_callbackFctName.constData(),
            Q_ARG(QVector<ito::JediAssignment>, assignments));
    }

    endRun();
};

//-------------------------------------------------------------------------------------
void CalltipRunnable::run()
{
    startRun();

    if (isOutdated())
    {
        return;
    }

    QVector<ito::JediCalltip> calltips;

    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* result = NULL;
    int lineOffset = 0;

    if (m_additionalImportString != "")
    {
        // add from itom import * as first line (this is afterwards removed from results)
        lineOffset = 1;
        result = PyObject_CallMethod(
            m_pPyModJedi,
            "calltips",
            "siis",
            (m_additionalImportString + "\n" + m_request.m_source).toUtf8().constData(),
            m_request.m_line + 1,
            m_request.m_col,
            m_request.m_path.toUtf8().constData()); // new ref
    }
    else
    {
        result = PyObject_CallMethod(
            m_pPyModJedi,
            "calltips",
            "siis",
            m_request.m_source.toUtf8().constData(),
            m_request.m_line,
            m_request.m_col,
            m_request.m_path.toUtf8().constData()); // new ref
    }

    if (result && PyList_Check(result))
    {
        PyObject* pycalltip = NULL;
        const char* calltipMethodName;
        PyObject* pyparams = NULL;
        int column;
        int bracketStartCol;
        int bracketStartLine;

        for (Py_ssize_t idx = 0; idx < PyList_Size(result); ++idx)
        {
            pycalltip = PyList_GetItem(result, idx); // borrowed ref

            if (PyTuple_Check(pycalltip))
            {
                if (PyArg_ParseTuple(
                        pycalltip,
                        "sO!iii",
                        &calltipMethodName,
                        &PyList_Type,
                        &pyparams,
                        &column,
                        &bracketStartLine,
                        &bracketStartCol))
                {
                    bool ok;
                    QStringList params = PythonQtConversion::PyObjToStringList(pyparams, false, ok);

                    if (ok)
                    {
                        calltips.append(ito::JediCalltip(
                            QLatin1String(calltipMethodName),
                            params,
                            column,
                            bracketStartLine - lineOffset,
                            bracketStartCol));
                    }
                    else
                    {
                        PyErr_Clear();
                        std::cerr
                            << "Error in param string list of calltip: invalid format of tuple\n"
                            << std::endl;
                    }
                }
                else
                {
                    PyErr_Clear();
                    std::cerr << "Error in calltip: invalid format of tuple\n" << std::endl;
                }
            }
            else
            {
                std::cerr << "Error in calltip: list of tuples required\n" << std::endl;
            }
        }

        Py_DECREF(result);
    }

    else
    {
        Py_XDECREF(result);
#ifdef _DEBUG
        std::cerr << "Error when getting calltips from jedi\n" << std::endl;
        PyErr_PrintEx(0);
#endif
    }

    PyGILState_Release(gstate);

    QObject* s = m_request.m_sender.data();

    if (s && m_request.m_callbackFctName != "")
    {
        QMetaObject::invokeMethod(
            s, m_request.m_callbackFctName.constData(), Q_ARG(QVector<ito::JediCalltip>, calltips));
    }

    endRun();
};

//-------------------------------------------------------------------------------------
void GetHelpRunnable::run()
{
    startRun();

    if (isOutdated())
    {
        return;
    }

    QVector<ito::JediGetHelp> helps;

    PyGILState_STATE gstate = PyGILState_Ensure();

    int lineOffset = 0;
    PyObject* result = nullptr;

    if (m_additionalImportString != "")
    {
        lineOffset = 1;
        // add from itom import * as first line (this is afterwards removed from results)
        result = PyObject_CallMethod(
            m_pPyModJedi,
            "get_help",
            "siis",
            (m_additionalImportString + "\n" + m_request.m_source).toUtf8().constData(),
            m_request.m_line + 1,
            m_request.m_col,
            m_request.m_path.toUtf8().constData()); // new ref
    }
    else
    {
        result = PyObject_CallMethod(
            m_pPyModJedi,
            "get_help",
            "siis",
            m_request.m_source.toUtf8().constData(),
            m_request.m_line,
            m_request.m_col,
            m_request.m_path.toUtf8().constData()); // new ref
    }

    if (result && PyList_Check(result))
    {
        PyObject* pygethelp = nullptr;
        const char* description;
        PyObject* pytooltips = nullptr;

        for (Py_ssize_t idx = 0; idx < PyList_Size(result); ++idx)
        {
            pygethelp = PyList_GetItem(result, idx); // borrowed ref

            if (PyTuple_Check(pygethelp))
            {
                if (PyArg_ParseTuple(pygethelp, "sO!", &description, &PyList_Type, &pytooltips))
                {
                    bool ok;
                    QStringList tooltips =
                        PythonQtConversion::PyObjToStringList(pytooltips, false, ok);

                    if (ok)
                    {
                        helps.append(ito::JediGetHelp(description, tooltips));
                    }
                }
                else
                {
                    PyErr_Clear();
                    std::cerr << "Error in jedi get_help: invalid format of tuple\n" << std::endl;
                }
            }
            else
            {
                std::cerr << "Error in jedi get_help: list of tuples required\n" << std::endl;
            }
        }

        Py_DECREF(result);
    }

    else
    {
        Py_XDECREF(result);
#ifdef _DEBUG
        std::cerr << "Error when getting help from jedi\n" << std::endl;
        PyErr_PrintEx(0);
#endif
    }


    PyGILState_Release(gstate);

    QObject* s = m_request.m_sender.data();
    if (s && m_request.m_callbackFctName != "")
    {
        QMetaObject::invokeMethod(
            s, m_request.m_callbackFctName.constData(), Q_ARG(QVector<ito::JediGetHelp>, helps));
    }

    endRun();
};

//--------------------------------------------------------------------------------
void JediRunnable::startRun()
{
    /*qDebug() << QDateTime::currentDateTime().toString("hh:mm:ss.zzz")
        << "start run. Type:" << m_type
        << "ID:" << getCurrentId()
        << "Outdated:" << isOutdated();*/
}

//--------------------------------------------------------------------------------
void JediRunnable::endRun()
{
    /*qDebug() << QDateTime::currentDateTime().toString("hh:mm:ss.zzz")
        << "End run. Type:" << m_type
        << "ID:" << getCurrentId();*/
}

//--------------------------------------------------------------------------------
bool JediRunnable::isOutdated() const
{
    return getCurrentId() != getMostRecentId();
}


} // end namespace ito
