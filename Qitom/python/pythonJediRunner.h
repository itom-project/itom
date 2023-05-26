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

/* includes */
#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must before include global.h)
    #define NO_IMPORT_ARRAY

    #include "python/pythonWrapper.h"
#endif

#include <qobject.h>
#include <qthreadpool.h>
#include <qscopedpointer.h>
#include <qqueue.h>
#include <QRunnable>
#include <qmutex.h>

#include "pythonJedi.h"


namespace ito
{

//-------------------------------------------------------------------------------------
//!< base class for all runnables, that are executed with PythonJediRunner
/*
    Every instance of a derived class of JediRunnable has a run() method,
    that is called by the thread pool (with one thread only) of the PythonJediRunner
    class. By this run method, a specific request to the Python library Jedi is
    executed. Since the thread pool only has one thread, Jedi is never called
    in parallel.

    However, the thread can be run in parallel to any other Python code execution.
    Therefore, every run method of a runnable has to lock the GIL before starting
    the request and release it if done.

    It might be, that the script editors enqueue more requests of the same type
    than can be currently handled. Therefore, the run method skips the call
    if there is a newer runnable in the queue.
*/
class JediRunnable : public QRunnable
{
public:
    enum Type
    {
        RunnableCalltip,
        RunnableCompletion,
        RunnableGoToAssignment,
        RunnableGetHelp
    };

    JediRunnable(
        const Type &type,
        PyObject *pPyModJedi,
        const QString &additionalImportString
    ) :
        m_type(type),
        m_pPyModJedi(pPyModJedi),
        m_additionalImportString(additionalImportString)
    {
        setAutoDelete(true);
    };

    virtual ~JediRunnable() {};

    virtual unsigned char getCurrentId() const
    {
        return m_currentId;
    }

    virtual unsigned char getMostRecentId() const = 0;

protected:
    bool isOutdated() const;
    void startRun();
    void endRun();

    Type m_type;
    QString m_additionalImportString;
    PyObject *m_pPyModJedi;
    unsigned char m_currentId;

    static QMutex m_mutex;
};

//-------------------------------------------------------------------------------------
//!< runnable that executes a completion call to Jedi by the thread pool of Python Jedi Runner.
class CompletionRunnable : public JediRunnable
{
public:
    CompletionRunnable(
        const QString &additionalImportString,
        PyObject *pPyModJedi,
        const JediCompletionRequest &request
    ) :
        JediRunnable(JediRunnable::RunnableCompletion, pPyModJedi, additionalImportString),
        m_request(request)
    {
        m_mutex.lock();

        if (mostRecentId < 255)
        {
            m_currentId = ++mostRecentId;
        }
        else
        {
            m_currentId = 0;
            mostRecentId = 0;
        }

        m_mutex.unlock();
    };


    virtual ~CompletionRunnable() {};

    void run();

    virtual unsigned char getMostRecentId() const
    {
        return CompletionRunnable::mostRecentId;
    }

private:
    JediCompletionRequest m_request;

    static unsigned char mostRecentId;
};

//-------------------------------------------------------------------------------------
//!< runnable that executes a goto definition / assignment call to Jedi by the thread pool of Python Jedi Runner.
class GoToAssignmentRunnable : public JediRunnable
{
public:
    GoToAssignmentRunnable(
        const QString &additionalImportString,
        PyObject *pPyModJedi,
        const JediAssignmentRequest &request
    ) :
        JediRunnable(JediRunnable::RunnableGoToAssignment, pPyModJedi, additionalImportString),
        m_request(request)
    {
        m_mutex.lock();

        if (mostRecentId < 255)
        {
            m_currentId = ++mostRecentId;
        }
        else
        {
            m_currentId = 0;
            mostRecentId = 0;
        }

        m_mutex.unlock();
    };

    virtual ~GoToAssignmentRunnable() {};

    void run();

    virtual unsigned char getMostRecentId() const
    {
        return GoToAssignmentRunnable::mostRecentId;
    }

private:
    JediAssignmentRequest m_request;

    static unsigned char mostRecentId;
};

//-------------------------------------------------------------------------------------
//!< runnable that executes a calltip call to Jedi by the thread pool of Python Jedi Runner.
class CalltipRunnable : public JediRunnable
{
public:
    CalltipRunnable(
        const QString &additionalImportString,
        PyObject *pPyModJedi,
        const JediCalltipRequest &request
    ) :
        JediRunnable(JediRunnable::RunnableCalltip, pPyModJedi, additionalImportString) ,
        m_request(request)
    {
        m_mutex.lock();

        if (mostRecentId < 255)
        {
            m_currentId = ++mostRecentId;
        }
        else
        {
            m_currentId = 0;
            mostRecentId = 0;
        }

        m_mutex.unlock();
    };

    virtual ~CalltipRunnable() {};

    void run();

    virtual unsigned char getMostRecentId() const
    {
        return CalltipRunnable::mostRecentId;
    }

private:
    JediCalltipRequest m_request;

    static unsigned char mostRecentId;
};

//-------------------------------------------------------------------------------------
//!< runnable that executes a calltip call to Jedi by the thread pool of Python Jedi Runner.
class GetHelpRunnable : public JediRunnable
{
public:
    GetHelpRunnable(
        const QString &additionalImportString,
        PyObject *pPyModJedi,
        const JediGetHelpRequest &request
    ) :
        JediRunnable(JediRunnable::RunnableGetHelp, pPyModJedi, additionalImportString),
        m_request(request)
    {
        m_mutex.lock();

        if (mostRecentId < 255)
        {
            m_currentId = ++mostRecentId;
        }
        else
        {
            m_currentId = 0;
            mostRecentId = 0;
        }

        m_mutex.unlock();
    };

    virtual ~GetHelpRunnable() {};

    void run();

    virtual unsigned char getMostRecentId() const
    {
        return GetHelpRunnable::mostRecentId;
    }

private:
    JediGetHelpRequest m_request;

    static unsigned char mostRecentId;
};

//-------------------------------------------------------------------------------------
//!< Thread-safe helper class for PythonEngine to manage calls to the Python Jedi package.
/* This class is initialized by the PythonEngine as singleton and opened in pythonStartup()
and closed in pythonShutdown().

Its methods are thread-safe and are usually called via wrapper methods in Python engine.
The idea is, that jedi is always triggered via the Python C-API from another thread,
using the Python GIL, such that other Python commands can be executed in parallel.

This is for instance important if somebody enters "import numpy" in the command line
and presses enter: Since numpy is an intense package, Jedi requires a little bit of time
when numpy is analyzed for the first time, such that the code execution would have to wait
for a couple of seconds. Using the thread-pool approach of this class, the execution is
started much faster.

Hint: The thread pool of this class is limited to one thread, such that no parallel
requests to jedi can be executed (desired behaviour).
*/
class PythonJediRunner : public QObject
{
    Q_OBJECT

public:
    PythonJediRunner(const QString &includeItomImportString);
    ~PythonJediRunner();

    //!< Tries to import itomJediLib (and the jedi package) and returns true if successful, else false.
    /*
    If the module and package is already imported, true is directly returned.
    This method is thread-safe.
    */
    bool tryToLoadJediIfNotYetDone();

    void setIncludeItomImportBeforeCodeAnalysis(bool include)
    {
        m_includeItomImportBeforeCodeAnalysis = include;
    }

    //!< Adds a new calltip request. Thread-safe.
    void addCalltipRequest(const JediCalltipRequest &request);

    //!< Adds a new completion request. Thread-safe.
    void addCompletionRequest(const JediCompletionRequest &request);

    //!< Adds a new goto assignment / definition request. Thread-safe.
    void addGoToAssignmentRequest(const JediAssignmentRequest &request);

    //!< Adds a new get-help request. Thread-safe.
    void addGetHelpRequest(const JediGetHelpRequest &request);

private:
    QString additionalImportString() const {
        return
            (m_includeItomImportBeforeCodeAnalysis ?
            m_includeItomImportString : "");
    }

    QScopedPointer<QThreadPool> m_threadPool;

    //!< Python package Jedi for auto completion and calltips (Jedi is tried to be loaded as late as possible)
    PyObject *m_pyModJedi;

    //!< defines, if it is already checked if Jedi could be loaded on this computer.
    bool  m_pyModJediChecked;

    //!< decides if itom is automatically included in every source file before it is handed to the syntax checker
    bool m_includeItomImportBeforeCodeAnalysis;

    //!< string that is prepended to each script before syntax check (if m_includeItomImportBeforeCodeAnalysis is true)
    QString m_includeItomImportString;
};

}; //end namespace ito
