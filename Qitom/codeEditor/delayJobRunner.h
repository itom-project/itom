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

    Further hints:
    ------------------------

    This file belongs to the code editor of itom. The code editor is
    in major parts a fork / rewritten version of the python-based source
    code editor PyQode from Colin Duquesnoy and others
    (see https://github.com/pyQode). PyQode itself is licensed under
    the MIT License (MIT).

    Some parts of the code editor of itom are also inspired by the
    source code editor of the Spyder IDE (https://github.com/spyder-ide),
    also licensed under the MIT License and developed by the Spyder Project
    Contributors.

*********************************************************************** */

#ifndef DELAYJOBRUNNER_H
#define DELAYJOBRUNNER_H

#include <qobject.h>
#include <qvariant.h>
#include <qtimer.h>
#include <qtextobject.h>

#define DELAY_JOB_RUNNER(base,T1,T2) ((DelayJobRunner<T1,T2>*)(base))
#define DELAY_JOB_RUNNER_ARGTEXTBLOCK(base,T1,T2) ((DelayJobRunnerArgTextBlock<T1,T2>*)(base))
#define DELAY_JOB_RUNNER_ARGTEXTCURSOR(base,T1,T2) ((DelayJobRunnerArgTextCursor<T1,T2>*)(base))
#define DELAY_JOB_RUNNER_NOARGS(base,T1,T2) ((DelayJobRunnerNoArgs<T1,T2>*)(base))
#define DELAY_JOB_RUNNER_GENERICARG(base,T1,T2,T3) ((DelayJobRunnerGenericArg<T1,T2,T3>*)(base))

namespace ito {

class DelayJobRunnerBase : public QObject
{
    Q_OBJECT
public:
    DelayJobRunnerBase(int delay = 500, QObject *parent = NULL) : QObject(parent), m_delay(delay)
    {
        connect(&m_timer, SIGNAL(timeout()), this, SLOT(execRequestedJob()));
    }
    virtual ~DelayJobRunnerBase() {}

    virtual void cancelRequests() = 0;

    int delay() const { return m_delay; }

    void setDelay(int delay) { m_delay = delay; }

protected:
    QTimer m_timer;
    int m_delay;


private slots:
    virtual void execRequestedJob() = 0;
};

/*
Utility class for running job after a certain delay. If a new request is
made during this delay, the previous request is dropped and the timer is
restarted for the new request.
We use this to implement a cooldown effect that prevents jobs from being
executed while the IDE is not idle.
A job is a simple callable.
*/
template <typename OBJECT, typename FUNC>
class DelayJobRunner : public DelayJobRunnerBase
{


public:
    //-------------------------------------------
    /*
    :param delay: Delay to wait before running the job. This delay applies
    to all requests and cannot be changed afterwards.
    */
    DelayJobRunner(int delay = 500, QObject *parent = NULL) :
        DelayJobRunnerBase(delay, parent),
        m_func(NULL),
        m_obj(NULL)
    {
        int i = 0;
    }

    virtual ~DelayJobRunner() {}

    void requestJob(OBJECT* obj, FUNC f, const QList<QVariant> &args)
    {
        cancelRequests();
        m_obj = obj;
        m_func = f;
        m_args = args;
        m_timer.start(m_delay);
    }

    virtual void cancelRequests()
    {
        m_timer.stop();
        m_obj = NULL;
        m_func = NULL;
        m_args.clear();
    }

private:
    OBJECT* m_obj;
    FUNC m_func;
    QList<QVariant> m_args;

    //-------------------------------------------
    /*
    Execute the requested job after the timer has timeout.
    */
    virtual void execRequestedJob()
    {
        m_timer.stop();
        if (m_obj)
        {
            (m_obj->*m_func)(m_args);
        }
    }
};

/*
Utility class for running job after a certain delay. If a new request is
made during this delay, the previous request is dropped and the timer is
restarted for the new request.
We use this to implement a cooldown effect that prevents jobs from being
executed while the IDE is not idle.
A job is a simple callable.
*/
template <typename OBJECT, typename FUNC>
class DelayJobRunnerArgTextBlock : public DelayJobRunnerBase
{


public:
    //-------------------------------------------
    /*
    :param delay: Delay to wait before running the job. This delay applies
    to all requests and cannot be changed afterwards.
    */
    DelayJobRunnerArgTextBlock(int delay = 500, QObject *parent = NULL) :
        DelayJobRunnerBase(delay, parent),
        m_func(NULL),
        m_obj(NULL)
    {
        int i = 0;
    }

    virtual ~DelayJobRunnerArgTextBlock() {}

    void requestJob(OBJECT* obj, FUNC f, const QTextBlock &block)
    {
        cancelRequests();
        m_obj = obj;
        m_func = f;
        m_block = block;
        m_timer.start(m_delay);
    }

    virtual void cancelRequests()
    {
        m_timer.stop();
        m_obj = NULL;
        m_func = NULL;
        m_block = QTextBlock();
    }

private:
    OBJECT* m_obj;
    FUNC m_func;
    QTextBlock m_block;

    //-------------------------------------------
    /*
    Execute the requested job after the timer has timeout.
    */
    virtual void execRequestedJob()
    {
        m_timer.stop();
        if (m_obj)
        {
            (m_obj->*m_func)(m_block);
        }
    }
};



/*
Utility class for running job after a certain delay. If a new request is
made during this delay, the previous request is dropped and the timer is
restarted for the new request.
We use this to implement a cooldown effect that prevents jobs from being
executed while the IDE is not idle.
A job is a simple callable.
*/
template <typename OBJECT, typename FUNC>
class DelayJobRunnerArgTextCursor : public DelayJobRunnerBase
{


public:
    //-------------------------------------------
    /*
    :param delay: Delay to wait before running the job. This delay applies
    to all requests and cannot be changed afterwards.
    */
    DelayJobRunnerArgTextCursor(int delay = 500, QObject *parent = NULL) :
        DelayJobRunnerBase(delay, parent),
        m_func(NULL),
        m_obj(NULL)
    {
    }

    virtual ~DelayJobRunnerArgTextCursor() {}

    void requestJob(OBJECT* obj, FUNC f, const QTextCursor &cursor)
    {
        cancelRequests();
        m_obj = obj;
        m_func = f;
        m_cursor = cursor;
        m_timer.start(m_delay);
    }

    virtual void cancelRequests()
    {
        m_timer.stop();
        m_obj = NULL;
        m_func = NULL;
        m_cursor = QTextCursor();
    }

private:
    OBJECT* m_obj;
    FUNC m_func;
    QTextCursor m_cursor;

    //-------------------------------------------
    /*
    Execute the requested job after the timer has timeout.
    */
    virtual void execRequestedJob()
    {
        m_timer.stop();
        if (m_obj)
        {
            (m_obj->*m_func)(m_cursor);
        }
    }
};



/*
Utility class for running job after a certain delay. If a new request is
made during this delay, the previous request is dropped and the timer is
restarted for the new request.
We use this to implement a cooldown effect that prevents jobs from being
executed while the IDE is not idle.
A job is a simple callable.
*/
template <typename OBJECT, typename FUNC, typename ARGTYPE>
class DelayJobRunnerGenericArg : public DelayJobRunnerBase
{


public:
    //-------------------------------------------
    /*
    :param delay: Delay to wait before running the job. This delay applies
    to all requests and cannot be changed afterwards.
    */
    DelayJobRunnerGenericArg(int delay = 500, QObject *parent = NULL) :
        DelayJobRunnerBase(delay, parent),
        m_func(NULL),
        m_obj(NULL)
    {
        int i = 0;
    }

    virtual ~DelayJobRunnerGenericArg() {}

    void requestJob(OBJECT* obj, FUNC f, const ARGTYPE &arg)
    {
        cancelRequests();
        m_obj = obj;
        m_func = f;
        m_arg = arg;
        m_timer.start(m_delay);
    }

    virtual void cancelRequests()
    {
        m_timer.stop();
        m_obj = NULL;
        m_func = NULL;
        m_arg = ARGTYPE();
    }

private:
    OBJECT* m_obj;
    FUNC m_func;
    ARGTYPE m_arg;

    //-------------------------------------------
    /*
    Execute the requested job after the timer has timeout.
    */
    virtual void execRequestedJob()
    {
        m_timer.stop();
        if (m_obj)
        {
            (m_obj->*m_func)(m_arg);
        }
    }
};



template <typename OBJECT, typename FUNC>
class DelayJobRunnerNoArgs : public DelayJobRunnerBase
{


public:
    //-------------------------------------------
    /*
    :param delay: Delay to wait before running the job. This delay applies
    to all requests and cannot be changed afterwards.
    */
    DelayJobRunnerNoArgs(int delay = 500, QObject *parent = NULL) :
        DelayJobRunnerBase(delay, parent),
        m_func(NULL),
        m_obj(NULL)
    {
        int i = 0;
    }

    virtual ~DelayJobRunnerNoArgs() {}

    void requestJob(OBJECT* obj, FUNC f)
    {
        cancelRequests();
        m_obj = obj;
        m_func = f;
        m_timer.start(m_delay);
    }

    virtual void cancelRequests()
    {
        m_timer.stop();
        m_obj = NULL;
        m_func = NULL;
    }

private:
    OBJECT* m_obj;
    FUNC m_func;

    //-------------------------------------------
    /*
    Execute the requested job after the timer has timeout.
    */
    virtual void execRequestedJob()
    {
        m_timer.stop();
        if (m_obj)
        {
            (m_obj->*m_func)();
        }
    }
};

} //end namespace ito


#endif
