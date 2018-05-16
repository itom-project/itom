#ifndef DELAYJOBRUNNER_H
#define DELAYJOBRUNNER_H

#include <qobject.h>
#include <qvariant.h>
#include <qtimer.h>
#include <qtextobject.h>

#define DELAY_JOB_RUNNER(base,T1,T2) ((DelayJobRunner<T1,T2>*)(base))
#define DELAY_JOB_RUNNER_ARGTEXTBLOCK(base,T1,T2) ((DelayJobRunnerArgTextBlock<T1,T2>*)(base))
#define DELAY_JOB_RUNNER_NOARGS(base,T1,T2) ((DelayJobRunnerNoArgs<T1,T2>*)(base))

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
        m_func(NULL)
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
        (m_obj->*m_func)(m_args);
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
        m_func(NULL)
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
        (m_obj->*m_func)(m_block);
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
        m_func(NULL)
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
        (m_obj->*m_func)();
    }
};


#endif