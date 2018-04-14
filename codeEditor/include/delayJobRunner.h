#ifndef DELAYJOBRUNNER_H
#define DELAYJOBRUNNER_H

#include <qobject.h>
#include <qvariant.h>
#include <qtimer.h>

#define DELAY_JOB_RUNNER(base,T1,T2) ((DelayJobRunner<T1,T2>*)(base))

class DelayJobRunnerBase : public QObject
{
    Q_OBJECT
public:
    DelayJobRunnerBase(int delay = 500, QObject *parent = NULL) : QObject(parent) 
    {
        connect(&m_timer, SIGNAL(timeout()), this, SLOT(execRequestedJob()));
    }
    virtual ~DelayJobRunnerBase() {}

    virtual void cancelRequests() = 0;

protected:
    QTimer m_timer;
    int m_delay;
    QList<QVariant> m_args;

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
    //DelayJobRunner(OBJECT obj = NULL, FUNC f = NULL, int delay = 500, QObject *parent = NULL);
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


#endif