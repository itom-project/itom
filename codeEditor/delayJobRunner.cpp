#include "delayJobRunner.h"

//-------------------------------------------
/*
:param delay: Delay to wait before running the job. This delay applies
to all requests and cannot be changed afterwards.
*/
template <typename OBJECT, typename FUNC>
DelayJobRunner<OBJECT,FUNC>::DelayJobRunner(OBJECT obj /*= NULL*/, FUNC f /*= NULL*/, int delay /*= 500*/, QObject *parent /*= NULL*/) :
    DelayJobRunnerBase(delay, obj),
    m_func(f)
{
    
}

//-------------------------------------------
template <typename OBJECT, typename FUNC>
DelayJobRunner<OBJECT,FUNC>::~DelayJobRunner()
{
}

//-------------------------------------------
template <typename OBJECT, typename FUNC>
void DelayJobRunner<OBJECT,FUNC>::cancelRequests()
{
    m_timer.stop();
    m_obj = NULL;
    m_funcs = NULL;
    m_args.clear();
}

//-------------------------------------------
//template <typename OBJECT, typename FUNC>
//void DelayJobRunner<OBJECT,FUNC>::requestJob(OBJECT obj, FUNC f, const QList<QVariant> &args)
//{
//    cancelRequests();
//    m_obj = obj;
//    m_func = f;
//    m_args = args;
//    m_timer.start(m_delay);
//}


//-------------------------------------------
/*
Execute the requested job after the timer has timeout.
*/
template <typename OBJECT, typename FUNC>
void DelayJobRunner<OBJECT,FUNC>::execRequestedJob()
{
    m_timer.stop();
    (m_obj->*m_func)(m_args);
}