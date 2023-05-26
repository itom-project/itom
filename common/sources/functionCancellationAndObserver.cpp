/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "functionCancellationAndObserver.h"

#include <qatomic.h>
#include <qmutex.h>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
class FunctionCancellationAndObserverPrivate
{
public:
    FunctionCancellationAndObserverPrivate(int progressMinimum, int progressMaximum) :
        m_progressMinimum(progressMinimum),
        m_progressMaximum(progressMaximum),
        m_cancellation(0),
        m_progressValue(progressMinimum),
        m_reason(FunctionCancellationAndObserver::ReasonGeneral)
    {

    }

    QAtomicInt m_cancellation;
    int m_progressMinimum;
    int m_progressMaximum;
    QAtomicInt m_progressValue;
    FunctionCancellationAndObserver::CancellationReason m_reason;

    QString m_progressText;
    QMutex m_readWriteMutex;
};

//--------------------------------------------------------------------
FunctionCancellationAndObserver::FunctionCancellationAndObserver(int progressMinimum /*= 0*/, int progressMaximum /*= 100*/, QObject *parent /*= NULL*/) :
    d_ptr(new FunctionCancellationAndObserverPrivate(progressMinimum, progressMaximum)),
    QObject(parent)
{
}

//--------------------------------------------------------------------
FunctionCancellationAndObserver::~FunctionCancellationAndObserver()
{

}

//--------------------------------------------------------------------
bool FunctionCancellationAndObserver::isCancelled() const
{
    Q_D(const FunctionCancellationAndObserver);
    return d->m_cancellation > 0;
}

//--------------------------------------------------------------------
FunctionCancellationAndObserver::CancellationReason FunctionCancellationAndObserver::cancellationReason()
{
    Q_D(FunctionCancellationAndObserver);
    QMutexLocker locker(&(d->m_readWriteMutex));
    return d->m_reason;
}

//--------------------------------------------------------------------
void FunctionCancellationAndObserver::requestCancellation(CancellationReason reason /*= CancellationReason::ReasonGeneral*/)
{
    Q_D(FunctionCancellationAndObserver);
    d->m_cancellation = 1;
    d->m_reason = reason;

    emit cancellationRequested();
}

//--------------------------------------------------------------------
void FunctionCancellationAndObserver::reset()
{
    Q_D(FunctionCancellationAndObserver);
    d->m_cancellation = 0;
    d->m_progressValue = d->m_progressMinimum;
    QMutexLocker locker(&(d->m_readWriteMutex));
    d->m_progressText = QString();
    d->m_reason = ReasonGeneral;

    emit progressTextChanged(d->m_progressText);
    emit progressValueChanged(d->m_progressValue);
    emit resetDone();
}

//--------------------------------------------------------------------
int FunctionCancellationAndObserver::progressMinimum() const
{
    Q_D(const FunctionCancellationAndObserver);
    return d->m_progressMinimum;
}

//--------------------------------------------------------------------
int FunctionCancellationAndObserver::progressMaximum() const
{
    Q_D(const FunctionCancellationAndObserver);
    return d->m_progressMaximum;
}

//--------------------------------------------------------------------
void FunctionCancellationAndObserver::setProgressValue(int value)
{
    Q_D(FunctionCancellationAndObserver);
    d->m_progressValue = qBound(d->m_progressMinimum, value, d->m_progressMaximum);
    emit progressValueChanged(value);
}

//--------------------------------------------------------------------
int FunctionCancellationAndObserver::progressValue() const
{
    Q_D(const FunctionCancellationAndObserver);
    return d->m_progressValue;
}

//--------------------------------------------------------------------
void FunctionCancellationAndObserver::setProgressText(const QString &text)
{
    Q_D(FunctionCancellationAndObserver);
    QMutexLocker locker(&(d->m_readWriteMutex));
    d->m_progressText = text;
    emit progressTextChanged(text);
}

//--------------------------------------------------------------------
QString FunctionCancellationAndObserver::progressText()
{
    Q_D(FunctionCancellationAndObserver);
    QMutexLocker locker(&(d->m_readWriteMutex));
    return d->m_progressText;
}

} //end namespace ito
