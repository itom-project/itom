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

#include "timerModel.h"

#include <qthread.h>

namespace ito
{

/*!
    \class TimerModel
    \brief model for management of all timer objects.
    This model will be is used as model for the view in the timer manager.
*/

//-------------------------------------------------------------------------------------
//! constructor
/*!
    initializes headers and its alignment
*/
TimerModel::TimerModel() :
    QAbstractItemModel(),
    m_iconRunning(QIcon(":/application/icons/timerRun.png")),
    m_iconStopped(QIcon(":/application/icons/timerStop.png")),
    m_timerId(-1),
    m_enableCount(0)
{
}

//-------------------------------------------------------------------------------------
//! destructor
TimerModel::~TimerModel()
{
}

//-------------------------------------------------------------------------------------
//! counts number of bookmarks in this model
/*!
    \return number of elements
*/
int TimerModel::rowCount(const QModelIndex &parent) const
{
    return m_timers.count();
}

//-------------------------------------------------------------------------------------
//! counts number of columns in this model (corresponds to number of header-elements)
/*!
    \return number of columns
*/
int TimerModel::columnCount(const QModelIndex &parent) const
{
    return 1;
}

//-------------------------------------------------------------------------------------
//! overwritten data method of QAbstractItemModel
/*!
    data method will be called by View-Widget in order to fill the table.

    \param index QModelIndex of item, whose content should be returned
    \return content of desired item and column
*/
QVariant TimerModel::data(const QModelIndex &index, int role) const
{
    if(!index.isValid() || index.row() >= m_timers.count())
    {
        return QVariant();
    }

    const TimerItem& item = m_timers[index.row()];
    auto strongTimer = item.timer.toStrongRef();

    if (!strongTimer)
    {
        return QVariant();
    }

    if (role == Qt::DisplayRole)
    {
        QString name;

        if (item.name == "")
        {
            name = tr("Timer ID: %1").arg(item.timerId);
        }
        else
        {
            name = tr("%1; ID: %2").arg(item.name).arg(item.timerId);
        }

        if (item.singleShot)
        {
            return tr("%1; interval: %2 ms; single shot").
                arg(name).
                arg(item.interval);
        }
        else
        {
            return tr("%1; interval: %2 ms").
                arg(name).
                arg(item.interval);
        }

    }
    else if (role == Qt::DecorationRole)
    {
        return item.started ? m_iconRunning : m_iconStopped;
    }
    else if (role == Qt::TextAlignmentRole)
    {
        return QVariant(Qt::Alignment(Qt::AlignLeft | Qt::AlignVCenter));
    }
    else if (role == Qt::UserRole)
    {
        return item.started;
    }

    return QVariant();
}

//-------------------------------------------------------------------------------------
//! returns QModelIndex for given row and column
/*!
    \param row row of desired entry, corresponds to index in m_bookmarks list
    \param column column of desired entry
    \param parent since this model is no tree model, parent always points to a
        "virtual" root element
    \return empty QModelIndex if row or column are out of bound, else returns new
        valid QModelIndex for that combination of row and column
*/
QModelIndex TimerModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!parent.isValid()) //root item
    {
        if (row < 0 || column < 0 || column >= 1 || row >= m_timers.count())
        {
            return QModelIndex();
        }
        else
        {
            return createIndex(row, column, nullptr);
        }
    }

    return QModelIndex();
}

//-------------------------------------------------------------------------------------
//! returns parent of given QModelIndex
/*!
    since this model is not a tree model, returns always an empty QModelIndex
*/
QModelIndex TimerModel::parent(const QModelIndex &index) const
{
    return QModelIndex();
}

//-------------------------------------------------------------------------------------
//! returns header element at given position
/*!
    \param section position in m_headers list
    \param orientation the model's orientation should be horizontal, no other orientation is supported
    \param role model is only prepared for DisplayRole
    \return name of header or empty QVariant value (if no header element available)
*/
QVariant TimerModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if( role == Qt::DisplayRole && orientation == Qt::Horizontal)
    {
        if(section == 0)
        {
            return tr("Timer");
        }
    }

    return QVariant();
}

//-------------------------------------------------------------------------------------
void TimerModel::registerNewTimer(const QWeakPointer<QTimer>& timer, const QString &name)
{
    QSharedPointer<QTimer> strongTimer = timer.toStrongRef();

    if (strongTimer)
    {
        TimerItem item;
        item.name = name;
        item.timer = timer;
        cacheItem(item);
        connect(timer.toStrongRef().data(), &QTimer::destroyed, this, &TimerModel::timerDestroyed);

        beginInsertRows(QModelIndex(), m_timers.count(), m_timers.count());
        m_timers.append(item);
        endInsertRows();
    }
}

//-------------------------------------------------------------------------------------
void TimerModel::timerDestroyed(QObject *timer)
{
    for (int idx = m_timers.size() - 1; idx >= 0; --idx)
    {
        if (m_timers[idx].timer.isNull() ||
            m_timers[idx].timer.toStrongRef().data() == timer)
        {
            beginRemoveRows(QModelIndex(), idx, idx);
            m_timers.removeAt(idx);
            endRemoveRows();
        }
    }
}

//-------------------------------------------------------------------------------------
void TimerModel::updateTimerData()
{
    for (int idx = 0; idx < m_timers.size(); ++idx)
    {
        if (cacheItem(m_timers[idx]))
        {
            emit dataChanged(createIndex(idx, 0), createIndex(idx, 0));
        }
    }
}

//-------------------------------------------------------------------------------------
void TimerModel::autoUpdateModel(bool enabled)
{
    m_enableCount += enabled ? 1 : -1;

    if (m_enableCount > 0 && m_timerId == -1)
    {
        m_timerId = startTimer(250);
    }
    else if (m_enableCount == 0 && m_timerId >= 0)
    {
        killTimer(m_timerId);
        m_timerId = -1;
    }
}

//-------------------------------------------------------------------------------------
void TimerModel::timerEvent(QTimerEvent *ev)
{
    updateTimerData();
}

//-------------------------------------------------------------------------------------
/*
    \return true if the cache was changed, else false
*/
bool TimerModel::cacheItem(TimerItem &item)
{
    auto strongTimer = item.timer.toStrongRef();

    if (strongTimer)
    {
        bool changed = false;
        bool started = strongTimer->isActive();
        int interval = strongTimer->interval();
        bool singleShot = strongTimer->isSingleShot();
        int timerId = strongTimer->timerId();

        if (item.interval != interval)
        {
            item.interval = interval;
            changed = true;
        }

        if (item.singleShot != singleShot)
        {
            item.singleShot = singleShot;
            changed = true;
        }

        if (item.started != started)
        {
            item.started = started;
            changed = true;
        }

        if (item.timerId != timerId)
        {
            item.timerId = timerId;
            changed = true;
        }

        return changed;
    }
    else
    {
        return true;
    }
}

//-------------------------------------------------------------------------------------
void TimerModel::timerStart(const QModelIndex &index)
{
    if (index.isValid() && index.row() >= 0 && index.row() < m_timers.count())
    {
        auto strongTimer = m_timers[index.row()].timer.toStrongRef();

        if (strongTimer)
        {
            QMetaObject::invokeMethod(strongTimer.data(), "start");

            QThread::msleep(50);
            updateTimerData();
        }
    }
}

//-------------------------------------------------------------------------------------
void TimerModel::timerStop(const QModelIndex &index)
{
    if (index.isValid() && index.row() >= 0 && index.row() < m_timers.count())
    {
        auto strongTimer = m_timers[index.row()].timer.toStrongRef();

        if (strongTimer)
        {
            QMetaObject::invokeMethod(strongTimer.data(), "stop");

            QThread::msleep(50);
            updateTimerData();
        }
    }
}

//-------------------------------------------------------------------------------------
void TimerModel::timerStopAll()
{
    foreach(auto item, m_timers)
    {
        auto strongTimer = item.timer.toStrongRef();

        if (strongTimer)
        {
            QMetaObject::invokeMethod(strongTimer.data(), "stop");
        }
    }

    QThread::msleep(50);
    updateTimerData();
}

} //end namespace ito
