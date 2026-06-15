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

#include "../common/sharedStructures.h"

#include <qabstractitemmodel.h>
#include <qlist.h>
#include <qicon.h>
#include <qstring.h>
#include <qobject.h>
#include <qsharedpointer.h>
#include <qtimer.h>
#include <qevent.h>
#include <qlist.h>


namespace ito
{

class TimerModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    TimerModel();
    ~TimerModel();

    QVariant data(const QModelIndex &index, int role) const;
    QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
    QModelIndex parent(const QModelIndex &index) const;
    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;

    void registerNewTimer(const QWeakPointer<QTimer>& timer, const QString &name);
    void updateTimerData();
    void autoUpdateModel(bool enabled);

    void timerStart(const QModelIndex &index);
    void timerStop(const QModelIndex &index);
    void timerStopAll();

protected:
    QVariant headerData(int section, Qt::Orientation orientation, int role) const;

    void timerEvent(QTimerEvent *ev);

private:
    //! item of TimerModel
    /*!
        this struct corresponds to one item in the TimerModel
    */
    struct TimerItem
    {
        TimerItem() : started(false), interval(0), singleShot(false), timerId(-1) {}
        QWeakPointer<QTimer> timer;
        QString name; //!< optional name of the name, can also be an empty string
        bool started; //!< cache value
        int interval; //!< cache value
        bool singleShot; //!< cache value
        int timerId; //!< cache value
    };

    bool cacheItem(TimerItem &item);

    /*!<  list of timers (TimerItem) which are currently available in this application */
    QList<TimerItem> m_timers;
    int m_timerId;
    int m_enableCount;
    QIcon m_iconRunning;
    QIcon m_iconStopped;
    QIcon m_iconUnknown;

private Q_SLOTS:
    void timerDestroyed(QObject *timer);
};

} //end namespace ito
