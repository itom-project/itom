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

#ifndef USERINTERACTIONWATCHER_H
#define USERINTERACTIONWATCHER_H

#include <qobject.h>

#include "../global.h"
#include "common/sharedStructuresQt.h"
#include "DataObject/dataobj.h"
#include "common/shape.h"

#include <qpolygon.h>
#include <qvector.h>
#include <qsharedpointer.h>

namespace ito
{

class UserInteractionWatcher : public QObject
{
    Q_OBJECT

public:
    UserInteractionWatcher(QWidget *plotWidget, ito::Shape::ShapeType type, int maxNrOfPoints, QSharedPointer<QVector<ito::Shape> > shapes, ItomSharedSemaphore *semaphore, QObject *parent = 0); //constructor
    virtual ~UserInteractionWatcher(); //destructor

private:
    const QWidget *m_pPlotWidget;
    ItomSharedSemaphore *m_pSemaphore;
    int m_maxNrOfPoints;
    bool m_waiting;
    QSharedPointer<QVector<ito::Shape> > m_shapes;

private slots:
    void plotWidgetDestroyed(QObject *obj);
    void userInteractionDone(int type, bool aborted, QVector<ito::Shape> shapes);

    public slots:


signals:
    void finished();
    void userInteractionStart(int type, bool start, int maxNrOfPoints);

};

} //end namespace ito

#endif
