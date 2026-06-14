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

#include "userInteractionWatcher.h"
#include <qwidget.h>

namespace ito
{

    UserInteractionWatcher::UserInteractionWatcher(QWidget *plotWidget, ito::Shape::ShapeType type, int maxNrOfPoints, QSharedPointer<QVector<ito::Shape> > shapes, ItomSharedSemaphore *semaphore, QObject *parent) :
    QObject(parent),
    m_pPlotWidget(plotWidget),
    m_pSemaphore(semaphore),
    m_maxNrOfPoints(maxNrOfPoints),
    m_shapes(shapes),
    m_waiting(true)
{
    connect(m_pPlotWidget, SIGNAL(destroyed(QObject*)), this, SLOT(plotWidgetDestroyed(QObject*)));

    if (shapes.data() == NULL)
    {
        if (m_pSemaphore)
        {
            m_pSemaphore->returnValue += ito::RetVal(ito::retError, 0, tr("The given shape storage is NULL.").toLatin1().data());
            m_pSemaphore->release();
            m_pSemaphore->deleteSemaphore();
            m_pSemaphore = NULL;
        }
        emit finished();
        return;
    }

    if (maxNrOfPoints < -1 || maxNrOfPoints == 0)
    {
        if (m_pSemaphore)
        {
            m_pSemaphore->returnValue += ito::RetVal(ito::retError, 0, tr("The maximum number of points must be -1 (infinite) or >= 1.").toLatin1().data());
            m_pSemaphore->release();
            m_pSemaphore->deleteSemaphore();
            m_pSemaphore = NULL;
        }
        emit finished();
        return;
    }

    if (!connect(m_pPlotWidget, SIGNAL(userInteractionDone(int, bool, QVector<ito::Shape>)), this, SLOT(userInteractionDone(int, bool, QVector<ito::Shape>))))
    {
        if (m_pSemaphore)
        {
            m_pSemaphore->returnValue += ito::RetVal(ito::retError, 0, tr("The given widget does not have the necessary signals and slots for a user interaction.").toLatin1().data());
            m_pSemaphore->release();
            m_pSemaphore->deleteSemaphore();
            m_pSemaphore = NULL;
        }
        emit finished();
    }
    else if (!connect(this, SIGNAL(userInteractionStart(int,bool,int)), m_pPlotWidget, SLOT(userInteractionStart(int,bool,int))) )
    {
        if (m_pSemaphore)
        {
            m_pSemaphore->returnValue += ito::RetVal(ito::retError, 0, tr("The given widget does not have the necessary signals and slots for a user interaction.").toLatin1().data());
            m_pSemaphore->release();
            m_pSemaphore->deleteSemaphore();
            m_pSemaphore = NULL;
        }
        emit finished();
    }
    else
    {
        emit userInteractionStart(type, true, m_maxNrOfPoints);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
UserInteractionWatcher::~UserInteractionWatcher()
{
    if (m_pSemaphore)
    {
        m_pSemaphore->release();
        m_pSemaphore->deleteSemaphore();
        m_pSemaphore = NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param obj
*/
void UserInteractionWatcher::plotWidgetDestroyed(QObject *obj)
{
    if (m_pSemaphore)
    {
        if (m_waiting)
        {
            m_pSemaphore->returnValue += ito::RetVal(ito::retError, 0, tr("User interaction terminated due to deletion of plot.").toLatin1().data());
        }
        m_pSemaphore->release();
        m_pSemaphore->deleteSemaphore();
        m_pSemaphore = NULL;
    }

    if (m_waiting)
    {
        emit finished();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param type
    \param aborted
    \param points
*/
void UserInteractionWatcher::userInteractionDone(int type, bool aborted, QVector<ito::Shape> shapes)
{
    int dims = 2; //m_dObjPtr ? m_dObjPtr->getDims() : 2;
    m_waiting = false;

    if (aborted)
    {
        m_shapes->clear();
    }
    else
    {
        *m_shapes = shapes;
    }

    if (m_pSemaphore)
    {
        if (aborted)
        {
            m_pSemaphore->returnValue += ito::RetVal(ito::retError, 0, tr("User interaction aborted.").toLatin1().data());
        }
        m_pSemaphore->release();
        m_pSemaphore->deleteSemaphore();
        m_pSemaphore = NULL;
    }

    emit finished();
}

} //end namespace ito
