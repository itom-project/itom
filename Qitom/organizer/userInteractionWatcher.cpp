/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

UserInteractionWatcher::UserInteractionWatcher(QWidget *plotWidget, int maxNrOfPoints, QSharedPointer<ito::DataObject> coords, ItomSharedSemaphore *semaphore, QObject *parent) :
    QObject(parent), 
    m_pPlotWidget(plotWidget), 
    m_pSemaphore(semaphore), 
    m_maxNrOfPoints(maxNrOfPoints), 
    m_coords(coords),
    m_waiting(true)
{
    connect(m_pPlotWidget, SIGNAL(destroyed(QObject*)), this, SLOT(plotWidgetDestroyed(QObject*)));

    if (coords.data() == NULL)
    {
        if (m_pSemaphore)
        {
            m_pSemaphore->returnValue += ito::RetVal(ito::retError,0,"The given data object is NULL.");
            m_pSemaphore->release();
            m_pSemaphore->deleteSemaphore();
            m_pSemaphore = NULL;
        }
        emit finished();
        return;
    }
        
    if (!connect(m_pPlotWidget, SIGNAL(userInteractionDone(int,bool,QPolygonF)), this, SLOT(userInteractionDone(int,bool,QPolygonF))) )
    {
        if (m_pSemaphore)
        {
            m_pSemaphore->returnValue += ito::RetVal(ito::retError,0,"The given widget does not have the necessary signals and slots for a user interaction.");
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
            m_pSemaphore->returnValue += ito::RetVal(ito::retError,0,"The given widget does not have the necessary signals and slots for a user interaction.");
            m_pSemaphore->release();
            m_pSemaphore->deleteSemaphore();
            m_pSemaphore = NULL;
        }
        emit finished();
    }
    else
    {
        emit userInteractionStart(1,true,m_maxNrOfPoints);
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
void UserInteractionWatcher::plotWidgetDestroyed(QObject *obj)
{
    if (m_pSemaphore)
    {
        if (m_waiting)
        {
            m_pSemaphore->returnValue += ito::RetVal(ito::retError,0,"User interaction terminated due to deletion of plot.");
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
void UserInteractionWatcher::userInteractionDone(int type, bool aborted, QPolygonF points)
{
    if (type == 1)
    {
        m_waiting = false;

        if (aborted) points.clear();

        int dims = 2; //m_dObjPtr ? m_dObjPtr->getDims() : 2;
        ito::DataObject output(dims, points.size(), ito::tFloat64);

        ito::float64 *ptr = (ito::float64*)output.rowPtr(0,0);
        int stride = points.size();

        for (int i = 0; i < points.size(); ++i)
        {
            ptr[i] = points[i].rx();
            ptr[i + stride] = points[i].ry();
        }

        *m_coords = output;

        if (m_pSemaphore)
        {
            if (aborted)
            {
                m_pSemaphore->returnValue += ito::RetVal(ito::retError,0,"User interaction aborted.");
            }
            m_pSemaphore->release();
            m_pSemaphore->deleteSemaphore();
            m_pSemaphore = NULL;
        }

        emit finished();

    }
}




} //end namespace ito