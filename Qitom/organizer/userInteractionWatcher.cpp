/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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
#include "common/sharedStructuresPrimitives.h"
#include <qwidget.h>

namespace ito
{

UserInteractionWatcher::UserInteractionWatcher(QWidget *plotWidget, int geomtriecType, int maxNrOfPoints, QSharedPointer<ito::DataObject> coords, ItomSharedSemaphore *semaphore, QObject *parent) :
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
        emit userInteractionStart(geomtriecType, true, m_maxNrOfPoints);
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
//! shortdesc
/*! longdesc

    \param type
    \param aborted
    \param points
*/
void UserInteractionWatcher::userInteractionDone(int type, bool aborted, QPolygonF points)
{
    int dims = 2; //m_dObjPtr ? m_dObjPtr->getDims() : 2;
    m_waiting = false;

    if (aborted)
    {
        *m_coords = ito::DataObject();
    }
    else
    {
        switch (type & ito::PrimitiveContainer::tTypeMask)
        {
        case ito::PrimitiveContainer::tMultiPointPick:
        {
            //in case of multi-point a 2xN data object is returned. Each column is the x,y coordinate of the clicked point
            ito::DataObject output(2, points.size(), ito::tFloat64);
            cv::Mat *mat = output.getCvPlaneMat(0);
            ito::float64 *x_ptr = mat->ptr<ito::float64>(0);
            ito::float64 *y_ptr = mat->ptr<ito::float64>(1);

            for (int i = 0; i < points.size(); ++i)
            {
                x_ptr[i] = points[i].rx();
                y_ptr[i] = points[i].ry();
            }
            *m_coords = output;
            break;
        }
        case ito::PrimitiveContainer::tSquare:
        case ito::PrimitiveContainer::tCircle:
        case ito::PrimitiveContainer::tPolygon:
        default:
        {
            *m_coords = ito::DataObject();
            break;
        }
        case ito::PrimitiveContainer::tPoint:
        case ito::PrimitiveContainer::tLine:
        case ito::PrimitiveContainer::tRectangle:
        case ito::PrimitiveContainer::tEllipse:
        {
            dims = 8;
            int elementCount = (points.size() * 2) / dims;

            ito::DataObject output(elementCount, dims, ito::tFloat64);
            cv::Mat *mat = output.getCvPlaneMat(0);
            ito::float64 *ptr;

            for (int i = 0; i < elementCount; i++)
            {
                ptr = mat->ptr<ito::float64>(i);
                ptr[0] = points[4 * i].rx();      //idx
                ptr[1] = points[4 * i].ry();      //type
                ptr[2] = points[4 * i + 1].rx();      //x1
                ptr[3] = points[4 * i + 1].ry();      //y1
                ptr[4] = points[4 * i + 2].rx();      //x2
                ptr[5] = points[4 * i + 2].ry();      //y2
                ptr[6] = 0.0; //points[i + 3].rx();      //???
                ptr[7] = 0.0; //points[i + 3].ry();      //???
            }
            *m_coords = output;
            break;
        }
        break;
        }
    }

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




} //end namespace ito