/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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

#include "../AbstractDObjPclFigure.h"

#include <qmetaobject.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractDObjPclFigure::update(void)
{
    ito::RetVal retval = ito::retOk;

    //!> do the real update work, here the transformation from source to displayed takes place
    retval += applyUpdate();

    //!> input data object is different from output data object so must cache it
/*
    if (m_inpType == ito::ParamBase::PointCloudPtr)
    {
        ito::PCLPointCloud *newDisplayed = (ito::PCLPointCloud*)(m_pOutput["displayed"]->getVal<void*>());

        if (m_dataPointerPC.contains("displayed") && newDisplayed == m_dataPointerPC["displayed"].data())
        {
            //contents remains the same
        }
        else if (newDisplayed == (ito::PCLPointCloud*)m_pInput["source"]->getVal<void*>())
        {
            //displayed is the same than source, source is already cached. Therefore we don't need to cache displayed
            m_dataPointerPC["displayed"].clear();
        }
        else
        {
            m_dataPointerPC["displayed"] = QSharedPointer<ito::PCLPointCloud>(new ito::PCLPointCloud(*newDisplayed));
        }
    }
    else if (m_inpType == ito::ParamBase::PolygonMeshPtr)
    {
        ito::PCLPolygonMesh *newDisplayed = (ito::PCLPolygonMesh*)(m_pOutput["displayed"]->getVal<void*>());

        if (m_dataPointerPM.contains("displayed") && newDisplayed == m_dataPointerPM["displayed"].data())
        {
            //contents remains the same
        }
        else if (newDisplayed == (ito::PCLPolygonMesh*)m_pInput["source"]->getVal<void*>())
        {
            //displayed is the same than source, source is already cached. Therefore we don't need to cache displayed
            m_dataPointerPM["displayed"].clear();
        }
        else
        {
            m_dataPointerPM["displayed"] = QSharedPointer<ito::PCLPolygonMesh>(new ito::PCLPolygonMesh(*newDisplayed));
        }    
    }
    else if (m_inpType == ito::ParamBase::DObjPtr)
    {
        ito::DataObject *newDisplayed = (ito::DataObject*)(m_pOutput["displayed"]->getVal<void*>());

        if (m_dataPointerDObj.contains("displayed") && newDisplayed == m_dataPointerDObj["displayed"].data())
        {
            //contents remains the same
        }
        else if (newDisplayed == (ito::DataObject*)m_pInput["source"]->getVal<void*>())
        {
            //displayed is the same than source, source is already cached. Therefore we don't need to cache displayed
            m_dataPointerDObj["displayed"].clear();
        }
        else
        {
            m_dataPointerDObj["displayed"] = QSharedPointer<ito::DataObject>(new ito::DataObject(*newDisplayed));
        }
    }
*/    
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
QSharedPointer<ito::DataObject> AbstractDObjPclFigure::getDataObject(void) const 
{
    ito::DataObject *dObj = m_pInput["dataObject"]->getVal<ito::DataObject*>();
    if (dObj)
    {
        return QSharedPointer<ito::DataObject>(new ito::DataObject(*dObj)); 
    }
    return QSharedPointer<ito::DataObject>();
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjPclFigure::setDataObject(QSharedPointer<ito::DataObject> source) 
{ 
    ito::RetVal retval = ito::retOk;
    QSharedPointer<ito::DataObject> oldSource; //possible backup for previous source, this backup must be alive until updateParam with the new one has been completely propagated

    if (m_dataPointerDObj.contains("dataObject"))
    {
        //check if pointer of shared incoming data object is different to pointer of previous data object
        //if so, free previous
        if (m_dataPointerDObj["dataObject"].data() != source.data())
        {
            oldSource = m_dataPointerDObj["dataObject"];
            if (oldSource)
                oldSource->lockWrite();
            // sometimes crash here when replacing the source
            m_dataPointerDObj["dataObject"] = source;
            if (oldSource)
                oldSource->unlock();
        }  
    }
    else
    {
        m_dataPointerDObj["dataObject"] = source;
    }
            
    ito::ParamBase thisParam("dataObject", ito::ParamBase::DObjPtr, (const char*)source.data());
    m_inpType = ito::ParamBase::DObjPtr;
    retval += updateParam(&thisParam, 1);

    updatePropertyDock();
}

#ifdef USEPCL
//----------------------------------------------------------------------------------------------------------------------------------
QSharedPointer<ito::PCLPointCloud> AbstractDObjPclFigure::getPointCloud(void) const 
{
    ito::PCLPointCloud *pc = m_pInput["pointCloud"]->getVal<ito::PCLPointCloud*>();
    if (pc)
    {
        return QSharedPointer<ito::PCLPointCloud>(new ito::PCLPointCloud(*pc)); 
    }
    return QSharedPointer<ito::PCLPointCloud>();
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjPclFigure::setPointCloud(QSharedPointer<ito::PCLPointCloud> source) 
{ 
    ito::RetVal retval = ito::retOk;
    QSharedPointer<ito::PCLPointCloud> oldSource; //possible backup for previous source, this backup must be alive until updateParam with the new one has been completely propagated

    if (m_dataPointerDObj.contains("pointCloud"))
    {
        //check if pointer of shared incoming data object is different to pointer of previous data object
        //if so, free previous
        if (m_dataPointerPC["pointCloud"].data() != source.data())
        {
            oldSource = m_dataPointerPC["pointCloud"];
//            if (oldSource)
//                oldSource->lockWrite();

            // sometimes crash here when replacing the source
            m_dataPointerPC["pointCloud"] = source;
//            if (oldSource)
//                oldSource->unlock();
        }  
    }
    else
    {
        m_dataPointerPC["pointCloud"] = source;
    }
            
    ito::ParamBase thisParam("pointCloud", ito::ParamBase::PointCloudPtr, (const char*)source.data());
    m_inpType = ito::ParamBase::PointCloudPtr;
    retval += updateParam(&thisParam, 1);

    updatePropertyDock();
}

//----------------------------------------------------------------------------------------------------------------------------------
QSharedPointer<ito::PCLPolygonMesh> AbstractDObjPclFigure::getPolygonMesh(void) const 
{
    ito::PCLPolygonMesh *pm = m_pInput["polgonMesh"]->getVal<ito::PCLPolygonMesh*>();
    if (pm)
    {
        return QSharedPointer<ito::PCLPolygonMesh>(new ito::PCLPolygonMesh(*pm)); 
    }
    return QSharedPointer<ito::PCLPolygonMesh>();
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjPclFigure::setPolygonMesh(QSharedPointer<ito::PCLPolygonMesh> source) 
{ 
    ito::RetVal retval = ito::retOk;
    QSharedPointer<ito::PCLPolygonMesh> oldSource; //possible backup for previous source, this backup must be alive until updateParam with the new one has been completely propagated

    if (m_dataPointerPM.contains("polgonMesh"))
    {
        //check if pointer of shared incoming data object is different to pointer of previous data object
        //if so, free previous
        if (m_dataPointerPM["polgonMesh"].data() != source.data())
        {
            oldSource = m_dataPointerPM["polgonMesh"];
//            if (oldSource)
//                oldSource->lockWrite();

            // sometimes crash here when replacing the source
            m_dataPointerPM["polgonMesh"] = source;
//            if (oldSource)
//                oldSource->unlock();

        }  
    }
    else
    {
        m_dataPointerPM["polgonMesh"] = source;
    }
            
    ito::ParamBase thisParam("polgonMesh", ito::ParamBase::PolygonMeshPtr, (const char*)source.data());
    m_inpType = ito::ParamBase::PolygonMeshPtr;
    retval += updateParam(&thisParam, 1);

    updatePropertyDock();
}

#endif // USEPCL
//----------------------------------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractDObjPclFigure::setLinePlot(const double /*x0*/, const double /*y0*/, const double /*x1*/, const double /*y1*/, const int /*destID*/)
{
    return ito::RetVal(ito::retError, 0, tr("Function \'spawnLinePlot\' not supported from this plot widget").toLatin1().data());

}

//----------------------------------------------------------------------------------------------------------------------------------
/*
//this source is invoked by any connected camera
void AbstractDObjFigure::setSource(QSharedPointer<ito::DataObject> source, ItomSharedSemaphore *waitCond) 
{ 
    ito::RetVal retval = ito::retOk;

    if (m_cameraConnected)
    {
        if (m_dataPointer.contains("source"))
        {
            //check if pointer of shared incoming data object is different to pointer of previous data object
            //if so, free previous
            if (m_dataPointer["source"].data() != source.data())
            {
                QSharedPointer<ito::DataObject> oldSource = m_dataPointer["source"];
                if (oldSource.data())
                {
                    oldSource->lockWrite();
                    m_dataPointer["source"] = source;
                    oldSource->unlock();
                }
                else
                {
                    m_dataPointer["source"] = source;
                }
            }
        }
        else
        {
            m_dataPointer["source"] = source;
        }
            
        ito::ParamBase thisParam("source", ito::ParamBase::DObjPtr, (const char*)source.data());
        retval += updateParam(&thisParam, 1);
    }

    if (waitCond)
    {
        waitCond->release();
        waitCond->deleteSemaphore();
        waitCond = NULL;
    }
}
*/

} //end namespace ito
