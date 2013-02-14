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

#include "AbstractDObjFigure.h"

#include <qmetaobject.h>

using namespace ito;

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractDObjFigure::update(void)
{
    ito::RetVal retval = ito::retOk;
    ito::DataObject *oldDisplayed = NULL;

     //lock the displayed object to prevent ugly surprises for the children
    if (m_dataPointer.contains("displayed"))
    {
        oldDisplayed = (ito::DataObject*)m_dataPointer["displayed"].data();
        oldDisplayed->lockWrite();
    }

    //!> do the real update work, here the transformation from source to displayed takes place
    retval += applyUpdate();

    //!> input data object is different from output data object so must cache it
/*
    if (m_pOutput["displayed"]->getVal<const char*>() == NULL) 
        return ito::RetVal(ito::retError, 0, QObject::tr("displayed object is null in update").toAscii().data()); // TODO: add object name to error message
    if (m_pInput["source"]->getVal<const char*>() == NULL)
        return ito::RetVal(ito::retError, 0, QObject::tr("source object is null in update").toAscii().data());  // TODO: add object name to error message
*/
    if (m_pOutput["displayed"]->getVal<const char*>() != m_pInput["source"]->getVal<const char*>())
    {
        m_dataPointer["displayed"] = QSharedPointer<ito::DataObject>(new ito::DataObject((*(ito::DataObject*)(m_pOutput["displayed"]->getVal<const char*>()))));
    }

    if (oldDisplayed)
    {
        oldDisplayed->unlock();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjFigure::setSource(QSharedPointer<ito::DataObject> source) 
{ 
    ito::RetVal retval = ito::retOk;
    QSharedPointer<ito::DataObject> oldSource; //possible backup for previous source, this backup must be alive until updateParam with the new one has been completely propagated

    if(m_cameraConnected)
    {
        retval += removeLiveSource(); //removes possibly existing live source
        m_cameraConnected = false;
    }

    if(m_dataPointer.contains("source"))
    {
        //check if pointer of shared incoming data object is different to pointer of previous data object
        //if so, free previous
        if(m_dataPointer["source"].data() != source.data())
        {
            oldSource = m_dataPointer["source"];
            oldSource->lockWrite();
            m_dataPointer["source"] = source;
            oldSource->unlock();
        }  
    }
	else
    {
        m_dataPointer["source"] = source;
    }
            
    ito::ParamBase thisParam("source", ito::ParamBase::DObjPtr, (const char*)source.data());
    retval += updateParam(&thisParam, 1);

    return;
}

//----------------------------------------------------------------------------------------------------------------------------------
QPointer<ito::AddInDataIO> AbstractDObjFigure::getCamera(void)
{
    if(m_pInput.contains("liveSource") && m_cameraConnected)
    {
        return QPointer<ito::AddInDataIO>( (ito::AddInDataIO*)(m_pInput["liveSource"]->getVal<void*>() ) );
    }
    else
    {
        return QPointer<ito::AddInDataIO>();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjFigure::setCamera(QPointer<ito::AddInDataIO> camera)
{
    ito::RetVal retval;
    if(camera && m_pInput.contains("liveSource") )
    {
        ito::Param *param = m_pInput["liveSource"];

        if(m_cameraConnected)
        {
            retval += removeLiveSource(); //removes existing live source
        }
        else
        {
            //delete current static dataObject, that recently has been displayed
            if(m_dataPointer.contains("source"))
            {
                m_dataPointer["source"] = QSharedPointer<ito::DataObject>();
            }
        }

        m_cameraConnected = true;
        param->setVal<void*>( (void*)camera );

        retval += apiConnectLiveData(camera, this); //increments reference of AddInDataIO
        retval += apiStartLiveData(camera, this);
        //QMetaObject::invokeMethod(camera, "startDeviceAndRegisterListener", Q_ARG(QObject*, this), Q_ARG(ItomSharedSemaphore*, NULL));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//this source is invoked by any connected camera
void AbstractDObjFigure::setSource(QSharedPointer<ito::DataObject> source, ItomSharedSemaphore *waitCond) 
{ 
    ito::RetVal retval = ito::retOk;

    if(m_cameraConnected)
    {
        if(m_dataPointer.contains("source"))
        {
            //check if pointer of shared incoming data object is different to pointer of previous data object
            //if so, free previous
            if(m_dataPointer["source"].data() != source.data())
            {
                QSharedPointer<ito::DataObject> oldSource = m_dataPointer["source"];
                if(oldSource.data())
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

    if(waitCond)
    {
        waitCond->release();
        ItomSharedSemaphore::deleteSemaphore(waitCond);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractDObjFigure::removeLiveSource()
{
    RetVal retval;
    if(m_pInput.contains("liveSource"))
    {
        ito::Param *param = m_pInput["liveSource"];
        ito::AddInDataIO* source = (ito::AddInDataIO*)(param->getVal<void*>());
        if(source)
        {
            retval += apiStopLiveData(source, this);
            retval += apiDisconnectLiveData(source, this);
        }
        param->setVal<void*>(NULL);
    }
    else
    {
        retval += RetVal(retWarning,0,tr("Figure does not contain an input slot for live sources").toAscii().data() );
    }
    return retval;
}



} //end namespace ito
