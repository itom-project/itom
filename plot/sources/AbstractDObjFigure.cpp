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

#include "../AbstractDObjFigure.h"
#include "../common/apiFunctionsGraphInc.h"

#include <qmetaobject.h>
#include <iostream>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
AbstractDObjFigure::AbstractDObjFigure(const QString &itomSettingsFile, AbstractFigure::WindowMode windowMode /*= AbstractFigure::ModeStandaloneInUi*/, QWidget *parent /*= 0*/) :
    AbstractFigure(itomSettingsFile, windowMode, parent),
    m_cameraConnected(false),
    m_currentDisplayedCameraChannel("")
{
    addInputParam(new ito::Param("source", ito::ParamBase::DObjPtr, NULL, QObject::tr("Source data for plot").toLatin1().data()));
    addInputParam(new ito::Param("liveSource", ito::ParamBase::HWRef, NULL, QObject::tr("Live data source for plot").toLatin1().data()));
    addOutputParam(new ito::Param("displayed", ito::ParamBase::DObjPtr, NULL, QObject::tr("Actual output data of plot").toLatin1().data()));
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractDObjFigure::~AbstractDObjFigure()
{
    removeLiveSource();
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractDObjFigure::update(void)
{
    ito::RetVal retval = ito::retOk;

    //!> do the real update work, here the transformation from source to displayed takes place
    retval += applyUpdate();

    //!> input data object is different from output data object so must cache it
    const ito::DataObject *newDisplayed = getOutputParam("displayed")->getVal<const ito::DataObject*>();

    if (m_dataPointer.contains("displayed") && newDisplayed == m_dataPointer["displayed"].data())
    {
        //contents remains the same
    }
    else if (newDisplayed == getInputParam("source")->getVal<const ito::DataObject*>())
    {
        //displayed is the same than source, source is already cached. Therefore we don't need to cache displayed
        m_dataPointer["displayed"].clear();
    }
    else
    {
        m_dataPointer["displayed"] = QSharedPointer<ito::DataObject>(new ito::DataObject(*newDisplayed));
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
QSharedPointer<ito::DataObject> AbstractDObjFigure::getAxisData(Qt::Axis axis) const
{
    if (axis == Qt::XAxis)
    {
        const ito::DataObject *dobj = getInputParam("xData")->getVal<const ito::DataObject*>();
        if (dobj)
        {
            return QSharedPointer<ito::DataObject>(new ito::DataObject(*dobj));
        }
    }

    return QSharedPointer<ito::DataObject>();
}

//----------------------------------------------------------------------------------------------------------------------------------
QSharedPointer<ito::DataObject> AbstractDObjFigure::getSource(void) const 
{
    const ito::DataObject *dObj = getInputParam("source")->getVal<const ito::DataObject*>();
    if (dObj)
    {
        return QSharedPointer<ito::DataObject>(new ito::DataObject(*dObj)); 
    }
    return QSharedPointer<ito::DataObject>();
}
//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractDObjFigure::setAxisData(QSharedPointer<ito::DataObject> data, Qt::Axis axis)
{
    ito::RetVal retval = ito::retOk;
    if (axis == Qt::XAxis)
    {
        if (m_dataPointer.contains("xData"))
        {

            if (m_dataPointer["xData"].data() != data.data())
            {
                QSharedPointer<ito::DataObject> oldSource = m_dataPointer["xData"]; //possible backup for previous xData, this backup must be alive until updateParam with the new one has been completely propagated
                m_dataPointer["xData"] = data;
            }
        }
        else
        {
            m_dataPointer["xData"] = data;
        }
        ito::ParamBase thisParam("xData", ito::ParamBase::DObjPtr, (const char*)data.data());
        retval += inputParamChanged(&thisParam);

        updatePropertyDock();
    }
    else
    {
        qWarning() << "AbstractDObjFigure::setAxisObj(...) ... invalid axis number " << axis;
        std::cerr << "AbstractDObjFigure::setAxisObj(...) ... invalid axis number " << axis << "\n" << std::endl;
    }

    return retval;
}
//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractDObjFigure::setSource(QSharedPointer<ito::DataObject> source) 
{ 
    ito::RetVal retval = ito::retOk;
    
    if (m_cameraConnected)
    {
        retval += removeLiveSource(); //removes possibly existing live source
        m_cameraConnected = false;
    }

    if (m_dataPointer.contains("source"))
    {
        //check if pointer of shared incoming data object is different to pointer of previous data object
        //if so, free previous
        if (m_dataPointer["source"].data() != source.data())
        {
            QSharedPointer<ito::DataObject> oldSource = m_dataPointer["source"]; //possible backup for previous source, this backup must be alive until updateParam with the new one has been completely propagated

            // sometimes crash here when replacing the source
            m_dataPointer["source"] = source;
        }  
    }
    else
    {
        m_dataPointer["source"] = source;
    }
            
    ito::ParamBase thisParam("source", ito::ParamBase::DObjPtr, (const char*)source.data());
    retval += inputParamChanged(&thisParam);

    updatePropertyDock();
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractDObjFigure::setLinePlot(const double /*x0*/, const double /*y0*/, const double /*x1*/, const double /*y1*/, const int /*destID*/)
{
    return ito::RetVal(ito::retError, 0, tr("Function \'spawnLinePlot\' not supported from this plot widget").toLatin1().data());

}

//----------------------------------------------------------------------------------------------------------------------------------
QSharedPointer<ito::DataObject> AbstractDObjFigure::getDisplayed(void)
{
    const ito::Param *p = getOutputParam("displayed");
    const ito::DataObject *dObj = p ? p->getVal<const ito::DataObject*>() : NULL;

    if (dObj)
    {
        return QSharedPointer<ito::DataObject>(new ito::DataObject(*dObj)); 
    }

    return QSharedPointer<ito::DataObject>();
}

//----------------------------------------------------------------------------------------------------------------------------------
QPointer<ito::AddInDataIO> AbstractDObjFigure::getCamera(void) const
{
    ito::Param *liveSource = getInputParam("liveSource");

    if (liveSource && m_cameraConnected)
    {
        return QPointer<ito::AddInDataIO>((liveSource->getVal<ito::AddInDataIO*>()));
    }
    else
    {
        return QPointer<ito::AddInDataIO>();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractDObjFigure::setCamera(QPointer<ito::AddInDataIO> camera)
{
    ito::RetVal retval;
    ito::Param *liveSource = getInputParam("liveSource");

    if (camera && liveSource)
    {
        if (m_cameraConnected)
        {
            retval += removeLiveSource(); //removes existing live source
        }
        else
        {
            //delete current static dataObject, that recently has been displayed
            if (m_dataPointer.contains("source"))
            {
                ito::ParamBase thisParam("source", ito::ParamBase::DObjPtr, (const char*)NULL);
                retval += inputParamChanged(&thisParam);

                m_dataPointer["source"] = QSharedPointer<ito::DataObject>();
            }
        }

        m_cameraConnected = true;
        liveSource->setVal<ito::AddInDataIO*>(camera.data());

        retval += apiConnectLiveData(camera, this); //increments reference of AddInDataIO
        retval += apiStartLiveData(camera, this);

        if (retval.containsError())
        {
            std::cerr << "Error while starting the live image.\n" << std::endl;
            if (retval.hasErrorMessage())
            {
                std::cerr << retval.errorMessage() << "\n" << std::endl;
            }
        }
        else if (retval.containsWarning())
        {
            std::cout << "Warning while starting the live image.\n" << std::endl;
            if (retval.hasErrorMessage())
            {
                std::cerr << retval.errorMessage() << "\n" << std::endl;
            }
        }
    }
    else if (!camera && m_cameraConnected)
    {
        retval += removeLiveSource(); //removes existing live source

        if (retval.containsError())
        {
            std::cerr << "Error while disconnecting the live image.\n" << std::endl;
            if (retval.hasErrorMessage())
            {
                std::cerr << retval.errorMessage() << "\n" << std::endl;
            }
        }
        else if (retval.containsWarning())
        {
            std::cout << "Warning while disconnecting the live image.\n" << std::endl;
            if (retval.hasErrorMessage())
            {
                std::cerr << retval.errorMessage() << "\n" << std::endl;
            }
        }
    }

    updatePropertyDock();
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
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
                m_dataPointer["source"] = source;
            }
        }
        else
        {
            m_dataPointer["source"] = source;
        }
            
        ito::ParamBase thisParam("source", ito::ParamBase::DObjPtr, (const char*)source.data());
        retval += inputParamChanged(&thisParam);
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
        waitCond->deleteSemaphore();
        waitCond = NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractDObjFigure::removeLiveSource()
{
    RetVal retval;
    ito::Param *liveSource = getInputParam("liveSource");

    if (liveSource)
    {
        ito::AddInDataIO* source = (liveSource->getVal<ito::AddInDataIO*>());

        if (source)
        {
            retval += apiStopLiveData(source, this);
            retval += apiDisconnectLiveData(source, this);
        }

        liveSource->setVal<ito::AddInDataIO*>(NULL);
        m_cameraConnected = false;
    }
    else
    {
        retval += RetVal(retWarning, 0, tr("Figure does not contain an input slot for live sources").toLatin1().data());
    }
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::AutoInterval AbstractDObjFigure::getXAxisInterval(void) const 
{ 
    return ito::AutoInterval(); 
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjFigure::setXAxisInterval(ito::AutoInterval) 
{ 
    return; 
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::AutoInterval AbstractDObjFigure::getYAxisInterval(void) const 
{ 
    return ito::AutoInterval(); 
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjFigure::setYAxisInterval(ito::AutoInterval) 
{ 
    return; 
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::AutoInterval AbstractDObjFigure::getZAxisInterval(void) const 
{ 
    return ito::AutoInterval(); 
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjFigure::setZAxisInterval(ito::AutoInterval) 
{ 
    return; 
}

//----------------------------------------------------------------------------------------------------------------------------------
QString AbstractDObjFigure::getColorMap(void) const 
{ 
    return QString(); 
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDObjFigure::setColorMap(QString) 
{ 
    return; 
}

//----------------------------------------------------------------------------------------------------------------------------------
//! plot-specific render function to enable more complex printing in subfigures ...
QPixmap AbstractDObjFigure::renderToPixMap(const int xsize, const int ysize, const int resolution)
{
    QPixmap emptyMap(xsize, ysize);
    emptyMap.fill(Qt::green);
    return emptyMap;
}
//-------------------------------------------------------------------------------------
ito::RetVal AbstractDObjFigure::setDisplayedCameraChannel(const QString& channel)
{
    ito::RetVal retValue(ito::retOk);
    QPointer<ito::AddInDataIO> liveSource = getCamera();
    if (liveSource)
    {
        bool isMultiChannel = false;
        isMultiChannel = liveSource->inherits("ito::AddInMultiChannelGrabber");
        if (isMultiChannel)
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QSharedPointer<ito::Param> channelListParam(new ito::Param("channelList", ito::ParamBase::StringList));
            if (QMetaObject::invokeMethod(liveSource, "getParam", Q_ARG(QSharedPointer<ito::Param>, channelListParam), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
            {
                bool timeout = false;

                while (!locker.getSemaphore()->wait(500))
                {
                    retValue += ito::RetVal(ito::retError, 0, tr("timeout while getting channelList parameter").toLatin1().data());
                    break;
                }
                if (!retValue.containsError())
                {
                    int len = 0;
                    const ito::ByteArray* channelList = channelListParam->getVal<const ito::ByteArray*>(len);
                    bool found = false;
                    for (int i = 0; i < len; i++)
                    {
                        if (QString(channelList[i].data()).compare(channel) == 0)
                        {
                            found = true;
                            m_currentDisplayedCameraChannel = channel;
                            emit cameraChannelChanged(channel, this);
                            break;
                        }
                    }
                    if (!found)
                    {
                        retValue += ito::RetVal(ito::retError, 0, tr("the connected camera does not contain a channel called %1.").arg(channel).toLatin1().data());
                    }
                }
            }
        }
        else
        {
            retValue +=ito::RetVal(ito::retWarning, 0, tr("could not set a camera channel, since the connected camera is not of type \"AddInMultiChannelGrabber\".").toLatin1().data());
            m_currentDisplayedCameraChannel = "";
        }


    }
    
    return retValue;
}
//-------------------------------------------------------------------------------------
QString AbstractDObjFigure::getDisplayedCameraChannel() const
{
    return m_currentDisplayedCameraChannel;
}

} //end namespace ito
