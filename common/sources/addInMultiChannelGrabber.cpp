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

#include "AddInMultiChannelGrabber.h"

#include <qmetatype.h>
#include <qcoreapplication.h>
#include <qmetaobject.h>
#include <qmap.h>
#include <qlist.h>
#include "common/helperCommon.h"


namespace ito
{
    class AddInMultiChannelGrabberPrivate
    {
    };
    //----------------------------------------------------------------------------------------------------------------------------------
     //! constructor
    AddInMultiChannelGrabber::AddInMultiChannelGrabber(const QByteArray &grabberName) :
        AddInAbstractGrabber(),
        m_defaultConfigReady(false)
    {
        dd = new AddInMultiChannelGrabberPrivate();

        ito::Param paramVal("name", ito::ParamBase::String | ito::ParamBase::Readonly, grabberName.data(), "GrabberName");
        paramVal.setMeta(new ito::StringMeta(ito::StringMeta::String, "General"), true);
        m_params.insert(paramVal.getName(), paramVal);

        paramVal = ito::Param("defaultChannel", ito::ParamBase::String, "", tr("indicates the current default channel").toLatin1().data());
        m_params.insert(paramVal.getName(), paramVal);

        paramVal = ito::Param("channelList", ito::ParamBase::StringList | ito::ParamBase::Readonly, {}, tr("names of the channels provided by the plugin").toLatin1().data());
        m_params.insert(paramVal.getName(), paramVal);


        //int roi[] = { 0, 0, 2048, 2048 };
        //paramVal = ito::Param("roi", ito::ParamBase::IntArray, 4, roi, tr("ROI (x,y,width,height) [this replaces the values x0,x1,y0,y1]").toLatin1().data());
        //ito::RectMeta *rm = new ito::RectMeta(ito::RangeMeta(roi[0], roi[0] + roi[2] - 1), ito::RangeMeta(roi[1], roi[1] + roi[3] - 1), "ImageFormatControl");
        //paramVal.setMeta(rm, true);
        //m_params.insert(paramVal.getName(), paramVal);

        //paramVal = ito::Param("sizex", ito::ParamBase::Int | ito::ParamBase::Readonly, 4, 4096, 4096, tr("size in x (cols) [px]").toLatin1().data());
        //paramVal.getMetaT<ito::IntMeta>()->setCategory("ImageFormatControl");
        //m_params.insert(paramVal.getName(), paramVal);

        //paramVal = ito::Param("sizey", ito::ParamBase::Int | ito::ParamBase::Readonly, 1, 4096, 4096, tr("size in y (rows) [px]").toLatin1().data());
        //paramVal.getMetaT<ito::IntMeta>()->setCategory("ImageFormatControl");
        //m_params.insert(paramVal.getName(), paramVal);

        //ito::StringMeta *m = new ito::StringMeta(ito::StringMeta::String, "mono8");
        //m->addItem("mono10");
        //m->addItem("mono12");
        //m->addItem("mono16");
        //paramVal = ito::Param("pixelFormat", ito::ParamBase::String, "mono8", tr("bitdepth of images: mono8, mono10, mono12, mono16, rgb32").toLatin1().data());
        //paramVal.setMeta(m, true);
        //m_params.insert(paramVal.getName(), paramVal);

    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! destructor
    AddInMultiChannelGrabber::~AddInMultiChannelGrabber()
    {
        DELETE_AND_SET_NULL(dd);
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    void AddInMultiChannelGrabber::initializeDefaultConfiguration(const QMap<QString, ChannelContainer>& channelContainerMap, const QMap<QString, ito::Param>& nonChannelSpecificParams /*QMap<QString, ito::Param>()*/)
    {
        ito::RetVal retValue(ito::retOk);
        bool channelSpecificParam = false;
        if (!m_defaultConfigReady)
        {
            assert(channelContainerMap.size() != 0);
            QMapIterator<QString, ChannelContainer> channel(channelContainerMap);
            QVector<ByteArray> channelList;
            QMap<QString, ito::Param>initialParams = m_params;
            while (channel.hasNext()) 
            {
                channel.next();
                QMapIterator<QString, ito::Param> channelParamIterator(channel.value().m_channelParam);
                while (channelParamIterator.hasNext())//iterate through channel params
                {
                    channelParamIterator.next();
                    if (!m_params.contains(channelParamIterator.key()))
                    {
                        m_params.insert(channelParamIterator.key(),channelParamIterator.value());
                    }
                    if (!m_paramChannelAvailabilityMap.contains(channelParamIterator.key()))
                    {
                        m_paramChannelAvailabilityMap.insert(channelParamIterator.key(), QStringList(channel.key()));
                    }
                    else
                    {
                        m_paramChannelAvailabilityMap[channelParamIterator.key()].append(channel.key());
                    }
                    
                }
                channelList.append(channel.key().toLatin1().data());
            }
            m_params["channelList"].setVal<ByteArray*>(channelList.data(), channelList.length());
            m_params["defaultChannel"].setVal<const char*>(channelContainerMap.firstKey().toLatin1().data());
            m_channels = channelContainerMap;

            QMapIterator<QString, ito::Param> nonChannelSpecificParamsIt(nonChannelSpecificParams);
            while (nonChannelSpecificParamsIt.hasNext())
            {
                nonChannelSpecificParamsIt.next();
                assert(!m_params.contains(nonChannelSpecificParamsIt.key()));
                assert(!m_paramChannelAvailabilityMap.contains(nonChannelSpecificParamsIt.key()));
                m_paramChannelAvailabilityMap.insert(nonChannelSpecificParamsIt.key(), QStringList());
                m_params.insert(nonChannelSpecificParamsIt.key(), nonChannelSpecificParamsIt.value());
                
            }
            QMapIterator<QString, ito::Param> initialParamsIt(initialParams);
            while (initialParamsIt.hasNext())
            {
                initialParamsIt.next();
                assert(!m_paramChannelAvailabilityMap.contains(initialParamsIt.key()));
                m_paramChannelAvailabilityMap.insert(nonChannelSpecificParamsIt.key(), QStringList());
            }
            m_defaultConfigReady = true;
            switchDefaultChannel();
            

        }
    }



    ////----------------------------------------------------------------------------------------------------------------------------------
    ////! sends m_image to all registered listeners.
    ///*!
    //This method is continuously called from timerEvent. Also call this method from your getVal-Method (usually with 0-timeout)

    //\param [in] waitMS indicates the time (in ms) that should be waiting until every registered live image source node received m_image. 0: no wait, -1: infinit waiting time, else time in milliseconds
    //\return retOk if everything was ok, retWarning if live image could not be invoked
    //*/
    ito::RetVal AddInMultiChannelGrabber::sendDataToListeners(int waitMS)
    {
        assert(m_defaultConfigReady);
        ito::RetVal retValue = ito::retOk;
        int size = m_autoGrabbingListeners.size();
        if (waitMS == 0)
        {
            QMultiMap<QString, QObject*>::iterator it = m_autoGrabbingListeners.begin();
            while (it != m_autoGrabbingListeners.end())
            {
                if (!QMetaObject::invokeMethod(it.value(), "setSource", Q_ARG(QSharedPointer<ito::DataObject>, QSharedPointer<ito::DataObject>(new ito::DataObject(m_channels[it.key().toLatin1().data()].data))), Q_ARG(ItomSharedSemaphore*, NULL)))
                {
                    retValue += ito::RetVal(ito::retWarning, 1001, tr("slot 'setSource' of live source node could not be invoked").toLatin1().data());
                }
                it++;
            }
        }
        else if (m_autoGrabbingListeners.size() > 0)
        {
            ItomSharedSemaphore** waitConds = new ItomSharedSemaphore*[size];
            int i = 0;
            QMultiMap<QString, QObject*>::iterator it = m_autoGrabbingListeners.begin();
            while (it != m_autoGrabbingListeners.end())
            {
                waitConds[i] = new ItomSharedSemaphore();
                // \todo On Linux a crash occurs here when closing the liveImage ... maybe the same reason why we get an error message on windows?
                if (it.value())
                {
                    if (!QMetaObject::invokeMethod(it.value(), "setSource", Q_ARG(QSharedPointer<ito::DataObject>, QSharedPointer<ito::DataObject>(new ito::DataObject(m_channels[it.key().toLatin1().data()].data))), Q_ARG(ItomSharedSemaphore*, waitConds[i])))
                    {
                        retValue += ito::RetVal(ito::retWarning, 1001, tr("slot 'setSource' of live source node could not be invoked").toLatin1().data());
                    }
                }
                it++;
                i++;
            }

            for (i = 0; i < size; i++)
            {
                if (!waitConds[i]->wait(waitMS))
                {
                    qDebug() << "timeout in number: " << i << "number of items: " << size;
                }
                waitConds[i]->deleteSemaphore();
                waitConds[i] = NULL;
            }

            delete[] waitConds;
            waitConds = NULL;
        }


        return retValue;
    }

    ito::RetVal ito::AddInMultiChannelGrabber::checkData(ito::DataObject *externalDataObject)
    {
        assert(m_defaultConfigReady);
        ito::RetVal retVal(ito::retOk);
        bool ok;
        unsigned int futureType;
        PixelFormat format;
        if (!externalDataObject)
        {

            QMutableMapIterator<QString, ChannelContainer> i(m_channels);
            while (i.hasNext()) {
                i.next();
                futureType = pixelFormatStringToEnum(i.value().m_channelParam["pixelFormat"].getVal<const char*>(), &ok);
                if (ok)
                {
                    int* roi = i.value().m_channelParam["roi"].getVal<int*>();
                    int height = roi[3];
                    int width = roi[2];
                    if (i.value().data.getDims() < 2 || i.value().data.getSize(0) != height || i.value().data.getSize(1) != width || i.value().data.getType() != futureType)
                    {
                        i.value().data = ito::DataObject(height, width, futureType);
                    }
                }
                else
                {
                    retVal += ito::RetVal(ito::retError, 0, tr("invalid pixel format").toLatin1().data());
                }

            }
        }
        else
        {
            char* channel = m_params["defaultChannel"].getVal<char*>();
            if (m_channels.contains(channel))
            {
                futureType = pixelFormatStringToEnum(m_channels[channel].m_channelParam["pixelFormat"].getVal<const char*>(), &ok);
                if (ok)
                {
                    int* roi = m_channels[channel].m_channelParam["roi"].getVal<int*>();
                    int width = roi[2];
                    int height = roi[3];
                    if (externalDataObject->getDims() == 0)
                    {
                        *externalDataObject = ito::DataObject(height, width, futureType);
                    }
                    else if (externalDataObject->calcNumMats() != 1)
                    {
                        return ito::RetVal(ito::retError, 0, tr("Error during check data, external dataObject invalid. Object has more or less than 1 plane. It must be of right size and type or an uninitilized image.").toLatin1().data());
                    }
                    else if (externalDataObject->getSize(externalDataObject->getDims() - 2) != height || externalDataObject->getSize(externalDataObject->getDims() - 1) != width || externalDataObject->getType() != futureType)
                    {
                        return ito::RetVal(ito::retError, 0, tr("Error during check data, external dataObject invalid. Object must be of right size and type or an uninitilized image.").toLatin1().data());
                    }
                }
                else
                {
                    retVal += ito::RetVal(ito::retError, 0, tr("invalid pixel format").toLatin1().data());
                }
            }
        }
        return retVal;
    }

    void AddInMultiChannelGrabber::addChannel(QString name)
    {
        ChannelContainer a(m_params["roi"], m_params["pixelFormat"], m_params["sizex"], m_params["sizey"]);
        m_channels[name] = a;
        const ByteArray* channelList = m_params["channelList"].getVal<const ByteArray*>();
        int len = 0;
        m_params["channelList"].getVal<const ByteArray*>(len);

        QVector<ByteArray> qVectorList(len, *channelList);
        qVectorList.append(ByteArray(name.toLatin1().data()));
        m_params["channelList"].setVal<ByteArray*>(qVectorList.data(), qVectorList.length());

    }

    ito::RetVal AddInMultiChannelGrabber::adaptDefaultChannelParams()
    {
        ito::RetVal retVal(ito::retOk);
        char* channel = m_params["defaultChannel"].getVal<char*>();
        return retVal;


    }

    ito::RetVal AddInMultiChannelGrabber::getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond)
    {
        assert(m_defaultConfigReady);
        ItomSharedSemaphoreLocker locker(waitCond);
        ito::RetVal retValue;
        QString key;
        bool hasIndex = false;
        int index;
        QString suffix;
        ParamMapIterator it;
        bool ok = false;

        //parse the given parameter-name (if you support indexed or suffix-based parameters)
        retValue += apiParseParamName(val->getName(), key, hasIndex, index, suffix);

        if (retValue == ito::retOk)
        {
            //gets the parameter key from m_params map (read-only is allowed, since we only want to get the value).
            if (!suffix.isEmpty() && m_channels.contains(suffix))
            {
                retValue += apiGetParamFromMapByKey(m_channels[suffix].m_channelParam, key, it, false);
                if (retValue.containsError())
                {
                    retValue = ito::RetVal(ito::retError, 0, tr("an error occured while searching parameter \"%0\" for channel \"%1\". Maybe this is a channel specific parameter.").arg(key).arg(suffix).toLatin1().data());
                }
            }
            else
            {
                retValue += apiGetParamFromMapByKey(m_params, key, it, false);
            }
        }
        if (!retValue.containsError())
        {
            retValue += getParameter(val, it, suffix, key, index, hasIndex, ok);
        }
        if (!retValue.containsError() && !ok)//the parameter was not processed by the plugin, so it is done here
        {
            *val = it.value();
        }

        if (waitCond)
        {
            waitCond->returnValue = retValue;
            waitCond->release();
        }

        return retValue;
    }

    ito::RetVal AddInMultiChannelGrabber::changeChannelForListerners(const QString& newChannel, QObject* obj)
    {
        assert(m_defaultConfigReady);
        ito::RetVal retValue(ito::retOk);
        bool found = false;
        if (obj)
        {

            QMultiMap<QString, QObject*>::iterator i = m_autoGrabbingListeners.begin();
            while (i != m_autoGrabbingListeners.end())
            {
                if (i.value() == obj)
                {
                    found = true;
                    m_autoGrabbingListeners.remove(i.key(), i.value());
                    m_autoGrabbingListeners.insert(newChannel, obj);
                    break;
                }

            }
            if (!found)
            {
                retValue += ito::RetVal(ito::retError, 0, tr("Could not find plot in m_autoGrabbingListeners").toLatin1().data());
            }
        }
        else
        {
            retValue += ito::RetVal(ito::retError, 0, "QObject not callable.");
        }
        return retValue;
    }
    //todo: update setParam... 
    ito::RetVal AddInMultiChannelGrabber::setParam(QSharedPointer<ito::ParamBase> val, ItomSharedSemaphore *waitCond/* = NULL*/)
    {
        assert(m_defaultConfigReady);
        ItomSharedSemaphoreLocker locker(waitCond);
        ito::RetVal retValue;
        bool hasIndex, ok;
        int index;
        QString suffix, key;
        ParamMapIterator it;
        retValue += ito::parseParamName(val->getName(), key, hasIndex, index, suffix);
        int cntStartedDevices = grabberStartedCount();
        if (!retValue.containsError())
        {
            if (!suffix.isEmpty() && m_channels.contains(suffix))
            {
                retValue += apiGetParamFromMapByKey(m_channels[suffix].m_channelParam, key, it, true);
                if (retValue.containsError())
                {
                    retValue = ito::RetVal(ito::retError, 0, tr("an error occured while searching parameter \"%0\" for channel \"%1\".Maybe this is a non channel specific parameter.").arg(key).arg(suffix).toLatin1().data());
                }
            }
            else
            {
                if (suffix.isEmpty())
                {
                    retValue += apiGetParamFromMapByKey(m_params, key, it, true);
                }
                else
                {
                    retValue += ito::RetVal(ito::retError, 0, tr("unknown channel: %1").arg(suffix).toLatin1().data());
                }
            }
        }
        if (!retValue.containsError())
        {
            retValue += apiValidateParam(*it, *val, false, true);
        }
        if (!retValue.containsError())
        {
            QStringList list;
            retValue += setParameter(val, it, suffix, key, index, hasIndex, ok, list);
            if (!retValue.containsError() && !ok)
            {
                if (key == "defaultChannel")
                {

                    if (m_channels.find(val->getVal<char*>()) != m_channels.end())
                    {
                        retValue += it->copyValueFrom(&(*val));
                        retValue += switchDefaultChannel();
                        //retValue += synchronizeParamswithChannelParams(previousChannel);
                    }
                    else
                    {
                        retValue += ito::RetVal(ito::retError, 0, tr("unknown channel: %1").arg(val->getVal<char*>()).toLatin1().data());
                    }
                }
                else if (key == "roi")
                {
                    if (!hasIndex)
                    {
                        retValue += it->copyValueFrom(&(*val));
                        list << "roi";
                    }
                    else
                    {
                        it->getVal<int*>()[index] = val->getVal<int>();
                        list << "roi";
                    }

                }
                else
                {
                    retValue += it->copyValueFrom(&(*val)); // it seems that the plugin does not process the param therefore it is copied here
                    if (m_channels[m_params["defaultChannel"].getVal<char*>()].m_channelParam.contains(key))
                    {
                        list << key;
                    }
                }
            }
            if (list.contains("roi")) //if key is roi sizex and sizey must be adapted
            {
                if (!suffix.isEmpty())
                {
                    QByteArrayList channelList;
                    channelList.append(suffix.toLatin1());
                    roiChanged(channelList);
                }
                else
                {
                    roiChanged();
                }

            }
            if (!retValue.containsError())
            {
                applyParamsToChannelParams(list);
                checkData();
            }
        }
        if (!retValue.containsError())
        {
            emit parametersChanged(m_params);
        }
        if (cntStartedDevices < grabberStartedCount())
        {
            if (cntStartedDevices != 0)
            {
                retValue += startDevice(NULL);
                setGrabberStarted(cntStartedDevices);
            }
        }
        if (waitCond)
        {
            waitCond->returnValue = retValue;
            waitCond->release();
        }
        return retValue;
    }
    ////----------------------------------------------------------------------------------------------------------------------------------
    ////! synchronizes m_params with the params of default channel container
    ///*!
    //This method synchronizes the parameters from the current selected channel container with m_params. Call this function after changing the default parameter.Parameters which are not available for the current default channel are set to readonly

    //\param [in] previousChannel indicates the name of the previous default channel. This is needed to check whether a parameter is no longer contained in the current channel, which was contained in the previous one.
    //\return retOk if everything was ok, else retError
    //*/
    //Todo: this function will be replaced by switchDefaultChannel 
    ito::RetVal AddInMultiChannelGrabber::switchDefaultChannel()
    {
        assert(m_defaultConfigReady);
        ito::RetVal retValue(ito::retOk);
        unsigned int flag = 0;
        const char* selectedChannel = m_params["defaultChannel"].getVal<const char*>();
        if (m_channels.contains(selectedChannel))
        {
            QMutableMapIterator<QString, ito::Param> itParam(m_params);
            while (itParam.hasNext())
            {
                itParam.next();
                if (m_paramChannelAvailabilityMap[itParam.key()].contains(selectedChannel))
                {
                    itParam.value() = m_channels[selectedChannel].m_channelParam[itParam.key()];
                }
                else if(!m_paramChannelAvailabilityMap[itParam.key()].isEmpty())
                {
                    flag = itParam.value().getFlags();
                    flag |= ito::ParamBase::Readonly;
                    itParam.value().setFlags(flag);
                }
            }
        }
        else
        {
            retValue += ito::RetVal(ito::retError, 0, tr("could not switch to channel %1. The channel is not registered.").arg(selectedChannel).toLatin1().data());
        }
        return retValue;
    }
    ////----------------------------------------------------------------------------------------------------------------------------------
    ////! copies value m_params to the channel params of the current default channel 
    ///*!
    //This method copies params of m_params to the params of the channel container. This function is usally called after setParam to apply the changed entries of m_params to the corresponding channel container. 
    //If a parameter is not found in the channel container nothing happens. This function updates also sizex and sizey if roi or nothing is passed as key.

    //\param [in] keyList indicates which params are copied. If the List is empty all Parameters of the current channel are updated.  
    //\return retOk if everything was ok, else retError
    //*/
    ito::RetVal AddInMultiChannelGrabber::applyParamsToChannelParams(QStringList keyList)
    {
        assert(m_defaultConfigReady);
        ito::RetVal retVal(ito::retOk);
        QString currentChannel = m_params["defaultChannel"].getVal<char*>();
        if (!keyList.isEmpty())
        {
            if (m_channels.contains(currentChannel))
            {
                QString tmp;
                foreach(tmp, keyList)
                {
                    if (m_channels[currentChannel].m_channelParam.contains(tmp))
                    {
                        if (m_params.contains(tmp))
                        {
                            m_channels[currentChannel].m_channelParam[tmp] = m_params[tmp];
                            if (tmp == "roi")
                            {
                                m_channels[currentChannel].m_channelParam["sizex"] = m_params["sizex"];
                                m_channels[currentChannel].m_channelParam["sizey"] = m_params["sizey"];
                            }
                        }
                        else
                        {
                            retVal = ito::RetVal(ito::retError, 0, tr("unknown parameter %1 in m_params").arg(tmp).toLatin1().data());
                        }
                    }
                }
            }

            else
            {
                retVal = ito::RetVal(ito::retError, 0, tr("unknown channel %1").arg(currentChannel).toLatin1().data());
            }
        }
        else
        {
            QMapIterator<QString, ito::Param> it(m_channels[m_params["defaultChannel"].getVal<char*>()].m_channelParam);
            while (it.hasNext())
            {
                it.next();
                const_cast<ito::Param&>(it.value()) = m_params[it.key()];

            }
        }
        return retVal;
    }

    ////----------------------------------------------------------------------------------------------------------------------------------
    ////! updates sizex and sizey
    ///*!
    //Call this function to update sizex and sizey. If the roi is changed via setParam this function will be call automatically.
    //Do not forget to apply the changes to the channel parameters by calling applyParamsToChannelParams after calling this function.

    //\param [in] keyList indicates which params are copied. If the List is empty all Parameters of the current channel are updated.  
    //\return retOk if everything was ok, else retError
    //*/
    void AddInMultiChannelGrabber::updateSizeXY()
    {
        assert(m_defaultConfigReady);
        const int* roi = m_params["roi"].getVal<const int*>();
        int height = roi[3];
        int width = roi[2];
        m_params["sizex"].setVal<int>(width);
        m_params["sizey"].setVal<int>(height);
    }

    template<typename T> ito::RetVal ito::AddInMultiChannelGrabber::setParamVal(  const QByteArray& paramName, const T & val, const QList<QByteArray>& channelList /*=QList()*/)
    {
        assert(m_defaultConfigReady);
        ito::RetVal retValue(ito::retOk);
        if (m_paramChannelAvailabilityMap.contains(paramName) && m_params.contains(paramName))
        {
            if (channelList.isEmpty()) //we want to update the param for all channels even if it is a global one
            {
                retValue = m_params[paramName].setVal<T>(val);
                QStringListIterator it (m_paramChannelAvailabilityMap[paramName]);
                while (it.hasNext()) // update param for all channels
                {
                    it.next();
                    retValue += m_channels[it.value()].m_channelParam[paramName].setVal<T>(val);
                }
            }
            else // we only want to update the param for a list of channels
            {       
                for (int i = 0; i < channelList.length(); i++)
                {
                    if (m_channels.contains(channelList[i]))
                    {
                        if (m_paramChannelAvailabilityMap[paramName].contains(channelList[i]))
                        {
                            retValue += m_channels[channelList[i]].m_channelParam[paramName].setVal<T>(val);
                        }
                    }
                    else
                    {
                        retValue += ito::RetVal(ito::retError, 0, QString("unknown channel %1").arg(channelList[i]));
                    }
                    
                }
                if (!retValue.containsError())
                {
                    //update m_params if the current default channel is listed in channelList or if the current default channel does not support the param (the param in m_params then is set to readonly) 
                    if (channelList.contains(m_params["defaultChannel"].getVal<const char*>()) || !m_paramChannelAvailabilityMap[paramName].contains(m_params["defaultChannel"].getVal<const char*>()))
                    {
                        retValue += m_params[paramName].setVal<T>(val);
                    }
                }
            }

        }
        else
        {
            retValue += ito::RetVal(ito::retError, 0, QString("could not find parameter %1. Maybe the parameter is not registered").arg(paramName));
        }
        if (paramName == "roi" && !retValue.containsError())
        {
            retValue += roiChanged(channelList);
        }
        
        return retValue;
    }

    ito::RetVal AddInMultiChannelGrabber::setParamMeta(const QByteArray& paramName, ito::ParamMeta* meta, bool takeOwnerShip ,const QList<QByteArray>& channelList/* = QList<ByteArray>()*/)
    {
        assert(m_defaultConfigReady);
        ito::RetVal retValue(ito::retOk);
        if (m_paramChannelAvailabilityMap.contains(paramName) && m_params.contains(paramName))
        {
            if (channelList.isEmpty()) //we want to update the param for all channels even if it is a global one
            {
                m_params[paramName].setMeta(meta, takeOwnerShip);
                QStringListIterator it(m_paramChannelAvailabilityMap[paramName]);
                while (it.hasNext()) // update param for all channels
                {
                    m_channels[it.next()].m_channelParam[paramName].setMeta(meta, takeOwnerShip);
                }
            }
            else // we only want to update the param for a list of channels
            {
                for (int i = 0; i < channelList.length(); i++)
                {
                    if (m_channels.contains(channelList[i]))
                    {
                        if (m_paramChannelAvailabilityMap[paramName].contains(channelList[i]))
                        {
                            m_channels[channelList[i]].m_channelParam[paramName].setMeta(meta, takeOwnerShip);
                        }
                    }
                    else
                    {
                        retValue += ito::RetVal(ito::retError, 0, QString("unknown channel %1").arg(QString(paramName)).toLatin1().data());
                    }

                }
                if (!retValue.containsError())
                {
                    //update m_params if the current default channel is listed in channelList or if the current default channel does not support the param (the param in m_params then is set to readonly) 
                    if (channelList.contains(m_params["defaultChannel"].getVal<const char*>()) || !m_paramChannelAvailabilityMap[paramName].contains(m_params["defaultChannel"].getVal<const char*>()))
                    {
                        m_params[paramName].setMeta(meta, takeOwnerShip);
                    }
                }
            }

        }
        else
        {
            retValue += ito::RetVal(ito::retError, 0, QString("could not find parameter %1. Maybe the parameter is not registered").arg(QString(paramName)).toLatin1().data());
        }

        return retValue;
    }
    
    ito::RetVal AddInMultiChannelGrabber::setParamFlags(const QByteArray& paramName, const unsigned int& flags, const QList<QByteArray>& channelList/* = QList<QByteArray>()*/)
    {
        assert(m_defaultConfigReady);
        ito::RetVal retValue(ito::retOk);
        if (m_paramChannelAvailabilityMap.contains(paramName) && m_params.contains(paramName))
        {
            if (channelList.isEmpty()) //we want to update the param for all channels even if it is a global one
            {
                m_params[paramName].setFlags(flags);
                QStringListIterator it(m_paramChannelAvailabilityMap[paramName]);
                while (it.hasNext()) // update param for all channels
                {
                    m_channels[it.next()].m_channelParam[paramName].setFlags(flags);
                }
            }
            else // we only want to update the param for a list of channels
            {
                for (int i = 0; i < channelList.length(); i++)
                {
                    if (m_channels.contains(channelList[i]))
                    {
                        if (m_paramChannelAvailabilityMap[paramName].contains(channelList[i]))
                        {
                            m_channels[channelList[i]].m_channelParam[paramName].setFlags(flags);
                        }
                    }
                    else
                    {
                        retValue += ito::RetVal(ito::retError, 0, QString("unknown channel %1").arg(QString(paramName)).toLatin1().data());
                    }

                }
                if (!retValue.containsError())
                {
                    //update m_params if the current default channel is listed in channelList or if the current default channel does not support the param (the param in m_params then is set to readonly) 
                    if (channelList.contains(m_params["defaultChannel"].getVal<const char*>()) || !m_paramChannelAvailabilityMap[paramName].contains(m_params["defaultChannel"].getVal<const char*>()))
                    {
                        m_params[paramName].setFlags(flags);
                    }
                }
            }

        }
        else
        {
            retValue += ito::RetVal(ito::retError, 0, QString("could not find parameter %1. Maybe the parameter is not registered").arg(QString(paramName)).toLatin1().data());
        }

        return retValue;
    }

    ito::RetVal AddInMultiChannelGrabber::roiChanged(const QList<QByteArray>& channelList /*= QList<QByteArray>()*/)
    {
        ito::RetVal retValue(ito::retOk);
        if (channelList.isEmpty())//assume the roi changed for all channels
        {
            QMapIterator<QString, ChannelContainer> it(m_channels);
            while (it.hasNext())
            {
                it.next();
                const int* roi = m_channels[it.key()].m_channelParam["roi"].getVal<const int*>();
                int height = roi[3];
                int width = roi[2];
                retValue += m_channels[it.key()].m_channelParam["sizex"].setVal<int>(width);
                retValue += m_channels[it.key()].m_channelParam["sizey"].setVal<int>(height);

            }
            const int* roi = m_params["roi"].getVal<const int*>();
            int height = roi[3];
            int width = roi[2];
            retValue += m_params["sizex"].setVal<int>(width);
            retValue += m_params["sizey"].setVal<int>(height);
        }
        else
        {
            for (int i = 0; i < channelList.length(); i++)
            {
                const int* roi = m_channels[channelList[i]].m_channelParam["roi"].getVal<const int*>();
                retValue += m_channels[channelList[i]].m_channelParam["sizex"].setVal<int>(roi[2]);
                retValue += m_channels[channelList[i]].m_channelParam["sizey"].setVal<int>(roi[3]);

            }
            if (channelList.contains(m_params["defaultChannel"].getVal<const char*>()))
            {
                const int* roi = m_params["roi"].getVal<const int*>();
                retValue += m_params["sizex"].setVal<int>(roi[2]);
                retValue += m_params["sizey"].setVal<int>(roi[3]);
            }
        }
        return retValue;
    }

}//end namespace ito