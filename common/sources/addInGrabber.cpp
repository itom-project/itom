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

#include "AddInGrabber.h"

#include <qmetatype.h>
#include <qcoreapplication.h>
#include <qmetaobject.h>
#include <qmap.h>
#include "common/helperCommon.h"


namespace ito
{
    class AddInAbstractGrabberPrivate
    {
    public:
        int m_nrLiveImageErrors; //number of consecutive errors when automatically grabbing the next image. If this number becomes bigger than a threshold, auto grabbing will be disabled.
    };
    class AddInGrabberPrivate
    {

    };
    class AddInMultiChannelGrabberPrivate
    {
    };

    /*!
    \class AddInAbstractGrabber
    \brief Inherit from AddInAbstractGrabber if you write a camera/grabber plugin. Please call the constructor of AddInAbstractGrabber within your plugin constructor.

    This class contains important variables and helper methods which simplify the creation of a camera plugin. Please consider that you should implement
    the methods checkImage() and retriveImage() (pure virtual in this class) in your own class.

    \see checkImage(), retrieveImage()
    */
    
    //----------------------------------------------------------------------------------------------------------------------------------
    //! constructor
    AddInAbstractGrabber::AddInAbstractGrabber() :
        AddInDataIO(),
        m_started(0)
    {
        dd = new AddInAbstractGrabberPrivate();
        dd->m_nrLiveImageErrors = 0;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! destructor
    AddInAbstractGrabber::~AddInAbstractGrabber()
    {
        DELETE_AND_SET_NULL(dd);
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! this method returns the size of a pixel for a given pixelFormat.
    int AddInAbstractGrabber::pixelFormatStringToBpp(const char* val)
    {
        bool ok;
        int format = pixelFormatStringToEnum(val, &ok);
        if (ok)
        {
            if (format == mono8)
            {
                return 8;
            }
            else if (format == mono10 || format == mono12 || format == mono16)
            {
                return 16;
            }
            else if(format == rgb32)
            {
                return 40;
            }
            else
            {
                return -1;
            }
        }
        else
        {
            return -1;
        }
        
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
    \class AddInAbstractGrabber
    \brief This method maps a string to a value of pixelFormat.

    This function maps a string to a pixel format by using QMetaType.
    */

    int AddInAbstractGrabber::pixelFormatStringToEnum(const char* val, bool* ok)
    {
#if QT_VERSION >= 0x050500
        const QMetaObject mo = staticMetaObject;
#else
        const QMetaObject mo = StaticQtMetaObject::get();
#endif
        QMetaEnum me = mo.enumerator(mo.indexOfEnumerator("PixelFormat"));
        int pixelFormat = me.keyToValue(val, ok);
        return pixelFormat;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    //! if any live image has been connected to this camera, this event will be regularly fired.
    /*!
        This event is continoulsy fired if auto grabbing is enabled. At first, the image is acquired (method acquire). Then
        the image is retrieved (retrieveImage) and finally the newly grabbed image is send to all registered listeners (sendImagetoListeners)
    */
    void AddInAbstractGrabber::timerEvent (QTimerEvent * /*event*/)
    {
        QCoreApplication::sendPostedEvents(this,0);
        ito::RetVal retValue = ito::retOk;

        if (m_autoGrabbingListeners.size() > 0) //verify that any liveImage is listening
        {
            retValue += acquire(0,NULL);

            if (!retValue.containsError())
            {
                retValue += retrieveData();
            }

            if (!retValue.containsError())
            {
                retValue += sendDataToListeners(200);
            }

            if (retValue.containsWarning())
            {
                if (retValue.hasErrorMessage())
                {
                    std::cout << "warning while sending live image: " << retValue.errorMessage() << "\n" << std::endl;
                }
                else
                {
                    std::cout << "warning while sending live image." << "\n" << std::endl;
                }

                dd->m_nrLiveImageErrors = 0;
            }
            else if (retValue.containsError())
            {
                if (retValue.hasErrorMessage())
                {
                    std::cout << "error while sending live image: " << retValue.errorMessage() << "\n" << std::endl;
                }
                else
                {
                    std::cout << "error while sending live image." << "\n" << std::endl;
                }

                dd->m_nrLiveImageErrors++;

                if (dd->m_nrLiveImageErrors > 10)
                {
                    disableAutoGrabbing();
                    dd->m_nrLiveImageErrors = 0;
                    std::cout << "Auto grabbing of grabber " << this->getIdentifier().toLatin1().data() << " was stopped due to consecutive errors in the previous tries\n" << std::endl;
                }
            }
            else
            {
                dd->m_nrLiveImageErrors = 0;
            }
        }
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    //! constructor
    AddInGrabber::AddInGrabber() :
        AddInAbstractGrabber()
    {
        dd = new AddInGrabberPrivate();
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! destructor
    AddInGrabber::~AddInGrabber()
    {
        DELETE_AND_SET_NULL(dd);
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInGrabber::checkData(ito::DataObject *externalDataObject)
    {
        int futureHeight = m_params["sizey"].getVal<int>();
        int futureWidth = m_params["sizex"].getVal<int>();
        int futureType;

        int bpp = m_params["bpp"].getVal<int>();
        if (bpp <= 8)
        {
            futureType = ito::tUInt8;
        }
        else if (bpp <= 16)
        {
            futureType = ito::tUInt16;
        }
        else if (bpp <= 32)
        {
            futureType = ito::tInt32;
        }
        else
        {
            futureType = ito::tFloat64;
        }
        if (!m_params.contains("sizez"))
        {

            if (externalDataObject == NULL)
            {
                if (m_data.getDims() < 2 || m_data.getSize(0) != (unsigned int)futureHeight || m_data.getSize(1) != (unsigned int)futureWidth || m_data.getType() != futureType)
                {
                    m_data = ito::DataObject(futureHeight, futureWidth, futureType);
                }
            }
            else
            {
                int dims = externalDataObject->getDims();
                if (externalDataObject->getDims() == 0)
                {
                    *externalDataObject = ito::DataObject(futureHeight, futureWidth, futureType);
                }
                else if (externalDataObject->calcNumMats() != 1)
                {
                    return ito::RetVal(ito::retError, 0, tr("Error during check data, external dataObject invalid. Object has more or less than 1 plane. It must be of right size and type or an uninitilized image.").toLatin1().data());
                }
                else if (externalDataObject->getSize(dims - 2) != (unsigned int)futureHeight || externalDataObject->getSize(dims - 1) != (unsigned int)futureWidth || externalDataObject->getType() != futureType)
                {
                    return ito::RetVal(ito::retError, 0, tr("Error during check data, external dataObject invalid. Object must be of right size and type or an uninitilized image.").toLatin1().data());
                }
            }
        }
        else
        {
            int numChannel = m_params["sizez"].getVal<int>();
            if (externalDataObject == NULL)
            {
                if (m_data.getDims() < 3 || m_data.getSize(0) != (unsigned int)numChannel || m_data.getSize(1) != (unsigned int)futureHeight || m_data.getSize(2) != (unsigned int)futureWidth || m_data.getType() != futureType)
                {
                    m_data = ito::DataObject(numChannel, futureHeight, futureWidth, futureType);
                }
            }
            else
            {
                int dims = externalDataObject->getDims();
                if (externalDataObject->getDims() == 0)
                {
                    *externalDataObject = ito::DataObject(numChannel, futureHeight, futureWidth, futureType);
                }
                else if (externalDataObject->getSize(dims - 3) != (unsigned int)numChannel || externalDataObject->getSize(dims - 2) != (unsigned int)futureHeight || externalDataObject->getSize(dims - 1) != (unsigned int)futureWidth || externalDataObject->getType() != futureType)
                {
                    return ito::RetVal(ito::retError, 0, tr("Error during check data, external dataObject invalid. Object must be of right size and type or an uninitilized image.").toLatin1().data());
                }
            }
        }

        return ito::retOk;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    //! sends m_image to all registered listeners.
    /*!
    This method is continuously called from timerEvent. Also call this method from your getVal-Method (usually with 0-timeout)

    \param [in] waitMS indicates the time (in ms) that should be waiting until every registered live image source node received m_image. 0: no wait, -1: infinit waiting time, else time in milliseconds
    \return retOk if everything was ok, retWarning if live image could not be invoked
    */
    ito::RetVal AddInGrabber::sendDataToListeners(int waitMS)
    {
        QObject *obj;
        ito::RetVal retValue = ito::retOk;
        //        int i=0;
        int size = m_autoGrabbingListeners.size();

        if (waitMS == 0)
        {
            foreach(obj, m_autoGrabbingListeners)
            {
                if (!QMetaObject::invokeMethod(obj, "setSource", Q_ARG(QSharedPointer<ito::DataObject>, QSharedPointer<ito::DataObject>(new ito::DataObject(m_data))), Q_ARG(ItomSharedSemaphore*, NULL)))
                {
                    retValue += ito::RetVal(ito::retWarning, 1001, tr("slot 'setSource' of live source node could not be invoked").toLatin1().data());
                }
            }
        }
        else if (m_autoGrabbingListeners.size() > 0)
        {
            ItomSharedSemaphore** waitConds = new ItomSharedSemaphore*[size];
            int i = 0;

            foreach(obj, m_autoGrabbingListeners)
            {
                waitConds[i] = new ItomSharedSemaphore();
                // \todo On Linux a crash occurs here when closing the liveImage ... maybe the same reason why we get an error message on windows?
                if (!QMetaObject::invokeMethod(obj, "setSource", Q_ARG(QSharedPointer<ito::DataObject>, QSharedPointer<ito::DataObject>(new ito::DataObject(m_data))), Q_ARG(ItomSharedSemaphore*, waitConds[i])))
                {
                    retValue += ito::RetVal(ito::retWarning, 1001, tr("slot 'setSource' of live source node could not be invoked").toLatin1().data());
                }

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

    //----------------------------------------------------------------------------------------------------------------------------------
    //! constructor
    AddInMultiChannelGrabber::AddInMultiChannelGrabber() :
        AddInAbstractGrabber()
    {
        dd = new AddInMultiChannelGrabberPrivate();

        ito::Param paramVal("name", ito::ParamBase::String | ito::ParamBase::Readonly, "DummyMultiChannelGrabber", "GrabberName");
        paramVal.setMeta(new ito::StringMeta(ito::StringMeta::String, "General"), true);
        m_params.insert(paramVal.getName(), paramVal);

        int roi[] = { 0, 0, 2048, 2048 };
        paramVal = ito::Param("roi", ito::ParamBase::IntArray, 4, roi, tr("ROI (x,y,width,height) [this replaces the values x0,x1,y0,y1]").toLatin1().data());
        ito::RectMeta *rm = new ito::RectMeta(ito::RangeMeta(roi[0], roi[0] + roi[2] - 1), ito::RangeMeta(roi[1], roi[1] + roi[3] - 1), "ImageFormatControl");
        paramVal.setMeta(rm, true);
        m_params.insert(paramVal.getName(), paramVal);

        paramVal = ito::Param("sizex", ito::ParamBase::Int | ito::ParamBase::Readonly, 4, 4096, 4096, tr("size in x (cols) [px]").toLatin1().data());
        paramVal.getMetaT<ito::IntMeta>()->setCategory("ImageFormatControl");
        m_params.insert(paramVal.getName(), paramVal);

        paramVal = ito::Param("sizey", ito::ParamBase::Int | ito::ParamBase::Readonly, 1, 4096, 4096, tr("size in y (rows) [px]").toLatin1().data());
        paramVal.getMetaT<ito::IntMeta>()->setCategory("ImageFormatControl");
        m_params.insert(paramVal.getName(), paramVal);

        ito::StringMeta *m = new ito::StringMeta(ito::StringMeta::String, "mono8");
        m->addItem("mono10");
        m->addItem("mono12");
        m->addItem("mono16");
        paramVal = ito::Param("pixelFormat", ito::ParamBase::String, "mono8", tr("bitdepth of images: mono8, mono10, mono12, mono16, rgb32").toLatin1().data());
        paramVal.setMeta(m, true);
        m_params.insert(paramVal.getName(), paramVal);

        paramVal = ito::Param("defaultChannel", ito::ParamBase::String, "", tr("indicates the current default channel").toLatin1().data());
        m_params.insert(paramVal.getName(), paramVal);

        paramVal = ito::Param("channelList", ito::ParamBase::StringList | ito::ParamBase::Readonly, {}, tr("names of the channels provided by the plugin").toLatin1().data());
        m_params.insert(paramVal.getName(), paramVal);
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! destructor
    AddInMultiChannelGrabber::~AddInMultiChannelGrabber()
    {
        DELETE_AND_SET_NULL(dd);
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
        ito::RetVal retValue = ito::retOk;
        int size = m_autoGrabbingListeners.size();
        if (waitMS == 0)
        {
            QMultiMap<QString, QObject*>::iterator it = m_autoGrabbingListeners.begin();
            while(it != m_autoGrabbingListeners.end())
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
        ito::RetVal retVal(ito::retOk);
        bool ok;
        unsigned int futureType;
        PixelFormat format;
        if (!externalDataObject)
        {

            QMutableMapIterator<QString, ChannelContainer> i(m_channels);
            while (i.hasNext()) {
                i.next();
                futureType = pixelFormatStringToEnum(i.value().m_channelParam["pixelFormat"].getVal<char*>(),&ok);
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
                futureType = pixelFormatStringToEnum(m_channels[channel].m_channelParam["pixelFormat"].getVal<char*>(), &ok);
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
        m_params["channelList"].setVal<ByteArray*>(qVectorList.data(),qVectorList.length());

    }

    ito::RetVal AddInMultiChannelGrabber::adaptDefaultChannelParams()
    {
        ito::RetVal retVal(ito::retOk);
        char* channel = m_params["defaultChannel"].getVal<char*>();
        return retVal;


    }

    ito::RetVal AddInMultiChannelGrabber::getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond)
    {
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

    ito::RetVal AddInMultiChannelGrabber::setParam(QSharedPointer<ito::ParamBase> val, ItomSharedSemaphore *waitCond/* = NULL*/)
    {
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
                    QString previousChannel = m_params["defaultChannel"].getVal<char*>();
                    
                    if (m_channels.find(val->getVal<char*>()) != m_channels.end())
                    {
                        retValue += it->copyValueFrom(&(*val));
                        retValue += synchronizeParamswithChannelParams(previousChannel);
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
                updateSizeXY();                
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
    ito::RetVal AddInMultiChannelGrabber::synchronizeParamswithChannelParams(QString previousChannel)
    {
        ito::RetVal retVal(ito::retOk);
        unsigned int flag = 0;
        QMapIterator<QString, ito::Param> itChannelParams(m_channels[m_params["defaultChannel"].getVal<char*>()].m_channelParam);
        bool channelDiffer = false;
        //Check if the previous channel had the same parameters
        QStringList defaulKeyList = m_channels[m_params["defaultChannel"].getVal<char*>()].m_channelParam.keys();
        qSort(defaulKeyList);
        QStringList previousKeyList = m_channels[previousChannel].m_channelParam.keys();
        qSort(previousKeyList);
        if (defaulKeyList != previousKeyList)
        {
            QStringList::ConstIterator itPreviousChannel;
            for (itPreviousChannel = previousKeyList.constBegin(); itPreviousChannel != previousKeyList.constEnd(); ++itPreviousChannel)
            {
                if (!defaulKeyList.contains(*itPreviousChannel))//if param from previous channel is not included in current channel, set param in m_params to readonly.
                {
                    flag = m_params[*itPreviousChannel].getFlags();
                    flag |= ito::ParamBase::Readonly;
                    m_params[*itPreviousChannel].setFlags(flag);
                    //if a param is in the current channel include which was not in the previous one, the readonly flag will be removed when copying the param from the channel container to m_params
                }
            }
        }
        while (itChannelParams.hasNext())
        {
            itChannelParams.next();
            if (m_params.contains(itChannelParams.key()))
            {
                m_params[itChannelParams.key()] = itChannelParams.value(); //copy param
            }
            else
            {
                retVal += ito::RetVal(ito::retError, 0, QString("channel parameter %1 not found in m_params").arg(itChannelParams.key()).toLatin1().data());
            }

        }        
        return retVal;
    }
    ////----------------------------------------------------------------------------------------------------------------------------------
    ////! copies value m_params to the channel params of the current default channel 
    ///*!
    //This method copies params of m_params to the params of the channel container. This function is usally called after setParam to apply the changed entries of m_params to the corresponding channel container. 
    //If a parameter is not found in the channel container nothing happens. This function updates also sizex and sizey if roi or nothing is passed as key.

    //\param [in] keyList indicates which params are copied. If the List is empty all Parameters of the current channel are updated.  
    //\return retOk if everything was ok, else retError
    //*/
    ito::RetVal AddInMultiChannelGrabber::applyParamsToChannelParams(QStringList keyList )
    {
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
        const int* roi = m_params["roi"].getVal<const int*>();
        int height = roi[3];
        int width = roi[2];
        m_params["sizex"].setVal<int>(width);
        m_params["sizey"].setVal<int>(height);
    }
} //end namespace ito

