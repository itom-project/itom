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
	int AddInAbstractGrabber::pixelFormatStringToBpp(char* val)
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

	int AddInAbstractGrabber::pixelFormatStringToEnum(char* val, bool* ok)
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
		QObject *obj;
		ito::RetVal retValue = ito::retOk;
		//        int i=0;
		int size = m_autoGrabbingListeners.size();
		char* defChannel = m_params["defaultChannel"].getVal<char*>();
		if (m_data.contains(defChannel))
		{
			if (waitMS == 0)
			{
				foreach(obj, m_autoGrabbingListeners)
				{
					if (!QMetaObject::invokeMethod(obj, "setSource", Q_ARG(QSharedPointer<ito::DataObject>, QSharedPointer<ito::DataObject>(new ito::DataObject(m_data[defChannel].data))), Q_ARG(ItomSharedSemaphore*, NULL)))
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
					if (!QMetaObject::invokeMethod(obj, "setSource", Q_ARG(QSharedPointer<ito::DataObject>, QSharedPointer<ito::DataObject>(new ito::DataObject(m_data[defChannel].data))), Q_ARG(ItomSharedSemaphore*, waitConds[i])))
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
		}
		else
		{
			retValue += ito::RetVal(ito::retError, 1001, tr("channel not found in channel map").toLatin1().data());
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

			QMutableMapIterator<QString, ChannelContainer> i(m_data);
			while (i.hasNext()) {
				i.next();
				futureType = pixelFormatStringToEnum(i.value().m_channelParam["pixelFormat"].getVal<char*>(),&ok);
				if (ok)
				{
					int* roi = i.value().m_channelParam["roi"].getVal<int*>();
					int height = roi[3] - roi[2];
					int width = roi[1] - roi[0];
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
			if (m_data.contains(channel))
			{
				futureType = pixelFormatStringToEnum(m_data[channel].m_channelParam["pixelFormat"].getVal<char*>(), &ok);
				if (ok)
				{
					int* roi = m_data[channel].m_channelParam["roi"].getVal<int*>();
					int width = roi[1] - roi[0];
					int height = roi[3] - roi[2];
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
		m_data[name] = a;

	}

	ito::RetVal ito::AddInMultiChannelGrabber::setBaseParam(ito::Param param, bool & ok)
	{
		QString paramName(param.getName());
		bool ok = paramName.compare("roi") || paramName.compare("pixelFormat") || paramName.compare("defaultChannel");
		if (ok)
		{
dfsfdsf
		}
		return ito::retOk;
	}

	ito::RetVal AddInMultiChannelGrabber::adaptDefaultChannelParams()
	{
		ito::RetVal retVal(ito::retOk);
		char* channel = m_params["defaultChannel"].getVal<char*>();
		return retVal;


	}

} //end namespace ito

