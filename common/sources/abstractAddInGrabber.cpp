/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut fuer Technische Optik (ITO),
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

#include "abstractAddInGrabber.h"

#include "common/helperCommon.h"
#include <qcoreapplication.h>
#include <qmap.h>
#include <qmetaobject.h>
#include <qmetatype.h>

namespace ito
{
class AbstractAddInGrabberPrivate
{
    public:
        AbstractAddInGrabberPrivate() : m_nrLiveImageErrors(0), m_started(0)
        {

        }

        // number of consecutive errors when automatically grabbing the next image. If this number
        // becomes bigger than a threshold, auto grabbing will be disabled.
        int m_nrLiveImageErrors;

        //! counter indicating how many times startDevice has been called
        /*!
            increment this variable every time startDevice is called (by incGrabberStarted())
            decrement this variable every time stopDevice is called (by decGrabberStarted())

            \sa grabberStartedCount, incGrabberStarted, decGrabberStarted, setGrabberStarted
        */
        int m_started;
};


/*!
\class AbstractAddInGrabber
\brief Inherit from AbstractAddInGrabber if you write a camera/grabber plugin. Please call the constructor of
AbstractAddInGrabber within your plugin constructor.

This class contains important variables and helper methods which simplify the creation of a camera plugin. Please
consider that you should implement the methods checkImage() and retriveImage() (pure virtual in this class) in your own
class.

\see checkImage(), retrieveImage()
*/

//-------------------------------------------------------------------------------------
//! constructor
AbstractAddInGrabber::AbstractAddInGrabber() : AddInDataIO(), d_ptr(new AbstractAddInGrabberPrivate())
{
}

//-------------------------------------------------------------------------------------
//! destructor
AbstractAddInGrabber::~AbstractAddInGrabber()
{
}

//-------------------------------------------------------------------------------------
//! this method gives the value range pixel for a given integer pixelFormat.
void AbstractAddInGrabber::minMaxBoundariesFromIntegerPixelFormat(const QByteArray& pixelFormat, int &min, int &max, bool &ok)
{
    ok = false;
    QByteArray pixelFormatLower = pixelFormat.toLower();

    if (pixelFormatLower == "mono8" || pixelFormatLower == "rgb8" || pixelFormatLower == "rgba8" ||
        pixelFormatLower == "rgb8planar" || pixelFormatLower == "rg8" || pixelFormatLower == "rg8packed" ||
        pixelFormatLower == "gb8")
    {
        min = 0.0;
        max = 255.0;
        ok = true;
    }
    else if (pixelFormatLower == "mono8s")
    {
        min = -128.0;
        max = 127.0;
        ok = true;
    }
    else if (pixelFormatLower == "mono10" || pixelFormatLower == "mono10packed" || pixelFormatLower == "rgb10planar")
    {
        min = 0.0;
        max = 1023.0;
        ok = true;
    }
    else if (pixelFormatLower == "mono12" || pixelFormatLower == "mono12packed" || pixelFormatLower == "rgb12planar")
    {
        min = 0.0;
        max = 4095.0;
        ok = true;
    }
    else if (pixelFormatLower == "mono14" || pixelFormatLower == "mono14packed")
    {
        min = 0.0;
        max = 16383.0;
        ok = true;
    }
    else if (pixelFormatLower == "mono16" || pixelFormatLower == "rgb16planar")
    {
        min = 0.0;
        max = 65535.0;
        ok = true;
    }
}
//-------------------------------------------------------------------------------------
/*!
\class AbstractAddInGrabber
\brief This method maps a string to a value of pixelFormat.

This function maps a string to a pixel format by using QMetaType.
*/

int AbstractAddInGrabber::itoDataTypeFromPixelFormat(const QByteArray &pixelFormat, bool *ok)
{
#if QT_VERSION >= 0x050500
    const QMetaObject mo = staticMetaObject;
#else
    const QMetaObject mo = StaticQtMetaObject::get();
#endif
    const QByteArray pixelFormatLower = pixelFormat.toLower();
    const char *val_ = pixelFormatLower.data();
    QMetaEnum me = mo.enumerator(mo.indexOfEnumerator("PixelFormat"));
    int dataType = me.keyToValue(val_, ok);

    return dataType;
}
//-------------------------------------------------------------------------------------
//! if any live image has been connected to this camera, this event will be regularly fired.
/*!
    This event is continoulsy fired if auto grabbing is enabled. At first, the image is acquired (method acquire). Then
    the image is retrieved (retrieveImage) and finally the newly grabbed image is send to all registered listeners
   (sendImagetoListeners)
*/
void AbstractAddInGrabber::timerEvent(QTimerEvent * /*event*/)
{
    Q_D(AbstractAddInGrabber);

    QCoreApplication::sendPostedEvents(this, 0);
    ito::RetVal retValue = ito::retOk;

    if (m_autoGrabbingListeners.size() > 0) // verify that any liveImage is listening
    {
        retValue += acquire(0, nullptr);

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
                std::cout << "warning while sending live image."
                          << "\n"
                          << std::endl;
            }

            d->m_nrLiveImageErrors = 0;
        }
        else if (retValue.containsError())
        {
            if (retValue.hasErrorMessage())
            {
                std::cout << "error while sending live image: " << retValue.errorMessage() << "\n" << std::endl;
            }
            else
            {
                std::cout << "error while sending live image."
                          << "\n"
                          << std::endl;
            }

            d->m_nrLiveImageErrors++;

            if (d->m_nrLiveImageErrors > 10)
            {
                disableAutoGrabbing();
                d->m_nrLiveImageErrors = 0;
                std::cout << "Auto grabbing of grabber " << this->getIdentifier().toLatin1().data()
                          << " was stopped due to consecutive errors in the previous tries\n"
                          << std::endl;
            }
        }
        else
        {
            d->m_nrLiveImageErrors = 0;
        }
    }
}

//-------------------------------------------------------------------------------------
/*!< returns the number of started devices \see m_started */
int AbstractAddInGrabber::grabberStartedCount() const
{
    Q_D(const AbstractAddInGrabber);
    return d->m_started;
}

//-------------------------------------------------------------------------------------
/*!< increments the number of started devices \see m_started */
void AbstractAddInGrabber::incGrabberStarted()
{
    Q_D(AbstractAddInGrabber);
    d->m_started++;

    if (d->m_started == 1)
    {
        runStatusChanged(true); //now, the device is started -> check if any listener is connected and if so start the auto grabbing timer (if flag is true, too)
    }
}

//-------------------------------------------------------------------------------------
/*!< decrements the number of started devices \see m_started */
void AbstractAddInGrabber::decGrabberStarted()
{
    Q_D(AbstractAddInGrabber);
    d->m_started--;

    if (d->m_started == 0)
    {
        runStatusChanged(false); //now, the device is stopped -> stop any possibly started auto grabbing listener
    }
}

//-------------------------------------------------------------------------------------
/*!< sets the number of started devices to a given value \see m_started */
void AbstractAddInGrabber::setGrabberStarted(int value)
{
    Q_D(AbstractAddInGrabber);
    d->m_started = value;
    runStatusChanged(value > 0);
}

} // end namespace ito
