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

#ifndef ADDINGRABBER_H
#define ADDINGRABBER_H

#include "addInInterface.h"

#include "../DataObject/dataobj.h"
#include "sharedStructuresQt.h"
#include "sharedStructures.h"

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{
    class AddInGrabberPrivate;

    class ITOMCOMMONQT_EXPORT AddInGrabber : public ito::AddInDataIO
    {
        Q_OBJECT

    private:
        //! counter indicating how many times startDevice has been called
        /*!
            increment this variable every time startDevice is called (by incGrabberStarted())
            decrement this variable every time stopDevice is called (by decGrabberStarted())

            \sa grabberStartedCount, incGrabberStarted, decGrabberStarted, setGrabberStarted
        */
        int m_started;

        AddInGrabberPrivate *dd;

    protected:
        void timerEvent (QTimerEvent *event);  /*!< this method is called every time when the auto-grabbing-timer is fired. Usually you don't have to overwrite this method. */

        //! implement this method in order to check if m_image should be (re)allocated with respect to the current sizex, sizey, bpp...
        /*!
            Call this method if the size or bitdepth of your camera has changed (e.g. in your constructor, too). In this method, compare if the new size
            is equal to the old one. If this is not the case, use the following example to set m_image to a newly allocated dataObject. The old dataObject
            is deleted automatically with respect to its internal reference counter:

            m_image = ito::DataObject(futureHeight,futureWidth,futureType);

            \see m_image
        */
        virtual ito::RetVal checkData(ito::DataObject *externalDataObject = NULL);

        //! implement this method in your camera plugin. In this method the image is grabbed and stored in the m_image variable.
        /*!
            Call this method in getVal(...) in order to get the image from the camera and deeply copy it the the m_image variable.
            This method is equally called from timerEvent.

            \return retOk if copy operation was successfull, else retWarning or retError
            \sa getVal, timerEvent
        */
        virtual ito::RetVal retrieveData(ito::DataObject *externalDataObject = NULL) = 0;

        ito::RetVal sendDataToListeners(int waitMS);  /*!< sends m_data to all registered listeners. */

        inline int grabberStartedCount() { return m_started; }  /*!< returns the number of started devices \see m_started */

        /*!< increments the number of started devices \see m_started */
        inline void incGrabberStarted()
        {
            m_started++;
            if(m_started == 1)
            {
                runStatusChanged(true); //now, the device is started -> check if any listener is connected and if so start the auto grabbing timer (if flag is true, too)
            }
        }

        /*!< decrements the number of started devices \see m_started */
        inline void decGrabberStarted()
        {
            m_started--;
            if(m_started == 0)
            {
                runStatusChanged(false); //now, the device is stopped -> stop any possibly started auto grabbing listener
            }
        }

        /*!< sets the number of started devices to a given value \see m_started */
        inline void setGrabberStarted(int value)
        {
            m_started = value;
            runStatusChanged( value > 0 );
        }

        ito::DataObject m_data; /*!< variable for the recently grabbed image */

    public:
        AddInGrabber();
        ~AddInGrabber();

    };
} //end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif
