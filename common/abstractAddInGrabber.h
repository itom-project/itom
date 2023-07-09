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

#pragma once

#include "addInInterface.h"

#include "../DataObject/dataobj.h"
#include "sharedStructuresQt.h"
#include "sharedStructures.h"

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{
    class AbstractAddInGrabberPrivate;

    class ITOMCOMMONQT_EXPORT AbstractAddInGrabber : public ito::AddInDataIO
    {
        Q_OBJECT
    public:

        // all items must be lowercase!
        enum PixelFormat
        {
            mono8 = ito::tUInt8,
            mono8s = ito::tInt8,
            mono10 = ito::tUInt16,
            mono10Packed = ito::tUInt16,
            mono12 = ito::tUInt16,
            mono12Packed = ito::tUInt16,
            mono14 = ito::tUInt16,
            mono14Packed = ito::tUInt16,
            mono16 = ito::tUInt16,
            rgb8 = ito::tRGBA32,
            rgba8 = ito::tRGBA32,
            rgb8Planar = ito::tUInt8,
            rgb10Planar = ito::tUInt16,
            rgb12Planar = ito::tUInt16,
            rgb16Planar = ito::tUInt16,
            rg8 = ito::tRGBA32,
            rg8Packed = ito::tRGBA32,
            gb8 = ito::tRGBA32,
            float32 = ito::tFloat32,
            float64 = ito::tFloat64,
            complex64 = ito::tComplex64,
            complex128 = ito::tComplex128,
        };

        Q_ENUM(PixelFormat)

    private:
        QScopedPointer<AbstractAddInGrabberPrivate> d_ptr;
        Q_DECLARE_PRIVATE(AbstractAddInGrabber);

    protected:
        /*!< this method is called every time when the auto-grabbing-timer is fired. Usually you don't have to overwrite this method. */
        void timerEvent (QTimerEvent *event);

        //! implement this method in your camera plugin. In this method the image is grabbed and stored in the m_image variable.
        /*!
            Call this method in getVal(...) in order to get the image from the camera and deeply copy it the the m_image variable.
            This method is equally called from timerEvent.

            \return retOk if copy operation was successfull, else retWarning or retError
            \sa getVal, timerEvent
        */
        virtual ito::RetVal retrieveData(ito::DataObject *externalDataObject = nullptr) = 0;

        virtual ito::RetVal sendDataToListeners(int waitMS) = 0; /*!< sends m_data to all registered listeners. */

        /*!< returns the number of started devices \see m_started */
        int grabberStartedCount() const;

        /*!< increments the number of started devices \see m_started */
        void incGrabberStarted();

        /*!< decrements the number of started devices \see m_started */
        void decGrabberStarted();

        /*!< sets the number of started devices to a given value \see m_started */
        void setGrabberStarted(int value);

    public:
        /*!< this method gives the value range pixel for a given integer pixelFormat */
        static void minMaxBoundariesFromIntegerPixelFormat(
            const QByteArray &pixelFormat,
            int& min,
            int& max,
            bool& ok);

        /*!< this method maps a string to a value of PixelFormat  */
        static int itoDataTypeFromPixelFormat(const QByteArray &pixelFormat, bool *ok = nullptr);

        AbstractAddInGrabber();
        ~AbstractAddInGrabber();
    };

} //end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)
