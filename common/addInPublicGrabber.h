/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut für Technische Optik (ITO),
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

#ifndef ADDINPUBLICGRABBER_H
#define ADDINPUBLICGRABBER_H

#include "addInGrabber.h"

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{

    class ITOMCOMMONQT_EXPORT AddInPublicGrabber : public ito::AddInGrabber
    {
        Q_OBJECT

    protected:
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

            \return retOk if copy operation was successful, else retWarning or retError
            \sa getVal, timerEvent
        */
        virtual ito::RetVal retrieveData(ito::DataObject *externalDataObject = NULL) = 0;

    public:
        /*!< Returns a direct pointer to the internal ito::DataObject managed by this class. This function acts as a getter to provide
             read or write access to the class's internal data structure without exposing the underlying implementation details.
        */
        ito::DataObject* getData();

        AddInPublicGrabber();
        ~AddInPublicGrabber();

    };
} //end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif
