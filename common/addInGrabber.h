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

#include "abstractAddInGrabber.h"

#include "../DataObject/dataobj.h"
#include "sharedStructuresQt.h"
#include "sharedStructures.h"

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{
    class AddInGrabberPrivate;

    class ITOMCOMMONQT_EXPORT AddInGrabber : public AbstractAddInGrabber
    {
        Q_OBJECT
    private:

        AddInGrabberPrivate *dd;
    protected:
        ito::DataObject m_data; /*!< variable for the recently grabbed image*/

        //! implement this method in order to check if m_image should be (re)allocated with respect to the current sizex, sizey, bpp...
        /*!
        Call this method if the size or bitdepth of your camera has changed (e.g. in your constructor, too). In this method, compare if the new size
        is equal to the old one. If this is not the case, use the following example to set m_image to a newly allocated dataObject. The old dataObject
        is deleted automatically with respect to its internal reference counter:

        m_image = ito::DataObject(futureHeight,futureWidth,futureType);

        \see m_image
        */
        virtual ito::RetVal checkData(ito::DataObject *externalDataObject = nullptr);

        virtual ito::RetVal sendDataToListeners(int waitMS); /*!< sends m_data to all registered listeners. */
    public:
        AddInGrabber();
        ~AddInGrabber();

    };

} //end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)
