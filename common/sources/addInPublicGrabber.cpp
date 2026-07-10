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

#include "addInPublicGrabber.h"

#include <qmetatype.h>
#include <qcoreapplication.h>

namespace ito
{
    //----------------------------------------------------------------------------------------------------------------------------------
    //! constructor
    AddInPublicGrabber::AddInPublicGrabber() :
        AddInGrabber()
    {
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! destructor
    AddInPublicGrabber::~AddInPublicGrabber()
    {
    }

    ito::DataObject* AddInPublicGrabber::getData()
    {
        return &m_data;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    // Now functions for multiliveimages
    ito::RetVal AddInPublicGrabber::checkData(ito::DataObject *externalDataObject)
    {
        return AddInGrabber::checkData(externalDataObject);
        /*
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
                    m_data = ito::DataObject(numChannel ,futureHeight, futureWidth, futureType);
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
        */
    }

    //----------------------------------------------------------------------------------------------------------------------------------
} //end namespace ito
