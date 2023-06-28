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

#include "addInGrabber.h"

#include "common/helperCommon.h"
#include <qcoreapplication.h>
#include <qmap.h>
#include <qmetaobject.h>
#include <qmetatype.h>


namespace ito
{
//-------------------------------------------------------------------------------------
class AddInGrabberPrivate
{
};

//-------------------------------------------------------------------------------------
//! constructor
AddInGrabber::AddInGrabber() : AbstractAddInGrabber()
{
    dd = new AddInGrabberPrivate();
}

//-------------------------------------------------------------------------------------
//! destructor
AddInGrabber::~AddInGrabber()
{
    DELETE_AND_SET_NULL(dd);
}
//-------------------------------------------------------------------------------------
ito::RetVal AddInGrabber::checkData(ito::DataObject *externalDataObject)
{
    const int futureHeight = m_params["sizey"].getVal<int>();
    const int futureWidth = m_params["sizex"].getVal<int>();
    int futureType;

    const int bpp = m_params["bpp"].getVal<int>();

    ito::float64 axisOffset[] = {0.0, 0.0};
    ito::float64 axisScale[] = {1.0, 1.0};
    QString axisUnit[] = {"<auto>", "<auto>"};
    QString axisDescription[] = {"<auto>", "<auto>"};
    QString valueDescription = "<auto>";
    QString valueUnit = "<auto>";

    // only if exists in plugin
    if (m_params.contains("axisOffset"))
    {
        axisOffset[0] = m_params["axisOffset"].getVal<ito::float64*>()[0];
        axisOffset[1] = m_params["axisOffset"].getVal<ito::float64*>()[1];
    }

    // only if exists in plugin
    if (m_params.contains("axisScale"))
    {
        axisScale[0] = m_params["axisScale"].getVal<ito::float64*>()[0];
        axisScale[1] = m_params["axisScale"].getVal<ito::float64*>()[1];
    }

    // only if exists in plugin
    if (m_params.contains("axisDescription"))
    {
        axisDescription[0] =
            QString::fromUtf8(m_params["axisDescription"].getVal<ito::ByteArray*>()[0].data());
        axisDescription[1] =
            QString::fromUtf8(m_params["axisDescription"].getVal<ito::ByteArray*>()[1].data());
    }

    // only if exists in plugin
    if (m_params.contains("axisUnit"))
    {
        axisUnit[0] = QString::fromUtf8(m_params["axisUnit"].getVal<ito::ByteArray*>()[0].data());
        axisUnit[1] = QString::fromUtf8(m_params["axisUnit"].getVal<ito::ByteArray*>()[1].data());
    }

    // only if exists in plugin
    if (m_params.contains("valueDescription"))
    {
        valueDescription = QString::fromLatin1(m_params["valueDescription"].getVal<char*>());
    }

    // only if exists in plugin
    if (m_params.contains("valueUnit"))
    {
        valueUnit = QString::fromLatin1(m_params["valueUnit"].getVal<char*>());
    }

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
            if (m_data.getDims() < 2 || m_data.getSize(0) != (unsigned int)futureHeight ||
                m_data.getSize(1) != (unsigned int)futureWidth || m_data.getType() != futureType)
            {
                m_data = ito::DataObject(futureHeight, futureWidth, futureType);
            }
            m_data.setAxisScale(0, axisScale[0]);
            m_data.setAxisScale(1, axisScale[1]);
            m_data.setAxisOffset(0, axisOffset[0]);
            m_data.setAxisOffset(1, axisOffset[1]);
            m_data.setAxisDescription(0, axisDescription[0].toLatin1().data());
            m_data.setAxisDescription(1, axisDescription[1].toLatin1().data());
            m_data.setAxisUnit(0, axisUnit[0].toLatin1().data());
            m_data.setAxisUnit(1, axisUnit[1].toLatin1().data());
            m_data.setValueDescription(valueDescription.toLatin1().data());
            m_data.setValueUnit(valueUnit.toLatin1().data());
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
                return ito::RetVal(ito::retError, 0,
                                   tr("Error during check data, external dataObject invalid. Object has more or less "
                                      "than 1 plane. It must be of right size and type or an uninitilized image.")
                                       .toLatin1()
                                       .data());
            }
            else if (externalDataObject->getSize(dims - 2) != (unsigned int)futureHeight ||
                     externalDataObject->getSize(dims - 1) != (unsigned int)futureWidth ||
                     externalDataObject->getType() != futureType)
            {
                return ito::RetVal(ito::retError, 0,
                                   tr("Error during check data, external dataObject invalid. Object must be of right "
                                      "size and type or an uninitilized image.")
                                       .toLatin1()
                                       .data());
            }
        }
    }
    else
    {
        int numChannel = m_params["sizez"].getVal<int>();
        if (externalDataObject == NULL)
        {
            if (m_data.getDims() < 3 || m_data.getSize(0) != (unsigned int)numChannel ||
                m_data.getSize(1) != (unsigned int)futureHeight || m_data.getSize(2) != (unsigned int)futureWidth ||
                m_data.getType() != futureType)
            {
                m_data = ito::DataObject(numChannel, futureHeight, futureWidth, futureType);
            }
            m_data.setAxisScale(0, axisScale[0]);
            m_data.setAxisScale(1, axisScale[1]);
            m_data.setAxisOffset(0, axisOffset[0]);
            m_data.setAxisOffset(1, axisOffset[1]);
            m_data.setAxisDescription(0, axisDescription[0].toLatin1().data());
            m_data.setAxisDescription(1, axisDescription[1].toLatin1().data());
            m_data.setAxisUnit(0, axisUnit[0].toLatin1().data());
            m_data.setAxisUnit(1, axisUnit[1].toLatin1().data());
            m_data.setValueDescription(valueDescription.toLatin1().data());
            m_data.setValueUnit(valueUnit.toLatin1().data());
        }
        else
        {
            int dims = externalDataObject->getDims();
            if (externalDataObject->getDims() == 0)
            {
                *externalDataObject = ito::DataObject(numChannel, futureHeight, futureWidth, futureType);
            }
            else if (externalDataObject->getSize(dims - 3) != (unsigned int)numChannel ||
                     externalDataObject->getSize(dims - 2) != (unsigned int)futureHeight ||
                     externalDataObject->getSize(dims - 1) != (unsigned int)futureWidth ||
                     externalDataObject->getType() != futureType)
            {
                return ito::RetVal(ito::retError, 0,
                                   tr("Error during check data, external dataObject invalid. Object must be of right "
                                      "size and type or an uninitilized image.")
                                       .toLatin1()
                                       .data());
            }
        }
    }


    return ito::retOk;
}
//-------------------------------------------------------------------------------------
//! sends m_image to all registered listeners.
/*!
This method is continuously called from timerEvent. Also call this method from your getVal-Method (usually with
0-timeout)

\param [in] waitMS indicates the time (in ms) that should be waiting until every registered live image source node
received m_image. 0: no wait, -1: infinit waiting time, else time in milliseconds \return retOk if everything was ok,
retWarning if live image could not be invoked
*/
ito::RetVal AddInGrabber::sendDataToListeners(int waitMS)
{
    QObject *obj;
    ito::RetVal retValue = ito::retOk;
    //        int i=0;
    int size = m_autoGrabbingListeners.size();

    if (waitMS == 0)
    {
        foreach (obj, m_autoGrabbingListeners)
        {
            if (!QMetaObject::invokeMethod(obj, "setSource",
                                           Q_ARG(QSharedPointer<ito::DataObject>,
                                                 QSharedPointer<ito::DataObject>(new ito::DataObject(m_data))),
                                           Q_ARG(ItomSharedSemaphore *, NULL)))
            {
                retValue +=
                    ito::RetVal(ito::retWarning, 1001,
                                tr("slot 'setSource' of live source node could not be invoked").toLatin1().data());
            }
        }
    }
    else if (m_autoGrabbingListeners.size() > 0)
    {
        ItomSharedSemaphore **waitConds = new ItomSharedSemaphore *[size];
        int i = 0;

        foreach (obj, m_autoGrabbingListeners)
        {
            waitConds[i] = new ItomSharedSemaphore();
            // \todo On Linux a crash occurs here when closing the liveImage ... maybe the same reason why we get an
            // error message on windows?
            if (!QMetaObject::invokeMethod(obj, "setSource",
                                           Q_ARG(QSharedPointer<ito::DataObject>,
                                                 QSharedPointer<ito::DataObject>(new ito::DataObject(m_data))),
                                           Q_ARG(ItomSharedSemaphore *, waitConds[i])))
            {
                retValue +=
                    ito::RetVal(ito::retWarning, 1001,
                                tr("slot 'setSource' of live source node could not be invoked").toLatin1().data());
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

} // end namespace ito
