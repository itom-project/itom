/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#ifndef HELPERGRABBER_H
#define HELPERGRABBER_H

#include "typeDefs.h"
#include "addInGrabber.h"
#include "sharedStructures.h"

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{

    //----------------------------------------------------------------------------------------------------------------------------------
    /** @class threadCamera
    *   @brief Helper class to give plugin-Developer an easy access to cameras in other threads
    *
    *   The threadCamera-Class can be used in filters and algorithms when a camera (framegrabber) in another thread is neccessary.
    *   Every capture procedure starts with the startDevice() to set the camera active and is ended with stopDevice().
    *   Between this a undefined number of captures can be done. A capture procedure constist either of
    *   threadCamera::acquire(...) + threadCamera::getVal(...) or
    *   threadCamera::acquire(...) + threadCamera::copyVal(...).
    *   Thereby the acquire only triggers an exposure and DO NOT wait until it is finished. This is done by getVal or copyVal!
    *   The acquire / getVal combination returns a shallow copy of the inner dataObject in the grabber. After the next acquire / getVal the content of
    *   the result will be overwritten though it is not deep copied earlier.
    *   The acquire / copyVal combination returns a deep copy of the grabber memory to the defined external dataObject.
    *
    * \warning check the shallow-copy / deep copy part
    * \author Wolfram Lyda (ITO)
    * \date 04.2012
    */

    //this class is deprecated. Please consider using ito::CameraThreadCtrl.
    class ITOMCOMMONQT_EXPORT threadCamera
    {
        private:
            ito::AddInGrabber *pMyCamera;                   /*! < Handle to the Grabber */
            ito::RetVal errorBuffer;                        /*! < Buffer containing unrecieved errors from the semaphore */
            ItomSharedSemaphoreLocker pMySemaphoreLocker;              /*! < Handle to the semaphore needed for thread save communication. Allocated in constructor*/
            inline ito::RetVal securityChecks();            /* ! < Checks if pMyCamera and pMySemaphore are != NULL and if pMySemaphore has already waited still last commant (see setPosAbs / setPosRel) */
            ito::RetVal waitForSemaphore(int timeOutMS = PLUGINWAIT);    /*! < Wait until camera-thread has finished the last command */

        protected:

        public:

            threadCamera(QVector<ito::ParamBase> *parameterVector, int paramNumber); /*! < Constructor */
            ~threadCamera();  /*! < Desctructor */

            ito::RetVal startDevice(int timeOutMS = PLUGINWAIT);                     /*! < Set camera active */
            ito::RetVal stopDevice(int timeOutMS = PLUGINWAIT);                      /*! < Set camera deactive */
            ito::RetVal acquire(const int trigger, int timeOutMS = PLUGINWAIT);      /*! < Trigger an exposure and return before image is done*/
            ito::RetVal getVal(ito::DataObject &dObj, int timeOutMS = PLUGINWAIT);   /*! < Get a shallow-copy of the dataObject */
            ito::RetVal copyVal(ito::DataObject &dObj, int timeOutMS = PLUGINWAIT);  /*! < Get a deep-copy of the dataObject */


            ito::RetVal getParam(ito::Param &val, int timeOutMS = PLUGINWAIT);      /*! < Get the parameter of the stage */
            ito::RetVal setParam(ito::ParamBase val, int timeOutMS = PLUGINWAIT);       /*! < Set the parameter of the stage */

            ito::RetVal getImageParams(int &bpp, int &xsize, int &ysize, int timeOutMS = PLUGINWAIT); /*! < Combined function to get the most important camera features */
    };


    class ITOMCOMMONQT_EXPORT CameraThreadCtrl
    {
        private:
            ito::AddInGrabber *m_pCamera;                   /*! < Handle to the Grabber */
            ito::RetVal waitForSemaphore(ItomSharedSemaphore *waitCond, int timeOutMS = PLUGINWAIT);    /*! < Wait until camera-thread has finished the last command */

        public:
            CameraThreadCtrl();                                                                  /*! < Constructor */
            CameraThreadCtrl(const ito::ParamBase &cameraParameter, ito::RetVal *retval = NULL); /*! < Constructor */
            CameraThreadCtrl(ito::AddInGrabber *camera, ito::RetVal *retval = NULL);             /*! < Constructor */
            CameraThreadCtrl(CameraThreadCtrl &other);
            ~CameraThreadCtrl();                                                                 /*! < Destructor */

            ito::RetVal startDevice(int timeOutMS = PLUGINWAIT);                     /*! < Set camera active */
            ito::RetVal stopDevice(int timeOutMS = PLUGINWAIT);                      /*! < Set camera deactive */
            ito::RetVal acquire(const int trigger, int timeOutMS = PLUGINWAIT);      /*! < Trigger an exposure and return before image is done*/
            ito::RetVal getVal(ito::DataObject &dObj, int timeOutMS = PLUGINWAIT);   /*! < Get a shallow-copy of the dataObject */
            ito::RetVal copyVal(ito::DataObject &dObj, int timeOutMS = PLUGINWAIT);  /*! < Get a deep-copy of the dataObject */

            ito::RetVal getParam(ito::Param &val, int timeOutMS = PLUGINWAIT);      /*! < Get the parameter of the stage */
            ito::RetVal setParam(ito::ParamBase val, int timeOutMS = PLUGINWAIT);       /*! < Set the parameter of the stage */

            ito::RetVal getImageParams(int &bpp, int &sizex, int &sizey, int timeOutMS = PLUGINWAIT); /*! < Combined function to get the most important camera features */
    };

}   // end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif