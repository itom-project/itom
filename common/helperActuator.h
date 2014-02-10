/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
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

#ifndef HELPERACTUATOR_H
#define HELPERACTUATOR_H

#include "typeDefs.h"
#include "addInInterface.h"
#include "sharedStructures.h"

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{

    //----------------------------------------------------------------------------------------------------------------------------------
    /** @class threadActuator
    *   @brief Helper class to give plugin-Developer an easy access to actuators in other threads
    *
    *   The threadActuator-Class can be used in filters and algorithms when a actuator in another thread is neccessary.
    *   To enable a slight asynchron usage of different hardware and algorithms the wait-Function can be executed seperatly
    *
    * \author Wolfram Lyda (ITO)
    * \date 04.2012
    */

    class ITOMCOMMONQT_EXPORT threadActuator
    {
        private:
            int axisNumbers;                /*! < Number of axis of the actuator */
            ito::AddInActuator *pMyMotor;   /*! < Handle to the actuator */
            ito::RetVal errorBuffer;        /*! < Buffer containing unrecieved errors from the semaphore */
            ItomSharedSemaphoreLocker pMySemaphoreLocker;  /*! < Handle to the semaphore needed for thread save communication. Allocated in constructor, deleted in destructor*/

            inline ito::RetVal securityChecks(); /* ! < Checks if pMyMotor and pMySemaphore are != NULL and if pMySemaphore has already waited still last commant (see setPosAbs / setPosRel) */

        protected:

        public:

            threadActuator(QVector<ito::ParamBase> *parameterVector, int paramNumber); /*! < Constructor */
            ~threadActuator();  /*! < Desctructor */

            ito::RetVal setPosRel(QVector<int> axisVec, QVector<double> stepSizeVec, int timeOutMS = PLUGINWAIT);  /*! < Move more than on axis relativ to current position */
            ito::RetVal setPosAbs(QVector<int> axisVec, QVector<double> posVec, int timeOutMS = PLUGINWAIT);       /*! < Move more than on axis absolute*/
            ito::RetVal setPosRel(int axis, double stepSize, int timeOutMS = PLUGINWAIT);                          /*! < Move a single axis relativ to current position */
            ito::RetVal setPosAbs(int axis, double pos, int timeOutMS = PLUGINWAIT);                               /*! < Move a single axi absolute*/
            ito::RetVal waitForSemaphore(int timeOutMS = PLUGINWAIT);                                              /*! < Wait until actuator-thread has finished the last command */

            ito::RetVal getPos(QVector<int> axisVec, QVector<double> &posVec, int timeOutMS = PLUGINWAIT);         /*! < Get the position of more than one axis */
            ito::RetVal getPos(int axis, double &pos, int timeOutMS = PLUGINWAIT);                                 /*! < Get the position of a single axis */

            ito::RetVal getParam(ito::Param &val, int timeOutMS = PLUGINWAIT);                                    /*! < Get the parameter of the stage */
            ito::RetVal setParam(ito::ParamBase val, int timeOutMS = PLUGINWAIT);                                     /*! < Set the parameter of the stage */
            ito::RetVal checkAxis(int axisNum);                                                                    /*! < Check if an axis is within the axis-range */

            ito::RetVal getErrorBuf(void);                                                                             /*! < Check if an axis is within the axis-range */
    };


}   // end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif