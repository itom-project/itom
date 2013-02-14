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

#include "helperActuator.h"
#include <iostream>

namespace ito
{
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail The constructor checks if parameterVector[paramNumber] is a valid actuator.
                If yes and if the actuator is accessable by getParam the actuatorhandle is stored in pMyMotor. The Semaphore for the invoke-method is also allocated here.
                If the actuator is invalid or not accessable the pMyMotor keeps beeing NULL.

        \param [in] parameterVector     is the ParameterVector (optional or mandatory ) of the filter / algorithm
        \param [in] paramNumber         is the zerobased number of the actuator in the parameterlist
        \return (void)
        \sa threadActuator
    */
    threadActuator::threadActuator(QVector<ito::ParamBase> *parameterVector, int paramNumber)
    {
        ito::RetVal retval(ito::retOk);

        errorBuffer = ito::retOk;
        pMyMotor = NULL;
        axisNumbers = 0;

        if(parameterVector->isEmpty())
            return;

        if(parameterVector->size() - 1 < paramNumber)
            return;

        if (reinterpret_cast<ito::AddInBase *>((*parameterVector)[paramNumber].getVal<void *>())->getBasePlugin()->getType() & (ito::typeActuator))
        {
            pMyMotor = (ito::AddInActuator *)(*parameterVector)[paramNumber].getVal<void *>();
        }

        if(pMyMotor == NULL)
        {
            return;
        }

        pMySemaphoreLocker = new ItomSharedSemaphore();

        QSharedPointer<ito::Param> qsParam(new ito::Param("numaxis", ito::ParamBase::Int));
        QMetaObject::invokeMethod(pMyMotor, "getParam", Q_ARG(QSharedPointer<ito::Param>, qsParam), Q_ARG(ItomSharedSemaphore *, pMySemaphoreLocker.getSemaphore()));

        while (!pMySemaphoreLocker.getSemaphore()->wait(PLUGINWAIT))
        {
            if (!pMyMotor->isAlive())
            {
                retval += ito::RetVal(ito::retError, 0, "timeout while getting numaxis parameter");
                break;
            }
        }

        retval += pMySemaphoreLocker.getSemaphore()->returnValue;

        if(retval.containsWarningOrError())
        {
            pMyMotor = NULL;
            return;
        }

        axisNumbers = (*qsParam).getVal<int>();

        return;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail The destructor. Deletes the semaphore after waiting a last time.
        \return (void)
        \sa threadActuator
    */
    threadActuator::~threadActuator()
    {
        if(pMySemaphoreLocker.getSemaphore())
        {
            while (!pMySemaphoreLocker.getSemaphore()->wait(PLUGINWAIT))
            {
                if (!pMyMotor->isAlive())
                {
                    std::cout << "The semaphore of a threadActuator could not be deleted and is now a ZOMBIE in your memory";
                    return;
                }
            }
        }

        return;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail This function is called by every subroutine of the threadActuator. It checks if the motor-handle and the semaphore handle is zero and if the semaphore has waited after last command.
                If the semaphore droppes or dropped to time-out it returns retError.

        \return retOk or retError
        \sa threadActuator
    */
    inline ito::RetVal threadActuator::securityChecks()
    {
        ito::RetVal retval(ito::retOk);

        if(!pMyMotor)
        {
            return ito::RetVal(ito::retError, 0, "Motor not correctly initialized");
        }

        if(pMySemaphoreLocker.getSemaphore())
        {
            while (!pMySemaphoreLocker.getSemaphore()->wait(PLUGINWAIT))
            {
                if (!pMyMotor->isAlive())
                {
                    retval += ito::RetVal(ito::retError, 0, "Timeout while Waiting for Semaphore");
                    break;
                }
            }
            if(pMySemaphoreLocker.getSemaphore()->returnValue.containsWarningOrError())
            {
                errorBuffer += pMySemaphoreLocker.getSemaphore()->returnValue;
                retval += ito::RetVal(ito::retWarning, 0, "Semaphore contained error");
            }
        }
        return retval;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail Move the axis in axisVec with a distance defined in stepSizeVec relative to current position.
                The axisVec and stepSizeVec must be same size. After the invoke-command this thread must wait / synchronize with the actuator-thread.
                Therefore the semaphore->wait is called via the function threadActuator::waitForSemaphore(timeOutMS)
                To enable the algorithm to process data during movement, the waitForSemaphore(timeOutMS) can be skipped by callWait = false.
                The threadActuator::waitForSemaphore(timeOutMS)-function must than be called by the algorithms afterwards / before the next command is send to the actuator.

        \param [in] axisVec         Vector with the axis to move
        \param [in] stepSizeVec     Vector with the distances for every axis
        \param [in] timeOutMS       TimeOut for the semaphore-wait, if(0) the waitForSemaphore is not called and must be called seperate by the algorithm

        \return retOk or retError
        \sa threadActuator
    */
    ito::RetVal threadActuator::setPosRel(QVector<int> axisVec, QVector<double> stepSizeVec, int timeOutMS)
    {
        ito::RetVal retval = securityChecks();
        if(retval.containsError())
        {
            return retval;
        }

        if(stepSizeVec.size() != axisVec.size())
        {
            return ito::RetVal(ito::retError, 0, "Error during setPosRel: Vectors differ in size");
        }

        pMySemaphoreLocker = new ItomSharedSemaphore();
        QMetaObject::invokeMethod(pMyMotor, "setPosRel", Q_ARG(QVector<int>, axisVec), Q_ARG(QVector<double>, stepSizeVec), Q_ARG(ItomSharedSemaphore *, pMySemaphoreLocker.getSemaphore()));
        if(timeOutMS)
        {
            return waitForSemaphore(timeOutMS);
        }

        return retval;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail Move the axis in axisVec to the positions given in posVec.
                The axisVec and posVec must be same size. After the invoke-command this thread must wait / synchronize with the actuator-thread.
                Therefore the semaphore->wait is called via the function threadActuator::waitForSemaphore(timeOutMS)
                To enable the algorithm to process data during movement, the waitForSemaphore(timeOutMS) can be skipped by callWait = false.
                The threadActuator::waitForSemaphore(timeOutMS)-function must than be called by the algorithms afterwards / before the next command is send to the actuator.

        \param [in] axisVec         Vector with the axis to move
        \param [in] posVec          Vector with the new absolute positions
        \param [in] timeOutMS       TimeOut for the semaphore-wait, if(0) the waitForSemaphore is not called and must be called seperate by the algorithm

        \return retOk or retError
        \sa threadActuator
    */
    ito::RetVal threadActuator::setPosAbs(QVector<int> axisVec, QVector<double> posVec, int timeOutMS)
    {
        ito::RetVal retval = securityChecks();
        if(retval.containsError())
        {
            return retval;
        }

        if(posVec.size() != axisVec.size())
        {
            return ito::RetVal(ito::retError, 0, "Error during setPosRel: Vectors differ in size");
        }

        pMySemaphoreLocker = new ItomSharedSemaphore();

        QMetaObject::invokeMethod(pMyMotor, "setPosAbs", Q_ARG(QVector<int>, axisVec), Q_ARG(QVector<double>, posVec), Q_ARG(ItomSharedSemaphore *, pMySemaphoreLocker.getSemaphore()));
        if(timeOutMS)
        {
            return waitForSemaphore(timeOutMS);
        }
        return retval;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail Move a single axis specified by axis  with a distance defined in stepSize relative to current position. After the invoke-command this thread must wait / synchronize with the actuator-thread.
                Therefore the semaphore->wait is called via the function threadActuator::waitForSemaphore(timeOutMS)
                To enable the algorithm to process data during movement, the waitForSemaphore(timeOutMS) can be skipped by callWait = false.
                The threadActuator::waitForSemaphore(timeOutMS)-function must than be called by the algorithms afterwards / before the next command is send to the actuator.

        \param [in] axis         Number of the axis
        \param [in] stepSize     Distances from current position
        \param [in] timeOutMS       TimeOut for the semaphore-wait, if(0) the waitForSemaphore is not called and must be called seperate by the algorithm

        \return retOk or retError
        \sa threadActuator
    */
    ito::RetVal threadActuator::setPosRel(int axis, double stepSize, int timeOutMS)
    {
        ito::RetVal retval = securityChecks();
        if(retval.containsError())
        {
            return retval;
        }

        pMySemaphoreLocker = new ItomSharedSemaphore();

        QMetaObject::invokeMethod(pMyMotor, "setPosRel", Q_ARG(const int, (const int) axis), Q_ARG(const double, (const double)(stepSize)), Q_ARG(ItomSharedSemaphore *, pMySemaphoreLocker.getSemaphore()));

        if(timeOutMS)
        {
            return waitForSemaphore(timeOutMS);
        }
        return retval;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail Move a single axis specified by axis to the position pos. After the invoke-command this thread must wait / synchronize with the actuator-thread.
                Therefore the semaphore->wait is called via the function threadActuator::waitForSemaphore(timeOutMS)
                To enable the algorithm to process data during movement, the waitForSemaphore(timeOutMS) can be skipped by callWait = false.
                The threadActuator::waitForSemaphore(timeOutMS)-function must than be called by the algorithms afterwards / before the next command is send to the actuator.

        \param [in] axis         Number of the axis
        \param [in] pos          New position of the axis
        \param [in] timeOutMS       TimeOut for the semaphore-wait, if(0) the waitForSemaphore is not called and must be called seperate by the algorithm

        \return retOk or retError
        \sa threadActuator
    */
    ito::RetVal threadActuator::setPosAbs(int axis, double pos, int timeOutMS)
    {
        ito::RetVal retval = securityChecks();
        if(retval.containsError())
        {
            return retval;
        }

        pMySemaphoreLocker = new ItomSharedSemaphore();

        QMetaObject::invokeMethod(pMyMotor, "setPosAbs", Q_ARG(const int, (const int) axis), Q_ARG(const double, (const double)(pos)), Q_ARG(ItomSharedSemaphore *, pMySemaphoreLocker.getSemaphore()));

        if(timeOutMS)
        {
            return waitForSemaphore(timeOutMS);
        }

        return retval;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail After the invoke-command this thread must wait / be synchronize with the actuator-thread.
                Therefore the wait-Function of pMySemaphore is called. If the actuator do not answer within timeOutMS and the pMyMotor is not alive anymore, the function returns a timeout.

        \param [in] timeOutMS    TimeOut for the semaphore-wait

        \return retOk or retError
        \sa threadActuator
    */
    ito::RetVal threadActuator::waitForSemaphore(int timeOutMS)
    {
        ito::RetVal retval(ito::retOk);

        if(!pMySemaphoreLocker.getSemaphore())
        {
            return ito::RetVal(ito::retError, 0, "Semaphore not correctly initialized");
        }

        while (!pMySemaphoreLocker.getSemaphore()->wait(timeOutMS))
        {
            if (!pMyMotor->isAlive())
            {
                return ito::RetVal(ito::retError, 0, "Timeout while Waiting for Semaphore");
            }
        }

        retval = pMySemaphoreLocker.getSemaphore()->returnValue;

        return retval;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail Get the position of a single axis specified by axis.

        \param [in] axis         Number of the axis
        \param [out] pos          position of the axis
        \param [in] timeOutMS    TimeOut for the semaphore-wait

        \return retOk or retError
        \sa threadActuator
    */
    ito::RetVal threadActuator::getPos(int axis, double &pos, int timeOutMS)
    {
        ito::RetVal retval = securityChecks();
        if(retval.containsError())
        {
            return retval;
        }

        QSharedPointer<double> posSP(new double);
        *posSP = 0.0;

        pMySemaphoreLocker = new ItomSharedSemaphore();

        QMetaObject::invokeMethod(pMyMotor, "getPos", Q_ARG(const int, (const int) axis), Q_ARG(QSharedPointer<double>, posSP), Q_ARG(ItomSharedSemaphore *, pMySemaphoreLocker.getSemaphore()));

        retval += waitForSemaphore(timeOutMS);
        pos = *posSP;

        return retval;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail Get the position of a number of axis specified by axisVec.

        \param [in] axisVec         Number of the axis
        \param [out] posVec         Vecotr with position of the axis
        \param [in] timeOutMS    TimeOut for the semaphore-wait

        \return retOk or retError
        \sa threadActuator
    */
    ito::RetVal threadActuator::getPos(QVector<int> axisVec, QVector<double> &posVec, int timeOutMS)
    {
        ito::RetVal retval = securityChecks();
        if(retval.containsError())
        {
            return retval;
        }

        posVec.clear();

        QSharedPointer<QVector<double> > posVecSP(new QVector<double>());

        for(int i = 0; i <  axisVec.size(); i++)
        {
            posVecSP->append(0.0);
        }

        pMySemaphoreLocker = new ItomSharedSemaphore();

        QMetaObject::invokeMethod(pMyMotor, "getPos", Q_ARG(QVector<int>, axisVec), Q_ARG(QSharedPointer<QVector<double> >, posVecSP), Q_ARG(ItomSharedSemaphore *, pMySemaphoreLocker.getSemaphore()));

        retval += waitForSemaphore(timeOutMS);

        for(int i = 0; i <  axisVec.size(); i++)
        {
            posVec.append((*posVecSP)[i]);
        }

        return retval;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail Get any parameter of the actuator defined by val.name. val must be initialised and name must be correct. After correct execution, val has the correct value.

        \param [in|out] val      Initialised tParam (correct name | in)
        \param [in] timeOutMS    TimeOut for the semaphore-wait

        \return retOk or retError
        \sa threadActuator
    */
    ito::RetVal threadActuator::getParam(ito::Param &val, int timeOutMS)
    {
        ito::RetVal retval = securityChecks();
        if(retval.containsError())
        {
            return retval;
        }
        QSharedPointer<ito::Param> qsParam(new ito::Param(val));

        pMySemaphoreLocker = new ItomSharedSemaphore();

        QMetaObject::invokeMethod(pMyMotor, "getParam", Q_ARG(QSharedPointer<ito::Param>, qsParam), Q_ARG(ItomSharedSemaphore *, pMySemaphoreLocker.getSemaphore()));

        retval += waitForSemaphore(timeOutMS);
        val = *qsParam;
        return retval;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail Get the parameter of the actuator defined by val.name to the value of val.

        \param [in] val         Initialised tParam (correct name | value)
        \param [in] timeOutMS    TimeOut for the semaphore-wait

        \return retOk or retError
        \sa threadActuator
    */
    ito::RetVal threadActuator::setParam(ito::ParamBase val, int timeOutMS)
    {
        ito::RetVal retval = securityChecks();
        if(retval.containsError())
        {
            return retval;
        }
        QSharedPointer<ito::ParamBase> qsParam(new ito::ParamBase(val));

        pMySemaphoreLocker = new ItomSharedSemaphore();

        QMetaObject::invokeMethod(pMyMotor, "setParam", Q_ARG(QSharedPointer<ito::ParamBase>, qsParam), Q_ARG(ItomSharedSemaphore *, pMySemaphoreLocker.getSemaphore()));

        retval += waitForSemaphore(timeOutMS);
        return retval;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail Check if a specific axis is within the axisSpace of this actuator

        \param [in] axisNum    Axisnumber

        \return retOk or retError
        \sa threadActuator
    */
    ito::RetVal threadActuator::checkAxis(int axisNum)
    {
        if(axisNum < 0 && (axisNum + 1) < axisNumbers)
        {
            return ito::retError;
        }
        else
        {
            return ito::retOk;
        }
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail Returns the last unrecieved errors and warnings and resets the internel errorBuffer.

        \return retOk or retError
        \sa threadActuator
    */
    ito::RetVal threadActuator::getErrorBuf(void)
    {
       ito::RetVal retval = errorBuffer;
       errorBuffer = ito::RetVal(ito::retOk);
       return retval;
    }
}
