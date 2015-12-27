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
#include "addInInterface.h"
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
* \date 04.2012
*/
class ITOMCOMMONQT_EXPORT PluginThreadCtrl
{
protected:
    ito::AddInBase *m_pPlugin;                   /*! < Handle to the plugin */
    ito::RetVal waitForSemaphore(ItomSharedSemaphore *waitCond, int timeOutMS = PLUGINWAIT);    /*! < Wait until camera-thread has finished the last command */

public:
    PluginThreadCtrl();                                                                  /*! < Constructor */
    PluginThreadCtrl(const ito::ParamBase &pluginParameter, ito::RetVal *retval = NULL); /*! < Constructor */
    PluginThreadCtrl(ito::AddInBase *plugin, ito::RetVal *retval = NULL);                /*! < Constructor */
    PluginThreadCtrl(const PluginThreadCtrl &other);
    virtual ~PluginThreadCtrl();                                                         /*! < Destructor */

    PluginThreadCtrl& operator =(const PluginThreadCtrl &other);

    ito::RetVal getParam(ito::Param &val, int timeOutMS = PLUGINWAIT);      /*! < Get the parameter of the plugin */
    ito::RetVal setParam(ito::ParamBase val, int timeOutMS = PLUGINWAIT);       /*! < Set the parameter of the plugin */
};


//-----------------------------------------------------------------------------------
class ITOMCOMMONQT_EXPORT DataIOThreadCtrl : public PluginThreadCtrl
{
public:
    DataIOThreadCtrl();                                                                  /*! < Constructor */
    DataIOThreadCtrl(const ito::ParamBase &pluginParameter, ito::RetVal *retval = NULL); /*! < Constructor */
    DataIOThreadCtrl(ito::AddInDataIO *plugin, ito::RetVal *retval = NULL);              /*! < Constructor */
    DataIOThreadCtrl(const DataIOThreadCtrl &other);
    virtual ~DataIOThreadCtrl();                                                         /*! < Destructor */

    ito::RetVal startDevice(int timeOutMS = PLUGINWAIT);                     /*! < Set camera active */
    ito::RetVal stopDevice(int timeOutMS = PLUGINWAIT);                      /*! < Set camera deactive */
    ito::RetVal acquire(const int trigger = 0, int timeOutMS = PLUGINWAIT);  /*! < Trigger an exposure and return before image is done*/
    ito::RetVal getVal(ito::DataObject &dObj, int timeOutMS = PLUGINWAIT);   /*! < Get a shallow-copy of the dataObject */
    ito::RetVal copyVal(ito::DataObject &dObj, int timeOutMS = PLUGINWAIT);  /*! < Get a deep-copy of the dataObject */

    ito::RetVal getImageParams(int &bpp, int &sizex, int &sizey, int timeOutMS = PLUGINWAIT); /*! < Combined function to get the most important camera features */
};

//-----------------------------------------------------------------------------------
class ITOMCOMMONQT_EXPORT ActuatorThreadCtrl : public PluginThreadCtrl
{
protected:
    int m_numAxes;

public:
    ActuatorThreadCtrl();                                                                  /*! < Constructor */
    ActuatorThreadCtrl(const ito::ParamBase &pluginParameter, ito::RetVal *retval = NULL); /*! < Constructor */
    ActuatorThreadCtrl(ito::AddInActuator *plugin, ito::RetVal *retval = NULL);            /*! < Constructor */
    ActuatorThreadCtrl(const ActuatorThreadCtrl &other);
    virtual ~ActuatorThreadCtrl();                                                         /*! < Destructor */

    ito::RetVal setPosRel(const QVector<int> &axes, const QVector<double> &relPositions, int timeOutMS = PLUGINWAIT);  /*! < Move more than on axis relativ to current position */
    ito::RetVal setPosAbs(const QVector<int> &axes, const QVector<double> &absPositions, int timeOutMS = PLUGINWAIT);  /*! < Move more than on axis absolute*/
    ito::RetVal setPosRel(int axis, double relPosition, int timeOutMS = PLUGINWAIT);                       /*! < Move a single axis relativ to current position */
    ito::RetVal setPosAbs(int axis, double absPosition, int timeOutMS = PLUGINWAIT);                       /*! < Move a single axi absolute*/

    ito::RetVal getPos(QVector<int> axes, QVector<double> &positions, int timeOutMS = PLUGINWAIT);         /*! < Get the position of more than one axis */
    ito::RetVal getPos(int axis, double &position, int timeOutMS = PLUGINWAIT);                            /*! < Get the position of a single axis */

    ito::RetVal checkAxis(int axisNum);                                                                    /*! < Check if an axis is within the axis-range */
};

}   // end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif