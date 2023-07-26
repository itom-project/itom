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

#ifndef PLUGINTHREADCTRL_H
#define PLUGINTHREADCTRL_H

#include "typeDefs.h"
#include "addInInterface.h"
#include "sharedStructures.h"

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
/*!
\class PluginThreadCtrl
\brief Base class for DataIOThreadCtrl and ActuatorThreadCtrl.

This base class only provides access to setParam and getParam of the covered plugin
in order to get or set internal parameters of the plugin. This is common for actuators
and dataIO instances.
*/
class ITOMCOMMONQT_EXPORT PluginThreadCtrl
{
protected:
    ito::AddInBase *m_pPlugin;                   /*!< Handle to the plugin */
    ItomSharedSemaphoreLocker m_semaphoreLocker; /*!< Handle to the semaphore needed for thread save communication. Allocated in constructor, deleted in destructor*/

public:
    //! default constructor. No plugin instance is currently under control.
    PluginThreadCtrl();

    //! Creates the control object for a plugin instance
    /*!
    This implementation gets the controlled instance from a ito::ParamBase object of param type ito::ParamBase::HWRef.
    Use this version, if the controlled plugin is passed to an algorithm or other plugin via a vector of mandatory or
    optional parameters in terms of ito::ParamBase objects.

    This constructor increments the reference of the controlled plugin such that the plugin is not deleted
    until the reference has been decremented in the destructor of this class.

    \param pluginParameter is a plugin parameter of type ito::ParamBase::HWRef.
    \param retval is an optional pointer to ito::RetVal. An error is set to this retval if the given plugin is no valid dataIO plugin instance.
    */
    PluginThreadCtrl(const ito::ParamBase &pluginParameter, ito::RetVal *retval = NULL);

    //! Creates the control object for a plugin instance
    /*!
    This implementation gets the controlled instance from the real pointer to a ito::AddInBase instance.

    This constructor increments the reference of the controlled plugin such that the plugin is not deleted
    until the reference has been decremented in the destructor of this class.

    \param plugin is the pointer to the controlled plugin.
    \param retval is an optional pointer to ito::RetVal. An error is set to this retval if the given plugin is no valid plugin instance.
    */
    PluginThreadCtrl(ito::AddInBase *plugin, ito::RetVal *retval = NULL);

    //! copy constructor. The reference counter of the covered plugin by other will be incremented another time.
    PluginThreadCtrl(const PluginThreadCtrl &other);

    //! destructor. Decrements the reference counter of the covered plugin and deletes it, if it drops to zero.
    virtual ~PluginThreadCtrl();

    //! assigment operator. Gets control over the plugin currently covered by other. Decrements the reference counter of the former plugin and increments it of the plugin controlled by other.
    PluginThreadCtrl& operator =(const PluginThreadCtrl &other);

    ito::RetVal getParam(ito::Param &val, int timeOutMS = PLUGINWAIT);      /*!< Get a parameter of the plugin */
    ito::RetVal setParam(ito::ParamBase val, int timeOutMS = PLUGINWAIT);   /*!< Set a parameter of the plugin */

    ito::RetVal waitForSemaphore(int timeOutMS = PLUGINWAIT);               /*!< Wait until plugin thread has finished the last command */
};


//-----------------------------------------------------------------------------------
/*!
\class DataIOThreadCtrl
\brief Helper class to give plugin developers an easy access to cameras in other threads.

The DataIOThreadCtrl-Class can be used in filters and algorithms when a camera (framegrabber) or other dataIO plugin should
be controlled by another thread. To create this controlling instance, at first, create an instance of
the dataIO plugin itself, then, pass it to the constructor of DataIOThreadCtrl. In the following example,
an instance of DummyGrabber is passed as first mandatory argument to the init method of your plugin. You would
like to create an easy thread control wrapper around this camera plugin:

\code
ito::RetVal retval;

//first alternative
ito::AddInDataIO *cam = paramsMand->at(0).getVal<ito::AddInDataIO*>();
DataIOThreadCtrl camSave(cam, &retval);

//second alternative
DataIOThreadCtrl camSave(paramsMand->at(0), &retval);
\endcode

Using this class, all main methods of the dataIO plugin can be directly called with an optional timeout value (in ms).
The thread control, timeout checks... is then done by this helper class.

To start the image acquisition of 'camSave', you have to call startDevice once. At the end of any acquisition
call stopDevice. Every single image is acquired using acquire, while the acquired image data is obtained via a
dataObject calling getVal or copyVal.

The acquire / getVal combination returns a shallow copy of the internal dataObject of the grabber. Therefore, the
content of the returned shallow copy might automatically change upon the next acquisition.
The acquire / copyVal combination returns a deep copy of the grabber memory to the defined external dataObject.

\code
retval += camSave.startDevice(2000);
if (!retval.containsError())
{
ito::DataObject image;
retval += camSave.setParam(ito::ParamBase("integration_time", ito::ParamBase::Double, 0.1));
for (int i = 0; i < 10; ++i)
{
    retval += camSave.acquire();
    retval += camSave.getVal(image);
}
retval += camSave.stopDevice();
}
\endcode
*/
class ITOMCOMMONQT_EXPORT DataIOThreadCtrl : public PluginThreadCtrl
{
public:
    //! default constructor. No dataIO instance is currently under control.
    DataIOThreadCtrl();

    //! Creates the control object for a dataIO plugin instance
    /*!
    This implementation gets the controlled instance from a ito::ParamBase object of param type ito::ParamBase::HWRef.
    Use this version, if the controlled plugin is passed to an algorithm or other plugin via a vector of mandatory or
    optional parameters in terms of ito::ParamBase objects.

    This constructor increments the reference of the controlled plugin such that the plugin is not deleted
    until the reference has been decremented in the destructor of this class.

    \param pluginParameter is a plugin parameter of type ito::ParamBase::HWRef.
    \param retval is an optional pointer to ito::RetVal. An error is set to this retval if the given plugin is no valid dataIO plugin instance.
    */
    DataIOThreadCtrl(const ito::ParamBase &pluginParameter, ito::RetVal *retval = NULL);

    //! Creates the control object for a dataIO plugin instance
    /*!
    This implementation gets the controlled instance from the real pointer to a ito::AddInDataIO instance.

    This constructor increments the reference of the controlled plugin such that the plugin is not deleted
    until the reference has been decremented in the destructor of this class.

    \param plugin is the pointer to the controlled plugin.
    \param retval is an optional pointer to ito::RetVal. An error is set to this retval if the given plugin is no valid dataIO plugin instance.
    */
    DataIOThreadCtrl(ito::AddInDataIO *plugin, ito::RetVal *retval = NULL);

    //! copy constructor. The reference counter of the covered plugin by other will be incremented another time.
    DataIOThreadCtrl(const DataIOThreadCtrl &other);

    //! destructor. Decrements the reference counter of the covered plugin and deletes it, if it drops to zero.
    virtual ~DataIOThreadCtrl();

    ito::RetVal startDevice(int timeOutMS = PLUGINWAIT);                     /*!< Set camera active */
    ito::RetVal stopDevice(int timeOutMS = PLUGINWAIT);                      /*!< Set camera deactive */
    ito::RetVal acquire(const int trigger = 0, int timeOutMS = PLUGINWAIT);  /*!< Trigger an exposure and return before image is done*/
    ito::RetVal getVal(ito::DataObject &dObj, int timeOutMS = PLUGINWAIT);   /*!< Get a shallow-copy of the dataObject */
    ito::RetVal copyVal(ito::DataObject &dObj, int timeOutMS = PLUGINWAIT);  /*!< Get a deep-copy of the dataObject */
    ito::RetVal enableAutoGrabbing(int timeOutMS = PLUGINWAIT);              /*!< Enables the timer for auto grabbing */
    ito::RetVal disableAutoGrabbing(int timeOutMS = PLUGINWAIT);             /*!< Disables the timer for auto grabbing */
    ito::RetVal setAutoGrabbingInterval(QSharedPointer<int> interval, int timeOutMS = PLUGINWAIT); /*!< Sets a new interval for the auto-grabbing timer (in ms). */
    bool getAutoGrabbing();                                                  /*!< Returns the state of m_autoGrabbingEnabled; consider this method as final */

    ito::RetVal getImageParams(int &bpp, int &sizex, int &sizey, int timeOutMS = PLUGINWAIT); /*!< Combined function to get the most important camera features */
};

//-----------------------------------------------------------------------------------
/*!
\class ActuatorThreadCtrl
\brief Helper class to give plugin developers an easy access to actuators in other threads.

The ActuatorThreadCtrl-Class can be used in filters and algorithms when an actuator plugin should
be controlled by another thread. To create this controlling instance, at first, create an instance of
the actuator plugin itself, then, pass it to the constructor of ActuatorThreadCtrl. In the following example,
an instance of DummyMotor is passed as first mandatory argument to the init method of your plugin. You would
like to create an easy thread control wrapper around this actuator plugin:

\code
ito::RetVal retval;
ActuatorThreadCtrl motSave(paramsMand->at(0), &retval);
\endcode

Using this class, all main methods of the actuator plugin can be directly called with an optional timeout value (in ms).
The thread control, timeout checks... is then done by this helper class.

Use getPos to obtain the current position (in mm or degree) for one multiple axes. Inversely, setPosRel or
setPosAbs will set the absolute or relative position(s) of one or multiple axes. Usually, setPosXXX will only
return after the position has been reached. If you want to continue within your code while the actuator is still
moving, check if your specific actuator has the 'async' parameter defined. If so, set it to 1 (see example below):

\code
double position;
motSave.setParam(ito::ParamBase("async", ito::ParamBase::Int, 1))
motSave.getPos(0, position);
motSave.setPosAbs(0, 10.5); //long movement

//since the movement is asynchrone, setPosAbs
//will immediately return. However, every subsequent
//call to the motSave instance will block until the previous
//movement is finished. E.g. another call of getPos will wait
//for this and you can therefore check if the movement has been
//done:
motSave.getPos(0, position);
\endcode
*/
class ITOMCOMMONQT_EXPORT ActuatorThreadCtrl : public PluginThreadCtrl
{
protected:
    int m_numAxes;

public:
    //! default constructor. No actuator instance is currently under control.
    ActuatorThreadCtrl();

    //! Creates the control object for an actuator plugin instance
    /*!
    This implementation gets the controlled instance from a ito::ParamBase object of param type ito::ParamBase::HWRef.
    Use this version, if the controlled plugin is passed to an algorithm or other plugin via a vector of mandatory or
    optional parameters in terms of ito::ParamBase objects.

    This constructor increments the reference of the controlled plugin such that the plugin is not deleted
    until the reference has been decremented in the destructor of this class.

    \param pluginParameter is a plugin parameter of type ito::ParamBase::HWRef.
    \param retval is an optional pointer to ito::RetVal. An error is set to this retval if the given plugin is no valid actuator plugin instance.
    */
    ActuatorThreadCtrl(const ito::ParamBase &pluginParameter, ito::RetVal *retval = NULL);

    //! Creates the control object for a actuator plugin instance
    /*!
    This implementation gets the controlled instance from the real pointer to a ito::Actuator instance.

    This constructor increments the reference of the controlled plugin such that the plugin is not deleted
    until the reference has been decremented in the destructor of this class.

    \param plugin is the pointer to the controlled plugin.
    \param retval is an optional pointer to ito::RetVal. An error is set to this retval if the given plugin is no valid actuator plugin instance.
    */
    ActuatorThreadCtrl(ito::AddInActuator *plugin, ito::RetVal *retval = NULL);

    //! copy constructor. The reference counter of the covered plugin by other will be incremented another time.
    ActuatorThreadCtrl(const ActuatorThreadCtrl &other);

    //! destructor. Decrements the reference counter of the covered plugin and deletes it, if it drops to zero.
    virtual ~ActuatorThreadCtrl();

    ito::RetVal setPosRel(const QVector<int> &axes, const QVector<double> &relPositions, int timeOutMS = PLUGINWAIT);  /*!< Move more than on axis relativ to current position */
    ito::RetVal setPosAbs(const QVector<int> &axes, const QVector<double> &absPositions, int timeOutMS = PLUGINWAIT);  /*!< Move more than on axis absolute*/
    ito::RetVal setPosRel(int axis, double relPosition, int timeOutMS = PLUGINWAIT);                       /*!< Move a single axis relativ to current position */
    ito::RetVal setPosAbs(int axis, double absPosition, int timeOutMS = PLUGINWAIT);                       /*!< Move a single axi absolute*/

    ito::RetVal getPos(QVector<int> axes, QVector<double> &positions, int timeOutMS = PLUGINWAIT);         /*!< Get the position of more than one axis */
    ito::RetVal getPos(int axis, double &position, int timeOutMS = PLUGINWAIT);                            /*!< Get the position of a single axis */


    ito::RetVal checkAxis(int axisNum);                                                              /*!< Check if an axis is within the axis-range */
};

}   // end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)
#endif //PLUGINTHREADCTRL_H
