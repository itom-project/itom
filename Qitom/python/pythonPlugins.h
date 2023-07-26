/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef PYTHONPLUGINS
#define PYTHONPLUGINS

#include <string>

#ifndef Q_MOC_RUN
    #include "python/pythonWrapper.h"
#endif

#include "../common/addInInterface.h"
//#include "pythonQtSignalMapper.h"

namespace ito
{
    class PythonQtSignalMapper; //forward declaration
/** @class PythonPlugins
*   @brief  class summing up the functionality of itom - hardware python plugins
*
*   There exist three different types of plugins for the itom program:
*       - actuator plugins  (everything that moves)
*       - dataIO plugins (in- and output of data, e.g. ADDA cards, frame grabber, cameras, spectrometers, ...)
*       - algo plugins (pure software evaluations / calculations)
*
*   The python interfaces for these plugins are declared in this class. For a more detailed description about
*   the plugin interface their handling and so on see the documentation of the according classes \ref AddInInterfaceBase,
*   \ref AddInBase, \ref AddInActuator, \ref AddInDataIO, \ref AddInAlgo and \ref AddInManager.
*/
class PythonPlugins
{
   public:
       typedef struct
       {
           PyObject_HEAD
           ito::AddInActuator *actuatorObj;
           PyObject* base;
           PyObject *weakreflist; /* List of weak references */
           PythonQtSignalMapper *signalMapper;
           bool userMutexLocked; //!< true if the user mutex has been recently locked by Python
       }
       PyActuatorPlugin;

       typedef struct
       {
           PyObject_HEAD
           ito::AddInDataIO *dataIOObj;
           PyObject* base;
           PyObject *weakreflist; /* List of weak references */
           PythonQtSignalMapper *signalMapper;
           bool userMutexLocked; //!< true if the user mutex has been recently locked by Python
       }
       PyDataIOPlugin;

       // Actuator
       static void PyActuatorPlugin_dealloc(PyActuatorPlugin *self);
       static PyObject *PyActuatorPlugin_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
       static int PyActuatorPlugin_init(PyActuatorPlugin *self, PyObject *args, PyObject *kwds);
       static PyObject* PyActuatorPlugin_repr(PyActuatorPlugin *self);

       static PyObject *PyActuatorPlugin_name(PyActuatorPlugin *self);
       static PyObject *PyActuatorPlugin_getParamList(PyActuatorPlugin *self);
       static PyObject *PyActuatorPlugin_getParamListInfo(PyActuatorPlugin *self,  PyObject *args);
       static PyObject *PyActuatorPlugin_getExecFuncsList(PyActuatorPlugin *self);
       static PyObject* PyActuatorPlugin_getExecFuncsInfo(PyActuatorPlugin* self, PyObject *args, PyObject *kwds);
       static PyObject *PyActuatorPlugin_getParam(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_getParamInfo(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_setParam(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_getType(PyActuatorPlugin *self);
       static PyObject *PyActuatorPlugin_execFunc(PyActuatorPlugin *self, PyObject *args, PyObject *kwds);
       static PyObject *PyActuatorPlugin_showConfiguration(PyActuatorPlugin *self);
       static PyObject *PyActuatorPlugin_showToolbox(PyActuatorPlugin *self);
       static PyObject *PyActuatorPlugin_hideToolbox(PyActuatorPlugin *self);

       static PyObject *PyActuatorPlugin_connect(PyActuatorPlugin *self, PyObject* args, PyObject* kwds);
       static PyObject *PyActuatorPlugin_disconnect(PyActuatorPlugin *self, PyObject* args, PyObject* kwds);
       static PyObject *PyActuatorPlugin_info(PyActuatorPlugin *self, PyObject* args);
       static PyObject *PyActuatorPlugin_setInterrupt(PyActuatorPlugin *self);

       static PyObject *PyActuatorPlugin_calib(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_setOrigin(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_getStatus(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_getPos(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_setPosAbs(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_setPosRel(PyActuatorPlugin *self, PyObject *args);

       static PyObject *PyActuatorPlugin_userMutex_tryLock(PyActuatorPlugin* self, PyObject* args, PyObject* kwds);
       static PyObject *PyActuatorPlugin_userMutex_unlock(PyActuatorPlugin* self);

       static PyObject* PyActuatorPlugin_getCurrentStatus(PyActuatorPlugin *self, void *closure);
       static PyObject* PyActuatorPlugin_getCurrentPositions(PyActuatorPlugin *self, void *closure);
       static PyObject* PyActuatorPlugin_getTargetPositions(PyActuatorPlugin *self, void *closure);

       static PyMemberDef  PyActuatorPlugin_members[];
       static PyMethodDef  PyActuatorPlugin_methods[];
       static PyGetSetDef  PyActuatorPlugin_getseters[];
       static PyTypeObject PyActuatorPluginType;
       static PyModuleDef  PyActuatorPluginModule;

       static void paramBaseVectorDeleter(QVector<ito::ParamBase> *obj)
       {
           delete obj;
       }

       //DataIO
       static void PyDataIOPlugin_dealloc(PyDataIOPlugin *self);
       static PyObject *PyDataIOPlugin_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
       static int PyDataIOPlugin_init(PyDataIOPlugin *self, PyObject *args, PyObject *kwds);
       static PyObject* PyDataIOPlugin_repr(PyDataIOPlugin *self);

       static PyObject *PyDataIOPlugin_name(PyDataIOPlugin *self);
       static PyObject *PyDataIOPlugin_getParamList(PyDataIOPlugin *self);
       static PyObject* PyDataIOPlugin_getParamListInfo(PyDataIOPlugin* self,  PyObject *args);
	   static PyObject* PyDataIOPlugin_getExecFuncsList(PyActuatorPlugin* self);
       static PyObject* PyDataIOPlugin_getExecFuncsInfo(PyDataIOPlugin* self, PyObject *args, PyObject *kwds);
       static PyObject *PyDataIOPlugin_getParam(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_getParamInfo(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_setParam(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_getType(PyDataIOPlugin *self);
       static PyObject *PyDataIOPlugin_execFunc(PyDataIOPlugin *self, PyObject *args, PyObject *kwds);
       static PyObject *PyDataIOPlugin_showConfiguration(PyDataIOPlugin *self);
       static PyObject *PyDataIOPlugin_showToolbox(PyDataIOPlugin *self);
       static PyObject *PyDataIOPlugin_hideToolbox(PyDataIOPlugin *self);

       static PyObject *PyDataIOPlugin_startDevice(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_stopDevice(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_acquire(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_stop(PyDataIOPlugin *self);
       static PyObject *PyDataIOPlugin_getVal(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_copyVal(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_setVal(PyDataIOPlugin *self, PyObject *args, PyObject* kwds);
       static PyObject *PyDataIOPlugin_enableAutoGrabbing(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_disableAutoGrabbing(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_setAutoGrabbing(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_getAutoGrabbing(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_setAutoGrabbingInterval(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_getAutoGrabbingInterval(PyDataIOPlugin *self);
       static PyObject *PyDataIOPlugin_connect(PyDataIOPlugin *self, PyObject *args, PyObject* kwds);
       static PyObject *PyDataIOPlugin_disconnect(PyDataIOPlugin *self, PyObject *args, PyObject* kwds);
       static PyObject *PyDataIOPlugin_info(PyDataIOPlugin* self, PyObject* args);

       static PyObject *PyDataIOPlugin_userMutex_tryLock(PyDataIOPlugin* self, PyObject* args, PyObject* kwds);
       static PyObject *PyDataIOPlugin_userMutex_unlock(PyDataIOPlugin* self);

       static PyMemberDef  PyDataIOPlugin_members[];
       static PyMethodDef  PyDataIOPlugin_methods[];
       static PyTypeObject PyDataIOPluginType;
       static PyModuleDef  PyDataIOPluginModule;
       static void PyDataIOPlugin_addTpDict(PyObject *tp_dict);
       static void PyActuatorPlugin_addTpDict(PyObject* tp_dict);

};

} //end namespace ito

#endif
