/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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
/* includes */
//python
// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#ifdef _DEBUG
    #undef _DEBUG
    #if (defined linux) | (defined CMAKE)
        #include "Python.h"
    #else
        #include "Python.h"
    #endif
    #define _DEBUG
#else
    #ifdef linux
        #include "Python.h"
    #else
        #include "Python.h"
    #endif
#endif

#include "../common/addInInterface.h"

namespace ito
{

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
       }
       PyActuatorPlugin;

       typedef struct
       {
           PyObject_HEAD
           ito::ActuatorAxis *axisObj;
           PyObject* base;
       }
       PyActuatorAxis;

       typedef struct
       {
           PyObject_HEAD
           ito::AddInDataIO *dataIOObj;
           PyObject* base;
       }
       PyDataIOPlugin;

       typedef struct
       {
           PyObject_HEAD
           ito::AddInAlgo *algoObj;
           PyObject* base;
       }
       PyAlgoPlugin;
       
       // Actuator
       static void PyActuatorPlugin_dealloc(PyActuatorPlugin *self);
       static PyObject *PyActuatorPlugin_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
       static int PyActuatorPlugin_init(PyActuatorPlugin *self, PyObject *args, PyObject *kwds);
       static PyObject* PyActuatorPlugin_repr(PyActuatorPlugin *self);

       static PyObject *PyActuatorPlugin_name(PyActuatorPlugin *self);
       static PyObject *PyActuatorPlugin_getParamList(PyActuatorPlugin *self);
	   static PyObject *PyActuatorPlugin_getParamListInfo(PyActuatorPlugin *self,  PyObject *args);
       static PyObject* PyActuatorPlugin_getExecFuncsInfo(PyActuatorPlugin* self, PyObject *args);
       static PyObject *PyActuatorPlugin_getParam(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_setParam(PyActuatorPlugin *self, PyObject *args);
	   static PyObject *PyActuatorPlugin_getType(PyActuatorPlugin *self);
       static PyObject *PyActuatorPlugin_execFunc(PyActuatorPlugin *self, PyObject *args, PyObject *kwds);

       static PyObject *PyActuatorPlugin_calib(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_setOrigin(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_getStatus(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_getPos(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_setPosAbs(PyActuatorPlugin *self, PyObject *args);
       static PyObject *PyActuatorPlugin_setPosRel(PyActuatorPlugin *self, PyObject *args);

       static PyMemberDef  PyActuatorPlugin_members[];
       static PyMethodDef  PyActuatorPlugin_methods[];
       static PyTypeObject PyActuatorPluginType;
       static PyModuleDef  PyActuatorPluginModule;

       // pending for deletion
/*
       // Actuator axis
       static void PyActuatorAxis_dealloc(PyActuatorAxis *self);
       static PyObject *PyActuatorAxis_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
       static int PyActuatorAxis_init(PyActuatorAxis *self, PyObject *args, PyObject *kwds);

       static PyObject *PyActuatorAxis_getStatus(PyActuatorAxis *self, PyObject *args);
       static PyObject *PyActuatorAxis_getPos(PyActuatorAxis *self, PyObject *args);
       static PyObject *PyActuatorAxis_setPosAbs(PyActuatorAxis *self, PyObject *args);
       static PyObject *PyActuatorAxis_setPosRel(PyActuatorAxis *self, PyObject *args);
	  
       static PyMemberDef  PyActuatorAxis_members[];
       static PyMethodDef  PyActuatorAxis_methods[];
       static PyTypeObject PyActuatorAxisType;
       static PyModuleDef  PyActuatorAxisModule;
*/
       //DataIO
       static void PyDataIOPlugin_dealloc(PyDataIOPlugin *self);
       static PyObject *PyDataIOPlugin_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
       static int PyDataIOPlugin_init(PyDataIOPlugin *self, PyObject *args, PyObject *kwds);
       static PyObject* PyDataIOPlugin_repr(PyDataIOPlugin *self);

       static PyObject *PyDataIOPlugin_name(PyDataIOPlugin *self);
       static PyObject *PyDataIOPlugin_getParamList(PyDataIOPlugin *self);
       static PyObject* PyDataIOPlugin_getParamListInfo(PyDataIOPlugin* self,  PyObject *args);
       static PyObject* PyDataIOPlugin_getExecFuncsInfo(PyDataIOPlugin* self, PyObject *args);
       static PyObject *PyDataIOPlugin_getParam(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_setParam(PyDataIOPlugin *self, PyObject *args);
	   static PyObject *PyDataIOPlugin_getType(PyDataIOPlugin *self);
       static PyObject *PyDataIOPlugin_execFunc(PyDataIOPlugin *self, PyObject *args, PyObject *kwds);

       static PyObject *PyDataIOPlugin_startDevice(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_stopDevice(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_acquire(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_getVal(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_copyVal(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_setVal(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_enableAutoGrabbing(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_disableAutoGrabbing(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_setAutoGrabbing(PyDataIOPlugin *self, PyObject *args);
       static PyObject *PyDataIOPlugin_getAutoGrabbing(PyDataIOPlugin *self, PyObject *args);
	   
       static PyMemberDef  PyDataIOPlugin_members[];
       static PyMethodDef  PyDataIOPlugin_methods[];
       static PyTypeObject PyDataIOPluginType;
       static PyModuleDef  PyDataIOPluginModule;


       // Algo
       static void PyAlgoPlugin_dealloc(PyAlgoPlugin *self);
       static PyObject *PyAlgoPlugin_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
       static int PyAlgoPlugin_init(PyAlgoPlugin *self, PyObject *args, PyObject *kwds);
       static PyObject *PyAlgoPlugin_name(PyAlgoPlugin *self);
       static PyObject *PyAlgoPlugin_getParamList(PyAlgoPlugin *self);
       static PyObject *PyAlgoPlugin_getParamListInfo(PyAlgoPlugin *self, PyObject *args);
       static PyObject* PyAlgoPlugin_getExecFuncsInfo(PyAlgoPlugin* self, PyObject *args);
       static PyObject *PyAlgoPlugin_getParam(PyAlgoPlugin *self, PyObject *args);
       static PyObject *PyAlgoPlugin_setParam(PyAlgoPlugin *self, PyObject *args);
	   static PyObject *PyAlgoPlugin_getType(PyAlgoPlugin *self);

       static PyMemberDef  PyAlgoPlugin_members[];
       static PyMethodDef  PyAlgoPlugin_methods[];
       static PyTypeObject PyAlgoPluginType;
       static PyModuleDef  PyAlgoPluginModule;
};

} //end namespace ito

#endif
