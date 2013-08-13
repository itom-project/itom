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

#include "pythonItom.h"
#include "pythonNumeric.h"
#include "pythonPlugins.h"
#include "pythonQtConversion.h"
#include "pythontParamConversion.h"
#include "pythonCommon.h"
#include "pythonProxy.h"
#include "pythonFigure.h"
#include "pythonPlotItem.h"

#include "pythonEngine.h"

#include "../helper/versionHelper.h"
#include "../../common/sharedFunctionsQt.h"

#include "../AppManagement.h"
#include "../organizer/uiOrganizer.h"
#include "../organizer/addInManager.h"

#include <qdir.h>
#include <qcoreapplication.h>
#include <qdesktopwidget.h>

QHash<unsigned int, QString> ito::PythonItom::m_gcTrackerList;

namespace ito
{


//----------------------------------------------------------------------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //
//                                                                                                              //
//          STATIC METHODS - - - STATIC METHODS - - - STATIC METHODS                                            //
//                                                                                                              //
//  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyOpenEmptyScriptEditor_doc,"scriptEditor() -> opens new, empty script editor window (undocked) \n\
\n\
Notes \n\
----- \n\
\n\
Opens a new and empty itom script editor window. The window is undocked and non blocking.");
PyObject* PythonItom::PyOpenEmptyScriptEditor(PyObject * /*pSelf*/, PyObject * /*pArgs*/)
{
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    QObject *sew = AppManagement::getScriptEditorOrganizer();
    if (sew == NULL)
    {
        return PyErr_Format(PyExc_RuntimeError, "gui not available");
    }

    QMetaObject::invokeMethod(sew,"openNewScriptWindow", Q_ARG(bool,false), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

    if (locker.getSemaphore()->wait(PLUGINWAIT))
    {
        Py_RETURN_NONE;
    }
    else
    {
        if (PyErr_CheckSignals() == -1) //!< check if key interrupt occured
        {
            return PyErr_Occurred();
        }
        return PyErr_Format(PyExc_RuntimeError, "timeout while opening empty script.");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyNewScript_doc, "newScript() -> opens an empty, new script in the current script window.\n\
\n\
Notes \n\
----- \n\
\n\
Creates a new itom script in the latest opened editor window.");
PyObject* PythonItom::PyNewScript(PyObject * /*pSelf*/, PyObject * /*pArgs*/)
{
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    QObject *sew = AppManagement::getScriptEditorOrganizer();
    if (sew == NULL)
    {
        return PyErr_Format(PyExc_RuntimeError, "gui not available");
    }

    QMetaObject::invokeMethod(sew,"newScript", Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

    if (locker.getSemaphore()->wait(PLUGINWAIT))
    {
        Py_RETURN_NONE;
    }
    else
    {
        if (PyErr_CheckSignals() == -1) //!< check if key interrupt occured
        {
            return PyErr_Occurred();
        }
        return PyErr_Format(PyExc_RuntimeError, "timeout while creating new script");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyOpenScript_doc,"openScript(filename) -> opens the given script in current script window.\n\
\n\
Parameters \n\
----------- \n\
filename : {str} \n\
    Path and File of the file to open. Unter windows not case sensitiv.\n\
\n\
Notes \n\
----- \n\
\n\
Open an existing itom script from the harddrive into the latest opened editor window.");
PyObject* PythonItom::PyOpenScript(PyObject * /*pSelf*/, PyObject *pArgs)
{
    const char* filename;
    if (PyArg_ParseTuple(pArgs, "s", &filename) == false)
    {
        return PyErr_Format(PyExc_RuntimeError, "no valid filename string available");
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    QObject *sew = AppManagement::getScriptEditorOrganizer();
    if (sew == NULL)
    {
        return PyErr_Format(PyExc_RuntimeError, "gui not available");
    }

    QMetaObject::invokeMethod(sew, "openScript", Q_ARG(QString,QString(filename)), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

    if (locker.getSemaphore()->wait(60000)) //longer time, since msgbox may appear
    {
        Py_RETURN_NONE;
    }
    else
    {
        if (PyErr_CheckSignals() == -1) //!< check if key interrupt occured
        {
            return PyErr_Occurred();
        }
        return PyErr_Format(PyExc_RuntimeError, "timeout while opening script");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotImage_doc,"plot(data, [className]) -> plots a dataObject in a newly created figure \n\
\n\
Plot an existing dataObject in not dockable, not blocking window. \n\
The style of the plot will depend on the object dimensions.\n\
If x-dim or y-dim are equal to 1, plot will be a lineplot else a 2D-plot.\n\
\n\
Parameters \n\
----------- \n\
data : {DataObject} \n\
    Is the data object whose region of interest will be plotted.\n\
className : {str}, optional \n\
    class name of desired plot (if not indicated default plot will be used (see application settings) \n\
");
PyObject* PythonItom::PyPlotImage(PyObject * /*pSelf*/, PyObject *pArgs, PyObject *pKwds)
{
    const char *kwlist[] = {"data", "className", NULL};
    PyObject *data = NULL;
    int areaIndex = 0;
    char* className = NULL;
    bool ok = true;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "O|s", const_cast<char**>(kwlist), &data, &className))
    {
        return NULL;
    }

    QSharedPointer<ito::DataObject> newDataObj(PythonQtConversion::PyObjGetDataObjectNewPtr(data, false, ok));
    if (!ok)
    {
        return PyErr_Format(PyExc_RuntimeError, "first argument cannot be converted to a dataObject");
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = 0;
    int areaRow = 0;
    QSharedPointer<unsigned int> figHandle(new unsigned int);
    *figHandle = 0; //new figure will be requested

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    QString defaultPlotClassName;
    if (className) defaultPlotClassName = className;

    QSharedPointer<unsigned int> objectID(new unsigned int);

    QMetaObject::invokeMethod(uiOrg, "figurePlot", Q_ARG(QSharedPointer<ito::DataObject>, newDataObj), Q_ARG(QSharedPointer<unsigned int>, figHandle), Q_ARG(QSharedPointer<unsigned int>, objectID), Q_ARG(int, areaRow), Q_ARG(int, areaCol), Q_ARG(QString, defaultPlotClassName), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
    if (!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        return PyErr_Format(PyExc_RuntimeError, "timeout while plotting data object");
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }
    
    //return Py_BuildValue("iO", *figHandle); //returns handle

    //return new instance of PyUiItem
    PyObject *args2 = PyTuple_New(0); //Py_BuildValue("OO", self, name);
    PyObject *kwds2 = PyDict_New();
    PyDict_SetItemString(kwds2, "objectID", PyLong_FromLong(*objectID));
    PythonPlotItem::PyPlotItem *pyPlotItem = (PythonPlotItem::PyPlotItem *)PyObject_Call((PyObject *)&PythonPlotItem::PyPlotItemType, args2, kwds2);
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    if (pyPlotItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create plotItem of plot widget");
        return NULL;
    }

    PyObject *res = Py_BuildValue("iO", *figHandle, (PyObject*)pyPlotItem); //returns handle
    Py_XDECREF(pyPlotItem);
    return res;
}

////----------------------------------------------------------------------------------------------------------------------------------
//PyDoc_STRVAR(pyCloseFigure_doc,"closeFigure(fig-handle|'all') -> closes the figure window with the given handle-number. \n\
//\n\
//Parameters \n\
//----------- \n\
//fig-handle : {int | 'all'} \n\
//    The number (ID) of the figure to close or all to close all.\n\
//\n\
//Notes \n\
//----- \n\
//\n\
//Closes the figure window with the given handle-number (type int) or closes all figures ('all').");
//PyObject* PythonItom::PyCloseFigure(PyObject * /*pSelf*/, PyObject *pArgs)
//{
//    int handle = 0; //0 = 'all', >0 = specific figure
//    const char* tag;
//
//    if (!PyArg_ParseTuple(pArgs, "I", &handle))
//    {
//        PyErr_Clear();
//        if (!PyArg_ParseTuple(pArgs, "s", &tag))
//        {
//            return PyErr_Format(PyExc_RuntimeError, "argument has to be a figure handle (unsigned int) or the string 'all'");
//        }
//
//        handle = 0;
//        if (!(strcmp(tag,"all") || strcmp(tag,"All") || strcmp(tag,"ALL")))
//        {
//            return PyErr_Format(PyExc_RuntimeError, "argument has to be a figure handle (unsigned int) or the string 'all'");
//        }
//    }
//    else
//    {
//        if (handle <= 0)
//        {
//            return PyErr_Format(PyExc_ValueError, "figure handle must be bigger than zero");
//        }
//    }
//
//    return PyErr_Format(PyExc_RuntimeError, "temporarily not implemented");
//
//    //QObject *figureOrganizer = AppManagement::getFigureOrganizer();
//    //ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
//
//    //QMetaObject::invokeMethod(figureOrganizer, "closeFigure", Q_ARG(unsigned int, static_cast<unsigned int>(handle)), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
//
//    //if (locker.getSemaphore()->wait(PLUGINWAIT))
//    //{
//    //    if (locker.getSemaphore()->returnValue == retError)
//    //    {
//    //        return PyErr_Format(PyExc_RuntimeError, "error while closing figure: \n%s", locker.getSemaphore()->returnValue.errorMessage());
//    //    }
//    //    else
//    //    {
//    //        Py_RETURN_NONE;
//    //    }
//    //}
//    //else
//    //{
//    //    if (PyErr_CheckSignals() == -1) //!< check if key interrupt occured
//    //    {
//    //        return PyErr_Occurred();
//    //    }
//    //    return PyErr_Format(PyExc_RuntimeError, "timeout while closing figure.");
//    //}
//}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyLiveImage_doc,"liveImage(cam, [className]) -> shows a camera live image in a newly created figure\n\
\n\
Creates a plot-image (2D) and automatically grabs images into this window.\n\
This function is not blocking.\n\
\n\
Parameters \n\
----------- \n\
cam : {dataIO-Instance} \n\
    Camera grabber device from which images are acquired.\n\
className : {str}, optional \n\
    class name of desired plot (if not indicated default plot will be used (see application settings)");
PyObject* PythonItom::PyLiveImage(PyObject * /*pSelf*/, PyObject *pArgs, PyObject *pKwds)
{
    const char *kwlist[] = {"cam", "className", NULL};
    PythonPlugins::PyDataIOPlugin *cam = NULL;
    int areaIndex = 0;
    char* className = NULL;
    bool ok = true;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "O!|s", const_cast<char**>(kwlist), &PythonPlugins::PyDataIOPluginType, &cam, &className))
    {
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = 0;
    int areaRow = 0;
    QSharedPointer<unsigned int> figHandle(new unsigned int);
    *figHandle = 0; //new figure will be requested

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    QString defaultPlotClassName;
    if (className) defaultPlotClassName = className;

    QSharedPointer<unsigned int> objectID(new unsigned int);

    QMetaObject::invokeMethod(uiOrg, "figureLiveImage", Q_ARG(AddInDataIO*, cam->dataIOObj), Q_ARG(QSharedPointer<unsigned int>, figHandle), Q_ARG(QSharedPointer<unsigned int>, objectID), Q_ARG(int, areaRow), Q_ARG(int, areaCol), Q_ARG(QString, defaultPlotClassName), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
    if (!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        return PyErr_Format(PyExc_RuntimeError, "timeout while showing live image of camera");
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }
    
    //return Py_BuildValue("i", *figHandle); //returns handle

    //return new instance of PyUiItem
    PyObject *args2 = PyTuple_New(0); //Py_BuildValue("OO", self, name);
    PyObject *kwds2 = PyDict_New();
    PyDict_SetItemString(kwds2, "objectID", PyLong_FromLong(*objectID));
    PythonPlotItem::PyPlotItem *pyPlotItem = (PythonPlotItem::PyPlotItem *)PyObject_Call((PyObject *)&PythonPlotItem::PyPlotItemType, args2, kwds2);
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    if (pyPlotItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create plotItem of plot widget");
        return NULL;
    }

    PyObject *res = Py_BuildValue("iO", *figHandle, (PyObject*)pyPlotItem); //returns handle
    Py_XDECREF(pyPlotItem);
    return res;
}

////----------------------------------------------------------------------------------------------------------------------------------
//PyDoc_STRVAR(pyLiveLine_doc,"liveLine(dataIO) -> shows grabber data in a live line view window  \n\
//\n\
//Parameters \n\
//----------- \n\
//dataIO : {Hardware-Pointer} \n\
//    Camera grabber device from which images are acquired.\n\
//plotName : {str}, optional \n\
//    class name of desired plot (if not indicated default plot will be used (see application settings) \n\
//\n\
//Notes \n\
//----- \n\
//\n\
//Creates a lineplot-window (1D) and automatically grabs lines into this window.\n\
//This function is not blocking.");
//PyObject* PythonItom::PyLiveLine(PyObject * /*pSelf*/, PyObject *pArgs)
//{
//    PyObject *grabber = NULL;
//    const char* plotClassName = NULL;
//
//    if (!PyArg_ParseTuple(pArgs, "O!|s", &PythonPlugins::PyDataIOPluginType, &grabber, &plotClassName))
//    {
//        return PyErr_Format(PyExc_RuntimeError, "argument is no dataIO device");
//    }
//
//    PythonPlugins::PyDataIOPlugin* elem = (PythonPlugins::PyDataIOPlugin*)grabber;
//    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
//    QString plotClassName2;
//    if (plotClassName)
//    {
//        plotClassName2 = QString(plotClassName);
//    }
//
//    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
//    QMetaObject::invokeMethod(uiOrg, "liveLine", Q_ARG(AddInDataIO*,elem->dataIOObj), Q_ARG(QString, plotClassName2), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
////    QMetaObject::invokeMethod(figureOrganizer, "liveLine", Q_ARG(AddInDataIO*,elem->dataIOObj), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
//
//    if (locker.getSemaphore()->wait(PLUGINWAIT))
//    {
//        if (locker.getSemaphore()->returnValue == retError)
//        {
//            PyErr_Format(PyExc_RuntimeError,"error while starting live line view: \n%s\n",locker.getSemaphore()->returnValue.errorMessage());
//            return NULL;
//        }
//        else if (locker.getSemaphore()->returnValue == retWarning)
//        {
//            std::cerr << "warning while starting live line view: \n" << "warning message: \n" << std::endl;
//            std::cerr << locker.getSemaphore()->returnValue.errorMessage() << std::endl;
//
//            Py_RETURN_NONE;
//        }
//        else
//        {
//            Py_RETURN_NONE;
//        }
//    }
//    else
//    {
//        if (PyErr_CheckSignals() == -1) //!< check if key interrupt occured
//        {
//            return PyErr_Occurred();
//        }
//        return PyErr_Format(PyExc_RuntimeError, "timeout while opening live line view");
//    }
//}
//
////----------------------------------------------------------------------------------------------------------------------------------
//PyDoc_STRVAR(pySetFigParam_doc,"setFigParam(figHandle, name, value OR name, value) -> sets the parameter 'name' of figure to value \n\
//\n\
//Parameters \n\
//----------- \n\
//figHandle : {int}, optional \n\
//    figHandle is the handle to the figure, which is returned by the plot-command (e.g.). \n\
//    If figHandle is not given, -1 is assumed, which means that the current figure is taken. \n\
//name :  {str} \n\
//    the name of the parameter to be changed.\n\
//value : {int} \n\
//    value is the new value for this parameter.\n\
//\n\
//Notes \n\
//----- \n\
//\n\
//This function sets the parameter 'name' of the specigied figure to value.\n\
//If 'help?' is used as name of the parameter, a list of available parameters is printed.");
//PyObject* PythonItom::PySetFigParam(PyObject * /*pSelf*/, PyObject *pArgs)
//{
//    const char *paramKey = "help?\0";
//    int figHandle = -1; //takes current figure handle
//    PyObject *paramValue = NULL;
//    //pArgs of type [unsigned int figHandle, string key, variant argument (one elem or tuple of elements)
//    if (PyTuple_Size(pArgs) <= 0)
//    {
//        paramKey = "help?\0";
//    }
//    else
//    {
//        if (!PyArg_ParseTuple(pArgs, "sO", &paramKey, &paramValue))
//        {
//            PyErr_Clear();
//            if (!PyArg_ParseTuple(pArgs, "i|sO", &figHandle, &paramKey, &paramValue))
//            {
//                return PyErr_Format(PyExc_RuntimeError, "argument is invalid, must be [int figureNumber], string key, variant argument");
//            }
//        }
//    }
//
//    QVariant argument = QVariant();
//
//    if (paramValue != NULL)
//    {
//        PythonQtConversion::convertPyObjectToQVariant(paramValue, argument);
//        if (PyErr_Occurred()) return NULL;
//    }
//
//    return PyErr_Format(PyExc_RuntimeError, "temporarily not implemented");
//
//    //QObject *figureOrganizer = AppManagement::getFigureOrganizer();
//    //ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
//
//    //QMetaObject::invokeMethod(figureOrganizer, "setFigureParameter", Q_ARG(int, figHandle), Q_ARG(QString, QString(paramKey)), Q_ARG(QVariant, argument), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
//
//    //if (locker.getSemaphore()->wait(PLUGINWAIT))
//    //{
//    //    Py_RETURN_NONE;
//    //}
//    //else
//    //{
//    //    if (PyErr_CheckSignals() == -1) //!< check if key interrupt occured
//    //    {
//    //        return PyErr_Occurred();
//    //    }
//    //    return PyErr_Format(PyExc_RuntimeError, "timeout while setting figure parameter");
//    //}
//
//}

////----------------------------------------------------------------------------------------------------------------------------------
//PyObject* PythonItom::convertPyObjectToQVariant(PyObject *argument, QVariant &qVarArg)
//{
//    if (PyList_Check(argument))
//    {
//        PyObject* tempArg = NULL;
//        PyObject* retValue = NULL;
//        QVariantList list;
//        for (Py_ssize_t i = 0; i < PyList_Size(argument); i++)
//        {
//            tempArg = PyList_GetItem(argument, i);
//            list.append(QVariant());
//            retValue = convertPyObjectToQVariant(tempArg, list[i]);
//
//            if (PyErr_Occurred())
//            {
//                return NULL;
//            }
//            if (retValue != NULL) Py_DECREF(retValue);
//        }
//
//        qVarArg = list;
//
//        Py_RETURN_NONE;
//    }
//
//    //check for elementary types char*, int, double
//    char* textArg;
//    if (PyLong_Check(argument))
//    {
//        qVarArg = (int)PyLong_AsLong(argument);
//    }
//    else if (PyFloat_Check(argument))
//    {
//        qVarArg = PyFloat_AsDouble(argument);
//    }
//    else if (PyArg_Parse(argument, "s", &textArg))
//    {
//        qVarArg = QString(textArg);
//    }
//    else
//    {
//        PyErr_Format(PyExc_ValueError, "argument does not fit to char*, int, long or double");
//        qVarArg = QVariant();
//    }
//
//    if (PyErr_Occurred())
//    {
//        return NULL;
//    }
//
//    Py_RETURN_NONE;
//
//}

////----------------------------------------------------------------------------------------------------------------------------------
//PyDoc_STRVAR(pyGetFigParam_doc,"getFigParam(figHandle, name, OR name) -> gets the parameter 'name' of given figure \n\
//\n\
//Parameters \n\
//----------- \n\
//figHandle : {int}, optinal \n\
//    figHandle is the handle to the figure, which is returned by the plot-command (e.g.). \n\
//    If figHandle is not given, -1 is assumed, which means that the current figure is taken. \n\
//name :  {str} \n\
//    the name of the parameter to be read out.\n\
//\n\
//Returns \n\
//------- \n\
//The value of the parameter 'name' or Py_Error.\n\
//\n\
//Notes \n\
//----- \n\
//\n\
//This function gets the value of parameter 'name' of the specigied figure.\n\
//If 'help?' is used as name of the parameter, a list of available parameters is printed.");
//PyObject* PythonItom::PyGetFigParam(PyObject * /*pSelf*/, PyObject *pArgs)
//{
//    return PyErr_Format(PyExc_RuntimeError, "temporary not implemented");
//
//    //const char *paramKey = "help?\0";
//    //int figHandle = -1; //takes current figure handle
//    ////pArgs of type [unsigned int figHandle, string key, variant argument (one elem or tuple of elements)
//    //if (PyTuple_Size(pArgs) <= 0)
//    //{
//    //    paramKey = "help?\0";
//    //}
//    //else
//    //{
//    //    if (!PyArg_ParseTuple(pArgs, "s", &paramKey))
//    //    {
//    //        PyErr_Clear();
//    //        if (!PyArg_ParseTuple(pArgs, "i|s", &figHandle, &paramKey))
//    //        {
//    //            return PyErr_Format(PyExc_RuntimeError, "argument is invalid, must be [int figureNumber], string key");
//    //        }
//    //    }
//    //}
//
//    //QObject *figureOrganizer = AppManagement::getFigureOrganizer();
//    //ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
//
//    //QSharedPointer<QVariant> returnValue = QSharedPointer<QVariant>(new QVariant());
//
//    //QMetaObject::invokeMethod(figureOrganizer, "getFigureParameter", Q_ARG(int, figHandle), Q_ARG(QString, QString(paramKey)), Q_ARG(QSharedPointer<QVariant>, returnValue), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
//
//    //if (locker.getSemaphore()->wait(PLUGINWAIT))
//    //{
//    //    //everything's ok
//    //}
//    //else
//    //{
//    //    if (PyErr_CheckSignals() == -1) //!< check if key interrupt occured
//    //    {
//    //        return PyErr_Occurred();
//    //    }
//    //    return PyErr_Format(PyExc_RuntimeError, "timeout while getting parameters");
//    //}
//
//    //PyObject* retObject = NULL;
//
//    ////now parse returnValue
//    //retObject = PythonQtConversion::QVariantToPyObject(*returnValue);
//
//    //return retObject;
//}

////----------------------------------------------------------------------------------------------------------------------------------
////returns new reference
//PyObject* PythonItom::convertQVariantToPyObject(QVariant value)
//{
//    QString stringValue;
//
//    switch(value.type())
//    {
//    case QVariant::Invalid:
//        Py_RETURN_NONE;
//        break;
//    case QVariant::Int:
//        return PyLong_FromLong(value.value<int>());
//        break;
//    case QVariant::Double:
//        return PyFloat_FromDouble(value.value<double>());
//        break;
//    case QVariant::String:
//        stringValue = value.value<QString>();
//        if (stringValue.isEmpty() || stringValue.isNull() || stringValue == "")
//        {
//            return PyUnicode_FromFormat("");
//        }
//        else
//        {
//            return PyUnicode_FromFormat("%s", stringValue.toAscii().data());
//        }
//        break;
//    case QVariant::List:
//        QVariantList list = value.value<QVariantList>();
//        PyObject* retObject = PyList_New(list.length());
//        for (int i = 0; i < list.length(); i++)
//        {
//            PyList_SetItem(retObject,i,convertQVariantToPyObject(list[i]));
//        }
//        return retObject;
//        break;
//    }
//
//    Py_RETURN_NONE;
//}

//----------------------------------------------------------------------------------------------------------------------------------
////bekommt ein array input, dessen werte werden umgeschrieben, zusätzlich wird ein undefiniertes array der größe x-y zurückgegeben
//PyObject* PythonItom::arrayManipulation(PyObject* pSelf, PyObject* pArgs)
//{
//   // FigureOrganizer *figureOrganizer = qobject_cast<FigureOrganizer*>(AppManagement::getFigureOrganizer());
//
//   // int j;
//   // QSharedPointer<blub> h;
//   // int64 time = cv::getTickCount();
//
//   // /*for (int i = 0; i < 10000; i++)
//   // {
//   //     j=2;
//   //     QMetaObject::invokeMethod(figureOrganizer, "test1", Qt::BlockingQueuedConnection, Q_ARG(int,j));
//   // }*/
//   //
//   // int64 time2 = cv::getTickCount();
//
//   // /*for (int i = 0; i < 10000; i++)
//   // {
//   //     j=2;
//   //     figureOrganizer->test1(j);
//   // }*/
//
//   // int64 time3 = cv::getTickCount();
//
//   ///* for (int i = 0; i < 10000; i++)
//   // {*/
//   //     h = QSharedPointer<blub>(new blub());
//   //     QMetaObject::invokeMethod(figureOrganizer, "test2", Qt::BlockingQueuedConnection, Q_ARG(QSharedPointer<blub>, h));
//   // //}
//
//   // int64 time4 = cv::getTickCount();
//
//   // double freq = cv::getTickFrequency();
//
//   // double meth1 = (double)(time2-time)/freq;
//   // double meth2 = (double)(time3-time2)/freq;
//   // double meth3 = (double)(time4-time3)/freq;
//
//
//   // Py_RETURN_NONE;
//
//
//	PyArrayObject* input;
//	int x,y;
//
//	if (!PyArg_ParseTuple(pArgs, "O!ii:arrayManipulation",  &PyArray_Type, &input, &x, &y))
//	{
//		//printPythonError(std::cout);
//		return NULL; /* PyArg_ParseTuple raised an exception */
//	}
//
//	int nx = PyArray_DIM(input,0);
//	int ny = PyArray_DIM(input,1);
//
//	double *value;
//
//	for (int i = 0; i < nx; i++)
//	{
//		for (int j = 0; j < ny; j++)
//		{
//			value = (double *) PyArray_GETPTR2(input,i,j);
//			*value = i*j;
//		}
//	}
//
//
//	PyArrayObject* a;
//
//
//
//	npy_intp a_dims[2];
//	a_dims[0] = x;
//	a_dims[1] = y;
//
//	if (0)
//	{
//	//version 1:
//		//a = (PyArrayObject*) (PyArray_SimpleNew(2,a_dims,NPY_INT));
//		a = (PyArrayObject*) (PyArray_ZEROS(2,a_dims,NPY_DOUBLE,0));
//	}else{
//	//version 2:
//		double* dataPtr = new double[x*y];
//		for (int i = 0; i < x * y; i++) dataPtr[i]=0.0;
//
//        //if (1)
//        //{
//        //    //!< owndata-flag should be set
//        //    a = (PyArrayObject*) (PyArray_New(&PyArray_Type, 2, a_dims, NPY_DOUBLE, NULL, (void *) dataPtr, NPY_OWNDATA, NPY_CARRAY, NULL));
//        //}
//        //else
//        //{
//            //!< owndata-flag not set
//		    a = (PyArrayObject*) (PyArray_SimpleNewFromData(2,a_dims,NPY_DOUBLE,(void *) dataPtr));
//
//            //PythonItom::getInstance()->attachPyArrayToGarbageCollector(*a, pyGarbageDeleteIfUnused);
//
//
//
//            //PyObject *capsule = PyCapsule_New((void *) dataPtr, "itom.a", numpyArray_deleteBorrowedData);
//            //PyArray_BASE(a) = capsule;
//            //Py_DECREF(capsule); //darf hier nicht geschehen
//        //}
//	}
//
//	return PyArray_Return(a);
//};
//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PyWidgetOrFilterHelp(bool getWidgetHelp, PyObject* pArgs)
{
    Py_ssize_t length = PyTuple_Size(pArgs);
    int output = 0;
    int listonly = 1;
    int userwithinfos = 0;
    int longest_name = 0;

    QString namefilter;
    const char* filterstring;

    switch(length)
    {
    case 0:
        namefilter.fromAscii(0);
        break;
    case 1: //!< copy filterstring only
        if (!PyArg_ParseTuple(pArgs, "s", &filterstring))
        {
            return PyErr_Format(PyExc_TypeError, "wrong input type");
        }
        namefilter.sprintf("%s",filterstring);

        if (namefilter.length())
        {
            listonly = 0;
        }
        break;
    case 2: //!< copy filterstring and toggle output
        if (!PyArg_ParseTuple(pArgs, "si", &filterstring, &output))
        {
            return PyErr_Format(PyExc_TypeError, "wrong input type");
        }
        namefilter.sprintf("%s", filterstring);

        if (namefilter.length())
        {
            listonly = 0;
        }
        break;
    case 3: //!< Valid input are filterstring, toggle output and listonlyflag
        if (!PyArg_ParseTuple(pArgs, "sii", &filterstring, &output, &userwithinfos))
        {
            return PyErr_Format(PyExc_TypeError, "wrong input type");
        }
        namefilter.sprintf("%s",filterstring);

        if (namefilter.length())
        {
            listonly = 0;
        }
        break;

    default://!< Valid input are filterstring, toggle output and listonlyflag only
        return PyErr_Format(PyExc_ValueError, "to many arguments");
        break;
    }

    if (namefilter.contains("*") && ((namefilter.indexOf("*") == (namefilter.length() - 1)) || (namefilter.indexOf("*") == 0)))
    {
        // This is executed if the '*' ist either the first or the last sign of the string
        listonly = 1;
        namefilter.remove("*");
    }

    PyErr_Clear();

//    QVector<ito::tParam> *paramsMand = NULL;
//    QVector<ito::tParam> *paramsOpt = NULL;

    ito::RetVal retval = 0;
    PyObject *result = NULL;
    PyObject *resultmand = NULL;
    PyObject *resultopt = NULL;
    PyObject *resulttemp = NULL;
    PyObject *item = NULL;

    ito::AddInManager *AIM = ito::AddInManager::getInstance();
    if (!AIM)
    {
        return PyErr_Format(PyExc_RuntimeError, "no addin-manager found");
    }

    const QHash<QString, ito::AddInAlgo::FilterDef *> *filtlist = AIM->getFilterList();
    const QHash<QString, ito::AddInAlgo::AlgoWidgetDef *> *widglist = AIM->getAlgoWidgetList();
    
    if (!widglist && getWidgetHelp)
    {
        return PyErr_Format(PyExc_RuntimeError, "no widget list found");
    }
    if (!filtlist && !getWidgetHelp)
    {
        return PyErr_Format(PyExc_RuntimeError, "no filterlist found");
    }

    QString contextName;
    QStringList keyList;

    if (getWidgetHelp)
    {

        keyList = widglist->keys();
        contextName = "widget";
    }
    else
    {
        keyList = filtlist->keys();
        contextName = "filter";
    }

    result = PyDict_New();

    if (keyList.size() < 1)
    {
        std::cout << "No "<< contextName.toAscii().data() <<" defined\n";
    }
    else
    {
        std::cout << "\n";
        QString filteredKey;

        //! first try to catch a perfect match with existing filters
        for (int n = 0; n < keyList.size(); n++)
        {
            filteredKey = keyList.value(n);

            if (!filteredKey.compare(namefilter, Qt::CaseSensitive) && !listonly && !userwithinfos)
            {
                resulttemp = PyDict_New();
                //QVector<ito::Param> paramsMand, paramsOpt, paramsOut;
                const ito::FilterParams *filterParams;

                std::cout << contextName.toUpper().toAscii().data() << "NAME:    "<< filteredKey.toAscii().data() << "\n";
                item = PythonQtConversion::QByteArrayToPyUnicodeSecure(filteredKey.toAscii());
                PyDict_SetItemString(resulttemp, "name", item);
                Py_DECREF(item);

                if (getWidgetHelp)
                {
                    ito::AddInAlgo::AlgoWidgetDef* wFunc = widglist->find(filteredKey).value();  
                    filterParams = AIM->getHashedFilterParams(wFunc->m_paramFunc);
                    //(*(wFunc->m_paramFunc))(&paramsMand, &paramsOpt);

                    std::cout << "DESCRIPTION    " << wFunc->m_description.toAscii().data() << "\n";
                    item = PythonQtConversion::QByteArrayToPyUnicodeSecure(wFunc->m_description.toAscii());
                    PyDict_SetItemString(resulttemp, "description", item);      
                    Py_DECREF(item);
                }
                else
                {
                    ito::AddInAlgo::FilterDef * fFunc = filtlist->find(filteredKey).value();  
                    filterParams = AIM->getHashedFilterParams(fFunc->m_paramFunc);
                    //(*(fFunc->m_paramFunc))(&paramsMand, &paramsOpt);

                    std::cout << "DESCRIPTION    " << fFunc->m_description.toAscii().data() << "\n";
                    item = PythonQtConversion::QByteArrayToPyUnicodeSecure(fFunc->m_description.toAscii());
                    PyDict_SetItemString(resulttemp, "description", item);
                    Py_DECREF(item);
                }

                if (filterParams)
                {

                    std::cout << "PARAMETERS:\n";
                    if (filterParams->paramsMand.size())
                    {
                        std::cout << "\nMandatory parameters:\n";
                        resultmand = PrntOutParams(&(filterParams->paramsMand), false, true, -1);
                        PyDict_SetItemString(resulttemp, "Mandatory Parameters", resultmand);
                        Py_DECREF(resultmand);
                    }
                    else
                    {
                        std::cout << "\nMandatory parameters: " <<  contextName.toAscii().data()  << " function has no mandatory parameters \n";
                    }
                    if (filterParams->paramsOpt.size())
                    {
                        std::cout << "\nOptional parameters:\n";
                        resultopt = ito::PrntOutParams(&(filterParams->paramsOpt), false, true, -1);
                        PyDict_SetItemString(resulttemp, "Optional Parameters", resultopt);
                        Py_DECREF(resultopt);
                    }
                    else
                    {
                        std::cout << "\nOptional parameters: " <<  contextName.toAscii().data()  << " function has no optional parameters \n";
                    }
                    if (filterParams->paramsOut.size())
                    {
                        std::cout << "\nOutput parameters:\n";
                        resultopt = ito::PrntOutParams(&(filterParams->paramsOut), false, true, -1);
                        PyDict_SetItemString(resulttemp, "Output Parameters", resultopt);
                        Py_DECREF(resultopt);
                    }
                    else
                    {
                        std::cout << "\nOutput parameters: " <<  contextName.toAscii().data()  << " function has no output parameters \n";
                    }
                }
                else
                {
                    std::cout << "PARAMETERS:\nError while loading parameter info.";
                }
                
                std::cout << "\n";
                PyDict_SetItemString(result, filteredKey.toAscii().data(), resulttemp);
                Py_DECREF(resulttemp);
            }
            else
                continue;
        }

        //! Now get the complete filterlist

        if (namefilter.length())
        {
            std::cout << contextName.toAscii().data() << ", which contain the given string: " << namefilter.toAscii().data() << "\n";
        }
        else
        {
            std::cout << "Complete "<< contextName.toAscii().data() << "list\n";
        }

        for (int n = 0; n < keyList.size(); n++)	// get the longest name in this list
        {
            filteredKey = keyList.value(n);
            if (filteredKey.contains(namefilter, Qt::CaseInsensitive))
            {
                if (longest_name < filteredKey.length())
                {
                    longest_name = filteredKey.length();
                }
            }

            
        }
        longest_name = (longest_name + 3) > 50 ? 50 : longest_name + 3;
        if (keyList.size())
        {
            keyList.sort();
            
            if (!userwithinfos)
            {
                std::cout <<"\n" << contextName.append("name").leftJustified(longest_name +1, ' ', false).toAscii().data() << " \tInfo-String\n";
            }
        }

        for (int n = 0; n < keyList.size(); n++)
        {
            filteredKey = keyList.value(n);

            if (filteredKey.contains(namefilter, Qt::CaseInsensitive))
            {
                resulttemp = PyDict_New();
                const ito::FilterParams *filterParams = NULL;

                QString descriptionString("");

                if (userwithinfos)
                    std::cout << contextName.toUpper().toAscii().data() << "NAME:    " << filteredKey.leftJustified(longest_name, ' ', false).toAscii().data() << " ";
                else
                    std::cout << filteredKey.leftJustified(longest_name, ' ', false).toAscii().data() << " ";

                item = PythonQtConversion::QByteArrayToPyUnicodeSecure(filteredKey.toAscii());
                PyDict_SetItemString(resulttemp, "name", item);
                Py_DECREF(item);

                if (getWidgetHelp)
                {
                    ito::AddInAlgo::AlgoWidgetDef* wFunc = widglist->find(filteredKey).value();  
                    filterParams = AIM->getHashedFilterParams(wFunc->m_paramFunc);
                    //(*(wFunc->m_paramFunc))(&paramsMand, &paramsOpt);
                    if (wFunc->m_description.length() > 0)
                    {
                         descriptionString = wFunc->m_description;
                    }
                }
                else
                {
                    ito::AddInAlgo::FilterDef * fFunc = filtlist->find(filteredKey).value();
                    filterParams = AIM->getHashedFilterParams(fFunc->m_paramFunc);
                    //(*(fFunc->m_paramFunc))(&paramsMand, &paramsOpt);
                    if (fFunc->m_description.length() > 0)
                    {
                         descriptionString = fFunc->m_description;
                    }
                }

                if (descriptionString.length() > 0)
                {
                    QString desString = descriptionString.trimmed();
                    //split desString until first \n
                    int linebreakIdx = desString.indexOf('\n');
                    if (linebreakIdx > 0)
                    {
                        desString = desString.left(linebreakIdx);
                    }

                    std::cout << "\t'" << desString.toAscii().data() << "'\n";
                    item = PythonQtConversion::QByteArrayToPyUnicodeSecure(descriptionString.toAscii());
                    PyDict_SetItemString(resulttemp, "description", item);
                    Py_DECREF(item);
                }
                else
                {
                    std::cout <<"\t' No description '\n";
                    item = PyUnicode_FromString("No description");
                    PyDict_SetItemString(resulttemp, "description", item);
                    Py_DECREF(item);
                }

                if (userwithinfos)
                {
                    std::cout << "PARAMETERS:\n";
                    if (filterParams)
                    {
                        if (filterParams->paramsMand.size())
                        {
                            std::cout << "\nMandatory parameters:\n";
                            resultmand = PrntOutParams(&(filterParams->paramsMand), false, true, -1);
                            PyDict_SetItemString(resulttemp, "Mandatory Parameters", resultmand);
                            Py_DECREF(resultmand);
                        }
                        else
                        {
                            std::cout << "\nMandatory parameters: " <<  contextName.toAscii().data()  << " function has no mandatory parameters \n";
                        }
                        if (filterParams->paramsOpt.size())
                        {
                            std::cout << "\nOptional parameters:\n";
                            resultopt = PrntOutParams(&(filterParams->paramsOpt), false, true, -1);
                            PyDict_SetItemString(resulttemp, "Optional Parameters", resultopt);
                            Py_DECREF(resultopt);
                        }
                        else
                        {
                            std::cout << "\nOptional parameters: " <<  contextName.toAscii().data()  << " function has no optional parameters \n";
                        }
                        if (filterParams->paramsOut.size())
                        {
                            std::cout << "\nOutput parameters:\n";
                            resultopt = PrntOutParams(&(filterParams->paramsOut), false, true, -1);
                            PyDict_SetItemString(resulttemp, "Output Parameters", resultopt);
                            Py_DECREF(resultopt);
                        }
                        else
                        {
                            std::cout << "\nOutput parameters: " <<  contextName.toAscii().data()  << " function has no output parameters \n";
                        }
                    }
                    else
                    {
                        std::cout << "Errors while loading parameter info";
                    }
                    std::cout << "\n";
                    PyDict_SetItemString(result, filteredKey.toAscii().data(), resulttemp);
                    Py_DECREF(resulttemp);
                }
            }
            else
                continue;
        }
    }

    std::cout << "\n";

    if (output)
    {
        return result;
    }
    else
    {
        Py_DECREF(result);
        Py_RETURN_NONE;
    }
}
//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFilterHelp_doc, "filterHelp([filterName, dictionary = 0, furtherInfos = 0]) -> generates an online help for the given filter(s). \n \
                               Generates an online help for the given widget or returns a list of available filter.\n\
\n\
Parameters \n\
----------- \n\
filterName : {str}, optional \n\
    is the fullname or a part of any filter-name which should be displayed. \n\
    If filterName is none or no filter matches filterName casesensitiv a list with all suitable filters is given. \n\
dictionary : {dict}, optional \n\
    if dictionary == 1, function returns an Py_Dictionary with parameters \n\
    Default value is 0.\n\
furtherInfos : {int}, optional \n\
    defines if a complete parameter-list of name-related filters to the given filterName should be displayed (1) \n\
\n\
Returns \n\
------- \n\
Returns none or a PyDictionary depending on the value of dictionary.");

PyObject* PythonItom::PyFilterHelp(PyObject* /*pSelf*/, PyObject* pArgs)
{
    return PyWidgetOrFilterHelp(false, pArgs);
}

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyWidgetHelp_doc,"widgetHelp([widgetName, dictionary = 0, furtherInfos = 0]) -> generates an online help for the given widget(s). \n \
Generates an online help for the given widget or returns a list of available widgets. \n\
\n\
Parameters \n\
----------- \n\
widgetName : {str}, optional \n\
    is the fullname or a part of any widget-name which should be displayed. \n\
    If widgetName is none or no widget matches widgetName casesensitiv a list with all suitable widgets is given. \n\
dictionary : {dict}, optional \n\
    if dictionary == 1, function returns an Py_Dictionary with parameters \n\
    Default value is 0.\n\
furtherInfos : {int}, optional \n\
    defines if a complete parameter-list of name-related widgets to the given widgetName should be displayed (1) \n\
\n\
Returns \n\
------- \n\
Returns none or a PyDictionary depending on the value of dictionary.");
PyObject* PythonItom::PyWidgetHelp(PyObject* /*pSelf*/, PyObject* pArgs)
{
    return PyWidgetOrFilterHelp(true, pArgs);
}

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginLoaded_doc,"pluginLoaded(pluginname) -> checks if a certain plugin was loaded.\n\
Checks if a specified plugin is loaded and returns the result as a boolean expression. \n\
\n\
Parameters \n\
----------- \n\
pluginname :  {str} \n\
    The name of a specified plugin as usually displayed in the plugin window.\n\
\n\
Returns \n\
------- \n\
True, if the plugin has been loaded and can be used, else False.");
PyObject* PythonItom::PyPluginLoaded(PyObject* /*pSelf*/, PyObject* pArgs)
{
    const char* pluginName = NULL;
    ito::RetVal retval = ito::retOk;
	
	if (!PyArg_ParseTuple(pArgs, "s", &pluginName))
	{
		return NULL;
	}

    ito::AddInManager *AIM = ito::AddInManager::getInstance();
    if (!AIM)
    {
        return PyErr_Format(PyExc_RuntimeError, "no addin-manager found");
    }

    int pluginNum = -1;
    int plugtype = -1;
    int version = -1;
    char * pTypeSTring = NULL;
    char * pAuthor = NULL;
    char * pDescription = NULL;
    char * pDetailDescription = NULL;

    retval = AIM->getPlugInInfo(pluginName, &plugtype, &pluginNum, &pTypeSTring, &pAuthor, &pDescription, &pDetailDescription, &version);

    if (pTypeSTring)
        free(pTypeSTring);
    if (pAuthor)
        free(pAuthor);
    if (pDescription)
        free(pDescription);
    if (pDetailDescription)
        free(pDetailDescription);

    if (retval.containsWarningOrError())
    {
        return Py_False;
    }

    return Py_True;
}

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginHelp_doc,"pluginHelp(pluginName [, dictionary = False]) -> generates an online help for the specified plugin.\n\
                              Gets (also print to console) the initialisation parameters of the plugin specified pluginName (str, as specified in the plugin window).\n\
If dictionary is True, a dict with all plugin parameters is returned.\n\
\n\
Parameters \n\
----------- \n\
pluginName : {str} \n\
    is the fullname of a plugin as specified in the plugin window.\n\
dictionary : {bool}, optional \n\
	if dictionary == True, function returns an dict with plugin parameters (default: False)\n\
\n\
Returns \n\
------- \n\
Returns None or a dict depending on the value of parameter dictionary.");
PyObject* PythonItom::PyPluginHelp(PyObject* /*pSelf*/, PyObject* pArgs, PyObject *pKwds)
{
	const char *kwlist[] = {"pluginName", "dictionary", NULL};
    const char* pluginName = NULL;
#if PY_VERSION_HEX < 0x03030000
	unsigned char output = 0;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "s|b", const_cast<char**>(kwlist), &pluginName, &output))
    {
        return NULL;
    }
#else //only python 3.3 or higher has the 'p' (bool, int) type string
	int output = 0; //this must be int, not bool!!! (else crash)

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "s|p", const_cast<char**>(kwlist), &pluginName, &output))
    {
        return NULL;
    }
#endif

    QVector<ito::Param> *paramsMand = NULL;
    QVector<ito::Param> *paramsOpt = NULL;

    ito::RetVal retval = 0;
    int pluginNum = -1;
    int plugtype = -1;
    int version = -1;
    char * pTypeSTring = NULL;
    char * pAuthor = NULL;
    char * pDescription = NULL;
    char * pDetailDescription = NULL;
    PyObject *result = NULL;
    PyObject *resultmand = NULL;
    PyObject *resultopt = NULL;
    PyObject *item = NULL;
    


    ito::AddInManager *AIM = ito::AddInManager::getInstance();
    if (!AIM)
    {
        return PyErr_Format(PyExc_RuntimeError, "no addin-manager found");
    }

    retval = AIM->getPlugInInfo(pluginName, &plugtype, &pluginNum, &pTypeSTring, &pAuthor, &pDescription, &pDetailDescription, &version);
    if (retval.containsWarningOrError())
    {
        if (pTypeSTring)
            free(pTypeSTring);
        if (pAuthor)
            free(pAuthor);
        if (pDescription)
            free(pDescription);
        if (pDetailDescription)
            free(pDetailDescription);

        if (retval.errorMessage())
        {
            return PyErr_Format(PyExc_RuntimeError, "Could not load plugin: %s with error message: \n%s\n", pluginName, retval.errorMessage());
        }
        else
        {
            return PyErr_Format(PyExc_RuntimeError, "Could not load plugin: %s\n", pluginName);
        }

    }
    else
    {
        result = PyDict_New();

        std::cout << "\n";

        if (pluginName)
        {
            std::cout << "NAME:\t " << pluginName << "\n";
            item = PythonQtConversion::QByteArrayToPyUnicodeSecure(pluginName);
            PyDict_SetItemString(result, "name", item); //PyUnicode_FromString(pluginName));
            Py_DECREF(item);
        }
        if (pTypeSTring)
        {
            std::cout << "TYPE:\t " << pTypeSTring << "\n";
            item = PythonQtConversion::QByteArrayToPyUnicodeSecure(pTypeSTring);
            PyDict_SetItemString(result, "type", item);
            Py_DECREF(item);
            free(pTypeSTring);
        }

        std::cout << "VERSION:\t " << version << "\n";
        item = PyFloat_FromDouble(version);
        PyDict_SetItemString(result, "version", item);
        Py_DECREF(item);

        if (pAuthor)
        {
            std::cout << "AUTHOR:\t " << pAuthor << "\n";
            item = PythonQtConversion::QByteArrayToPyUnicodeSecure(pAuthor);
            PyDict_SetItemString(result, "author", item);
            Py_DECREF(item);
            free(pAuthor);
        }
        if (pDescription)
        {
            std::cout << "INFO:\t\t " << pDescription << "\n";
            item = PythonQtConversion::QByteArrayToPyUnicodeSecure(pDescription);
            PyDict_SetItemString(result, "description", item);
            Py_DECREF(item);
            free(pDescription);
        }
        if (pDetailDescription)
        {
            std::cout << "\nDETAILS:\n" << pDetailDescription << "\n";
            item = PythonQtConversion::QByteArrayToPyUnicodeSecure(pDetailDescription);
            PyDict_SetItemString(result, "detaildescription", item);
            Py_DECREF(item);
            free(pDetailDescription);
        }
    }

    

    PyObject *noneText = PyUnicode_FromString("None");

    switch(plugtype)
    {
        default:
        break;
        case ito::typeDataIO:
        case ito::typeActuator:
            retval = AIM->getInitParams(pluginName, plugtype, &pluginNum, paramsMand, paramsOpt);
            if (retval.containsWarningOrError())
            {
                Py_DECREF(result);

                if (retval.errorMessage())
                {
                    return PyErr_Format(PyExc_RuntimeError, "Could not load plugin: %s with error message: \n%s\n", pluginName, retval.errorMessage());
                }
                else
                {
                    return PyErr_Format(PyExc_RuntimeError, "Could not load plugin: %s\n", pluginName);
                }
            }

            std::cout << "\nINITIALISATION PARAMETERS:\n";
            if (paramsMand)
            {
               if ((*paramsMand).size())
               {
                  std::cout << "\n Mandatory parameters:\n";
                  resultmand = PrntOutParams(paramsMand, false, true, -1);
                  PyDict_SetItemString(result, "Mandatory Parameters", resultmand);
                  Py_DECREF(resultmand);
               }
               else
               {
                   std::cout << "  Initialisation function has no mandatory parameters \n";
               }
            }
            else
            {
                   std::cout << "  Initialisation function has no mandatory parameters \n";
            }
            if (paramsOpt)
            {
                if ((*paramsOpt).size())
                {
                    std::cout << "\n Optional parameters:\n";
                    resultopt = PrntOutParams(paramsOpt, false, true, -1);
                    PyDict_SetItemString(result, "Optional Parameters", resultopt);
                    Py_DECREF(resultopt);
                }
                else
                {
                    std::cout << "  Initialisation function has no optional parameters \n";
                }
            }
            else
            {
                    std::cout << "  Initialisation function has no optional parameters \n";
            }
            std::cout << "\n";
            std::cout << "\nFor more information use the member functions 'getParamListInfo()' and 'getExecFuncInfo()'\n\n";
            break;

        case ito::typeAlgo:
        {
            
            if (pluginNum >= 0 && pluginNum < AIM->getAlgList()->size())
            {
                pluginNum += AIM->getActList()->size();

                ito::AddInAlgo *algoInst = (ito::AddInAlgo *)((ito::AddInInterfaceBase *)AIM->getAddInPtr(pluginNum))->getInstList()[0];

                QHash<QString, ito::AddInAlgo::FilterDef *> funcList;
                algoInst->getFilterList(funcList);

                if (funcList.size() > 0)
                {
                    std::cout << "\nThis is the container for following filters:\n";
                    QStringList keyList = funcList.keys();
                    keyList.sort();

                    PyObject *algorithmlist = PyDict_New();
                    for (int algos = 0; algos < keyList.size(); algos++)
                    {
                        item = PythonQtConversion::QByteArrayToPyUnicodeSecure(keyList.value(algos).toAscii());
                        PyDict_SetItemString(algorithmlist, keyList.value(algos).toAscii().data(), item);
                        Py_DECREF(item);
                        std::cout << "> " << algos << "  " << keyList.value(algos).toAscii().data() << "\n";
                    }
                    PyDict_SetItemString(result, "filter", algorithmlist);
                    Py_DECREF(algorithmlist);
                    std::cout << "\nFor more information use 'filterHelp(\"filterName\")'\n\n";
                }
                else
                {
                    PyDict_SetItemString(result, "filter", noneText);
                }

                QHash<QString, ito::AddInAlgo::AlgoWidgetDef *> widgetList;
                algoInst->getAlgoWidgetList(widgetList);

                if (widgetList.size() > 0)
                {
                    std::cout << "\nThis is the container for following widgets:\n";
                    QStringList keyList = widgetList.keys();
                    keyList.sort();

                    PyObject *widgetlist = PyDict_New();
                    for (int widgets = 0; widgets < keyList.size(); widgets++)
                    {
                        item = PythonQtConversion::QByteArrayToPyUnicodeSecure(keyList.value(widgets).toAscii());
                        PyDict_SetItemString(widgetlist, keyList.value(widgets).toAscii().data(), item);
                        Py_DECREF(item);
                        std::cout << "> " << widgets << "  " << keyList.value(widgets).toAscii().data() << "\n";
                    }
                    PyDict_SetItemString(result, "widgets", widgetlist);
                    Py_DECREF(widgetlist);
                    std::cout << "\nFor more information use 'widgetHelp(\"widgetName\")'\n";
                }
                else
                {
                    PyDict_SetItemString(result, "widgets", noneText);
                }

            }
            else
            {
                PyDict_SetItemString(result, "widgets", noneText);
            }
        }
        break;
    }

    Py_DECREF(noneText);

    if (output > 0)
    {
        return result;
    }
    else
    {
        Py_DECREF(result);
        Py_RETURN_NONE;
    }
}
//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyITOMVersion_doc,"version([toggle-output [, include-plugins]])) -> retrieve complete information about itom version numbers\n\
\n\
Parameters \n\
----------- \n\
toggle-output : {bool}, optional\n\
    default = false\n\
    if true, output will be written to a dictionary else to console.\n\
dictionary : {bool}, optional \n\
    default = false\n\
    if true, add informations about plugIn versions.\n\
\n\
Returns \n\
------- \n\
None (display outPut) or PyDictionary with version information.\n\
\n\
Notes \n\
----- \n\
\n\
Retrieve complete version information of ITOM and if specified version information of loaded plugins\n\
and print it either to the console or to a PyDictionary.");
PyObject* PythonItom::PyITOMVersion(PyObject* /*pSelf*/, PyObject* pArgs)
{
    bool toogleOut = false;
    bool addPlugIns = false;

    int length = PyTuple_Size(pArgs);

    if (length == 1) //!< copy name + object
    {
        if (!PyArg_ParseTuple(pArgs, "b", &toogleOut))
        {
            return PyErr_Format(PyExc_TypeError, "wrong input type, must be (bool, bool)");
        }
    }
    else if (length == 2) //!< copy name + object + asBinary
    {
        if (!PyArg_ParseTuple(pArgs, "bb", &toogleOut, &addPlugIns))
        {
            return PyErr_Format(PyExc_TypeError, "wrong input type, must be (bool, bool)");
        }
    }
    else if (length > 2)
    {
        return PyErr_Format(PyExc_ValueError, "Only two optional parameters. Not more!");
    }

    PyObject* myDic = PyDict_New();
    PyObject* myTempDic = PyDict_New();
    PyObject* key = NULL;
    PyObject* value = NULL;

    int ret = 0;

    QMap<QString, QString> versionMap = ito::getItomVersionMap();
    QMapIterator<QString, QString> i(versionMap);

    while (i.hasNext()) 
    {
        i.next();

        key = PythonQtConversion::QStringToPyObject(i.key());
        value = PythonQtConversion::QStringToPyObject(i.value());
        ret = PyDict_SetItem(myTempDic, key, value);

        Py_DECREF(key);
        Py_DECREF(value);
    }

    /*QList<QPair<QString, QString> > versionList = ito::retrieveITOMVERSIONMAP();

    for (int i = 0; i < versionList.size(); i++)
    {
        key = PythonQtConversion::QStringToPyObject(versionList[i].first);
        value = PythonQtConversion::QStringToPyObject(versionList[i].second);
        ret = PyDict_SetItem(myTempDic, key, value);

        Py_DECREF(key);
        Py_DECREF(value);
    }*/

    ret = PyDict_SetItemString(myDic, "itom", myTempDic);
    Py_XDECREF(myTempDic);

    if (addPlugIns)
    {
        PyObject* myTempDic = PyDict_New();
        char buf[7] = {0};
        ito::AddInManager *aim = ito::AddInManager::getInstance();
        ito::AddInInterfaceBase  *curAddInInterface = NULL;
        if (aim != NULL)
        {
            PyObject* info = NULL;
            PyObject* license = NULL;
            for (int i = 0; i < aim->getNumTotItems(); i++)
            {
                
                curAddInInterface = reinterpret_cast<ito::AddInInterfaceBase*>(aim->getAddInPtr(i));
                if (curAddInInterface)
                {
                    info = PyDict_New();

                    QString currentName = curAddInInterface->objectName();
                    key = PythonQtConversion::QStringToPyObject(currentName);
                    license = PythonQtConversion::QStringToPyObject(curAddInInterface->getLicenseInfo());
                    
                    int val = curAddInInterface->getVersion();
                    int first = MAJORVERSION(val);              
                    int middle =MINORVERSION(val);               
                    int last =PATCHVERSION(val); 
                    _snprintf(buf, 7, "%i.%i.%i", first, middle, last);
                    value = PyUnicode_FromString(buf);

                    ret = PyDict_SetItemString(info, "version", value);
                    ret = PyDict_SetItemString(info, "license", license);

                    ret = PyDict_SetItem(myTempDic, key, info);

                    Py_DECREF(key);
                    Py_DECREF(value);
                    Py_DECREF(license);
                    Py_XDECREF(info);
                }
            }
        }

        ret = PyDict_SetItemString(myDic, "plugins", myTempDic);
        Py_XDECREF(myTempDic);

    }

    if (toogleOut)
    {
        return myDic;
    }
    else
    {
        std::cout << "\n";
        
        PyObject* myKeys = PyDict_Keys(myDic);
        Py_ssize_t size = PyList_Size(myKeys);
        bool check = true;

        for (Py_ssize_t i = 0; i < size; i++)
        {
            std::cout << "\n ----------------------------------------------------------------------------------------------------------------------------------------\n";
            PyObject* currentKey = PyList_GET_ITEM(myKeys, i);
            QString key("");
            key = PythonQtConversion::PyObjGetString(currentKey, true, check);

            if (!check)
            {
                continue;
            }

            std::cout << key.toAscii().toUpper().data() << ":\n";

            PyObject* currentDict = PyDict_GetItem(myDic, currentKey);
            
            PyObject* mySubKeys = PyDict_Keys(currentDict);

            int longest = 0;
            int compensatorMax = 30; 

            for (Py_ssize_t m = 0; m < PyList_Size(mySubKeys); m++)
            {      
                PyObject* currentSubKey = PyList_GET_ITEM(mySubKeys, m);
                int temp = PyUnicode_GET_SIZE(currentSubKey);
                longest = temp > longest ? temp : longest;
            }
            longest += 3;
            longest = longest > compensatorMax ? compensatorMax : longest;

            for (Py_ssize_t m = 0; m < PyList_Size(mySubKeys); m++)
            {
                QString subKey("");
                PyObject* currentSubKey = PyList_GET_ITEM(mySubKeys, m);
                subKey = PythonQtConversion::PyObjGetString(currentSubKey, true, check);
                if (!check)
                {
                    continue;
                }

                int compensator = longest + (longest - subKey.length())*0.2;
                subKey = subKey.append(":").leftJustified(compensator);

                QString subVal("");
                PyObject* currentSubVal = PyDict_GetItem(currentDict, currentSubKey);

                if (PyDict_Check(currentSubVal))
                {
                    PyObject* curVal = PyDict_GetItemString(currentSubVal, "version");

                    if (PyLong_Check(curVal))
                    {
                        subVal = QString("%1").arg(PythonQtConversion::PyObjGetInt(curVal, true,check));
                    }
                    else 
                    {
                        subVal = PythonQtConversion::PyObjGetString(curVal, true, check);
                    }

                    subVal = QString("%1").arg(PythonQtConversion::PyObjGetString(curVal, false,check));
                    subVal.append("\t(license: ");

                    curVal = PyDict_GetItemString(currentSubVal, "license");
                    
                    if (PyLong_Check(curVal))
                    {
                        subVal.append(QString("%1").arg(PythonQtConversion::PyObjGetInt(curVal, true,check)));
                    }
                    else 
                    {
                        subVal.append(PythonQtConversion::PyObjGetString(curVal, true, check));
                    }

                    subVal.append(")");
                }
                else if (PyLong_Check(currentSubVal))
                {
                    subVal = QString("%1").arg(PythonQtConversion::PyObjGetInt(currentSubVal, true,check));
                }
                else 
                {
                    subVal = PythonQtConversion::PyObjGetString(currentSubVal, true, check);
                }

                if (!check)
                {
                    continue;
                }

                std::cout << subKey.toAscii().data() <<"\t" << subVal.toAscii().data() << "\n";

            }

            Py_XDECREF(mySubKeys);
            
        }

        
        Py_DECREF(myKeys);

        Py_DECREF(myDic);
        Py_RETURN_NONE;
    }

}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyAddButton_doc,"addButton(toolbarName, buttonName, code [, icon, argtuple]) -> adds a button to a toolbar in the main window \n\
\n\
Parameters \n\
----------- \n\
toolbarName : {str} \n\
    The name of the toolbar.\n\
buttonName : {str} \n\
    The name (str, identifier) of the button to create.\n\
code : {str, Method, Function}\n\
    The code to be executed if button is pressed.\n\
icon : {str}, optional \n\
    The filename of an icon-file. This can also be relative to the application directory of 'itom'.\n\
argtuple : {tuple}, optional \n\
    Arguments, which will be passed to method (in order to avoid cyclic references try to only use basic element types).\n\
\n\
Notes \n\
----- \n\
\n\
This function adds a button to a toolbar in the main window.\n\
If the button is pressed the simple python command specified by python-code is executed.\n\
If the toolbar specified by toolbarName do not exist, the toolbar is created.\n\
The button representation will be the optional icon or if icon is not loadable 'buttonName' will be displayed.\n\
\n\
ITOM comes with basic icons addressable by ':/.../iconname.png', e.g. ':/gui/icons/close.png'.\n\
Find available via iconbrower under 'Editor-Menu/Edit/iconbrower' or crtl-b");
PyObject* PythonItom::PyAddButton(PyObject* /*pSelf*/, PyObject* pArgs, PyObject *kwds)
{
    //int type = 0; //BUTTON (default)
    const char* toolbarName = NULL;
    const char* name = NULL;
    PyObject* code = NULL;
    const char* icon = NULL;
    const char *kwlist[] = {"toolbarName","buttonName", "code", "icon", "argtuple", NULL};

    QString qkey;
    QString qkey2;
    QString qname;
    QString qicon;
    QString qcode = "";
    PyObject *argtuple = NULL;
    bool ok = false;
    RetVal retValue(retOk);

    if (!PyArg_ParseTupleAndKeywords(pArgs, kwds, "ss|OsO!", const_cast<char**>(kwlist), &toolbarName, &name, &code, &icon, &PyTuple_Type, &argtuple))
    {
        PyErr_Clear();

        if (!PyArg_ParseTupleAndKeywords(pArgs, kwds, "ss|OsO!", const_cast<char**>(kwlist), &toolbarName, &name, &code, &icon, &PyList_Type, &argtuple))
        {
            return NULL;
        }
        //return PyErr_Format(PyExc_TypeError, "wrong length or type of arguments. Type help(addMenu) for more information.");
    }

    if (code)
    {
        qcode = PythonQtConversion::PyObjGetString(code,true,ok);
    }

    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine
    qkey = QString(name);
    qname = QString(name);
    qicon = QString(icon);

    if (qkey == "")
    {
        retValue += RetVal(retError,0,QObject::tr("Button must have a valid name.").toAscii().data());
    }
    else
    {
        if (!code)
        {
            retValue += RetVal(retError,0,QObject::tr("Any type of code (String or callable method or function) must be indicated.").toAscii().data());
        }
        else if (!ok) //check whether code is a method or function
        {
            if (PyMethod_Check(code) || PyFunction_Check(code))
            {
                //create hash-string
                qkey2 = qkey + "_" +  QString::number(pyEngine->m_pyFuncWeakRefHashesAutoInc++);
                qcode = ":::itomfcthash:::" + qkey2;
                if (pyEngine->m_pyFuncWeakRefHashes.contains(qkey2))
                {
                    retValue += RetVal(retError,0,QObject::tr("The given button name is already associated to a python method or function. The button can not be created.").toAscii().data());
                }
                else
                {
                    PyObject *arguments = PyTuple_New(1);
                    Py_INCREF(code);
                    PyTuple_SetItem(arguments,0,code); //steals ref
                    PyObject *proxy = PyObject_CallObject((PyObject *) &PythonProxy::PyProxyType, arguments); //new ref
                    Py_DECREF(arguments);
                        
                    if (proxy)
                    {
                        if (argtuple)
                        {
                            if (PyTuple_Check(argtuple))
                            {
                                Py_INCREF(argtuple);
                            }
                            else //list
                            {
                                argtuple = PyList_AsTuple(argtuple); //returns new reference
                            }
                        }
                        pyEngine->m_pyFuncWeakRefHashes[qkey2] = QPair<PyObject*,PyObject*>(proxy,argtuple);
                    }
                    else
                    {
                        retValue += RetVal(retError,0,QObject::tr("Could not create a itom.proxy-object  of the given callable method or function.").toAscii().data());
                    }
                }
            }
            else
            {
                retValue += RetVal(retError,0,QObject::tr("The code parameter must either be a python code snippet or a callable method or function object.").toAscii().data());
            }
        }
    }

    if (!retValue.containsError())
    {
        emit pyEngine->pythonAddToolbarButton(toolbarName, qname, qicon, qcode); //queued
        //emit pyEngine->pythonAddMenuElement(type, qkey, qname, qcode, qicon); //queued
    }

    if (!PythonCommon::transformRetValToPyException(retValue)) return NULL;
    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyRemoveButton_doc,"removeButton(toolbarName, buttonName) -> removes a button from a given toolbar. \n\
\n\
Parameters \n\
----------- \n\
toolbarName : {str} \n\
    The name of the toolbar.\n\
buttonName : {str} \n\
    The name (str, identifier) of the button to remove.\n\
\n\
Notes \n\
----- \n\
\n\
This function removes a button from a toolbar in the main window.\n\
If the toolbar is empty after removal, it is deleted.");
PyObject* PythonItom::PyRemoveButton(PyObject* /*pSelf*/, PyObject* pArgs)
{
    const char* toolbarName;
    const char* buttonName;

    if (! PyArg_ParseTuple(pArgs, "ss", &toolbarName, &buttonName))
    {
        return PyErr_Format(PyExc_TypeError, "wrong length or type of arguments. Type help(removeButton) for more information.");
    }

    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine
    emit pyEngine->pythonRemoveToolbarButton(toolbarName, buttonName); //queued connection

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyAddMenu_doc,"addMenu(type, key [, name, code, icon, argtuple]) -> adds an element to the menu bar. \n\
\n\
Parameters \n\
----------- \n\
type : {Int}\n\
    The type of the menu-element (BUTTON:0 [default], SEPARATOR:1, MENU:2). Use the corresponding constans in module 'itom'.\n\
key : {str} \n\
    A slash-separated string where every sub-element is the key-name for the menu-element in the specific level.\n\
name : {str}, optional \n\
    The text of the menu-element. If not indicated, the last sub-element of key is taken.\n\
code : {str, Method, Function}, optional \n\
    The code to be executed if menu element is pressed.\n\
icon : {str}, optional \n\
    The filename of an icon-file. This can also be relative to the application directory of 'itom'.\n\
argtuple : {tuple}, optional \n\
    Arguments, which will be passed to method (in order to avoid cyclic references try to only use basic element types).\n\
\n\
Notes \n\
----- \n\
\n\
This function adds an element to the main window menu bar. \n\
The root element of every menu-list must be a MENU-element. Such a MENU-element can contain sub-elements. \n\
The following sub-elements can be either another MENU, a SEPARATOR or a BUTTON. Only the BUTTON itself \n\
triggers a signal, which then executes the code, given by a string or a reference to a callable python method \n\
or function. Remember, that this reference is only stored as a weak pointer. \n\
If you want to directly add a sub-element, you can give a slash-separated string in the key-parameter. \n\
Every sub-component of this string then represents the menu-element in its specific level. Only the element in the last \n\
can be something else than MENU.\n\
\n\
ITOM comes with basic icons addressable by ':/.../iconname.png', e.g. ':/gui/icons/close.png'.\n\
Find available via iconbrower under 'Editor-Menu/Edit/iconbrower' or crtl-b");
PyObject* PythonItom::PyAddMenu(PyObject* /*pSelf*/, PyObject* args, PyObject *kwds)
{
    int type = 0; //BUTTON (default)
    const char* key = NULL;
    const char* name = NULL;
    PyObject* code = NULL;
    const char* icon = NULL;
    const char *kwlist[] = {"type","key","name", "code", "icon", "argtuple", NULL};

    QString qkey;
    QString qkey2;
    QString qname;
    QString qicon;
    QString qcode = "";
    PyObject *argtuple = NULL;
    bool ok = false;
    RetVal retValue(retOk);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "is|sOsO!", const_cast<char**>(kwlist), &type, &key, &name, &code, &icon, &PyTuple_Type, &argtuple))
    {
        PyErr_Clear();
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "is|sOsO!", const_cast<char**>(kwlist), &type, &key, &name, &code, &icon, &PyList_Type, &argtuple))
        {
            return NULL;
        }
        //return PyErr_Format(PyExc_TypeError, "wrong length or type of arguments. Type help(addMenu) for more information.");
    }

    if (code)
    {
        qcode = PythonQtConversion::PyObjGetString(code,true,ok);
    }

    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine
    qkey = QString(key);
    qname = QString(name);
    qicon = QString(icon);

    QStringList sl = qkey.split("/");
    if (qname == "" && sl.size() > 0)
    {
        qname = sl[ sl.size() - 1];
    }

    if (qkey == "")
    {
        retValue += RetVal(retError,0,QObject::tr("Menu element must have a valid key.").toAscii().data());
    }
    else
    {

        //check type
        switch(type)
        {
        case 0: //BUTTON
            {
            if (!code)
            {
                retValue += RetVal(retError,0,QObject::tr("For menu elements of type 'BUTTON' any type of code (String or callable method or function) must be indicated.").toAscii().data());
            }
            else if (!ok) //check whether code is a method or function
            {
                if (PyMethod_Check(code) || PyFunction_Check(code))
                {
                    //create hash-string
                    qkey2 = qkey + "_" +  QString::number(pyEngine->m_pyFuncWeakRefHashesAutoInc++);
                    qcode = ":::itomfcthash:::" + qkey2;
                    if (pyEngine->m_pyFuncWeakRefHashes.contains(qkey2))
                    {
                        retValue += RetVal(retError,0,QObject::tr("The given key is already associated to a python method or function. The menu element can not be created.").toAscii().data());
                    }
                    else
                    {
                        PyObject *arguments = PyTuple_New(1);
                        Py_INCREF(code);
                        PyTuple_SetItem(arguments,0,code); //steals ref
                        PyObject *proxy = PyObject_CallObject((PyObject *) &PythonProxy::PyProxyType, arguments); //new ref
                        Py_DECREF(arguments);
                        
                        if (proxy)
                        {
                            if (argtuple)
                            {
                                if (PyTuple_Check(argtuple))
                                {
                                    Py_INCREF(argtuple);
                                }
                                else //List
                                {
                                    argtuple = PyList_AsTuple(argtuple); //returns new reference
                                }
                            }
                            pyEngine->m_pyFuncWeakRefHashes[qkey2] = QPair<PyObject*,PyObject*>(proxy,argtuple);
                        }
                        else
                        {
                            retValue += RetVal(retError,0,QObject::tr("Could not create a itom.proxy-object  of the given callable method or function.").toAscii().data());
                        }
                    }
                }
                else
                {
                    retValue += RetVal(retError,0,QObject::tr("The code parameter must either be a python code snippet or a callable method or function object.").toAscii().data());
                }
            }
            break;
            }
        case 1: //SEPARATOR
            {
            if (ok && qcode != "")
            {
                retValue += RetVal(retWarning,0,QObject::tr("A menu element of type 'separator' can not execute some code. Code argument is ignored.").toAscii().data());
                qcode = "";
            }
            else if (!ok && code != NULL && code != Py_None)
            {
                retValue += RetVal(retWarning,0,QObject::tr("A menu element of type 'separator' can not execute any function or method. Code argument is ignored.").toAscii().data());
            }
            break;
            }
        case 2: //MENU
            {
            if (ok && qcode != "")
            {
                retValue += RetVal(retWarning,0,QObject::tr("A menu element of type 'menu' can not execute some code. Code argument is ignored.").toAscii().data());
                qcode = "";
            }
            else if (!ok && code != NULL && code != Py_None)
            {
                retValue += RetVal(retWarning,0,QObject::tr("A menu element of type 'menu' can not execute any function or method. Code argument is ignored.").toAscii().data());
            }
            break;
            }
        }
    }

    if (!retValue.containsError())
    {
        emit pyEngine->pythonAddMenuElement(type, qkey, qname, qcode, qicon); //queued
    }

    if (!PythonCommon::transformRetValToPyException(retValue)) return NULL;
    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyRemoveMenu_doc,"removeMenu(key) -> remove a menu element with the given key. \n\
\n\
Parameters \n\
----------- \n\
key : {str} \n\
    The name (str, identifier) of the menu entry to remove.\n\
\n\
Notes \n\
----- \n\
\n\
This function remove a menu element with the given key. \n\
key is a slash separated list. The sub-components then \n\
lead the way to the final element, which should be removed.");
PyObject* PythonItom::PyRemoveMenu(PyObject* /*pSelf*/, PyObject* args, PyObject *kwds)
{
    const char* keyName;
    const char *kwlist[] = {"key", NULL};
    static QString prefix = ":::itomfcthash:::";
    QString qkey;
    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", const_cast<char**>(kwlist), &keyName))
    {
        return PyErr_Format(PyExc_TypeError, "wrong length or type of arguments. Type help(removeMenu) for more information.");
    }

    qkey = QString(keyName);
    if (qkey == "")
    {
        return PyErr_Format(PyExc_KeyError, "The given key name must not be empty.");
    }

    //check if hashValue is in m_pyFuncWeakRefHashes and delete it and all hashValues which start with the given hashValue (hence its childs)
    QHash<QString,QPair<PyObject*,PyObject*> >::iterator it = pyEngine->m_pyFuncWeakRefHashes.begin();
    while(it != pyEngine->m_pyFuncWeakRefHashes.end())
    {
        if (it.key().startsWith(qkey)) //hashValue.startsWith(it.key()))
        {
            Py_XDECREF(it->first);
            Py_XDECREF(it->second);
            it = pyEngine->m_pyFuncWeakRefHashes.erase(it);
        }
        else
        {
            ++it;
        }
    }
        
    emit pyEngine->pythonRemoveMenuElement(qkey); //queued connection

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static */PyObject* PythonItom::PyCheckSignals(PyObject* /*pSelf*/)
{
    int result = PyErr_CheckSignals();
    return Py_BuildValue("i", result);
    //Py_RETURN_NONE;
}

/*static */PyObject* PythonItom::PyProcessEvents(PyObject* /*pSelf*/)
{
    QCoreApplication::processEvents(QEventLoop::AllEvents);
    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine
    if (pyEngine)
    {
        QCoreApplication::sendPostedEvents(pyEngine,0);
    }
    Py_RETURN_NONE;
}

/*static */PyObject* PythonItom::PyGetDebugger(PyObject* /*pSelf*/)
{
    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine
    if (pyEngine)
    {
        Py_INCREF(pyEngine->itomDbgInstance);
        return pyEngine->itomDbgInstance;

    }
    Py_RETURN_NONE;
}

/*static */PyObject* PythonItom::PyGCStartTracking(PyObject * /*pSelf*/)
{
    PyObject *gc = PyImport_AddModule("gc"); //borrowed ref
    PyObject *gc_collect = NULL;
    PyObject *obj_list = NULL;
    PyObject *t = NULL;
    bool ok;
    if (gc)
    {
        gc_collect = PyObject_CallMethod(gc, "collect","");
        Py_XDECREF(gc_collect);
        obj_list = PyObject_CallMethod(gc, "get_objects", "");
        if (!obj_list)
        {
            PyErr_SetString(PyExc_RuntimeError, "call to gc.get_objects() failed");
            return NULL;
        }

        m_gcTrackerList.clear();
        for (Py_ssize_t i = 0; i < PyList_Size(obj_list); i++)
        {
            t = PyList_GET_ITEM(obj_list,i); //borrowed
            m_gcTrackerList[ (size_t)t] = QString("%1 [%2]").arg(t->ob_type->tp_name).arg(PythonQtConversion::PyObjGetString(t, false, ok)); //t->ob_type->tp_name;
        }
        Py_DECREF(obj_list);
        std::cout << m_gcTrackerList.count() << " elements tracked" << std::endl;
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "module gc could not be imported");
        return NULL;
    }
    Py_RETURN_NONE;
}

/*static */PyObject* PythonItom::PyGCEndTracking(PyObject * /*pSelf*/)
{
    PyObject *gc = PyImport_AddModule("gc"); //borrowed ref
    PyObject *gc_collect = NULL;
    PyObject *obj_list = NULL;
    PyObject *t = NULL;
    QString temp;
    bool ok;
    int n;
    if (m_gcTrackerList.count() == 0)
    {
        PyErr_SetString(PyExc_RuntimeError,"tracker has not been started. Call gcStartTracking() first.");
        return NULL;
    }

    if (gc)
    {
        gc_collect = PyObject_CallMethod(gc, "collect","");
        Py_XDECREF(gc_collect);
        obj_list = PyObject_CallMethod(gc, "get_objects", "");
        if (!obj_list)
        {
            PyErr_SetString(PyExc_RuntimeError, "call to gc.get_objects() failed");
            return NULL;
        }

        for (Py_ssize_t i = 0; i < PyList_Size(obj_list); i++)
        {
            t = PyList_GET_ITEM(obj_list,i); //borrowed
            n = m_gcTrackerList.remove((size_t)t);
            if (n == 0)
            {
                temp = QString("%1 [%2]").arg(t->ob_type->tp_name).arg(PythonQtConversion::PyObjGetString(t, false, ok));
                std::cout << "New Element. Addr:" << (size_t)t << " Type: " << temp.toAscii().data() << "\n" << std::endl;
            }
        }

         QHashIterator<unsigned int, QString> i(m_gcTrackerList);
         while (i.hasNext()) 
         {
             i.next();
             std::cout << "Del Element. Addr:" << i.key() << " Type: " << i.value().toAscii().data() << "\n" << std::endl;
         }

         m_gcTrackerList.clear();


        Py_DECREF(obj_list);
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError,"module gc could not be imported");
        return NULL;
    }
    Py_RETURN_NONE;
}


//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(getScreenInfo_doc,"getScreenInfo() -> returns dictionary with information about all available screens. \n\
\n\
Returns \n\
------- \n\
ScreenInfo : {PyDict} \n\
    Returns a PyDictionary containing:\n\
screenCount : {int} \n\
    number of available screens \n\
primaryScreen : {int} \n\
    index (0-based) of primary screen \n\
geometry : {tuple} \n\
    tuple with dictionaries for each screen containing data for width (w), height (h) and its top-left-position (x, y)\n\
\n\
Notes \n\
----- \n\
\n\
This function returns a PyDictionary which contains informations about the current screen configuration of this PC.");
PyObject* PythonItom::PyGetScreenInfo(PyObject* /*pSelf*/)
{
    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine
    if (pyEngine)
    {
        PyObject *res = PyDict_New();
        int nScreens = pyEngine->m_pDesktopWidget->screenCount();
        int primaryScreen = pyEngine->m_pDesktopWidget->primaryScreen();

        PyObject* geom = PyTuple_New(nScreens);
        PyObject* subgeom = NULL;
        PyObject* item = NULL;
        QRect rec;
        for (int i = 0; i < nScreens; i++)
        {
            subgeom = PyDict_New();
            rec = pyEngine->m_pDesktopWidget->screenGeometry(i);
            item = PyLong_FromLong(rec.x());
            PyDict_SetItemString(subgeom, "x", item);
            Py_DECREF(item);
            item = PyLong_FromLong(rec.y());
            PyDict_SetItemString(subgeom, "y", item);
            Py_DECREF(item);
            item = PyLong_FromLong(rec.width());
            PyDict_SetItemString(subgeom, "w", item);
            Py_DECREF(item);
            item = PyLong_FromLong(rec.height());
            PyDict_SetItemString(subgeom, "h", item);
            Py_DECREF(item);
            PyTuple_SetItem(geom,i, subgeom); //steals reference
        }

        item = PyLong_FromLong(nScreens);
        PyDict_SetItemString(res, "screenCount", item);
        Py_DECREF(item);
        item = PyLong_FromLong(primaryScreen);
        PyDict_SetItemString(res, "primaryScreen", item);
        Py_DECREF(item);
        PyDict_SetItemString(res, "geometry", geom);
        Py_DECREF(geom);

        return res;
    }
    else
    {
        Py_RETURN_NONE;
    }

}

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pySaveMatlabMat_doc,"saveMatlabMat(filename, dictionary[, matrixName]) -> saves strings, numbers, arrays or combinations to a Matlab matrix. \n\
\n\
Parameters \n\
----------- \n\
filename : {str} \n\
    Filename to which the data will be written (.mat will be added if not available)\n\
dictionary : {dictionary, list, tuple} \n\
    dictionary, list or tuple with elements of type number, string, array (dataObject, numpy.ndarray, npDataObject...)\n\
matrix-name : {string or list or tuple of strings}, optional \n\
    string or list or tuple of string (same length than object-sequence)\n\
\n\
Notes \n\
----- \n\
\n\
This function saves strings, numbers, arrays or combinations to a Matlab matrix (file). \n\
List or Tuples will be included to new dictionary (element-wise entry with name matrix1,...,matrixN or names given by last optional matrix-name sequence).");
PyObject * PythonItom::PySaveMatlabMat(PyObject * /*pSelf*/, PyObject *pArgs)
{
    PyObject* scipyIoModule = PyImport_ImportModule("scipy.io"); // borrowed reference

    if (scipyIoModule == NULL)
    {
        return PyErr_Format(PyExc_ImportError, "scipy-module and scipy.io-module could not be loaded.");
    }

    //Arguments must be: filename -> string,
    //                   dict or sequence, whose elements will be put into a dict with default-name-sets or
    //                      single-py-object, which will be put into a dict with default-name

    PyObject *element = NULL;
    PyObject *saveDict = NULL;
    PyObject *tempItem = NULL;
    const char *filename = NULL;
    PyObject *matrixNames = NULL;
    const char *matrixName = "matrix";
    const char *tempName = NULL;
    char *key = NULL;
    PyObject *matrixNamesTuple = NULL;
    PyObject *matlabData = NULL;
    PyObject *matrixNamesItem = NULL;


    if (!PyArg_ParseTuple(pArgs, "sO|O", &filename, &element, &matrixNames))
    {
        Py_XDECREF(scipyIoModule);
        return PyErr_Format(PyExc_TypeError, "wrong arguments. Required arguments are: filename (string), dict or other python object [e.g. array] [, matrix-name (string)]");
    }

    if (element == Py_None)
    {
        Py_XDECREF(scipyIoModule);
        return PyErr_Format(PyExc_ValueError, "Python element must not be None");
    }


    if (!PyDict_Check(element))
    {
        if (!PyTuple_Check(element) && !PyList_Check(element))
        {
            if (matrixNames == NULL)
            {
                tempName = matrixName;
            }
            else
            {
                matrixNamesTuple = PyTuple_Pack(1, matrixNames);
                if (!PyArg_ParseTuple(matrixNamesTuple, "s", &tempName))
                {
                    Py_XDECREF(scipyIoModule);
                    Py_XDECREF(matrixNamesTuple);
                    return PyErr_Format(PyExc_TypeError, "if matrix name is indicated, it must be one unicode string");
                }
                Py_XDECREF(matrixNamesTuple);
            }

            saveDict = PyDict_New();
            matlabData = PyMatlabMatDataObjectConverter(element);
            PyDict_SetItemString(saveDict, tempName, matlabData);
            Py_DECREF(matlabData);

        }
        else
        {
            if (matrixNames == NULL)
            {
                tempName = matrixName;
            }
            else if (!PyArg_ParseTuple(matrixNames, "s", &tempName))
            {
                PyErr_Clear();

                if (!PySequence_Check(matrixNames) || PySequence_Size(matrixNames) < PySequence_Size(element))
                {
                    Py_XDECREF(scipyIoModule);
                    return PyErr_Format(PyExc_TypeError, "if matrix name is indicated, it must be a sequence of unicode strings (same length than elements)");
                }
                else
                {
                    for (Py_ssize_t i = 0; i < PySequence_Size(matrixNames); i++)
                    {
                        matrixNamesItem = PySequence_GetItem(matrixNames,i); //new reference
                        if (!PyUnicode_Check(matrixNamesItem) && !PyBytes_Check(matrixNamesItem))
                        {
                            Py_XDECREF(scipyIoModule);
                            Py_XDECREF(matrixNamesItem);
                             return PyErr_Format(PyExc_TypeError, "each element of matrix names sequence must be a unicode object");
                        }
                        Py_XDECREF(matrixNamesItem);
                    }
                }
            }

            saveDict = PyDict_New();
            int sizeIter = 32; //max buffer length for number in key "matrix%i" with i being number
            

            for (Py_ssize_t i = 0; i < PySequence_Size(element); i++)
            {
                tempItem = PySequence_GetItem(element, i); //new reference

                if (tempName == matrixName)
                {
                    key = (char*)calloc(strlen(tempName) + sizeIter + 1, sizeof(char));
                    snprintf(key, strlen(tempName) + sizeIter, "%s%i", matrixName, ((int)i + 1));
                    matlabData = PyMatlabMatDataObjectConverter(tempItem);
                    PyDict_SetItemString(saveDict, key, matlabData);
                    Py_DECREF(matlabData);
                    free(key);
                }
                else
                {
                    matrixNamesItem = PySequence_GetItem(matrixNames, i); //new reference
                    matrixNamesTuple  = PyTuple_Pack(1, matrixNamesItem);
                    PyArg_ParseTuple(matrixNamesTuple, "s", &tempName);
                    matlabData = PyMatlabMatDataObjectConverter(tempItem);
                    PyDict_SetItemString(saveDict, tempName, matlabData);
                    Py_DECREF(matlabData);
                    Py_XDECREF(matrixNamesTuple);
                    Py_XDECREF(matrixNamesItem);
                }

                Py_XDECREF(tempItem);
            }

        }
    }
    else
    {
        saveDict = PyMatlabMatDataObjectConverter(element);
    }

    Py_INCREF(Py_True);
    Py_INCREF(Py_False);
    if (!PyObject_CallMethodObjArgs(scipyIoModule, PyUnicode_FromString("savemat"), PyUnicode_FromString(filename), saveDict, Py_True, PyUnicode_FromString("5"), Py_True, Py_False, PyUnicode_FromString("row"), NULL))
    {
        Py_XDECREF(saveDict);
        Py_DECREF(Py_True);
        Py_DECREF(Py_False);
        Py_XDECREF(scipyIoModule);
        return NULL;
    }

    Py_XDECREF(saveDict);
    Py_DECREF(Py_True);
    Py_DECREF(Py_False);
    Py_XDECREF(scipyIoModule);

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns new reference to element and extracts tags, if possible
/*!
    If element is of type npDataObject or dataObject, the following new reference is returned:
    dict{"dataObject": org-dataObject, "tags": tags, "axisScales": axisScales ... }

    \param name description
    \return description
    \sa (see also) keywords (comma-separated)
*/
PyObject* PythonItom::PyMatlabMatDataObjectConverter(PyObject *element)
{
    PyObject *item = NULL;
    PyObject *newElement = NULL;
    if (element && Py_TYPE(element) == &PythonNpDataObject::PyNpDataObjectType)
    {
        newElement = PythonNpDataObject::PyNpDataObject_getTagDict((PythonNpDataObject::PyNpDataObject*)element, NULL);
        //Py_INCREF(element);
        PyDict_SetItemString(newElement, "dataObject", element);
        item = PyUnicode_FromString("npDataObject");
        PyDict_SetItemString(newElement, "itomMetaInformation", item);
        Py_DECREF(item);
    }
    else if (element && Py_TYPE(element) == &PythonDataObject::PyDataObjectType)
    {
        newElement = PythonDataObject::PyDataObject_getTagDict((PythonDataObject::PyDataObject*)element, NULL);
        //Py_INCREF(element);
        PyDict_SetItemString(newElement, "dataObject", element);
        item = PyUnicode_FromString("dataObject");
        PyDict_SetItemString(newElement, "itomMetaInformation", item);
        Py_DECREF(item);
    }
    else if (element)
    {
        Py_INCREF(element);
        newElement = element;
    }
    else
    {
        PyErr_Format(PyExc_ValueError, "element is NULL");
        return NULL;
    }

    if (PyDict_Check(newElement))
    {
        //check that tags is no empty dictionary, this cannot be stored to matlab
        PyObject *temp = PyDict_GetItemString(newElement, "tags");
        if (temp && PyDict_Check(temp) && PyDict_Size(temp) <= 0)
        {
            item = PyLong_FromLong(0);
            PyDict_SetItemString(temp, "None", item);
            Py_DECREF(item);
        }
    }

    return newElement;

}

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyLoadMatlabMat_doc,"loadMatlabMat(filename) -> loads matlab mat-file by using scipy methods and returns the loaded dictionary. \n\
\n\
Parameters \n\
----------- \n\
filename : {str} \n\
    Filename from which the data will be imported (.mat will be added if not available)\n\
\n\
Returns \n\
------- \n\
PyDictionary with content of the Matlab-file\n\
\n\
Notes \n\
----- \n\
\n\
This function loads matlab mat-file by using scipy methods and returns the loaded dictionary.");
PyObject * PythonItom::PyLoadMatlabMat(PyObject * /*pSelf*/, PyObject *pArgs)
{
    PyObject* scipyIoModule = PyImport_ImportModule("scipy.io"); // new reference
    PyObject* resultLoadMat = NULL;

    if (scipyIoModule == NULL)
    {
        return PyErr_Format(PyExc_ImportError, "scipy-module and scipy.io-module could not be loaded.");
    }

    //Arguments must be: filename -> string

    const char *filename = NULL;


    if (!PyArg_ParseTuple(pArgs, "s", &filename))
    {
        Py_XDECREF(scipyIoModule);
        return PyErr_Format(PyExc_ValueError, "wrong arguments. Required argument is: filename (string)");
    }

    PyObject *kwdDict = PyDict_New();
    PyObject *argTuple = PyTuple_New(1);
    PyTuple_SetItem(argTuple, 0, PyUnicode_FromString(filename));
    PyDict_SetItemString(kwdDict, "squeeze_me",Py_True);
    PyObject *callable = PyObject_GetAttr(scipyIoModule, PyUnicode_FromString("loadmat"));
    resultLoadMat = PyObject_Call(callable, argTuple, kwdDict);
    Py_DECREF(kwdDict);
    Py_DECREF(argTuple);
    //resultLoadMat = PyObject_CallMethodObjArgs(scipyIoModule, PyUnicode_FromString("loadmat"), PyUnicode_FromString(filename), NULL);


    if (resultLoadMat)
    {
        //parse every element of dictionary and check if it is a numpy.ndarray. If so, transforms it to c-style contiguous form
        if (PyDict_Check(resultLoadMat))
        {
            PyObject *key = NULL;
            PyObject *value = NULL;
            Py_ssize_t pos = 0;

            while (PyDict_Next(resultLoadMat, &pos, &key, &value)) //borrowed reference to key and value
            {
                if (PyArray_Check(value))
                {
                    if (PyArray_SIZE((PyArrayObject*)value) == 1) //this is either a single value or a matlab-struct
                    {
                        PyObject *item = PyArray_ToList((PyArrayObject*)value); //new ref

                        if (item && (PyLong_Check(item) || PyFloat_Check(item))) //keep it
                        {
                            PyDict_SetItem(resultLoadMat, key, item);
                        }
                        else if (value && PyArray_HASFIELDS((PyArrayObject*)value))
                        {
                            //it may be that this is a struct which has been generated earlier from a npDataObject or dataObject
                            PyArray_Descr * descr = PyArray_DESCR((PyArrayObject*)value);

                            if (descr->fields != NULL) //fields is a dictionary with "fieldname" => type-description for this field
                            {
                                if (PyDict_Contains(descr->fields, PyUnicode_FromString("itomMetaInformation")))
                                {
                                    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
                                    PyObject *result = PyObject_CallMethodObjArgs(pyEngine->itomFunctions, PyUnicode_FromString("importMatlabMatAsDataObject"), value, NULL);

                                    if (result == NULL || PyErr_Occurred())
                                    {
                                        Py_XDECREF(result);
                                        Py_XDECREF(scipyIoModule);
                                        PyErr_Print();
                                        return PyErr_Format(PyExc_RuntimeError, "error while parsing imported dataObject or npDataObject.");
                                    }
                                    PyDict_SetItem(resultLoadMat, key, result);
                                    Py_XDECREF(result);
                                }
                            }

                        }

                        Py_XDECREF(item);

                    }
                    else //this should be an ordinary numpy.array
                    {
                        PyObject *newArr = (PyObject*)PyArray_GETCONTIGUOUS((PyArrayObject*)value); //should be new reference
                        PyDict_SetItem(resultLoadMat, key, newArr);
                        Py_DECREF(newArr);
                    }
                }
            }
        }
    }

    Py_XDECREF(scipyIoModule);

    return resultLoadMat;
}

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFilter_doc,"filter(name [, furtherParameters, ...]) -> invoke a filter (or algorithm) function within an algorithm-plugin. \n\
\n\
Parameters \n\
----------- \n\
name : {str} \n\
    The name of the filter\n\
furtherParameters : {variant} \n\
    Further parameters depend on the filter-methods itself (give the mandatory and then optional parameters in their defined order).\n\
\n\
Returns \n\
------- \n\
output parameters : {variant} \n\
    The returned values depend on the definition of each filter. In general it is a tuple of all output parameters that are defined by the filter function.\n\
\n\
Notes \n\
----- \n\
\n\
This function is used to invoke itom filter-functions or algorithms, declared within itom-algorithm plugins.\n\
The parameters (arguments) depends on the specific filter function (see filterHelp(name)),\n\
By filterHelp() a list of available filter functions is retrieved.");
PyObject * PythonItom::PyFilter(PyObject * /*pSelf*/, PyObject *pArgs, PyObject *kwds)
{
    int length = PyTuple_Size(pArgs);
    PyObject *params = NULL;
    ito::RetVal ret = ito::retOk;

    if (length == 0)
    {
        PyErr_Format(PyExc_ValueError, "no filter specified");
        return NULL;
    }
    PyObject *tempObj = PyTuple_GetItem(pArgs, 0);
    QString key;
    if (PyUnicode_Check(tempObj))
    {
        bool ok = false;
        key = PythonQtConversion::PyObjGetString(tempObj,false,ok);
    }
    else
    {
        PyErr_Format(PyExc_TypeError, "first argument must be the filter name! Wrong argument type!");
        return NULL;
    }

    ito::AddInManager *aim = ito::AddInManager::getInstance();
    const QHash<QString, ito::AddInAlgo::FilterDef *> *flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->constFind(key);
    if (cfit == flist->constEnd())
    {
        PyErr_SetString(PyExc_ValueError, "unknown filter, please check typing!");
        return NULL;
    }
    ito::AddInAlgo::FilterDef * fFunc = cfit.value();
    QVector<ito::ParamBase> paramsMandBase, paramsOptBase, paramsOutBase;

    const ito::FilterParams* filterParams = aim->getHashedFilterParams(fFunc->m_paramFunc);
    if (filterParams == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "parameters of filter could not be found.");
        return NULL;
    }

    params = PyTuple_GetSlice(pArgs, 1, PyTuple_Size(pArgs));

    //parses python-parameters with respect to the default values given py (*it).paramsMand and (*it).paramsOpt and returns default-initialized ParamBase-Vectors paramsMand and paramsOpt.
    ret += parseInitParams(&(filterParams->paramsMand), &(filterParams->paramsOpt), params, kwds, paramsMandBase, paramsOptBase);
    //makes deep copy from default-output parameters (*it).paramsOut and returns it in paramsOut (ParamBase-Vector)
    ret += copyParamVector(&(filterParams->paramsOut), paramsOutBase);

    if (ret.containsError())
    {
        PyErr_Format(PyExc_RuntimeError, "error while parsing parameters.");
        return NULL;
    }
    Py_DECREF(params);

    ret = (*(fFunc->m_filterFunc))(&paramsMandBase, &paramsOptBase, &paramsOutBase);

    if (!PythonCommon::transformRetValToPyException(ret))
    {
        return NULL;
    }
    else
    {
        if (paramsOutBase.size() == 0)
        {
            Py_RETURN_NONE;
        }
        else if (paramsOutBase.size() == 1)
        {
            PyObject* out = PythonParamConversion::ParamBaseToPyObject(paramsOutBase[0]); //new ref
            if (!PythonCommon::transformRetValToPyException(ret))
            {
                return NULL;
            }
            else
            {
                return out;
            }
        }
        else
        {
            //parse output vector to PyObject-Tuple
            PyObject* out = PyTuple_New(paramsOutBase.size());
            PyObject* temp;
            Py_ssize_t i = 0;

            foreach(const ito::ParamBase &p, paramsOutBase)
            {
                temp = PythonParamConversion::ParamBaseToPyObject(p); //new ref
                if (temp)
                {
                    PyTuple_SetItem(out,i,temp); //steals ref
                    i++;
                }
                else
                {
                    break;
                }
            }

            if (!PythonCommon::transformRetValToPyException(ret))
            {
                Py_DECREF(out);
                return NULL;
            }
            else
            {
                return out;
            }
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pySaveDataObject_doc,"saveDataObject(filename, dataObject [, tagsAsBinary]) -> save a dataObject to harddrive. \n\
\n\
Parameters \n\
----------- \n\
filename : {str} \n\
    Filename and Path of the destination (.ido will be added if no .*-ending is available)\n\
dataObject : {DataObject} \n\
    An allocated dataObject of n-Dimensions.\n\
tagsAsBinary : {bool}, optional \n\
    Optional tag to toogle if numeric-tags should be saved (metaData) as binary or by default as string.\n\
\n\
Notes \n\
----- \n\
\n\
This function writes an ito::dataObject to the file specified by filename. The data is stored as binary. \n\
The value of string-Tags is encoded to avoid XML-conflics. The value of numerical-Tags are saved as string\n\
with 15 significat digits (>32bit, tagsAsBinary == False, default) or as binary (tagsAsBinary == True).\n\
Tagnames which contains special characters leads to XML-conflics.");
PyObject* PythonItom::PySaveDataObject(PyObject* /*pSelf*/, PyObject* pArgs)
{
    ito::RetVal ret(ito::retOk);
    int length = PyTuple_Size(pArgs);
    const char* folderfilename;
    PyObject *pyDataObject = NULL;
    PyObject *pyBool = NULL;
    bool asBin = false; // defaults metaData as string (false)

    if (length < 2)
    {
        return PyErr_Format(PyExc_ValueError, "Mandatory parameters are a String for the Folder/Filename and a dataObject");
    }
    else if (length == 2) //!< copy name + object
    {
        if (!PyArg_ParseTuple(pArgs, "sO", &folderfilename, &pyDataObject))
        {
            return PyErr_Format(PyExc_TypeError, "wrong input type");
        }
        Py_INCREF(pyDataObject);
    }
    else if (length == 3) //!< copy name + object + asBinary
    {
        if (!PyArg_ParseTuple(pArgs, "sOO", &folderfilename, &pyDataObject, &pyBool))
        {
            return PyErr_Format(PyExc_TypeError, "wrong input type");
        }
        Py_INCREF(pyDataObject);
        Py_INCREF(pyBool);
    }
    else
    {
        return PyErr_Format(PyExc_ValueError, "To many arguments");
    }

    PythonDataObject::PyDataObject* elem = (PythonDataObject::PyDataObject*)pyDataObject;

    if (pyBool == NULL)
    {
        if (pyBool == Py_True)   // do not change the filename
        {
            asBin = true;
        }
        Py_XDECREF(pyBool);
    }

    ret += ito::saveDOBJ2XML(elem->dataObject, folderfilename, false, asBin);


    Py_XDECREF(pyDataObject);

    if (ret.containsError())
    {
        if (ret.errorMessage())
        {
            return PyErr_Format(PyExc_RuntimeError, "Could not save dataObject: error message: \n%s\n", ret.errorMessage());
        }
        else
        {
            return PyErr_Format(PyExc_RuntimeError, "Could not save dataObject");
        }
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyLoadDataObject_doc,"loadDataObject(filename, dataObject [, doNotAppendIDO]) -> load a dataObject from the harddrive to an existing dataObject. \n\
\n\
Parameters \n\
----------- \n\
filename : {str} \n\
    Filename and Path of the destination (.ido will be added if not available)\n\
dataObject : {DataObject} \n\
    A pre-allocated dataObject (empty Objects are allowed).\n\
doNotAppendIDO : {bool}, optional \n\
    Optional tag to avoid adding -ido-Tag, default is False.\n\
\n\
Notes \n\
----- \n\
\n\
This function reads an ito::dataObject from the file specified by filename. \n\
MetaData saveType (string, binary) are extracted from the file and restored within the object.\n\
The value of string-Tags must be encoded to avoid XML-conflics.\n\
Tagnames which contains special characters leads to XML-conflics.");
PyObject* PythonItom::PyLoadDataObject(PyObject* /*pSelf*/, PyObject* pArgs)
{
    ito::RetVal ret(ito::retOk);
    int length = PyTuple_Size(pArgs);
    const char* folderfilename;
    PyObject *pyDataObject = NULL;
    PyObject *pyBool = NULL;
    bool appendEnding = true;

    if (length < 2)
    {
        return PyErr_Format(PyExc_ValueError, "Mandatory parameters are a String for the Folder/Filename and a dataObject");
    }
    else if (length == 2) //!< copy name + object
    {
        if (!PyArg_ParseTuple(pArgs, "sO", &folderfilename, &pyDataObject))
        {
            return PyErr_Format(PyExc_TypeError, "wrong input type");
        }
        Py_INCREF(pyDataObject);
    }
    else if (length == 3) //!< copy name + object + asBinary
    {
        if (!PyArg_ParseTuple(pArgs, "sOO", &folderfilename, &pyDataObject, &pyBool))
        {
            return PyErr_Format(PyExc_TypeError, "wrong input type");
        }
        Py_INCREF(pyDataObject);
        Py_INCREF(pyBool);
    }
    else
    {
        return PyErr_Format(PyExc_ValueError, "To many arguments");
    }

    PythonDataObject::PyDataObject* elem = (PythonDataObject::PyDataObject*)pyDataObject;

    if (pyBool == NULL)
    {
        if (pyBool == Py_False)   // do not change the filename
        {
            appendEnding = false;
        }
        Py_XDECREF(pyBool);
    }

    ret += ito::loadXML2DOBJ(elem->dataObject, folderfilename, false, appendEnding);

    Py_XDECREF(pyDataObject);


    if (ret.containsError())
    {
        if (ret.errorMessage())
        {
            return PyErr_Format(PyExc_RuntimeError, "Could not load dataObject: error message: \n%s\n", ret.errorMessage());
        }
        else
        {
            return PyErr_Format(PyExc_RuntimeError, "Could not load dataObject.");
        }
    }

    Py_RETURN_NONE;
}



//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(getDefaultScaleAbleUnits_doc,"getDefaultScaleAbleUnits() -> Get a PythonList with standard scaleable units. \n\
\n\
Returns \n\
------- \n\
List with strings containing all scaleable units : {PyList}\n\
\n\
Notes \n\
----- \n\
\n\
Get a PythonList with standard scaleable units. Used together with itom.ScaleValueAndUnit(...)");
PyObject* PythonItom::getDefaultScaleAbleUnits(PyObject * /*pSelf*/)
{
    PyObject *myList = PyList_New(0);
    PyList_Append(myList, PyUnicode_FromString("mm"));
    PyList_Append(myList, PyUnicode_FromString("m"));
    PyList_Append(myList, PyUnicode_FromString("V"));
    PyList_Append(myList, PyUnicode_FromString("s"));
    PyList_Append(myList, PyUnicode_FromString("g"));
    PyList_Append(myList, PyUnicode_FromString("cd"));
    PyList_Append(myList, PyUnicode_FromString("A"));
    PyList_Append(myList, PyUnicode_FromString("%"));

    return myList;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(ScaleValueAndUnit_doc,"ScaleValueAndUnit(ScaleableUnits, value, valueUnit) -> Scale a value and its unit and returns [value, 'Unit'] \n\
\n\
Parameters \n\
----------- \n\
ScaleableUnits : {PyList of Strings} \n\
    A string list with all scaleable units\n\
value : {double} \n\
    The value to be scaled\n\
valueUnit : {str} \n\
    The value unit to be scaled\n\
\n\
Returns \n\
------- \n\
PyTuple with scaled value and scaled unit\n\
\n\
Notes \n\
----- \n\
\n\
Rescale a value with SI-unit (e.g. 0.01 mm to 10 micrometer). Used together with itom.getDefaultScaleAbleUnits()");
PyObject* PythonItom::ScaleValueAndUnit(PyObject * /*pSelf*/, PyObject *pArgs)
{
    QStringList myQlist;

    int length = PyTuple_Size(pArgs);
    double value = 0.0;
    double valueOut = 0.0;
    const char* unitString = NULL;
    PyObject *myList = NULL;
    QString unitIn("");
    QString unitOut("");

    if (length < 3)
    {
        PyErr_Format(PyExc_ValueError, "Inputarguments are scaleable units, value, unit");
        return NULL;
    }
    else if (length == 3)
    {
        if (!PyArg_ParseTuple(pArgs, "O!ds", &PyList_Type, &myList, &value, &unitString))
        //if (!PyArg_ParseTuple(pArgs, "ds", &value, &unitString))
        {
            PyErr_Format(PyExc_RuntimeError, "Inputarguments are ...");
            return NULL;
        }
        Py_INCREF(myList);
    }
    else
    {
        PyErr_Format(PyExc_RuntimeError, "to many input parameters specified");
        return NULL;
    }

    for (int i = 0; i < PyList_Size(myList); i++)
    {
        myQlist.append(PythonQtConversion::PyObjGetString(PyList_GetItem(myList, i)));
    }

    if (unitString)
    {
        unitIn.append(unitString);
    }

    ito::RetVal ret = ito::formatDoubleWithUnit(myQlist, unitIn, value, valueOut, unitOut);
    Py_XDECREF(myList);

    return Py_BuildValue("ds", valueOut, unitOut.toLatin1().data());
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(getAppPath_doc,"getAppPath() -> returns absolute path of application base directory.\n\
\n\
Returns \n\
------- \n\
Path : {str}\n\
    string with absolute path of this application\n\
\n\
Notes \n\
----- \n\
\n\
This function returns the absolute path of application base directory.\n\
The return value is independent of the current working diractory");
PyObject* PythonItom::getAppPath(PyObject* /*pSelf*/)
{
    return PythonQtConversion::QStringToPyObject(QDir::cleanPath(QCoreApplication::applicationDirPath()));
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(getCurrentPath_doc,"getCurrentPath() -> returns absolute path of current working directory.\n\
\n\
Returns \n\
------- \n\
Path : {str}\n\
    string with current working path\n\
\n\
Notes \n\
----- \n\
\n\
This function returns the current working path of the application.");
PyObject* PythonItom::getCurrentPath(PyObject* /*pSelf*/)
{
    return PythonQtConversion::QStringToPyObject(QDir::cleanPath(QDir::currentPath()));
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(setCurrentPath_doc,"setCurrentPath(newPath) -> sets absolute path of current working directory \n\
\n\
Parameters \n\
----------- \n\
newPath : {str} \n\
    The new working path of this application\n\
\n\
Returns \n\
------- \n\
Success : {bool} \n\
    True in case of success else False\n\
\n\
Notes \n\
----- \n\
\n\
sets absolute path of current working directory returns True if currentPath could be changed, else False.");
PyObject* PythonItom::setCurrentPath(PyObject* /*pSelf*/, PyObject* pArgs)
{
    PyObject *pyObj = NULL;
    if (!PyArg_ParseTuple(pArgs, "O", &pyObj))
    {
        PyErr_Format(PyExc_RuntimeError, "method requires a string as argument");
        return NULL;
    }

    bool ok;
    QString path;
    path = PythonQtConversion::PyObjGetString(pyObj,true,ok);
    if (ok == false)
    {
        PyErr_Format(PyExc_RuntimeError, "newPath parameter could not be interpreted as string.");
        return NULL;
    }
    if (!QDir::setCurrent(path))
    {
        Py_RETURN_FALSE;
    }
    else
    {
        PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
        if (pyEngine) emit pyEngine->pythonCurrentDirChanged();
        Py_RETURN_TRUE;
    }

}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonItom::setApplicationCursor(PyObject* pSelf, PyObject* pArgs)
{
    int i = -1;
    if (!PyArg_ParseTuple(pArgs, "|i", &i))
    {
        return NULL;
    }

    if (i > Qt::LastCursor)
    {
        return PyErr_Format(PyExc_RuntimeError, "Cursor number must be in range [-1,%i]", Qt::LastCursor);
    }


    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (i >= 0 && pyEngine)
    {
        Qt::CursorShape shape = (Qt::CursorShape)i;
        emit pyEngine->pythonSetCursor(shape);
    }
    else if (pyEngine)
    {
        emit pyEngine->pythonResetCursor();
    }
    Py_RETURN_NONE;
}

///*static*/ PyObject* PythonItom::PyGetGlobalDict(PyObject* /*pSelf*/)
//{
//	PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
//	if (pyEngine)
//	{
//		PyObject *dict = pyEngine->getMainDictionary();
//		if (dict)
//		{
//			Py_INCREF(dict);
//			return dict;
//		}
//		PyErr_Format(PyExc_RuntimeError, "The global dictionary is not available.");
//        return NULL;
//	}
//	else
//	{
//		PyErr_Format(PyExc_RuntimeError, "Python Engine is not available.");
//        return NULL;
//	}
//}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyItom_FigureClose_doc,"close(handle|'all') -> method to close any specific or all open figures (unless any figure-instance still keeps track of them)\n\
\n\
This method closes and deletes any specific figure (given by handle) or all opened figures. This method always calls the static method \n\
close of class figure.\n\
\n\
Parameters \n\
----------- \n\
handle : {dataIO-Instance} \n\
    any figure handle (>0) or 'all' in order to close all opened figures \n\
\n\
Notes \n\
------- \n\
If any instance of class 'figure' still keeps a reference to any figure, it is only closed and deleted if the last instance is deleted, too. \n\
\n\
See Also \n\
--------- \n\
figure.close");


//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyLoadIDC_doc,"loadIDC(filename) -> loads a pickled idc-file and returns the content as dictionary\n\
\n\
This methods loads the given idc-file using the method load from the python-buildin module pickle and returns the loaded dictionary.\n\
\n\
Parameters \n\
----------- \n\
filename : {String} \n\
    absolute filename or filename relative to the current directory. \n\
\n\
See Also \n\
--------- \n\
pickle.load");
PyObject* PythonItom::PyLoadIDC(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds)
{
    const char *kwlist[] = {"filename", NULL};
    char* filename = NULL;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "s", const_cast<char**>(kwlist), &filename))
    {
        return NULL;
    }

    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (pyEngine)
    {
        QFileInfo info(filename);

        if (info.exists())
        {
            PyObject *dict = PyDict_New();
            RetVal retval = pyEngine->unpickleDictionary(dict, filename, true);

            if (!PythonCommon::transformRetValToPyException(retval))
            {
                Py_DECREF(dict);
                return NULL;
            }

            return dict;
        }
        else
        {
            return PyErr_Format(PyExc_RuntimeError, "The file '%s' does not exist", info.absoluteFilePath().toAscii().data());
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "Python Engine not available");
        return NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pySaveIDC_doc,"saveIDC(filename, dict [,overwriteIfExists = True]) -> saves the given dictionary as pickled idc-file.\n\
\n\
This method saves the given dictionary as pickled icd-file using the method dump from the builtin module pickle.\n\
\n\
Parameters \n\
----------- \n\
filename : {string} \n\
    absolute filename or filename relative to the current directory. \n\
dict : {dict} \n\
    dictionary which should be pickled. \n\
overwriteIfExists : {bool}, default: True \n\
    if True, an existing file will be overwritten. \n\
\n\
See Also \n\
--------- \n\
pickle.dump");
PyObject* PythonItom::PySaveIDC(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds)
{
    const char *kwlist[] = {"filename", "dict", "overwriteIfExists", NULL};
    char* filename = NULL;
    PyObject *dict = NULL;

#if PY_VERSION_HEX < 0x03030000
	unsigned char overwriteIfExists = 1;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "sO!|b", const_cast<char**>(kwlist), &filename, &PyDict_Type, &dict, &overwriteIfExists)) //all borrowed
    {
        return NULL;
    }
#else //only python 3.3 or higher has the 'p' (bool, int) type string
	int overwriteIfExists = 1; //this must be int, not bool!!! (else crash)

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "sO!|p", const_cast<char**>(kwlist), &filename, &PyDict_Type, &dict, &overwriteIfExists)) //all borrowed
    {
        return NULL;
    }
#endif

    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (pyEngine)
    {
        QFileInfo info(filename);

        if (!info.exists() || (info.exists() && (overwriteIfExists > 0)))
        {
            PyObject *dict = PyDict_New();
            RetVal retval = pyEngine->pickleDictionary(dict, filename);

            if (!PythonCommon::transformRetValToPyException(retval))
            {
                Py_DECREF(dict);
                return NULL;
            }

            Py_RETURN_NONE;
        }
        else
        {
            return PyErr_Format(PyExc_RuntimeError, "The file '%s' cannot be overwritten", info.absoluteFilePath().toAscii().data());
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "Python Engine not available");
        return NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //
//                                                                                                              //
//          PYTHON MODULES - - - PYTHON TYPES - - - PYTHON MODULES                                              //
//                                                                                                              //
//  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------------------------------------------------------------
PyMethodDef PythonItom::PythonMethodItom[] = {
    // "Python name", C Ffunction Code, Argument Flags, __doc__ description
    {"scriptEditor", (PyCFunction)PythonItom::PyOpenEmptyScriptEditor, METH_NOARGS, pyOpenEmptyScriptEditor_doc},
    {"newScript", (PyCFunction)PythonItom::PyNewScript, METH_NOARGS, pyNewScript_doc},
    {"openScript", (PyCFunction)PythonItom::PyOpenScript, METH_VARARGS, pyOpenScript_doc},
    {"plot", (PyCFunction)PythonItom::PyPlotImage, METH_VARARGS | METH_KEYWORDS, pyPlotImage_doc},
    {"liveImage", (PyCFunction)PythonItom::PyLiveImage, METH_VARARGS | METH_KEYWORDS, pyLiveImage_doc},
	{"close", (PyCFunction)PythonFigure::PyFigure_close, METH_VARARGS, pyItom_FigureClose_doc}, /*class static figure.close(...)*/
    /*{"liveLine", (PyCFunction)PythonItom::PyLiveLine, METH_VARARGS, pyLiveLine_doc},
    {"closeFigure", (PyCFunction)PythonItom::PyCloseFigure, METH_VARARGS, pyCloseFigure_doc},
    {"setFigParam", (PyCFunction)PythonItom::PySetFigParam, METH_VARARGS, pySetFigParam_doc},
    {"getFigParam", (PyCFunction)PythonItom::PyGetFigParam, METH_VARARGS, pyGetFigParam_doc},*/
    {"filter", (PyCFunction)PythonItom::PyFilter, METH_VARARGS | METH_KEYWORDS, pyFilter_doc},
    {"filterHelp", (PyCFunction)PythonItom::PyFilterHelp, METH_VARARGS, pyFilterHelp_doc},
    {"widgetHelp", (PyCFunction)PythonItom::PyWidgetHelp, METH_VARARGS, pyWidgetHelp_doc},
    {"pluginHelp", (PyCFunction)PythonItom::PyPluginHelp, METH_VARARGS | METH_KEYWORDS, pyPluginHelp_doc},
    {"pluginLoaded", (PyCFunction)PythonItom::PyPluginLoaded, METH_VARARGS, pyPluginLoaded_doc},
    {"version", (PyCFunction)PythonItom::PyITOMVersion, METH_VARARGS, pyITOMVersion_doc},
    {"saveDataObject", (PyCFunction)PythonItom::PySaveDataObject, METH_VARARGS, pySaveDataObject_doc},
    {"loadDataObject", (PyCFunction)PythonItom::PyLoadDataObject, METH_VARARGS, pyLoadDataObject_doc},
    {"addButton", (PyCFunction)PythonItom::PyAddButton, METH_VARARGS | METH_KEYWORDS, pyAddButton_doc},
    {"removeButton", (PyCFunction)PythonItom::PyRemoveButton, METH_VARARGS, pyRemoveButton_doc},
    {"addMenu", (PyCFunction)PythonItom::PyAddMenu, METH_VARARGS | METH_KEYWORDS, pyAddMenu_doc},
    {"removeMenu", (PyCFunction)PythonItom::PyRemoveMenu, METH_VARARGS | METH_KEYWORDS, pyRemoveMenu_doc},
    {"saveMatlabMat", (PyCFunction)PythonItom::PySaveMatlabMat, METH_VARARGS, pySaveMatlabMat_doc},
    {"loadMatlabMat", (PyCFunction)PythonItom::PyLoadMatlabMat, METH_VARARGS, pyLoadMatlabMat_doc},
    {"scaleDoubleUnit", (PyCFunction)PythonItom::ScaleValueAndUnit, METH_VARARGS, ScaleValueAndUnit_doc},
    {"getDefaultScaleableUnits", (PyCFunction)PythonItom::getDefaultScaleAbleUnits, METH_NOARGS, getDefaultScaleAbleUnits_doc},
    {"getAppPath", (PyCFunction)PythonItom::getAppPath, METH_NOARGS, getAppPath_doc},
    {"getCurrentPath", (PyCFunction)PythonItom::getCurrentPath, METH_NOARGS, getCurrentPath_doc},
    {"setCurrentPath", (PyCFunction)PythonItom::setCurrentPath, METH_VARARGS, setCurrentPath_doc},
    {"checkSignals", (PyCFunction)PythonItom::PyCheckSignals, METH_NOARGS, NULL},
    {"processEvents", (PyCFunction)PythonItom::PyProcessEvents, METH_NOARGS, NULL},
    {"getDebugger", (PyCFunction)PythonItom::PyGetDebugger, METH_NOARGS, "getDebugger() -> returns new reference to debugger instance"},
    {"gcStartTracking", (PyCFunction)PythonItom::PyGCStartTracking, METH_NOARGS, "gcStartTracking() -> stores the current object list of the garbage collector."},
    {"gcEndTracking", (PyCFunction)PythonItom::PyGCEndTracking, METH_NOARGS, "gcEndTracking() -> compares the current object list of the garbage collector with the recently saved list."},
	//{"getGlobalDict", (PyCFunction)PythonItom::PyGetGlobalDict, METH_NOARGS, "getGlobalDict() -> returns borrowed reference to global dictionary of itom python instance"},
    {"getScreenInfo", (PyCFunction)PythonItom::PyGetScreenInfo, METH_NOARGS, getScreenInfo_doc},
    {"setApplicationCursor", (PyCFunction)PythonItom::setApplicationCursor, METH_VARARGS, NULL},
    {"loadIDC", (PyCFunction)PythonItom::PyLoadIDC, METH_VARARGS | METH_KEYWORDS, pyLoadIDC_doc},
    {"saveIDC", (PyCFunction)PythonItom::PySaveIDC, METH_VARARGS | METH_KEYWORDS, pySaveIDC_doc},
    {NULL, NULL, 0, NULL}
};

PyModuleDef PythonItom::PythonModuleItom = {
    PyModuleDef_HEAD_INIT, "itom", NULL, -1, PythonItom::PythonMethodItom,
    NULL, NULL, NULL, NULL
};

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonItom::PyInitItom(void)
{
    PyObject *m = PyModule_Create(&PythonModuleItom);
    if (m != NULL)
    {
        PyModule_AddObject(m, "numeric", PyModule_Create(&PythonNumeric::PythonModuleItomNumeric));

        //constants for addMenu
        PyModule_AddObject(m, "BUTTON",     PyLong_FromLong(0)); //steals reference to value
        PyModule_AddObject(m, "SEPARATOR",  PyLong_FromLong(1)); //steals reference to value
        PyModule_AddObject(m, "MENU",       PyLong_FromLong(2)); //steals reference to value

        //equivalent:
        //PyObject_SetAttrString(m, "TEST", PyLong_FromLong(0));
        //PyObject_GenericSetAttr(m, PyUnicode_FromString("TEST2"), PyLong_FromLong(-1));

    }
    return m;
}

} //end namespace ito

//----------------------------------------------------------------------------------------------------------------------------------
