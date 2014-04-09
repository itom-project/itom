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

#include "pythonFigure.h"

#include "structmember.h"

#include "../global.h"
#include "../organizer/uiOrganizer.h"

#include "pythonQtConversion.h"
#include "AppManagement.h"
#include "pythonPlotItem.h"

#include <qsharedpointer.h>
#include <qmessagebox.h>
#include <qmetaobject.h>


namespace ito
{
// -------------------------------------------------------------------------------------------------------------------------
//
//  PyFigure
//
// -------------------------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------------------------------OK
void PythonFigure::PyFigure_dealloc(PyFigure* self)
{
    self->guardedFigHandle.clear(); //if reference of semaphore drops to zero, the static method threadSafeDeleteUi of UiOrganizer is called that will finally delete the figure

    DELETE_AND_SET_NULL(self->signalMapper);

    //Py_TYPE(self)->tp_free((PyObject*)self);
    PythonUi::PyUiItemType.tp_dealloc((PyObject*)self);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonFigure::PyFigure_new(PyTypeObject *type, PyObject * args, PyObject * kwds)
{
    PyFigure *self = (PyFigure*)PythonUi::PyUiItemType.tp_new(type,args,kwds);
    if (self != NULL)
    {
        self->guardedFigHandle.clear(); //default: invalid
        self->rows = 1;
        self->cols = 1;
        self->currentSubplotIdx = 0;
        self->signalMapper = NULL;
    }

    return (PyObject *)self;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigureInit_doc,"figure([handle, [rows = 1, cols = 1]]) -> creates figure window.\n\
\n\
The class itom.figure represents a standalone figure window, that can have various subplots. If an instance of this class \n\
is created without further parameters a new figure is created and opened having one subplot area (currently empty) and the numeric \n\
handle to this figure is returned:: \n\
    \n\
    h = figure() \n\
\n\
Subplots are arranged in a regular grid whose size is defined by the optional parameters 'rows' and 'cols'. If you create a figure \n\
instance with a given handle, the instance is either a reference to an existing figure that has got this handle or if it does not exist, \n\
a new figure with the desired handle is opened and the handle is returned, too. \n\
\n\
Parameters \n\
------------- \n\
handle : {int} \n\
    numeric handle of the desired figure. \n\
rows : {int, default: 1} \n\
    number of rows this figure should have (defines the size of the subplot-grid) \n\
cols : {int, default: 1} \n\
    number of columns this figure should have (defines the size of the subplot-grid)");
int PythonFigure::PyFigure_init(PyFigure *self, PyObject *args, PyObject *kwds)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    const char *kwlist[] = {"handle", "rows", "cols", NULL};

    int handle = -1;
    unsigned int rows = 1;
    unsigned int cols = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds,"|iII",const_cast<char**>(kwlist), &handle, &rows, &cols))
    {
        return -1;
    }

    QSharedPointer< QSharedPointer<unsigned int> > guardedFigHandle(new QSharedPointer<unsigned int>());
    QSharedPointer<unsigned int> initSlotCount(new unsigned int);
    QSharedPointer<unsigned int> objectID(new unsigned int);
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue;

    if (handle != -1)
    {
        *guardedFigHandle = QSharedPointer<unsigned int>(new unsigned int);
        **guardedFigHandle = handle;
    }

    *initSlotCount = 0;

    QSharedPointer<int> rows_(new int);
    QSharedPointer<int> cols_(new int);
    *rows_ = rows;
    *cols_ = cols;

    QMetaObject::invokeMethod(uiOrga, "createFigure",Q_ARG(QSharedPointer< QSharedPointer<unsigned int> >,guardedFigHandle), Q_ARG(QSharedPointer<unsigned int>, initSlotCount), Q_ARG(QSharedPointer<unsigned int>, objectID), Q_ARG(QSharedPointer<int>,rows_), Q_ARG(QSharedPointer<int>,cols_), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if (!locker.getSemaphore()->wait(60000))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while opening figure");
        return -1;
    }
    
    retValue = locker.getSemaphore()->returnValue;
    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return -1;
    }

    self->guardedFigHandle = *guardedFigHandle;
    DELETE_AND_SET_NULL(self->signalMapper);
    self->signalMapper = new PythonQtSignalMapper(*initSlotCount);

    self->rows = *rows_;
    self->cols = *cols_;

    PyObject *args2 = PyTuple_New(3);
    PyTuple_SetItem(args2,0,PyLong_FromLong(*objectID));
    PyTuple_SetItem(args2,1, PyUnicode_FromString("<figure>"));
    PyTuple_SetItem(args2,2, PyUnicode_FromString("FigureClass"));
    int result = PythonUi::PyUiItemType.tp_init((PyObject*)self,args2,NULL);
    Py_DECREF(args2);

    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonFigure::PyFigure_repr(PyFigure *self)
{
    PyObject *result;
    if (self->guardedFigHandle.isNull())
    {
        result = PyUnicode_FromFormat("Figure(empty)");
    }
    else
    {
        result = PyUnicode_FromFormat("Figure(handle: %i)", *(self->guardedFigHandle));
    }
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigurePlot_doc,"plot(data, [areaIndex, className]) -> plots a dataObject in the current or given area of this figure\n\
Plot an existing dataObject in not dockable, not blocking window. \n\
The style of the plot will depend on the object dimensions.\n\
If x-dim or y-dim are equal to 1, plot will be a lineplot else a 2D-plot.\n\
\n\
Parameters\n\
-----------\n\
data : {DataObject} \n\
    Is the data object whose region of interest will be plotted.\n\
areaIndex: {int}, optional \n\
    \n\
className : {str}, optional \n\
    class name of desired plot (if not indicated default plot will be used (see application settings) \n\
");
PyObject* PythonFigure::PyFigure_plot(PyFigure *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"data", "areaIndex", "className", NULL};
    PyObject *data = NULL;
    int areaIndex = self->currentSubplotIdx;
    char* className = NULL;
    bool ok = true;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|is", const_cast<char**>(kwlist), &data, &areaIndex, &className))
    {
        return NULL;
    }

    QSharedPointer<ito::DataObject> newDataObj(PythonQtConversion::PyObjGetDataObjectNewPtr(data, false, ok));
    if (!ok)
    {
        PyErr_SetString(PyExc_RuntimeError, "first argument cannot be converted to a dataObject");
        return NULL;
    }

    if (areaIndex > self->cols * self->rows)
    {
        PyErr_SetString(PyExc_RuntimeError, "areaIndex is bigger than the maximum number of subplot areas in this figure");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = areaIndex % self->cols;
    int areaRow = (areaIndex - areaCol) / self->rows;

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    QString defaultPlotClassName;
    if (className)
    {
        defaultPlotClassName = className;
    }
    QSharedPointer<unsigned int> objectID(new unsigned int);

    QMetaObject::invokeMethod(uiOrg, "figurePlot", Q_ARG(QSharedPointer<ito::DataObject>, newDataObj), Q_ARG(QSharedPointer<unsigned int>, self->guardedFigHandle), Q_ARG(QSharedPointer<unsigned int>, objectID), Q_ARG(int, areaRow), Q_ARG(int, areaCol), Q_ARG(QString, defaultPlotClassName), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
    if (!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while plotting data object");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }

    //return new instance of PyUiItem
    PyObject *args2 = PyTuple_New(0); //Py_BuildValue("OO", self, name);
    PyObject *kwds2 = PyDict_New();
    PyDict_SetItemString(kwds2, "objectID", PyLong_FromLong(*objectID));
    PyDict_SetItemString(kwds2, "figure", (PyObject*)self);
    PythonPlotItem::PyPlotItem *pyPlotItem = (PythonPlotItem::PyPlotItem *)PyObject_Call((PyObject *)&PythonPlotItem::PyPlotItemType, args2, kwds2);
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    if (pyPlotItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create plotItem of plot widget");
        return NULL;
    }

    return (PyObject*)pyPlotItem;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigureLiveImage_doc,"liveImage(cam, [areaIndex, className]) -> shows a camera live image in the current or given area of this figure\n\
Creates a plot-image (2D) and automatically grabs images into this window.\n\
This function is not blocking.\n\
\n\
Parameters\n\
-----------\n\
cam : {dataIO-Instance} \n\
    Camera grabber device from which images are acquired.\n\
areaIndex: {int}, optional \n\
    \n\
className : {str}, optional \n\
    class name of desired plot (if not indicated default plot will be used (see application settings) \n\
");
/*static*/ PyObject* PythonFigure::PyFigure_liveImage(PyFigure *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"cam", "areaIndex", "className", NULL};
    PythonPlugins::PyDataIOPlugin *cam = NULL;
    int areaIndex = self->currentSubplotIdx;
    char* className = NULL;
    bool ok = true;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|is", const_cast<char**>(kwlist), &PythonPlugins::PyDataIOPluginType, &cam, &areaIndex, &className))
    {
        return NULL;
    }

    if (areaIndex > self->cols * self->rows)
    {
        PyErr_SetString(PyExc_RuntimeError, "areaIndex is bigger than the maximum number of subplot areas in this figure");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = areaIndex % self->cols;
    int areaRow = (areaIndex - areaCol) / self->rows;

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    QString defaultPlotClassName;
    if (className)
    {
        defaultPlotClassName = className;
    }
    QSharedPointer<unsigned int> objectID(new unsigned int);

    QMetaObject::invokeMethod(uiOrg, "figureLiveImage", Q_ARG(AddInDataIO*, cam->dataIOObj), Q_ARG(QSharedPointer<unsigned int>, self->guardedFigHandle), Q_ARG(QSharedPointer<unsigned int>, objectID), Q_ARG(int, areaRow), Q_ARG(int, areaCol), Q_ARG(QString, defaultPlotClassName), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
    if (!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing live image");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }
    
    //return new instance of PyUiItem
    PyObject *args2 = PyTuple_New(0); //Py_BuildValue("OO", self, name);
    PyObject *kwds2 = PyDict_New();
    PyDict_SetItemString(kwds2, "objectID", PyLong_FromLong(*objectID));
    PyDict_SetItemString(kwds2, "figure", (PyObject*)self);
    PythonPlotItem::PyPlotItem *pyPlotItem = (PythonPlotItem::PyPlotItem *)PyObject_Call((PyObject *)&PythonPlotItem::PyPlotItemType, args2, kwds2);
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    if (pyPlotItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create plotItem of plot widget");
        return NULL;
    }

    return (PyObject*)pyPlotItem;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigureShow_doc,"show() -> shows figure \n\
\n\
\n\
");
PyObject* PythonFigure::PyFigure_show(PyFigure *self, PyObject *args)
{
    int modalLevel = 0; //no modal

    if (!PyArg_ParseTuple(args, ""))
    {
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if (uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if (*(self->guardedFigHandle) <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid figure handle.");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    QSharedPointer<int> retCodeIfModal(new int);
    *retCodeIfModal = -1;
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "showDialog", Q_ARG(unsigned int, *(self->guardedFigHandle)) , Q_ARG(int,modalLevel), Q_ARG(QSharedPointer<int>, retCodeIfModal), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if (!locker.getSemaphore()->wait(30000))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing dialog");
        return NULL;
    }
    
    retValue = locker.getSemaphore()->returnValue;
    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    if (*retCodeIfModal >= 0)
    {
        return Py_BuildValue("i",*retCodeIfModal);
    }
    else
    {
        Py_RETURN_NONE;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigureHide_doc, "hide() -> hides figure without deleting it\n\
\n\
\n\
");
PyObject* PythonFigure::PyFigure_hide(PyFigure *self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if (uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if (*(self->guardedFigHandle) <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid figure handle.");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "hideDialog", Q_ARG(unsigned int, *(self->guardedFigHandle)), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if (!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while hiding figure");
        return NULL;
    }
    
    retValue = locker.getSemaphore()->returnValue;
    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigureSubplot_doc,"subplot(index) -> returns plotItem of desired subplot\n\
\n\
This method closes and deletes any specific figure (given by handle) or all opened figures. \n\
\n\
Parameters \n\
----------- \n\
index : {unsigned int} \n\
    index to desired subplot. The subplot at the top, left position has the index 0 whereas the index is incremented row-wise.");
/*static*/ PyObject* PythonFigure::PyFigure_getSubplot(PyFigure *self, PyObject *args)
{
    unsigned int index = 0;
    if (!PyArg_ParseTuple(args, "I", &index))
    {
        return NULL;
    }

    if (index >= (unsigned int)(self->cols * self->rows))
    {
        return PyErr_Format(PyExc_RuntimeError,"index exceeds maximum number of existing subplots. The allowed range is [0,%i]", (self->cols * self->rows - 1));
    }

    //return new instance of PyUiItem
    PyObject *arg2 = Py_BuildValue("OI", self, index);
    PythonPlotItem::PyPlotItem *plotItem = (PythonPlotItem::PyPlotItem *)PyObject_CallObject((PyObject *)&PythonPlotItem::PyPlotItemType, arg2);
    Py_DECREF(arg2);

    if (plotItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create plotItem of requested subplot");
        return NULL;
    }

    if (PyErr_Occurred())
    {
        Py_XDECREF(plotItem);
        plotItem = NULL;
    }

    return (PyObject *)plotItem;

}

//----------------------------------------------------------------------------------------------------------------------------------
//   getter / setters
//----------------------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonFigure::PyFigure_getHandle(PyFigure *self, void * /*closure*/)
{
    if (self->guardedFigHandle.isNull())
    {
        PyErr_SetString(PyExc_RuntimeError,"invalid figure");
        return NULL;
    }
    return Py_BuildValue("i", *(self->guardedFigHandle));
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigure_docked_doc, "dock status of figure (True|False) \n\
\n\
this attribute controls the dock appearance of this figure. If it is docked, the figure is integrated into the main window \n\
of itom, else it is a independent window. \n\
");
/*static*/ PyObject* PythonFigure::PyFigure_getDocked(PyFigure *self, void *closure)
{
    ito::RetVal retValue = retOk;
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if (uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if (*(self->guardedFigHandle) <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid figure handle.");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    QSharedPointer<bool> docked(new bool);

    QMetaObject::invokeMethod(uiOrga, "getDockedStatus", Q_ARG(unsigned int, *(self->guardedFigHandle)), Q_ARG(QSharedPointer<bool>, docked), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if (!locker.getSemaphore()->wait(5000))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting dock status");
        return NULL;
    }
    
    retValue = locker.getSemaphore()->returnValue;
    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    if (*docked)
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ int PythonFigure::PyFigure_setDocked(PyFigure *self, PyObject *value, void *closure)
{
    bool ok;
    bool docked = PythonQtConversion::PyObjGetBool(value,false,ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_RuntimeError, "docked attribute must be set to True or False");
        return -1;
    }

    ito::RetVal retValue = retOk;
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if (uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return -1;
    }

    if (*(self->guardedFigHandle) <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid figure handle.");
        return -1;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    QMetaObject::invokeMethod(uiOrga, "setDockedStatus", Q_ARG(unsigned int, *(self->guardedFigHandle)), Q_ARG(bool, docked), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if (!locker.getSemaphore()->wait(5000))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting dock status");
        return -1;
    }
    
    retValue = locker.getSemaphore()->returnValue;
    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return -1;
    }

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
//   static
//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigure_Close_doc,"close(handle|'all') -> static method to close any specific or all open figures (unless any figure-instance still keeps track of them)\n\
\n\
This method closes and deletes any specific figure (given by handle) or all opened figures. \n\
\n\
Parameters \n\
----------- \n\
handle : {dataIO-Instance} \n\
    any figure handle (>0) or 'all' in order to close all opened figures \n\
\n\
Notes \n\
------- \n\
If any instance of class 'figure' still keeps a reference to any figure, it is only closed and deleted if the last instance is deleted, too.");
/*static*/ PyObject* PythonFigure::PyFigure_close(PyFigure * /*self*/, PyObject *args)
{
    PyObject *arg = NULL;
    if (!PyArg_ParseTuple(args, "O", &arg))
    {
        return NULL;
    }

    bool ok;
    int handle;
    QString text;
    handle = PythonQtConversion::PyObjGetInt(arg,false,ok);
    if (!ok)
    {
        handle = 0;
        text = PythonQtConversion::PyObjGetString(arg,false,ok);
        if (!ok || text != "all")
        {
            PyErr_SetString(PyExc_RuntimeError, "argument must be either a figure handle or 'all'");
            return NULL;
        }
    }
    else if (handle <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "figure handle must be bigger than zero");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if (uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "figureClose", Q_ARG(unsigned int, handle), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if (!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while closing figures");
        return NULL;
    }
    
    retValue = locker.getSemaphore()->returnValue;
    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyMethodDef PythonFigure::PyFigure_methods[] = {
    {"show", (PyCFunction)PyFigure_show,     METH_VARARGS, pyFigureShow_doc},
    {"hide", (PyCFunction)PyFigure_hide, METH_NOARGS, pyFigureHide_doc},
    {"plot", (PyCFunction)PyFigure_plot, METH_VARARGS |METH_KEYWORDS, pyFigurePlot_doc},
    {"liveImage", (PyCFunction)PyFigure_liveImage, METH_VARARGS | METH_KEYWORDS, pyFigureLiveImage_doc},
    {"subplot", (PyCFunction)PyFigure_getSubplot, METH_VARARGS, pyFigureSubplot_doc},

    {"close", (PyCFunction)PyFigure_close, METH_VARARGS | METH_STATIC, pyFigure_Close_doc},
    //{"isVisible", (PyCFunction)PyUi_isVisible, METH_NOARGS, pyUiIsVisible_doc},
    //
    //{"getDouble",(PyCFunction)PyUi_getDouble, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetDouble_doc},
    //{"getInt",(PyCFunction)PyUi_getInt, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetInt_doc},
    //{"getItem",(PyCFunction)PyUi_getItem, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetItem_doc},
    //{"getText",(PyCFunction)PyUi_getText, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetText_doc},
    //{"msgInformation", (PyCFunction)PyUi_msgInformation, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiMsgInformation_doc},
    //{"msgQuestion", (PyCFunction)PyUi_msgQuestion, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiMsgQuestion_doc},
    //{"msgWarning", (PyCFunction)PyUi_msgWarning, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiMsgWarning_doc},
    //{"msgCritical", (PyCFunction)PyUi_msgCritical, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiMsgCritical_doc},
    //{"getExistingDirectory", (PyCFunction)PyUi_getExistingDirectory, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetExistingDirectory_doc},
    //{"getOpenFileName", (PyCFunction)PyUi_getOpenFileName, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiGetOpenFileName_doc},
    //{"getSaveFileName", (PyCFunction)PyUi_getSaveFileName, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiGetSaveFileName_doc},
    //{"createNewPluginWidget", (PyCFunction)PyUi_createNewAlgoWidget, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiCreateNewPluginWidget_doc},
    {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyMemberDef PythonFigure::PyFigure_members[] = {
    {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyModuleDef PythonFigure::PyFigureModule = {
    PyModuleDef_HEAD_INIT,
    "figure",
    "itom figure type in python",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

//----------------------------------------------------------------------------------------------------------------------------------
PyGetSetDef PythonFigure::PyFigure_getseters[] = {
    {"handle", (getter)PyFigure_getHandle, NULL, "returns handle of figure", NULL},
    {"docked", (getter)PyFigure_getDocked, (setter)PyFigure_setDocked, pyFigure_docked_doc, NULL},
    {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyTypeObject PythonFigure::PyFigureType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "itom.figure",             /* tp_name */
    sizeof(PyFigure),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PyFigure_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    (reprfunc)PyFigure_repr,         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0, /* tp_getattro */
    0,  /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    pyFigureInit_doc /*"dataObject objects"*/,           /* tp_doc */
    0,                       /* tp_traverse */
    0,                       /* tp_clear */
    0,            /* tp_richcompare */
    0,                       /* tp_weaklistoffset */
    0,                       /* tp_iter */
    0,                       /* tp_iternext */
    PyFigure_methods,             /* tp_methods */
    PyFigure_members,             /* tp_members */
    PyFigure_getseters,            /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyFigure_init,      /* tp_init */
    0,                         /* tp_alloc */
    PyFigure_new /*PyType_GenericNew*/ /*PythonStream_new,*/                 /* tp_new */
};

//----------------------------------------------------------------------------------------------------------------------------------
void PythonFigure::PyFigure_addTpDict(PyObject *tp_dict)
{
    //PyObject *value;
    //QMetaObject metaObject = QMessageBox::staticMetaObject;
    //QMetaEnum metaEnum = metaObject.enumerator(metaObject.indexOfEnumerator("StandardButtons"));
    //QString key;
    ////auto-parsing of StandardButtons-enumeration for key-value-pairs
    //for (int i = 0 ; i < metaEnum.keyCount() ; i++)
    //{
    //    value = Py_BuildValue("i", metaEnum.value(i));
    //    key = metaEnum.key(i);
    //    key.prepend("MsgBox"); //Button-Constants will be accessed by ui.MsgBoxOk, ui.MsgBoxError...
    //    PyDict_SetItemString(tp_dict, key.toLatin1().data(), value);
    //    Py_DECREF(value);
    //}

    ////add dialog types
    //value = Py_BuildValue("i", 0);
    //PyDict_SetItemString(tp_dict, "TYPEDIALOG", value);
    //Py_DECREF(value);
    //value = Py_BuildValue("i", 1);
    //PyDict_SetItemString(tp_dict, "TYPEWINDOW", value);
    //Py_DECREF(value);
    //value = Py_BuildValue("i", 2);
    //PyDict_SetItemString(tp_dict, "TYPEDOCKWIDGET", value);
    //Py_DECREF(value);

    ////add button orientation
    //value = Py_BuildValue("i", 0);
    //PyDict_SetItemString(tp_dict, "BUTTONBAR_NO", value);
    //Py_DECREF(value);
    //value = Py_BuildValue("i", 1);
    //PyDict_SetItemString(tp_dict, "BUTTONBAR_HORIZONTAL", value);
    //Py_DECREF(value);
    //value = Py_BuildValue("i", 2);
    //PyDict_SetItemString(tp_dict, "BUTTONBAR_VERTICAL", value);
    //Py_DECREF(value);
}

} //end namespace ito

