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
// ------------------------------------------------------------------------------------
//
//  PyFigure
//
// ------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
void PythonFigure::PyFigure_dealloc(PyFigure* self)
{
    // if reference of semaphore drops to zero, the static method threadSafeDeleteUi
    // of UiOrganizer is called that will finally delete the figure
    self->guardedFigHandle.clear();

    DELETE_AND_SET_NULL(self->signalMapper);

    //Py_TYPE(self)->tp_free((PyObject*)self);
    PythonUi::PyUiItemType.tp_dealloc((PyObject*)self);
}

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigureInit_doc,"figure(handle = -1, rows = 1, cols = 1, x0 = -1, y0 = -1, width = -1, height = -1) -> figure \n\
\n\
Creates a new figure window.\n\
\n\
The class :class:`figure` represents a standalone figure window, that can have \n\
various subplots. If an instance of this class is created without further parameters, \n\
a new figure is created and opened having one subplot area (currently empty) and the \n\
integer handle to this figure is returned:: \n\
    \n\
    h: int = figure() \n\
\n\
Subplots are arranged in a regular grid whose size is defined by the optional \n\
parameters ``rows`` and ``cols``. If you create a figure instance with a given handle, \n\
the instance is either a reference to an existing figure that has got this handle or if \n\
it does not exist, a new figure with the desired handle is opened and the handle is \n\
returned, too. \n\
\n\
Using the parameters ``width`` and ``height``, it is possible to control the size of the figure. \n\
If one of both parameters are not set or <= 0 (default), no size adjustment is done at all. \n\
\n\
The size and position control can afterwards done using the property ``geometry`` of \n\
the figure. \n\
\n\
Parameters \n\
---------- \n\
handle : int \n\
    integer handle of the desired figure. \n\
rows : int, optional \n\
    number of rows this figure should have (defines the size of the subplot-grid) \n\
cols : int, optional \n\
    number of columns this figure should have (defines the size of the subplot-grid) \n\
x0 : int, optional \n\
    If ``x0`` is != -1, its left position is set to this value. \n\
y0 : int, optional \n\
    If ``y0`` is != -1, its top position is set to this value. \n\
width : int, optional \n\
    If ``width`` is != -1, the width of the figure window is set to this value. \n\
height : int, optional \n\
    If ``height`` is != -1, the height of the figure window is set to this value. \n\
\n\
Returns \n\
------- \n\
figure \n\
    is the reference to the newly created figure object.");
int PythonFigure::PyFigure_init(PyFigure *self, PyObject *args, PyObject *kwds)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    const char *kwlist[] = { "handle", "rows", "cols", "x0", "y0", "width", "height", NULL };

    int handle = -1;
    unsigned int rows = 1;
    unsigned int cols = 1;
    int x0 = std::numeric_limits<int>::min();
    int y0 = x0;
    int width = -1;
    int height = -1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds,"|iIIiiii",const_cast<char**>(kwlist), &handle, &rows, &cols, &x0, &y0, &width, &height))
    {
        return -1;
    }

    QSharedPointer< QSharedPointer<unsigned int> > guardedFigHandle(new QSharedPointer<unsigned int>());
    QSharedPointer<unsigned int> objectID(new unsigned int);
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue;

    if (handle != -1)
    {
        *guardedFigHandle = QSharedPointer<unsigned int>(new unsigned int);
        **guardedFigHandle = handle;
    }


    QSharedPointer<int> rows_(new int);
    QSharedPointer<int> cols_(new int);
    *rows_ = rows;
    *cols_ = cols;

    QSize size;
    if (width >= 1 && height >= 1)
    {
        size = QSize(width, height);
    }

    QPoint offset;
    if (x0 > std::numeric_limits<int>::min() && y0 > std::numeric_limits<int>::min())
    {
        offset = QPoint(x0, y0);
    }

    QMetaObject::invokeMethod(
        uiOrga,
        "createFigure",
        Q_ARG(QSharedPointer<QSharedPointer<uint> >,guardedFigHandle),
        Q_ARG(QSharedPointer<uint>, objectID),
        Q_ARG(QSharedPointer<int>,rows_),
        Q_ARG(QSharedPointer<int>,cols_),
        Q_ARG(QPoint, offset),
        Q_ARG(QSize, size),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

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
    self->signalMapper = new PythonQtSignalMapper();

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

//-------------------------------------------------------------------------------------
PyObject* PythonFigure::PyFigure_repr(PyFigure *self)
{
    PyObject *result;
    if (self->guardedFigHandle.isNull())
    {
        result = PyUnicode_FromFormat("Figure(empty)");
    }
    else
    {
        UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

        if (uiOrga == NULL)
        {
            result = PyUnicode_FromFormat("Figure(handle: %i, unknown status)", *(self->guardedFigHandle));
        }
        else
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QSharedPointer<bool> exist(new bool);

            QMetaObject::invokeMethod(
                uiOrga,
                "handleExist",
                Q_ARG(uint, *(self->guardedFigHandle)), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
                Q_ARG(QSharedPointer<bool>, exist),
                Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
            );

            if (!locker.getSemaphore()->wait(PLUGINWAIT))
            {
                result = PyUnicode_FromFormat("Figure(handle: %i, unknown status)", *(self->guardedFigHandle));
            }
            else
            {
                if (*exist == true)
                {
                    result = PyUnicode_FromFormat("Figure(handle: %i)", *(self->guardedFigHandle));
                }
                else
                {
                    result = PyUnicode_FromFormat("Figure(handle: %i, figure is not longer available)", *(self->guardedFigHandle));
                }
            }
        }
    }

    return result;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigurePlot_doc,"plot(data, areaIndex = currentAreaIndex, className = \"\", properties = {}) -> plotItem \n\
\n\
Plots a dataObject, pointCloud or polygonMesh in the current or given area of this figure.\n\
\n\
Plots an existing :class:`dataObject`, :class:`pointCloud` or :class:`polygonMesh` in \n\
the newly created plot. The style of the plot depends on the object dimensions.\n\
\n\
If no ``className`` is given, the type of the plot is chosen depending on the type and  \n\
the size of the object. The defaults for several plot classes can be adjusted in the  \n\
property dialog of itom. \n\
\n\
You can also set a class name of your preferred plot plugin (see also property dialog of itom). \n\
If your preferred plot is not able to display the given object, a warning is returned and the \n\
default plot type is used again. For :class:`dataObject`, it is also possible to simply set \n\
``className`` to ``1D``, ``2D`` or ``2.5D`` in order to choose the default plot type depending \n\
on these aliases. For :class:`pointCloud` and :class:`polygonMesh` only the alias ``2.5D`` is valid. \n\
\n\
Every plot has several properties that can be configured in the Qt Designer (if the \n\
plot is embedded in a GUI), or by the property toolbox in the plot itself or by using \n\
the :meth:`~itom.uiItem.info` method of the corresponding :class:`itom.uiItem` instance. \n\
\n\
Use the ``properties`` argument to pass a dictionary with properties you want to set \n\
to certain values. \n\
\n\
Parameters\n\
----------\n\
data : dataObject or pointCloud or polygonMesh \n\
    Is the data object whose region of interest will be plotted.\n\
areaIndex : int, optional \n\
    Area index where the plot canvas should be created (if subplots exists). \n\
    The default ``areaIndex`` is the current subplot area, hence, ``0`` if \n\
    only one plot area exists in the figure. \n\
className : str, optional \n\
    class name of desired plot (if not indicated or if the ``className`` cannot be found, \n\
    the default plot will be used (see application settings) \n\
properties : dict, optional \n\
    Optional dictionary of properties that will be directly applied to the \n\
    plot widget.\n\
\n\
Returns \n\
------- \n\
plotHandle : plotItem \n\
    Handle of the subplot. This handle is used to control the properties of the plot, \n\
    connect signals to it or call slots of the plot.");
PyObject* PythonFigure::PyFigure_plot(PyFigure *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"data", "areaIndex", "className", "properties", NULL};
    PyObject *data = NULL;
    PyObject *propDict = NULL;
    int areaIndex = self->currentSubplotIdx;
    char* className = NULL;
    bool ok = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|isO!", const_cast<char**>(kwlist), &data, &areaIndex, &className, &PyDict_Type, &propDict))
    {
        return NULL;
    }

    ito::UiDataContainer dataCont;
    ito::RetVal retval2;

    //at first try to strictly convert to a point cloud or polygon mesh (non strict conversion not available for this)
    //if this fails, try to non-strictly convert to data object, such that numpy arrays are considered as well.
#if ITOM_POINTCLOUDLIBRARY > 0
    dataCont = QSharedPointer<ito::PCLPointCloud>(PythonQtConversion::PyObjGetPointCloudNewPtr(data, true, ok));
    if (!ok)
    {
        dataCont = QSharedPointer<ito::PCLPolygonMesh>(PythonQtConversion::PyObjGetPolygonMeshNewPtr(data, true, ok));
    }
#else
    ok = false;
#endif

    if (!ok)
    {
        dataCont = PythonQtConversion::PyObjGetSharedDataObject(data, false, ok, &retval2);
    }

    if (!ok)
    {
#if ITOM_POINTCLOUDLIBRARY > 0
        return PyErr_Format(PyExc_RuntimeError, "first argument cannot be converted to dataObject, pointCloud or polygonMesh (%s).", retval2.errorMessage());
#else
        return PyErr_Format(PyExc_RuntimeError, "first argument cannot be converted to dataObject (%s).", retval2.errorMessage());
#endif
    }

    if (areaIndex >= self->cols * self->rows || areaIndex < 0)
    {
        PyErr_Format(PyExc_RuntimeError, "areaIndex out of range [0, %i]. The figure has %i rows and %i columns.", (self->cols * self->rows - 1), self->rows, self->cols);
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = areaIndex % self->cols;
    int areaRow = (areaIndex - areaCol) / self->cols;

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    QString defaultPlotClassName;
    if (className && strlen(className) > 0)
    {
        defaultPlotClassName = className;
    }
    QSharedPointer<unsigned int> objectID(new unsigned int);
    QVariantMap properties;

    if (propDict)
    {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        QVariant valueV;
        QString keyS;

        while (PyDict_Next(propDict, &pos, &key, &value)) //key and value are borrowed
        {
            keyS = PythonQtConversion::PyObjGetString(key,true,ok);
            valueV = PythonQtConversion::PyObjToQVariant(value);
            if(valueV.isValid())
            {
                properties[keyS] = valueV;
            }
            else
            {
                PyErr_SetString(PyExc_RuntimeError, "at least one property value could not be parsed to QVariant.");
                return NULL;
            }
        }
    }
    ito::UiDataContainer xAxisCont;
    QMetaObject::invokeMethod(uiOrg, "figurePlot", Q_ARG(ito::UiDataContainer&, dataCont),Q_ARG(ito::UiDataContainer&, xAxisCont), Q_ARG(QSharedPointer<uint>, self->guardedFigHandle), Q_ARG(QSharedPointer<uint>, objectID), Q_ARG(int, areaRow), Q_ARG(int, areaCol), Q_ARG(QString, defaultPlotClassName), Q_ARG(QVariantMap, properties), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

    if (!locker.getSemaphore()->wait(PLUGINWAIT * 5))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while plotting object");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }

    //return new instance of PyUiItem
    PyObject *args2 = PyTuple_New(0); //Py_BuildValue("OO", self, name);
    PyObject *kwds2 = PyDict_New();
    PyObject *objectIdObj = PyLong_FromLong(*objectID);
    PyDict_SetItemString(kwds2, "objectID", objectIdObj);
    Py_XDECREF(objectIdObj);
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
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigurePlot1_doc, "plot1(data, xData = None, areaIndex = currentAreaIndex, className = \"\", properties = {}) -> plotItem \n\
\n\
Creates a 1D plot of a dataObject ``data`` in the current or given area of this figure.\n\
\n\
If ``xData`` is given, the plot uses this vector for the values of the ``x-axis`` of \n\
the plot.\n\
\n\
The plot type of this function is ``1D`` (see method :meth:`figure.plot`). If a \n\
``className`` is given, that does not support the given type of ``data`` (or ``xData``) \n\
a warning is returned and the default plot class for the given data is used again. \n\
\n\
Every plot has several properties that can be configured in the Qt Designer (if the plot is \n\
embedded in a GUI), or by the property toolbox in the plot itself or by using the \n\
:meth:`~uiItem.info` method of the corresponding :class:`uiItem` instance. \n\
\n\
Use the ``properties`` argument to pass a dictionary with properties you want to set. \n\
\n\
Parameters\n\
----------\n\
data : dataObject \n\
    Is the data object whose region of interest will be plotted.\n\
xData : dataObject, optional \n\
    1D plots can optionally accept this :class:`dataObject`. If given, the \n\
    values are not displayed on an equally distributed x-scale but with \n\
    the values given by ``xData``. \n\
areaIndex : int, optional \n\
    Area index where the plot canvas should be created (if subplots exists). \n\
    The default ``areaIndex`` is the current subplot area, hence, ``0`` if \n\
    only one plot area exists in the figure. \n\
className : str, optional \n\
    class name of desired plot (if not indicated or if the ``className`` cannot be found, \n\
    the default plot will be used (see application settings) \n\
properties : dict, optional \n\
    Optional dictionary of properties that will be directly applied to the \n\
    plot widget.\n\
\n\
Returns \n\
------- \n\
plotHandle : plotItem \n\
    Handle of the subplot. This handle is used to control the properties of the plot, \n\
    connect signals to it or call slots of the plot.");
PyObject* PythonFigure::PyFigure_plot1(PyFigure *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = { "data", "xData", "areaIndex","className", "properties", NULL };
    PyObject *data = NULL;
    PyObject *xData = NULL;
    PyObject *propDict = NULL;
    char* className = NULL;
    int areaIndex = self->currentSubplotIdx;
    bool ok = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OisO!", const_cast<char**>(kwlist), &data, &xData ,&areaIndex, &className ,&PyDict_Type, &propDict))
    {
        return NULL;
    }

    ito::UiDataContainer dataCont;
    ito::RetVal retval2;

    //at first try to strictly convert to a point cloud or polygon mesh (non strict conversion not available for this)
    //if this fails, try to non-strictly convert to data object, such that numpy arrays are considered as well.
#if ITOM_POINTCLOUDLIBRARY > 0
    dataCont = QSharedPointer<ito::PCLPointCloud>(PythonQtConversion::PyObjGetPointCloudNewPtr(data, true, ok));
    if (!ok)
    {
        dataCont = QSharedPointer<ito::PCLPolygonMesh>(PythonQtConversion::PyObjGetPolygonMeshNewPtr(data, true, ok));
    }
#else
    ok = false;
#endif
    ito::UiDataContainer xDataCont;
    if (!ok)
    {
        dataCont = PythonQtConversion::PyObjGetSharedDataObject(data, false, ok);
        if (ok && xData)
        {
            xDataCont = PythonQtConversion::PyObjGetSharedDataObject(xData, false, ok);

            if (!ok)
            {
                PyErr_SetString(PyExc_RuntimeError, "2nd parameter (xData) cannot be converted to dataObject.");
                return NULL;
            }
        }
    }

    if (!ok)
    {
#if ITOM_POINTCLOUDLIBRARY > 0
        return PyErr_Format(PyExc_RuntimeError, "1st parameter (data) cannot be converted to dataObject, pointCloud or polygonMesh (%s).", retval2.errorMessage());
#else
        return PyErr_Format(PyExc_RuntimeError, "1st parameter (data) cannot be converted to dataObject (%s).", retval2.errorMessage());
#endif
    }

    if (areaIndex >= self->cols * self->rows || areaIndex < 0)
    {
        PyErr_Format(PyExc_RuntimeError, "areaIndex out of range [0, %i]. The figure has %i rows and %i columns.", (self->cols * self->rows - 1), self->rows, self->cols);
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = areaIndex % self->cols;
    int areaRow = (areaIndex - areaCol) / self->cols;

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    QSharedPointer<unsigned int> objectID(new unsigned int);
    QVariantMap properties;

    if (propDict)
    {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        QVariant valueV;
        QString keyS;

        while (PyDict_Next(propDict, &pos, &key, &value)) //key and value are borrowed
        {
            keyS = PythonQtConversion::PyObjGetString(key, true, ok);
            valueV = PythonQtConversion::PyObjToQVariant(value);
            if (valueV.isValid())
            {
                properties[keyS] = valueV;
            }
            else
            {
                PyErr_SetString(PyExc_RuntimeError, "at least one property value could not be parsed to QVariant.");
                return NULL;
            }
        }
    }
    QString name(className);
    if (name.compare("2d", Qt::CaseInsensitive) == 0 || name.compare("2.5d", Qt::CaseInsensitive) == 0)
    {
        PyErr_Format(PyExc_RuntimeError, "invalid className parameter %s. Use the plot, plot2 or plot25 command instead to get a other dimensional representation", className);
        return NULL;
    }
    else if (name.length() == 0)
    {
        name = "1d";
    }
    else
    {
        name = "1d:" + name; //to be sure, that only plots from the 1d category are used (className must be compatible to 1d -> checked in FigureWidget::plot
    }

    QMetaObject::invokeMethod(uiOrg, "figurePlot", Q_ARG(ito::UiDataContainer&, dataCont), Q_ARG(ito::UiDataContainer&, xDataCont), Q_ARG(QSharedPointer<uint>, self->guardedFigHandle), Q_ARG(QSharedPointer<uint>, objectID), Q_ARG(int, areaRow), Q_ARG(int, areaCol), Q_ARG(QString, name), Q_ARG(QVariantMap, properties), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    if (!locker.getSemaphore()->wait(PLUGINWAIT * 5))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while plotting object");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }

    //return new instance of PyUiItem
    PyObject *args2 = PyTuple_New(0); //Py_BuildValue("OO", self, name);
    PyObject *kwds2 = PyDict_New();
    PyObject *objectIdObj = PyLong_FromLong(*objectID);
    PyDict_SetItemString(kwds2, "objectID", objectIdObj);
    Py_XDECREF(objectIdObj);
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
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigurePlot2_doc, "plot2(data, areaIndex = currentAreaIndex, className = \"\", properties = {}) -> plotItem \n\
\n\
Creates a 2D plot of a dataObject ``data`` in the current or given area of this figure.\n\
\n\
This method plots an existing :class:`dataObject` in the new 2D plot. \n\
The style of the plot depends on the object dimensions. The plot type of this function \n\
is ``2D`` (see method :meth:`figure.plot`).\n\
\n\
If the 2D plot is not able to display the given object, a warning is returned and the \n\
default plot type is used again.\n\
\n\
Every plot has several properties that can be configured in the Qt Designer (if the plot is \n\
embedded in a GUI), or by the property toolbox in the plot itself or by using the \n\
:meth:`~uiItem.info` method of the corresponding :class:`uiItem` instance. \n\
\n\
Use the ``properties`` argument to pass a dictionary with properties you want to set. \n\
\n\
Parameters\n\
----------\n\
data : dataObject \n\
    is the data, that should be plotted. If a ``className`` it must support dataObjects \n\
    as accepted data type. Else, the default ``className`` for 2D :class:`dataObject` \n\
    is chosen (see itom application settings for default plot types. \n\
areaIndex : int, optional \n\
    Area index where the plot canvas should be created (if subplots exists). \n\
    The default ``areaIndex`` is the current subplot area, hence, ``0`` if \n\
    only one plot area exists in the figure. \n\
className : str, optional \n\
    class name of desired plot (if not indicated or if the ``className`` cannot be found, \n\
    the default plot will be used (see application settings) \n\
properties : dict, optional \n\
    Optional dictionary of properties that will be directly applied to the \n\
    plot widget.\n\
\n\
Returns \n\
------- \n\
plotHandle : plotItem \n\
    Handle of the subplot. This handle is used to control the properties of the plot, \n\
    connect signals to it or call slots of the plot.");
PyObject* PythonFigure::PyFigure_plot2(PyFigure *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = { "data", "areaIndex", "properties", NULL };
    PyObject *data = NULL;
    PyObject *propDict = NULL;
    char* className = NULL;
    int areaIndex = self->currentSubplotIdx;
    bool ok = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|isO!", const_cast<char**>(kwlist), &data, &areaIndex, &className ,&PyDict_Type, &propDict))
    {
        return NULL;
    }

    ito::UiDataContainer dataCont;
    ito::RetVal retval2;

    //at first try to strictly convert to a point cloud or polygon mesh (non strict conversion not available for this)
    //if this fails, try to non-strictly convert to data object, such that numpy arrays are considered as well.
#if ITOM_POINTCLOUDLIBRARY > 0
    dataCont = QSharedPointer<ito::PCLPointCloud>(PythonQtConversion::PyObjGetPointCloudNewPtr(data, true, ok));
    if (!ok)
    {
        dataCont = QSharedPointer<ito::PCLPolygonMesh>(PythonQtConversion::PyObjGetPolygonMeshNewPtr(data, true, ok));
    }
#else
    ok = false;
#endif
    ito::UiDataContainer xDataCont;
    if (!ok)
    {
        dataCont = PythonQtConversion::PyObjGetSharedDataObject(data, false, ok);

    }

    if (!ok)
    {
#if ITOM_POINTCLOUDLIBRARY > 0
        return PyErr_Format(PyExc_RuntimeError, "first argument cannot be converted to dataObject, pointCloud or polygonMesh (%s).", retval2.errorMessage());
#else
        return PyErr_Format(PyExc_RuntimeError, "first argument cannot be converted to dataObject (%s).", retval2.errorMessage());
#endif
    }

    if (areaIndex >= self->cols * self->rows || areaIndex < 0)
    {
        PyErr_Format(PyExc_RuntimeError, "areaIndex out of range [0, %i]. The figure has %i rows and %i columns.", (self->cols * self->rows - 1), self->rows, self->cols);
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = areaIndex % self->cols;
    int areaRow = (areaIndex - areaCol) / self->cols;

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    QSharedPointer<unsigned int> objectID(new unsigned int);
    QVariantMap properties;

    if (propDict)
    {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        QVariant valueV;
        QString keyS;

        while (PyDict_Next(propDict, &pos, &key, &value)) //key and value are borrowed
        {
            keyS = PythonQtConversion::PyObjGetString(key, true, ok);
            valueV = PythonQtConversion::PyObjToQVariant(value);
            if (valueV.isValid())
            {
                properties[keyS] = valueV;
            }
            else
            {
                PyErr_SetString(PyExc_RuntimeError, "at least one property value could not be parsed to QVariant.");
                return NULL;
            }
        }
    }
    QString name(className);
    if (name.compare("1d", Qt::CaseInsensitive) == 0 || name.compare("2.5d", Qt::CaseInsensitive) == 0)
    {
        PyErr_Format(PyExc_RuntimeError, "invalid className parameter %s. Use the plot, plot1 or plot25 command instead to get a other dimensional representation", className);
        return NULL;
    }
    else if (name.length() == 0)
    {
        name = "2d";
    }
    else
    {
        name = "2d:" + name; //to be sure, that only plots from the 2d category are used (className must be compatible to 2d -> checked in FigureWidget::plot
    }

    QMetaObject::invokeMethod(uiOrg, "figurePlot", Q_ARG(ito::UiDataContainer&, dataCont), Q_ARG(ito::UiDataContainer&, xDataCont), Q_ARG(QSharedPointer<uint>, self->guardedFigHandle), Q_ARG(QSharedPointer<uint>, objectID), Q_ARG(int, areaRow), Q_ARG(int, areaCol), Q_ARG(QString, name), Q_ARG(QVariantMap, properties), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if (!locker.getSemaphore()->wait(PLUGINWAIT * 5))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while plotting object");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }

    //return new instance of PyUiItem
    PyObject *args2 = PyTuple_New(0); //Py_BuildValue("OO", self, name);
    PyObject *kwds2 = PyDict_New();
    PyObject *objectIdObj = PyLong_FromLong(*objectID);
    PyDict_SetItemString(kwds2, "objectID", objectIdObj);
    Py_XDECREF(objectIdObj);
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
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigurePlot25_doc, "plot25(data, areaIndex = currentAreaIndex, className = \"\", properties = {}) -> plotItem \n\
\n\
Creates a 2.5D plot of a dataObject, pointCloud or polygonMesh in the current or given area of this figure.\n\
\n\
This method plots ``data`` object in the new plot. The style of the plot depends on \n\
the object dimensions, its plot type is ``2.5D`` (see method :meth:`figure.plot`).\n\
\n\
If the 2.5D plot is not able to display the given object, a warning is returned and \n\
the default plot type is used again.\n\
\n\
Every plot has several properties that can be configured in the Qt Designer (if the plot is \n\
embedded in a GUI), or by the property toolbox in the plot itself or by using the \n\
:meth:`~uiItem.info` method of the corresponding :class:`uiItem` instance. \n\
\n\
Use the ``properties`` argument to pass a dictionary with properties you want to set. \n\
\n\
Parameters\n\
----------\n\
data : dataObject or pointCloud or polygonMesh \n\
    is the data, that should be plotted. If a ``className`` is given, only the \n\
    type of data, supported by this class, can be displayed. Else, the default \n\
    ``className`` for the kind of data is chosen (see itom application settings \n\
    for default plot types. \n\
areaIndex : int, optional \n\
    Area index where the plot canvas should be created (if subplots exists). \n\
    The default ``areaIndex`` is the current subplot area, hence, ``0`` if \n\
    only one plot area exists in the figure. \n\
className : str, optional \n\
    class name of desired plot (if not indicated or if the ``className`` cannot be found, \n\
    the default plot will be used (see application settings) \n\
properties : dict, optional \n\
    Optional dictionary of properties that will be directly applied to the \n\
    plot widget.\n\
\n\
Returns \n\
------- \n\
plotHandle : plotItem \n\
    Handle of the subplot. This handle is used to control the properties of the plot, \n\
    connect signals to it or call slots of the plot.");
PyObject* PythonFigure::PyFigure_plot25(PyFigure *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = { "data", "areaIndex", "properties", NULL };
    PyObject *data = NULL;
    PyObject *propDict = NULL;
    char* className = NULL;
    int areaIndex = self->currentSubplotIdx;
    bool ok = false;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|isO!", const_cast<char**>(kwlist), &data, &areaIndex, &className ,&PyDict_Type, &propDict))
    {
        return NULL;
    }

    ito::UiDataContainer dataCont;
    ito::RetVal retval2;

    //at first try to strictly convert to a point cloud or polygon mesh (non strict conversion not available for this)
    //if this fails, try to non-strictly convert to data object, such that numpy arrays are considered as well.
#if ITOM_POINTCLOUDLIBRARY > 0
    dataCont = QSharedPointer<ito::PCLPointCloud>(PythonQtConversion::PyObjGetPointCloudNewPtr(data, true, ok));
    if (!ok)
    {
        dataCont = QSharedPointer<ito::PCLPolygonMesh>(PythonQtConversion::PyObjGetPolygonMeshNewPtr(data, true, ok));
    }
#else
    ok = false;
#endif
    ito::UiDataContainer xDataCont;
    if (!ok)
    {
        dataCont = PythonQtConversion::PyObjGetSharedDataObject(data, false, ok);

    }

    if (!ok)
    {
#if ITOM_POINTCLOUDLIBRARY > 0
        return PyErr_Format(PyExc_RuntimeError, "first argument cannot be converted to dataObject, pointCloud or polygonMesh (%s).", retval2.errorMessage());
#else
        return PyErr_Format(PyExc_RuntimeError, "first argument cannot be converted to dataObject (%s).", retval2.errorMessage());
#endif
    }

    if (areaIndex >= self->cols * self->rows || areaIndex < 0)
    {
        PyErr_Format(PyExc_RuntimeError, "areaIndex out of range [0, %i]. The figure has %i rows and %i columns.", (self->cols * self->rows - 1), self->rows, self->cols);
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = areaIndex % self->cols;
    int areaRow = (areaIndex - areaCol) / self->cols;

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    QSharedPointer<unsigned int> objectID(new unsigned int);
    QVariantMap properties;

    if (propDict)
    {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        QVariant valueV;
        QString keyS;

        while (PyDict_Next(propDict, &pos, &key, &value)) //key and value are borrowed
        {
            keyS = PythonQtConversion::PyObjGetString(key, true, ok);
            valueV = PythonQtConversion::PyObjToQVariant(value);
            if (valueV.isValid())
            {
                properties[keyS] = valueV;
            }
            else
            {
                PyErr_SetString(PyExc_RuntimeError, "at least one property value could not be parsed to QVariant.");
                return NULL;
            }
        }
    }
    QString name(className);
    if (name.compare("1d", Qt::CaseInsensitive) == 0 || name.compare("2d", Qt::CaseInsensitive) == 0)
    {
        PyErr_Format(PyExc_RuntimeError, "invalid className parameter %s. Use the plot, plot1 or plot2 command instead to get a other dimensional representation", className);
        return NULL;
    }
    else if (name.length() == 0)
    {
        name = "2.5d";
    }
    else
    {
        name = "2.5d:" + name; //to be sure, that only plots from the 2.5d category are used (className must be compatible to 2.5d -> checked in FigureWidget::plot
    }

    QMetaObject::invokeMethod(uiOrg, "figurePlot", Q_ARG(ito::UiDataContainer&, dataCont), Q_ARG(ito::UiDataContainer&, xDataCont), Q_ARG(QSharedPointer<uint>, self->guardedFigHandle), Q_ARG(QSharedPointer<uint>, objectID), Q_ARG(int, areaRow), Q_ARG(int, areaCol), Q_ARG(QString, name), Q_ARG(QVariantMap, properties), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if (!locker.getSemaphore()->wait(PLUGINWAIT * 5))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while plotting object");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }

    //return new instance of PyUiItem
    PyObject *args2 = PyTuple_New(0); //Py_BuildValue("OO", self, name);
    PyObject *kwds2 = PyDict_New();
    PyObject *objectIdObj = PyLong_FromLong(*objectID);
    PyDict_SetItemString(kwds2, "objectID", objectIdObj);
    Py_XDECREF(objectIdObj);
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
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigureLiveImage_doc,"liveImage(cam, areaIndex = currentAreaIndex, className = \"\", properties = {}) -> plotItem \n\
\n\
Shows a camera live image in the current or given area of this figure. \n\
\n\
If no ``className`` is given, the type of the plot is chosen depending on the type \n\
and the size of the object. The defaults for several plot classes can be adjusted in \n\
the property dialog of itom. \n\
\n\
You can also set a class name of your preferred plot plugin (see also property dialog \n\
of itom). If your preferred plot is not able to display the given object, a warning is \n\
returned and the default plot type is used again. For dataObjects, it is also possible \n\
to simply set ``className`` to `1D` or `2D` in order to choose the default plot type \n\
depending on these aliases. \n\
\n\
Every plot has several properties that can be configured in the Qt Designer (if the \n\
plot is embedded in a GUI), or by the property toolbox in the plot itself or by using \n\
the :meth:`~itom.uiItem.info` method of the corresponding :class:`itom.uiItem` instance. \n\
\n\
Use the ``properties`` argument to pass a dictionary with properties you want to set to \n\
certain values. \n\
\n\
Parameters\n\
----------\n\
cam : dataIO \n\
    Camera grabber device from which images are acquired.\n\
areaIndex : int, optional \n\
    Area index where the plot canvas should be created (if subplots exists). \n\
    The default ``areaIndex`` is the current subplot area, hence, ``0`` if \n\
    only one plot area exists in the figure. \n\
className : str, optional \n\
    class name of desired plot (if not indicated or if the ``className`` cannot be found, \n\
    the default plot will be used (see application settings) \n\
properties : dict, optional \n\
    Optional dictionary of properties that will be directly applied to the \n\
    plot widget.\n\
\n\
Returns \n\
------- \n\
plotHandle : plotItem \n\
    Handle of the subplot. This handle is used to control the properties of the plot, \n\
    connect signals to it or call slots of the plot.");
/*static*/ PyObject* PythonFigure::PyFigure_liveImage(PyFigure *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"cam", "areaIndex", "className", "properties", NULL};
    PythonPlugins::PyDataIOPlugin *cam = NULL;
    int areaIndex = self->currentSubplotIdx;
    char* className = NULL;
    PyObject* propDict = NULL;
    bool ok = true;

    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwds,
        "O!|isO!",
        const_cast<char**>(kwlist),
        &PythonPlugins::PyDataIOPluginType,
        &cam,
        &areaIndex,
        &className,
        &PyDict_Type,
        &propDict))
    {
        return NULL;
    }

    if (areaIndex >= self->cols * self->rows || areaIndex < 0)
    {
        PyErr_Format(
            PyExc_RuntimeError,
            "areaIndex out of range [0, %i]. The figure has %i rows and %i columns.",
            (self->cols * self->rows - 1),
            self->rows,
            self->cols);
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = areaIndex % self->cols;
    int areaRow = (areaIndex - areaCol) / self->cols;

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    QString defaultPlotClassName;
    if (className)
    {
        defaultPlotClassName = className;
    }
    QSharedPointer<unsigned int> objectID(new unsigned int);
    QVariantMap properties;

    if (propDict)
    {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        QVariant valueV;
        QString keyS;
        while (PyDict_Next(propDict, &pos, &key, &value))
        {
            keyS = PythonQtConversion::PyObjGetString(key,true,ok);
            valueV = PythonQtConversion::PyObjToQVariant(value);
            if(valueV.isValid())
            {
                properties[keyS] = valueV;
            }
            else
            {
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "at least one property value could not be parsed to QVariant.");
                return NULL;
            }
        }
    }

    QMetaObject::invokeMethod(
        uiOrg,
        "figureLiveImage",
        Q_ARG(AddInDataIO*, cam->dataIOObj),
        Q_ARG(QSharedPointer<uint>, self->guardedFigHandle),
        Q_ARG(QSharedPointer<uint>, objectID),
        Q_ARG(int, areaRow),
        Q_ARG(int, areaCol),
        Q_ARG(QString, defaultPlotClassName),
        Q_ARG(QVariantMap, properties),
        Q_ARG(ItomSharedSemaphore*,locker.getSemaphore())
    );

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
    PyObject *args2 = PyTuple_New(0); // new ref
    PyObject *kwds2 = PyDict_New(); // new ref
    PyObject *objectIdObj = PyLong_FromLong(*objectID);
    PyDict_SetItemString(kwds2, "objectID", objectIdObj);
    Py_XDECREF(objectIdObj);
    PyDict_SetItemString(kwds2, "figure", (PyObject*)self);
    PythonPlotItem::PyPlotItem *pyPlotItem = \
        (PythonPlotItem::PyPlotItem *)PyObject_Call((PyObject *)&PythonPlotItem::PyPlotItemType, args2, kwds2);
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    if (pyPlotItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create plotItem of plot widget");
        return NULL;
    }

    return (PyObject*)pyPlotItem;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigureMatplotlib_doc,"matplotlibFigure(areaIndex = currentAreaIndex, properties = {}) -> plotItem \n\
\n\
Creates a matplotlib canvas at the given area in the figure. \n\
\n\
Creates and returns a matplotlib canvas at the given area or returns an existing one. \n\
This canvas can be used as canvas argument for :class:`matplotlib.pyplot.figure` of \n\
matplotlib and is internally used by the itom backend of matplotlib. \n\
\n\
Parameters\n\
----------\n\
areaIndex : int, optional \n\
    Area index where the plot canvas should be created (if subplots exists). \n\
    The default ``areaIndex`` is the current subplot area, hence, ``0`` if \n\
    only one plot area exists in the figure. \n\
properties : dict, optional \n\
    Optional dictionary of properties that will be directly applied to the \n\
    plot widget.\n\
\n\
Returns \n\
------- \n\
plotHandle : plotItem \n\
    Handle of the subplot. This handle is used to control the properties of the plot, connect to \n\
    its signals or call slots of the plot.");
PyObject* PythonFigure::PyFigure_matplotlib(PyFigure *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"areaIndex", "properties", NULL};
    int areaIndex = self->currentSubplotIdx;
    PyObject* propDict = NULL;
    bool ok = true;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO!", const_cast<char**>(kwlist), &areaIndex, &PyDict_Type, &propDict))
    {
        return NULL;
    }

    if (areaIndex >= self->cols * self->rows || areaIndex < 0)
    {
        PyErr_Format(
            PyExc_RuntimeError,
            "areaIndex out of range [0, %i]. The figure has %i rows and %i columns.",
            (self->cols * self->rows - 1),
            self->rows,
            self->cols);
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = areaIndex % self->cols;
    int areaRow = (areaIndex - areaCol) / self->cols;

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    QString defaultPlotClassName;

    QSharedPointer<unsigned int> objectID(new unsigned int);
    QVariantMap properties;

    if (propDict)
    {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        QVariant valueV;
        QString keyS;
        while (PyDict_Next(propDict, &pos, &key, &value))
        {
            keyS = PythonQtConversion::PyObjGetString(key,true,ok);
            valueV = PythonQtConversion::PyObjToQVariant(value);
            if(valueV.isValid())
            {
                properties[keyS] = valueV;
            }
            else
            {
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "at least one property value could not be parsed to QVariant.");
                return NULL;
            }
        }
    }

	if (self->guardedFigHandle.isNull() || *(self->guardedFigHandle) <= 0)
	{
		PyErr_SetString(PyExc_RuntimeError, "invalid figure");
		return NULL;
	}

    QMetaObject::invokeMethod(
        uiOrg,
        "figureDesignerWidget",
        Q_ARG(QSharedPointer<uint>, self->guardedFigHandle),
        Q_ARG(QSharedPointer<uint>, objectID),
        Q_ARG(int, areaRow),
        Q_ARG(int, areaCol),
        Q_ARG(QString, "MatplotlibPlot"),
        Q_ARG(QVariantMap, properties),
        Q_ARG(ItomSharedSemaphore*,locker.getSemaphore())
    );

    if (!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while creating matplotlib canvas.");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }

    //return new instance of PyUiItem
    PyObject *args2 = PyTuple_New(0); //Py_BuildValue("OO", self, name);
    PyObject *kwds2 = PyDict_New();
    PyObject *objectIdObj = PyLong_FromLong(*objectID);
    PyDict_SetItemString(kwds2, "objectID", objectIdObj);
    Py_XDECREF(objectIdObj);
    PyDict_SetItemString(kwds2, "figure", (PyObject*)self);
    PythonPlotItem::PyPlotItem *pyPlotItem = \
        (PythonPlotItem::PyPlotItem *)PyObject_Call((PyObject *)&PythonPlotItem::PyPlotItemType, args2, kwds2);
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    if (pyPlotItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create plotItem of plot widget");
        return NULL;
    }

    return (PyObject*)pyPlotItem;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigurePlotly_doc, "plotlyFigure(areaIndex = currentAreaIndex, properties = {}) -> plotItem \n\
\n\
Creates a plotly canvas at the given area in the figure. \n\
\n\
Creates and returns a plotly canvas at the given area or returns an existing one. \n\
If the itom plotly renderer is used, this renderer calls this method to send the \n\
html output to this widget. \n\
\n\
Parameters\n\
----------\n\
areaIndex : int, optional \n\
    Area index where the plot canvas should be created (if subplots exists). \n\
    The default ``areaIndex`` is the current subplot area, hence, ``0`` if \n\
    only one plot area exists in the figure. \n\
properties : dict, optional \n\
    Optional dictionary of properties that will be directly applied to the \n\
    plot widget.\n\
\n\
Returns \n\
------- \n\
plotHandle : plotItem \n\
    Handle of the subplot. This handle is used to control the properties of the plot, connect to \n\
    its signals or call slots of the plot.");
PyObject* PythonFigure::PyFigure_plotly(PyFigure *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = { "areaIndex", "properties", NULL };
    int areaIndex = self->currentSubplotIdx;
    PyObject* propDict = NULL;
    bool ok = true;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO!", const_cast<char**>(kwlist), &areaIndex, &PyDict_Type, &propDict))
    {
        return NULL;
    }

    if (areaIndex >= self->cols * self->rows || areaIndex < 0)
    {
        PyErr_Format(
            PyExc_RuntimeError,
            "areaIndex out of range [0, %i]. The figure has %i rows and %i columns.",
            (self->cols * self->rows - 1),
            self->rows,
            self->cols);
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = areaIndex % self->cols;
    int areaRow = (areaIndex - areaCol) / self->cols;

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    QString defaultPlotClassName;

    QSharedPointer<unsigned int> objectID(new unsigned int);
    QVariantMap properties;

    if (propDict)
    {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        QVariant valueV;
        QString keyS;
        while (PyDict_Next(propDict, &pos, &key, &value))
        {
            keyS = PythonQtConversion::PyObjGetString(key, true, ok);
            valueV = PythonQtConversion::PyObjToQVariant(value);
            if (valueV.isValid())
            {
                properties[keyS] = valueV;
            }
            else
            {
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "at least one property value could not be parsed to QVariant.");
                return NULL;
            }
        }
    }

    if (self->guardedFigHandle.isNull() || *(self->guardedFigHandle) <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "invalid figure");
        return NULL;
    }

    QMetaObject::invokeMethod(
        uiOrg,
        "figureDesignerWidget",
        Q_ARG(QSharedPointer<uint>, self->guardedFigHandle),
        Q_ARG(QSharedPointer<uint>, objectID),
        Q_ARG(int, areaRow),
        Q_ARG(int, areaCol),
        Q_ARG(QString, "PlotlyPlot"),
        Q_ARG(QVariantMap, properties),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    );

    if (!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while creating plotlyPlot canvas.");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }

    //return new instance of PyUiItem
    PyObject *args2 = PyTuple_New(0); //Py_BuildValue("OO", self, name);
    PyObject *kwds2 = PyDict_New();
    PyObject *objectIdObj = PyLong_FromLong(*objectID);
    PyDict_SetItemString(kwds2, "objectID", objectIdObj);
    Py_XDECREF(objectIdObj);
    PyDict_SetItemString(kwds2, "figure", (PyObject*)self);
    PythonPlotItem::PyPlotItem *pyPlotItem = \
        (PythonPlotItem::PyPlotItem *)PyObject_Call((PyObject *)&PythonPlotItem::PyPlotItemType, args2, kwds2);
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    if (pyPlotItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create plotItem of plot widget");
        return NULL;
    }

    return (PyObject*)pyPlotItem;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigureShow_doc,"show() \n\
\n\
Shows the figure if it is currently hidden.");
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

	if (self->guardedFigHandle.isNull() || *(self->guardedFigHandle) <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid figure handle.");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    QSharedPointer<int> retCodeIfModal(new int);
    *retCodeIfModal = -1;
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "showDialog", Q_ARG(uint, *(self->guardedFigHandle)) , Q_ARG(int,modalLevel), Q_ARG(QSharedPointer<int>, retCodeIfModal), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

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

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigureHide_doc, "hide() \n\
\n\
Hides the figure, but does not delete it.");
PyObject* PythonFigure::PyFigure_hide(PyFigure *self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if (uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

	if (self->guardedFigHandle.isNull() || *(self->guardedFigHandle) <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid figure handle.");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "hideDialog", Q_ARG(uint, *(self->guardedFigHandle)), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

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

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigureSubplot_doc,"subplot(index) -> plotItem \n\
\n\
Returns :class:`plotItem` of desired subplot.\n\
\n\
Parameters \n\
---------- \n\
index : int \n\
    index to desired subplot in the range ``[0, n)``, where n is the number of subplots. \n\
    The subplot at the top, left position has the index 0 and the index is \n\
    incremented row-wise. \n\
\n\
Returns \n\
------- \n\
plotItem \n\
    The plot item of the desired subplot.");
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

//-------------------------------------------------------------------------------------
//   getter / setters
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigure_handle_doc, "int : gets the handle of this figure.");
PyObject* PythonFigure::PyFigure_getHandle(PyFigure *self, void * /*closure*/)
{
    if (self->guardedFigHandle.isNull())
    {
        PyErr_SetString(PyExc_RuntimeError,"invalid figure");
        return NULL;
    }

    return Py_BuildValue("i", *(self->guardedFigHandle));
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigure_docked_doc, "bool : gets or sets the docked status of this figure. \n\
\n\
This attribute controls the dock appearance of this figure. If it is docked, the \n\
figure is integrated into the main window of itom, else it is a independent window. \n\
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

    QMetaObject::invokeMethod(uiOrga, "getDockedStatus", Q_ARG(uint, *(self->guardedFigHandle)), Q_ARG(QSharedPointer<bool>, docked), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

    if (!locker.getSemaphore()->wait(PLUGINWAIT))
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

//-------------------------------------------------------------------------------------
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

    QMetaObject::invokeMethod(uiOrga, "setDockedStatus", Q_ARG(uint, *(self->guardedFigHandle)), Q_ARG(bool, docked), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

    if (!locker.getSemaphore()->wait(PLUGINWAIT))
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

//-------------------------------------------------------------------------------------
//   static
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigure_Close_doc,"close(handle) -> None \\\n\
close(all = \"all\") -> None \n\
\n\
Closes a specific or all opened figures. \n\
\n\
This method closes and deletes either one specific figure (if ``handle`` is given \n\
and valid), or all opened figures (if the string argument ``\"all\"`` is given). \n\
All figure can only be closed, if no other figure references this figure (e.g. \n\
line cut of an image plot (2D). \n\
\n\
Parameters \n\
---------- \n\
handle : int \n\
    a valid figure handle, whose reference figure should be closed. \n\
    This figure handle is for instance obtained by the first value of the \n\
    returned tuple of :meth:`plot`, :meth:`plot1`, :meth:`plot2` among others. \n\
all : {\"all\"} \n\
    Pass the string ``\"all\"``  if all closeable opened figures should be closed. \n\
\n\
Notes \n\
----- \n\
If a :class:`figure` instance still keeps a reference to any figure, it is only closed \n\
and will be deleted after that the last referencing instance has been deleted. \n\
\n\
See Also \n\
-------- \n\
itom.close");
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

    QMetaObject::invokeMethod(uiOrga, "figureClose", Q_ARG(uint, handle), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

    if (!locker.getSemaphore()->waitAndProcessEvents(-1))
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

//-------------------------------------------------------------------------------------
PyMethodDef PythonFigure::PyFigure_methods[] = {
    {"show", (PyCFunction)PyFigure_show,     METH_VARARGS, pyFigureShow_doc},
    {"hide", (PyCFunction)PyFigure_hide, METH_NOARGS, pyFigureHide_doc},
    {"plot", (PyCFunction)PyFigure_plot, METH_VARARGS |METH_KEYWORDS, pyFigurePlot_doc},
    {"plot1",(PyCFunction)PyFigure_plot1, METH_VARARGS | METH_KEYWORDS, pyFigurePlot1_doc},
    {"plot2",(PyCFunction)PyFigure_plot2, METH_VARARGS | METH_KEYWORDS, pyFigurePlot2_doc},
    {"plot25",(PyCFunction)PyFigure_plot25, METH_VARARGS | METH_KEYWORDS, pyFigurePlot25_doc},
    {"liveImage", (PyCFunction)PyFigure_liveImage, METH_VARARGS | METH_KEYWORDS, pyFigureLiveImage_doc},
    {"matplotlibFigure", (PyCFunction)PyFigure_matplotlib, METH_VARARGS | METH_KEYWORDS, pyFigureMatplotlib_doc},
    {"plotlyFigure", (PyCFunction)PyFigure_plotly, METH_VARARGS | METH_KEYWORDS, pyFigurePlotly_doc},
    {"subplot", (PyCFunction)PyFigure_getSubplot, METH_VARARGS, pyFigureSubplot_doc},
    {"close", (PyCFunction)PyFigure_close, METH_VARARGS | METH_STATIC, pyFigure_Close_doc},
    {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyMemberDef PythonFigure::PyFigure_members[] = {
    {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyModuleDef PythonFigure::PyFigureModule = {
    PyModuleDef_HEAD_INIT,
    "figure",
    "itom figure type in python",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

//-------------------------------------------------------------------------------------
PyGetSetDef PythonFigure::PyFigure_getseters[] = {
    {"handle", (getter)PyFigure_getHandle, NULL, pyFigure_handle_doc, NULL},
    {"docked", (getter)PyFigure_getDocked, (setter)PyFigure_setDocked, pyFigure_docked_doc, NULL},
    {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
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
