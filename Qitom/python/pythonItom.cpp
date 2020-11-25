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

#include "pythonItom.h"
#include "pythonNumeric.h"
#include "pythonPlugins.h"
#include "pythonQtConversion.h"
#include "pythontParamConversion.h"
#include "pythonCommon.h"
#include "pythonProxy.h"
#include "pythonFigure.h"
#include "pythonPlotItem.h"
#include "pythonRgba.h"
#include "pythonProgressObserver.h"

#include "pythonEngine.h"

#include "../helper/versionHelper.h"
#include "../../common/sharedFunctionsQt.h"

#include "../AppManagement.h"
#include "../organizer/uiOrganizer.h"
#include "../../AddInManager/addInManager.h"
#include "../organizer/userOrganizer.h"
#include "../organizer/paletteOrganizer.h"
#include "../organizer/designerWidgetOrganizer.h"
#include "../organizer/processOrganizer.h"

#include <qdir.h>
#include <qcoreapplication.h>
#include <qdesktopwidget.h>
#include <qstringlist.h>
#include <qresource.h>

#include <QtCore/qpluginloader.h>

#include "opencv2/core/core_c.h"

QHash<size_t, QString> ito::PythonItom::m_gcTrackerList;

namespace ito
{


//-------------------------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //
//                                                                                                              //
//          STATIC METHODS - - - STATIC METHODS - - - STATIC METHODS                                            //
//                                                                                                              //
//  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyOpenEmptyScriptEditor_doc,"scriptEditor()\n\
\n\
Opens new, empty script editor window (undocked).\n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if the current user has no permission to open a new script.");
PyObject* PythonItom::PyOpenEmptyScriptEditor(PyObject * /*pSelf*/, PyObject * /*pArgs*/)
{
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    QObject *sew = AppManagement::getScriptEditorOrganizer();
    if (sew == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Gui not available");
        return NULL;
    }

    QMetaObject::invokeMethod(sew, "openNewScriptWindow", Q_ARG(bool,false), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

    if (locker.getSemaphore()->wait(PLUGINWAIT))
    {
        if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
        {
            return NULL;
        }
        Py_RETURN_NONE;
    }
    else
    {
        if (PyErr_CheckSignals() == -1) //!< check if key interrupt occured
        {
            return PyErr_Occurred();
        }
        PyErr_SetString(PyExc_RuntimeError, "Timeout while opening empty script.");
        return NULL;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyNewScript_doc, "newScript()\n\
\n\
Opens an empty, new script in the current script window.\n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if the current user has no permission to open a new script."); 
PyObject* PythonItom::PyNewScript(PyObject * /*pSelf*/, PyObject * /*pArgs*/)
{
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    QObject *sew = AppManagement::getScriptEditorOrganizer();
    if (sew == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Gui not available");
        return NULL;
    }

    QMetaObject::invokeMethod(sew,"newScript", Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

    if (locker.getSemaphore()->wait(PLUGINWAIT))
    {
        if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
        {
            return NULL;
        }
        Py_RETURN_NONE;
    }
    else
    {
        if (PyErr_CheckSignals() == -1) //!< check if key interrupt occured
        {
            return PyErr_Occurred();
        }
        PyErr_SetString(PyExc_RuntimeError, "Timeout while creating new script");
        return NULL;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyOpenScript_doc,"openScript(filename) \n\
\n\
Open the given script in current script window.\n\
\n\
Open the python script indicated by *filename* in a new tab in the current, \n\
latest opened editor window. Filename can be either a string with a relative \n\
or absolute filename to the script to open or any object with a ``__file__`` \n\
attribute. This attribute is then read and used as path. \n\
\n\
The relative filename is relative with respect to the current directory. \n\
\n\
Parameters \n\
----------- \n\
filename : str or obj \n\
    Relative or absolute filename to a python script that is then opened \n\
    (in the current editor window). Alternatively an object with a \n\
    ``__file__`` attribute is allowed. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if the current user has no permission to open a script.");
PyObject* PythonItom::PyOpenScript(PyObject * /*pSelf*/, PyObject *pArgs)
{
    const char* filename;
    QByteArray filename2;
    if (PyArg_ParseTuple(pArgs, "s", &filename) == false)
    {
        PyErr_Clear();

        //check if argument is a PyObject with a __file__ argument
        PyObject *obj = NULL;
        if (!PyArg_ParseTuple(pArgs, "O", &obj))
        {
            return NULL;
        }
        else if (PyObject_HasAttrString(obj, "__file__"))
        {
            PyObject *__file__ = PyObject_GetAttrString(obj, "__file__"); //new reference
            bool ok;
            QString f = PythonQtConversion::PyObjGetString(__file__,true,ok);
            Py_DECREF(__file__);
            __file__ = NULL;

            if (ok)
            {
                filename2 = f.toLatin1();
                filename = filename2.data(); //be carefull, filename is borrowed from filename2
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "__file__ attribute of given argument could not be parsed as string.");
                return NULL;
            }
        }
        else
        { 
            PyErr_SetString(PyExc_ValueError, "Argument is no filename string and no other object that has a __file__ attribute.");
            return NULL;
        }
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    QObject *sew = AppManagement::getScriptEditorOrganizer();
    if (sew == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Gui not available");
        return NULL;
    }

    QMetaObject::invokeMethod(sew, "openScript", Q_ARG(QString,QString(filename)), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

    if (locker.getSemaphore()->wait(60000)) //longer time, since msgbox may appear
    {
        if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
        {
            return NULL;
        }
        Py_RETURN_NONE;
    }
    else
    {
        if (PyErr_CheckSignals() == -1) //!< check if key interrupt occurred
        {
            return PyErr_Occurred();
        }
        PyErr_SetString(PyExc_RuntimeError, "Timeout while opening script");
        return NULL;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyShowHelpViewer_doc, "showHelpViewer(collectionFile = \"\") \n\
\n\
Opens the itom help viewer and displays the itom user documentation or another desired documentation. \n\
\n\
The user documentation is shown in the help viewer window. If ``collectionFile`` \n\
is given, this user-defined collection file is displayed in this help viewer.\n\
\n\
Parameters \n\
----------- \n\
collectionFile : str, optional \n\
	If given, the indicated Qt collection file (.qch) will be loaded in the help viewer.\n\
    Per default, the user documentation is loaded (pass an empty string or nothing).");
PyObject* PythonItom::PyShowHelpViewer(PyObject *pSelf, PyObject *pArgs)
{
	const char* collectionFile = NULL;
	if (!PyArg_ParseTuple(pArgs, "|s", &collectionFile))
	{
		return NULL;
	}

	QObject *mainWindow = AppManagement::getMainWindow();
	if (mainWindow)
	{
		QString collection = (collectionFile ? QLatin1String(collectionFile) : QLatin1String(""));
		QMetaObject::invokeMethod(mainWindow, "showAssistant", Q_ARG(QString, collection));
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Main window is not available");
		return NULL;
	}

	Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyClearCommandLine_doc, "clc() \n\
\n\
Clears the itom command line (if available).");
PyObject* PythonItom::PyClearCommandLine(PyObject *pSelf)
{
    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine
    if (pyEngine)
    {
        emit pyEngine->clearCommandLine();
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotImage_doc,"plot(data, className = \"\", properties = {}) -> Tuple[int, plotItem] \n\
\n\
Plots a dataObject, pointCloud or polygonMesh in a new figure window \n\
\n\
Plots an existing :class:`dataObject`, :class:`pointCloud` or :class:`polygonMesh` in a \n\
dockable, not blocking window. The style of the plot depends on the object dimensions.\n\
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
Every plot has several properties that can be configured in the Qt Designer (if the plot is \n\
embedded in a GUI), or by the property toolbox in the plot itself or by using the info() method \n\
of the corresponding :class:`~itom.uiItem` instance. \n\
\n\
Use the ``properties`` argument to pass a :obj:`dict` with properties you want to set. \n\
\n\
Parameters \n\
----------- \n\
data : dataObject or pointCloud or polygonMesh \n\
    Is the data object, point cloud or polygonal mesh, that will be plotted.\n\
className : str, optional \n\
    class name of desired plot (if not indicated or if the className can not be found, the default \n\
    plot will be used (see application settings)). Depending on the object, you can also set ``className`` \n\
    to ``1D``, ``2D`` or ``2.5D`` for displaying the object in the default plot of \n\
	the indicated categories. If nothing is given, the plot category is guessed from ``data``.\n\
properties : dict, optional \n\
    optional dictionary of properties that will be directly applied to the plot widget. \n\
\n\
Returns \n\
-------- \n\
index : int \n\
    This index is the figure index of the plot figure that is opened by this command. Use \n\
    ``figure(index)`` to get a reference to the figure window of this plot. The plot can \n\
    be closed by ``close(index)``. \n\
plotHandle: plotItem \n\
    Handle of the plot. This handle is used to control the properties of the plot, connect to \n\
    its signals or call slots of the plot. \n\
\n\
See Also \n\
--------- \n\
liveImage, plotItem, plot1, plot2, plot25");
PyObject* PythonItom::PyPlotImage(PyObject * /*pSelf*/, PyObject *pArgs, PyObject *pKwds)
{
    const char *kwlist[] = {"data", "className", "properties", NULL};
    PyObject *data = NULL;
    PyObject *propDict = NULL;
//    int areaIndex = 0;
    char* className = NULL;
    bool ok = false;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "O|sO!", const_cast<char**>(kwlist), &data, &className, &PyDict_Type, &propDict))
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
        dataCont = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(data, false, ok, &retval2));
    }

    if (!ok)
    {
#if ITOM_POINTCLOUDLIBRARY > 0
        return PyErr_Format(PyExc_RuntimeError, "1st parameter (data) cannot be converted to dataObject, pointCloud or polygonMesh (%s).", retval2.errorMessage());
#else
        return PyErr_Format(PyExc_RuntimeError, "1st parameter (data) cannot be converted to dataObject (%s).", retval2.errorMessage());
#endif
    }

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
                PyErr_SetString(PyExc_RuntimeError, "At least one property value could not be parsed to QVariant.");
                return NULL;
            }
        }
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = 0;
    int areaRow = 0;
    ito::UiDataContainer xDataCont;
    QSharedPointer<unsigned int> figHandle(new unsigned int);
    *figHandle = 0; //new figure will be requested

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    QString defaultPlotClassName;
    if (className) defaultPlotClassName = className;

    QSharedPointer<unsigned int> objectID(new unsigned int);

    QMetaObject::invokeMethod(uiOrg, "figurePlot", Q_ARG(ito::UiDataContainer&, dataCont), Q_ARG(ito::UiDataContainer&, xDataCont), Q_ARG(QSharedPointer<uint>, figHandle), Q_ARG(QSharedPointer<uint>, objectID), Q_ARG(int, areaRow), Q_ARG(int, areaCol), Q_ARG(QString, defaultPlotClassName), Q_ARG(QVariantMap, properties), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
    if (!locker.getSemaphore()->wait(PLUGINWAIT * 5))
    {
        PyErr_SetString(PyExc_RuntimeError, "Timeout while plotting object");
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
    PythonPlotItem::PyPlotItem *pyPlotItem = (PythonPlotItem::PyPlotItem *)PyObject_Call((PyObject *)&PythonPlotItem::PyPlotItemType, args2, kwds2);
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    if (pyPlotItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create plotItem of plot widget");
        return NULL;
    }

    //try to get figure handle of plot, to set the baseItem of the plot to the figure.
    //this is important such that one can connect to signals of the plot (e.g. userInteractionDone) since signals can only be handled if the last item
    //in the baseItem-chain is of type itom.ui or itom.figure
    args2 = PyTuple_New(0);
    kwds2 = PyDict_New();
    PyDict_SetItemString(kwds2, "handle", PyLong_FromLong(*figHandle));
    PythonFigure::PyFigure *pyFigure = (PythonFigure::PyFigure *)PyObject_Call((PyObject *)&PythonFigure::PyFigureType, args2, kwds2);
    Py_XDECREF(pyPlotItem->uiItem.baseItem);
    pyPlotItem->uiItem.baseItem = (PyObject*)pyFigure;
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    PyObject *res = Py_BuildValue("iO", *figHandle, (PyObject*)pyPlotItem); //returns handle
    Py_XDECREF(pyPlotItem);
    return res;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlot1d_doc, "plot1(data, xData = None, className = \"\", properties = {}) \n\
\n\
Plots a dataObject as an 1d plot in a new figure window. \n\
\n\
This method plots an existing :class:`dataObject` ``data`` in a dockable, not blocking \n\
window. \n\
\n\
If ``xData`` is given, the plot uses this vector for the values of the x axis of the plot.\n\
\n\
The plot type of this function is ``1D`` (see method :meth:`plot`).\n\
\n\
Every plot has several properties that can be configured in the Qt Designer (if the plot is \n\
embedded in a GUI), or by the property toolbox in the plot itself or by using the \n\
:meth:`~uiItem.info` method of the corresponding :class:`uiItem` instance. \n\
\n\
Use the ``properties`` argument to pass a dictionary with properties you want to set. \n\
\n\
Parameters \n\
----------- \n\
data : dataObject \n\
    Is the :class:`dataObject` whose region of interest will be plotted.\n\
xData : dataObject, optional \n\
    Is the :class:`dataObject` whose values are used for the axis.\n\
className : str, optional \n\
    class name of the desired 1D plot (if not indicated, the default 1D plot will be used, \n\
    see application settings) \n\
properties : dict, optional \n\
    optional dictionary of properties that will be directly applied to the plot widget. \n\
\n\
Returns \n\
-------- \n\
index : int \n\
    This index is the figure index of the plot figure that is opened by this command. \n\
    Use ``figure(index)`` to get a reference to the figure window of this plot. The \n\
    plot can be closed by ``close(index)``. \n\
plotHandle: plotItem \n\
    Handle of the plot. This handle is used to control the properties of the plot, \n\
    connect signals to it or call slots of the plot. \n\
\n\
See Also \n\
---------- \n\
liveImage, plotItem, plot, plot2, plot25");

//-------------------------------------------------------------------------------------
PyObject* PythonItom::PyPlot1d(PyObject * /*pSelf*/, PyObject *pArgs, PyObject *pKwds)
{
    const char *kwlist[] = { "data", "xData", "className","properties", NULL };
    PyObject *data = NULL;
    PyObject *propDict = NULL;
    char* className = NULL;
    //    int areaIndex = 0;
    PyObject *xData = NULL;
    bool ok = false;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "O|OsO!", const_cast<char**>(kwlist), &data, &xData, &className,&PyDict_Type, &propDict))
    {
        return NULL;
    }

    ito::UiDataContainer dataCont;
    ito::UiDataContainer xDataCont; //= QSharedPointer<ito::DataObject>(); //this is a null pointer
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
        dataCont = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(data, false, ok));
        if (ok && xData)
        {
            xDataCont = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(xData, false, ok));

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
        PyErr_SetString(PyExc_RuntimeError, "1st parameter (data) cannot be converted to dataObject, pointCloud or polygonMesh.");
#else
        PyErr_SetString(PyExc_RuntimeError, "1st parameter (data) cannot be converted to dataObject.");
#endif
        return NULL;
    }

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
                PyErr_SetString(PyExc_RuntimeError, "At least one property value could not be parsed to QVariant.");
                return NULL;
            }
        }
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = 0;
    int areaRow = 0;
    QSharedPointer<unsigned int> figHandle(new unsigned int);
    *figHandle = 0; //new figure will be requested

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    QSharedPointer<unsigned int> objectID(new unsigned int);

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

    QMetaObject::invokeMethod(uiOrg, "figurePlot", Q_ARG(ito::UiDataContainer&, dataCont), Q_ARG(ito::UiDataContainer&, xDataCont), Q_ARG(QSharedPointer<uint>, figHandle), Q_ARG(QSharedPointer<uint>, objectID), Q_ARG(int, areaRow), Q_ARG(int, areaCol), Q_ARG(QString, name), Q_ARG(QVariantMap, properties), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    if (!locker.getSemaphore()->wait(PLUGINWAIT * 5))
    {
        PyErr_SetString(PyExc_RuntimeError, "Timeout while plotting object");
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
    PythonPlotItem::PyPlotItem *pyPlotItem = (PythonPlotItem::PyPlotItem *)PyObject_Call((PyObject *)&PythonPlotItem::PyPlotItemType, args2, kwds2);
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    if (pyPlotItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create plotItem of plot widget");
        return NULL;
    }

    //try to get figure handle of plot, to set the baseItem of the plot to the figure.
    //this is important such that one can connect to signals of the plot (e.g. userInteractionDone) since signals can only be handled if the last item
    //in the baseItem-chain is of type itom.ui or itom.figure
    args2 = PyTuple_New(0);
    kwds2 = PyDict_New();
    PyDict_SetItemString(kwds2, "handle", PyLong_FromLong(*figHandle));
    PythonFigure::PyFigure *pyFigure = (PythonFigure::PyFigure *)PyObject_Call((PyObject *)&PythonFigure::PyFigureType, args2, kwds2);
    Py_XDECREF(pyPlotItem->uiItem.baseItem);
    pyPlotItem->uiItem.baseItem = (PyObject*)pyFigure;
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    PyObject *res = Py_BuildValue("iO", *figHandle, (PyObject*)pyPlotItem); //returns handle
    Py_XDECREF(pyPlotItem);
    return res;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlot2d_doc, "plot2(data, properties = {}) \n\
\n\
Plots a dataObject in a new figure window.\n\
\n\
This method plots an existing :class:`dataObject` in a dockable, not blocking window. \n\
The style of the plot depends on the object dimensions. The plot type of this function \n\
is ``2D``.\n\
\n\
Every plot has several properties that can be configured in the Qt Designer (if the \n\
plot is embedded in a GUI), or by the property toolbox in the plot itself or by using \n\
the :meth:`~itom.uiItem.info` method of the corresponding :class:`itom.uiItem` instance. \n\
\n\
Use the ``properties`` argument to pass a dictionary with properties you want to set \n\
to certain values. \n\
\n\
Parameters \n\
----------- \n\
data : dataObject \n\
    Is the :class:`dataObject` whose region of interest will be plotted.\n\
className : str, optional \n\
    class name of the desired `2D` plot (if not indicated default plot will be used, \n\
    see application settings) \n\
properties : dict, optional \n\
    optional dictionary of properties that will be directly applied to the plot widget. \n\
\n\
Returns \n\
-------- \n\
index : int \n\
    This index is the figure index of the plot figure that is opened by this command. \n\
    Use ``figure(index)`` to get a reference to the figure window of this plot. The \n\
    plot can be closed by ``close(index)``. \n\
plotHandle: plotItem \n\
    Handle of the plot. This handle is used to control the properties of the plot, \n\
    connect signals to it or call slots of the plot. \n\
\n\
See Also \n\
---------- \n\
liveImage, plotItem, plot, plot1, plot25");

//-------------------------------------------------------------------------------------
PyObject* PythonItom::PyPlot2d(PyObject * /*pSelf*/, PyObject *pArgs, PyObject *pKwds)
{
    const char *kwlist[] = { "data", "className", "properties", NULL };
    PyObject *data = NULL;
    PyObject *propDict = NULL;
    char* className = NULL;
    //    int areaIndex = 0;
    bool ok = false;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "O|sO!", const_cast<char**>(kwlist), &data, &className, &PyDict_Type, &propDict))
    {
        return NULL;
    }

    ito::UiDataContainer dataCont;
    ito::UiDataContainer xDataCont; //= QSharedPointer<ito::DataObject>(); //this is a null pointer
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
        dataCont = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(data, false, ok));
    }

    if (!ok)
    {
#if ITOM_POINTCLOUDLIBRARY > 0
        PyErr_SetString(PyExc_RuntimeError, "1st parameter (data) cannot be converted to dataObject, pointCloud or polygonMesh.");
#else
        PyErr_SetString(PyExc_RuntimeError, "1st parameter (data) cannot be converted to dataObject.");
#endif
        return NULL;
    }

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
                PyErr_SetString(PyExc_RuntimeError, "At least one property value could not be parsed to QVariant.");
                return NULL;
            }
        }
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = 0;
    int areaRow = 0;
    QSharedPointer<unsigned int> figHandle(new unsigned int);
    *figHandle = 0; //new figure will be requested

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();


    QSharedPointer<unsigned int> objectID(new unsigned int);
    QString name(className);
    if (name.compare("1d", Qt::CaseInsensitive) == 0 || name.compare("2.5d", Qt::CaseInsensitive) == 0)
    {
        PyErr_Format(PyExc_RuntimeError, "invalid className parameter %s. Use the plot, plot1 or plot25 command instead to get another dimensional representation", className);
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

    QMetaObject::invokeMethod(uiOrg, "figurePlot", Q_ARG(ito::UiDataContainer&, dataCont), Q_ARG(ito::UiDataContainer&, xDataCont), Q_ARG(QSharedPointer<uint>, figHandle), Q_ARG(QSharedPointer<uint>, objectID), Q_ARG(int, areaRow), Q_ARG(int, areaCol), Q_ARG(QString, name), Q_ARG(QVariantMap, properties), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    if (!locker.getSemaphore()->wait(PLUGINWAIT * 5))
    {
        PyErr_SetString(PyExc_RuntimeError, "Timeout while plotting object");
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
    PythonPlotItem::PyPlotItem *pyPlotItem = (PythonPlotItem::PyPlotItem *)PyObject_Call((PyObject *)&PythonPlotItem::PyPlotItemType, args2, kwds2);
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    if (pyPlotItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create plotItem of plot widget");
        return NULL;
    }

    //try to get figure handle of plot, to set the baseItem of the plot to the figure.
    //this is important such that one can connect to signals of the plot (e.g. userInteractionDone) since signals can only be handled if the last item
    //in the baseItem-chain is of type itom.ui or itom.figure
    args2 = PyTuple_New(0);
    kwds2 = PyDict_New();
    PyDict_SetItemString(kwds2, "handle", PyLong_FromLong(*figHandle));
    PythonFigure::PyFigure *pyFigure = (PythonFigure::PyFigure *)PyObject_Call((PyObject *)&PythonFigure::PyFigureType, args2, kwds2);
    Py_XDECREF(pyPlotItem->uiItem.baseItem);
    pyPlotItem->uiItem.baseItem = (PyObject*)pyFigure;
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    PyObject *res = Py_BuildValue("iO", *figHandle, (PyObject*)pyPlotItem); //returns handle
    Py_XDECREF(pyPlotItem);
    return res;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlot25d_doc, "plot25(data, className = \"\", properties = {}) \n\
\n\
Plots a dataObject, pointCloud or polygonMesh in a new figure window. \n\
\n\
This method plots the ``data`` object in a dockable, not blocking window. \n\
The style of the plot depends on the object dimensions, its plot type is ``2.5D``.\n\
\n\
Every plot has several properties that can be configured in the Qt Designer (if the \n\
plot is embedded in a GUI), or by the property toolbox in the plot itself or by using \n\
the :meth:`~itom.uiItem.info` method of the corresponding :class:`itom.uiItem` instance. \n\
\n\
Use the ``properties`` argument to pass a dictionary with properties you want to set to  \n\
desired values. \n\
\n\
Parameters \n\
----------- \n\
data : dataObject or pointCloud or polygonMesh \n\
    is the object, that is plotted.\n\
className : str, optional \n\
    class name of the desired `2.5D` plot (if not indicated default plot will be used, \n\
    see application settings) \n\
properties : dict, optional \n\
    optional dictionary of properties that will be directly applied to the plot widget. \n\
\n\
Returns \n\
-------- \n\
index : int \n\
    This index is the figure index of the plot figure that is opened by this command. \n\
    Use ``figure(index)`` to get a reference to the figure window of this plot. The \n\
    plot can be closed by ``close(index)``. \n\
plotHandle: plotItem \n\
    Handle of the plot. This handle is used to control the properties of the plot, \n\
    connect signals to it or call slots of the plot. \n\
\n\
See Also \n\
---------- \n\
liveImage, plotItem, plot, plot1, plot2");

//-------------------------------------------------------------------------------------
PyObject* PythonItom::PyPlot25d(PyObject * /*pSelf*/, PyObject *pArgs, PyObject *pKwds)
{
    const char *kwlist[] = { "data", "className","properties", NULL };
    PyObject *data = NULL;
    PyObject *propDict = NULL;
    char* className = NULL;
    //    int areaIndex = 0;
    bool ok = false;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "O|sO!", const_cast<char**>(kwlist), &data, &className, &PyDict_Type, &propDict))
    {
        return NULL;
    }

    ito::UiDataContainer dataCont;
    ito::UiDataContainer xDataCont; //= QSharedPointer<ito::DataObject>(); //this is a null pointer
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
        dataCont = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(data, false, ok));
    }

    if (!ok)
    {
#if ITOM_POINTCLOUDLIBRARY > 0
        PyErr_SetString(PyExc_RuntimeError, "1st parameter (data) cannot be converted to dataObject, pointCloud or polygonMesh.");
#else
        PyErr_SetString(PyExc_RuntimeError, "1st parameter (data) cannot be converted to dataObject.");
#endif
        return NULL;
    }

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
                PyErr_SetString(PyExc_RuntimeError, "At least one property value could not be parsed to QVariant.");
                return NULL;
            }
        }
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    int areaCol = 0;
    int areaRow = 0;
    QSharedPointer<unsigned int> figHandle(new unsigned int);
    *figHandle = 0; //new figure will be requested

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();


    QSharedPointer<unsigned int> objectID(new unsigned int);

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

    QMetaObject::invokeMethod(uiOrg, "figurePlot", Q_ARG(ito::UiDataContainer&, dataCont), Q_ARG(ito::UiDataContainer&, xDataCont), Q_ARG(QSharedPointer<uint>, figHandle), Q_ARG(QSharedPointer<uint>, objectID), Q_ARG(int, areaRow), Q_ARG(int, areaCol), Q_ARG(QString, name), Q_ARG(QVariantMap, properties), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    if (!locker.getSemaphore()->wait(PLUGINWAIT * 5))
    {
        PyErr_SetString(PyExc_RuntimeError, "Timeout while plotting object");
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
    PythonPlotItem::PyPlotItem *pyPlotItem = (PythonPlotItem::PyPlotItem *)PyObject_Call((PyObject *)&PythonPlotItem::PyPlotItemType, args2, kwds2);
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    if (pyPlotItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create plotItem of plot widget");
        return NULL;
    }

    //try to get figure handle of plot, to set the baseItem of the plot to the figure.
    //this is important such that one can connect to signals of the plot (e.g. userInteractionDone) since signals can only be handled if the last item
    //in the baseItem-chain is of type itom.ui or itom.figure
    args2 = PyTuple_New(0);
    kwds2 = PyDict_New();
    PyDict_SetItemString(kwds2, "handle", PyLong_FromLong(*figHandle));
    PythonFigure::PyFigure *pyFigure = (PythonFigure::PyFigure *)PyObject_Call((PyObject *)&PythonFigure::PyFigureType, args2, kwds2);
    Py_XDECREF(pyPlotItem->uiItem.baseItem);
    pyPlotItem->uiItem.baseItem = (PyObject*)pyFigure;
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    PyObject *res = Py_BuildValue("iO", *figHandle, (PyObject*)pyPlotItem); //returns handle
    Py_XDECREF(pyPlotItem);
    return res;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyLiveImage_doc,"liveImage(cam, className = \"\", properties = {}) \n\
\n\
Shows a camera live image in a new figure window. \n\
\n\
This method creates a plot-image (2D) and automatically grabs images into this window.\n\
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
Parameters \n\
----------- \n\
cam : dataIO \n\
    Camera grabber device from which images are acquired.\n\
className : str, optional \n\
    class name of desired plot (if not indicated or if the ``className`` can not be found, \n\
    the default plot will be used (see application settings) \n\
properties : dict, optional \n\
    optional dictionary of properties that will be directly applied to the plot widget. \n\
\n\
Returns \n\
-------- \n\
index : int \n\
    This index is the figure index of the plot figure that is opened by this command. \n\
    Use ``figure(index)`` to get a reference to the figure window of this plot. The \n\
    plot can be closed by ``close(index)``. \n\
plotHandle: plotItem \n\
    Handle of the plot. This handle is used to control the properties of the plot, \n\
    connect signals to it or call slots of the plot. \n\
\n\
See Also \n\
--------- \n\
plot, plotItem, plot1, plot2, plot25");
PyObject* PythonItom::PyLiveImage(PyObject * /*pSelf*/, PyObject *pArgs, PyObject *pKwds)
{
    const char *kwlist[] = {"cam", "className", "properties", NULL};
    PythonPlugins::PyDataIOPlugin *cam = NULL;
    PyObject *propDict = NULL;
    int areaIndex = 0;
    char* className = NULL;
    bool ok = true;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "O!|sO!", const_cast<char**>(kwlist), &PythonPlugins::PyDataIOPluginType, &cam, &className, &PyDict_Type, &propDict))
    {
        return NULL;
    }

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
                PyErr_SetString(PyExc_RuntimeError, "At least one property value could not be parsed to QVariant.");
                return NULL;
            }
        }
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

    QMetaObject::invokeMethod(uiOrg, "figureLiveImage", Q_ARG(AddInDataIO*, cam->dataIOObj), Q_ARG(QSharedPointer<uint>, figHandle), Q_ARG(QSharedPointer<uint>, objectID), Q_ARG(int, areaRow), Q_ARG(int, areaCol), Q_ARG(QString, defaultPlotClassName), Q_ARG(QVariantMap, properties), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
    if (!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "Timeout while showing live image of camera");
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
    PythonPlotItem::PyPlotItem *pyPlotItem = (PythonPlotItem::PyPlotItem *)PyObject_Call((PyObject *)&PythonPlotItem::PyPlotItemType, args2, kwds2);
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    if (pyPlotItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create plotItem of plot widget");
        return NULL;
    }

    //try to get figure handle of plot, to set the baseItem of the plot to the figure.
    //this is important such that one can connect to signals of the plot (e.g. userInteractionDone) since signals can only be handled if the last item
    //in the baseItem-chain is of type itom.ui or itom.figure
    args2 = PyTuple_New(0);
    kwds2 = PyDict_New();
    PyDict_SetItemString(kwds2, "handle", PyLong_FromLong(*figHandle));
    PythonFigure::PyFigure *pyFigure = (PythonFigure::PyFigure *)PyObject_Call((PyObject *)&PythonFigure::PyFigureType, args2, kwds2);
    Py_XDECREF(pyPlotItem->uiItem.baseItem);
    pyPlotItem->uiItem.baseItem = (PyObject*)pyFigure;
    Py_DECREF(args2);
    Py_DECREF(kwds2);

    PyObject *res = Py_BuildValue("iO", *figHandle, (PyObject*)pyPlotItem); //returns handle
    Py_XDECREF(pyPlotItem);
    return res;
}

//-------------------------------------------------------------------------------------
PyObject* PyWidgetOrFilterHelp(bool getWidgetHelp, PyObject* pArgs, PyObject *pKwds)
{
    const char *kwlistFilter[] = {"filterName", "dictionary", "furtherInfos", NULL};
    const char *kwlistWidget[] = {"widgetName", "dictionary", "furtherInfos", NULL};

    char **kwlist = getWidgetHelp ? const_cast<char**>(kwlistWidget) : const_cast<char**>(kwlistFilter);

    const char *filterstring = NULL;
    int retDict = 0; //dictionary
    int userwithinfos = 0; //furtherInfos

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds,"|sii", kwlist, &filterstring, &retDict, &userwithinfos))
    {
        return NULL;
    }

    int longest_name = 0;
    int listonly = 1;
    QString namefilter;

    if (filterstring == NULL)
    {
        namefilter.fromLatin1(0);
    }
    else
    {
        namefilter.sprintf("%s",filterstring);

        if (namefilter.length())
        {
            listonly = 0;
        }
    }

    if (namefilter.contains("*") && ((namefilter.indexOf("*") == (namefilter.length() - 1)) || (namefilter.indexOf("*") == 0)))
    {
        // This is executed if the '*' ist either the first or the last sign of the string
        listonly = 1;
        namefilter.remove("*");
    }

    PyErr_Clear();

    ito::RetVal retval = ito::retOk;
    PyObject *result = NULL;
    PyObject *resultmand = NULL;
    PyObject *resultopt = NULL;
    PyObject *resulttemp = NULL;
    PyObject *item = NULL;

    ito::AddInManager *AIM = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    if (!AIM)
    {
        PyErr_SetString(PyExc_RuntimeError, "No addin-manager found");
        return NULL;
    }

    const QHash<QString, ito::AddInAlgo::FilterDef *> *filtlist = AIM->getFilterList();
    const QHash<QString, ito::AddInAlgo::AlgoWidgetDef *> *widglist = AIM->getAlgoWidgetList();
    
    if (!widglist && getWidgetHelp)
    {
        PyErr_SetString(PyExc_RuntimeError, "No widget list found");
        return NULL;
    }
    if (!filtlist && !getWidgetHelp)
    {
        PyErr_SetString(PyExc_RuntimeError, "No filterlist found");
        return NULL;
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
        if (!retDict)
        {
            std::cout << "No "<< contextName.toLatin1().data() <<" defined\n";
        }
    }
    else
    {
        if (!retDict)
        {
            std::cout << "\n";
        }
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

                if (!retDict)
                {
                    std::cout << contextName.toUpper().toLatin1().data() << "NAME:    "<< filteredKey.toLatin1().data() << "\n";
                }
                else
                {
                    item = PythonQtConversion::QByteArrayToPyUnicodeSecure(filteredKey.toLatin1());
                    PyDict_SetItemString(resulttemp, "name", item);
                    Py_DECREF(item);
                }

                if (getWidgetHelp)
                {
                    ito::AddInAlgo::AlgoWidgetDef* wFunc = widglist->find(filteredKey).value();  
                    filterParams = AIM->getHashedFilterParams(wFunc->m_paramFunc);

                    if (!retDict)
                    {
                        std::cout << "DESCRIPTION:    " << wFunc->m_description.toLatin1().data() << "\n";
                    }
                    else
                    {
                        item = PythonQtConversion::QByteArrayToPyUnicodeSecure(wFunc->m_description.toLatin1());
                        PyDict_SetItemString(resulttemp, "description", item);      
                        Py_DECREF(item);
                    }
                }
                else
                {
                    ito::AddInAlgo::FilterDef * fFunc = filtlist->find(filteredKey).value(); 
                    ito::AddInAlgo::FilterDefExt *fFuncExt = dynamic_cast<ito::AddInAlgo::FilterDefExt*>(fFunc);

                    filterParams = AIM->getHashedFilterParams(fFunc->m_paramFunc);

                    if (!retDict)
                    {
                        std::cout << "DESCRIPTION:    " << fFunc->m_description.toLatin1().data() << "\n";

                        if (fFuncExt)
                        {
                            std::cout << "OBSERVATION: \n* An observer can be passed to this filter.\n";

                            if (fFuncExt->m_hasStatusInformation)
                            {
                                std::cout << "* This filter can provide status information.\n";
                            }
                            else
                            {
                                std::cout << "* This filter cannot provide status information.\n";
                            }

                            if (fFuncExt->m_isCancellable)
                            {
                                std::cout << "* This filter can be interrupted.\n";
                            }
                            else
                            {
                                std::cout << "* This filter cannot be interrupted.\n";
                            }
                        }
                        else
                        {
                            std::cout << "OBSERVATION: \n* No observer can be passed to this filter.\n* It does not provide any status information.\n* It cannot be cancelled.\n";
                        }
                    }
                    else
                    {
                        item = PythonQtConversion::QByteArrayToPyUnicodeSecure(fFunc->m_description.toLatin1());
                        PyDict_SetItemString(resulttemp, "description", item);
                        Py_DECREF(item);

                        if (fFuncExt)
                        {
                            QByteArray text = "Observer possible.";
                            std::cout << "FEATURES: An observer can be passed to this filter.";

                            if (fFuncExt->m_hasStatusInformation)
                            {
                                text += " Status information.";
                            }
                            else
                            {
                                text += " No status information.";
                            }

                            if (fFuncExt->m_isCancellable)
                            {
                                text += " Cancellation possible.";
                            }
                            else
                            {
                                text += " No cancellation.";
                            }
                            item = PythonQtConversion::QByteArrayToPyUnicodeSecure(text);
                            PyDict_SetItemString(resulttemp, "observation", item);
                            Py_DECREF(item);
                        }
                        else
                        {
                            item = PythonQtConversion::QByteArrayToPyUnicodeSecure("No observer. No status information. No cancellation.");
                            PyDict_SetItemString(resulttemp, "observation", item);
                            Py_DECREF(item);
                        }
                    }
                }

                if (filterParams)
                {
                    if (!retDict)
                    {
                        std::cout << "PARAMETERS:\n";
                    }
                    if (filterParams->paramsMand.size())
                    {
                        if (!retDict)
                        {
                            std::cout << "\nMandatory parameters:\n";
                            resultmand = printOutParams(&(filterParams->paramsMand), false, true, -1);
                            Py_DECREF(resultmand);
                        }
                        else
                        {
                            resultmand = printOutParams(&(filterParams->paramsMand), false, true, -1, false);
                            PyDict_SetItemString(resulttemp, "Mandatory Parameters", resultmand);
                            Py_DECREF(resultmand);
                        }
                        
                    }
                    else if(!retDict)
                    {
                        std::cout << "\nMandatory parameters: " <<  contextName.toLatin1().data()  << " function has no mandatory parameters \n";
                    }

                    if (filterParams->paramsOpt.size())
                    {
                        if (!retDict)
                        {
                            std::cout << "\nOptional parameters:\n";
                            resultopt = ito::printOutParams(&(filterParams->paramsOpt), false, true, -1);
                            Py_DECREF(resultopt);
                        }
                        else
                        {
                            resultopt = ito::printOutParams(&(filterParams->paramsOpt), false, true, -1, false);
                            PyDict_SetItemString(resulttemp, "Optional Parameters", resultopt);
                            Py_DECREF(resultopt);
                        }
                    }
                    else if(!retDict)
                    {
                        std::cout << "\nOptional parameters: " <<  contextName.toLatin1().data()  << " function has no optional parameters \n";
                    }

                    if (filterParams->paramsOut.size())
                    {
                        if (!retDict)
                        {
                            std::cout << "\nOutput parameters:\n";
                            resultopt = ito::printOutParams(&(filterParams->paramsOut), false, true, -1);
                            Py_DECREF(resultopt);
                        }
                        else
                        {
                            resultopt = ito::printOutParams(&(filterParams->paramsOut), false, true, -1, false);
                            PyDict_SetItemString(resulttemp, "Output Parameters", resultopt);
                            Py_DECREF(resultopt);
                        }
                    }
                    else if (!retDict)
                    {
                        std::cout << "\nOutput parameters: " <<  contextName.toLatin1().data()  << " function has no output parameters \n";
                    }
                }
                else if (!retDict)
                {
                    std::cout << "PARAMETERS:\nError while loading parameter info.";
                }
                
                if (!retDict)
                {
                    std::cout << "\n";
                }
                PyDict_SetItemString(result, filteredKey.toLatin1().data(), resulttemp);
                Py_DECREF(resulttemp);
            }
            else
            {
                continue;
            }
        }

        //! Now get the complete filterlist

        if (namefilter.length() && !retDict)
        {
            std::cout << contextName.toLatin1().data() << ", which contain the given string: " << namefilter.toLatin1().data() << "\n";
        }
        else if (!retDict)
        {
            std::cout << "Complete "<< contextName.toLatin1().data() << "list\n";
        }

        for (int n = 0; n < keyList.size(); n++)    // get the longestKeySize name in this list
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
            
            if (!userwithinfos && !retDict)
            {
                std::cout <<"\n" << contextName.append("name").leftJustified(longest_name +1, ' ', false).toLatin1().data() << " \tInfo-String\n";
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

                if (!retDict)
                {
                    if (userwithinfos)
                        std::cout << contextName.toUpper().toLatin1().data() << "NAME:    " << filteredKey.leftJustified(longest_name, ' ', false).toLatin1().data() << " ";
                    else
                        std::cout << filteredKey.leftJustified(longest_name, ' ', false).toLatin1().data() << " ";
                }

                item = PythonQtConversion::QByteArrayToPyUnicodeSecure(filteredKey.toLatin1());
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

                    if (!retDict)
                    {
                        std::cout << "\t'" << desString.toLatin1().data() << "'\n";
                    }
                    else
                    {
                        item = PythonQtConversion::QByteArrayToPyUnicodeSecure(descriptionString.toLatin1());
                        PyDict_SetItemString(resulttemp, "description", item);
                        Py_DECREF(item);
                    }
                }
                else
                {
                    if (!retDict)
                    {
                        std::cout <<"\t' No description '\n";
                    }
                    else
                    {
                        item = PyUnicode_FromString("No description");
                        PyDict_SetItemString(resulttemp, "description", item);
                        Py_DECREF(item);
                    }
                }

                if (userwithinfos)
                {
                    if (!retDict)
                    {
                        std::cout << "PARAMETERS:\n";
                    }

                    if (filterParams)
                    {
                        if (filterParams->paramsMand.size())
                        {
                            if (!retDict)
                            {
                                std::cout << "\nMandatory parameters:\n";
                                resultmand = printOutParams(&(filterParams->paramsMand), false, true, -1);
                                Py_DECREF(resultmand);
                            }
                            else
                            {
                                resultmand = printOutParams(&(filterParams->paramsMand), false, true, -1, false);
                                PyDict_SetItemString(resulttemp, "Mandatory Parameters", resultmand);
                                Py_DECREF(resultmand);
                            }
                        }
                        else if (!retDict)
                        {
                            std::cout << "\nMandatory parameters: " <<  contextName.toLatin1().data()  << " function has no mandatory parameters \n";
                        }
                        if (filterParams->paramsOpt.size())
                        {
                            if (!retDict)
                            {
                                std::cout << "\nOptional parameters:\n";
                                resultopt = printOutParams(&(filterParams->paramsOpt), false, true, -1);
                                Py_DECREF(resultopt);
                            }
                            else
                            {
                                resultopt = printOutParams(&(filterParams->paramsOpt), false, true, -1, false);
                                PyDict_SetItemString(resulttemp, "Optional Parameters", resultopt);
                                Py_DECREF(resultopt);
                            }
                        }
                        else if (!retDict)
                        {
                            std::cout << "\nOptional parameters: " <<  contextName.toLatin1().data()  << " function has no optional parameters \n";
                        }
                        if (filterParams->paramsOut.size())
                        {
                            if (!retDict)
                            {
                                std::cout << "\nOutput parameters:\n";
                                resultopt = printOutParams(&(filterParams->paramsOut), false, true, -1);
                                Py_DECREF(resultopt);
                            }
                            else
                            {
                                resultopt = printOutParams(&(filterParams->paramsOut), false, true, -1, false);
                                PyDict_SetItemString(resulttemp, "Output Parameters", resultopt);
                                Py_DECREF(resultopt);
                            }
                        }
                        else if (!retDict)
                        {
                            std::cout << "\nOutput parameters: " <<  contextName.toLatin1().data()  << " function has no output parameters \n";
                        }
                    }
                    else if (!retDict)
                    {
                        std::cout << "Errors while loading parameter info";
                    }

                    if (!retDict)
                    {
                        std::cout << "\n";
                    }

                    PyDict_SetItemString(result, filteredKey.toLatin1().data(), resulttemp);
                    Py_DECREF(resulttemp);
                }
            }
            else
            {
                continue;
            }
        }
    }

    if (!retDict)
    {
        std::cout << "\n";
    }

    if (retDict)
    {
        return result;
    }
    else
    {
        Py_DECREF(result);
        Py_RETURN_NONE;
    }
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFilterHelp_doc, "filterHelp(filterName = \"\", dictionary = 0, furtherInfos = 0) -> Optional[dict] \n\
\n\
Print outs an online help for the given filter(s) or return help information as dictionary. \n\
\n\
This method prints information about one specific filter (algorithm) or a list of \n\
filters to the console output. If one specific filter, defined in an algorithm plugin, \n\
can be found that case-sensitively fits the given ``filterName``, its full documentation \n\
is printed or returned. Else, a list of filters is printed whose name contains the \n\
given ``filterName``.\n\
\n\
Parameters \n\
----------- \n\
filterName : str, optional \n\
    is the fullname or a part of any filter name which should be displayed. \n\
    If ``filterName`` is empty or no filter matches ``filterName`` (case sensitive) \n\
    a list with all suitable filters is given. \n\
dictionary : int, optional \n\
    if ``1``, a dictionary with all relevant information about the documentation of \n\
    this filter is returned as dictionary and nothing is printed to the command line \n\
    (default: 0). \n\
furtherInfos : int, optional \n\
    Usually, filters or algorithms whose name only contains the given ``filterName`` \n\
    are only listed at the end of the information text. If this parameter is set \n\
    to ``1`` (default: ``0``), the full information for all these filters is printed \n\
    as well. \n\
\n\
Returns \n\
------- \n\
info : dict \n\
    This dictionary is only returned, if ``dictionary`` is set to ``1``. Else ``None`` \n\
    is returned. The dictionary contains relevant information about the desired ``filterName``.");

PyObject* PythonItom::PyFilterHelp(PyObject* /*pSelf*/, PyObject* pArgs, PyObject *pKwds)
{
    return PyWidgetOrFilterHelp(false, pArgs, pKwds);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyWidgetHelp_doc,"widgetHelp(widgetName = \"\", dictionary = 0, furtherInfos = 0) -> Optional[dict] \n\
\n\
Print outs an online help for the given widget(s) or return help information as dictionary. \n\
\n\
This method prints information about one specific widget (defined in an algorithm plugin) \n\
or a list of widgets to the console output. If one specific widget \n\
can be found that case-sensitively fits the given ``widgetName``, its full documentation \n\
is printed or returned. Else, a list of widgets is printed whose name contains the \n\
given ``widgetName``.\n\
\n\
Parameters \n\
----------- \n\
widgetName : str, optional \n\
    is the fullname or a part of any widget name which should be displayed. \n\
    If ``widgetName`` is empty or no widget matches ``widgetName`` (case sensitive) \n\
    a list with all suitable widgets is given. \n\
dictionary : int, optional \n\
    if ``1``, a dictionary with all relevant information about the documentation of \n\
    this widget is returned as dictionary and nothing is printed to the command line \n\
    (default: 0). \n\
furtherInfos : int, optional \n\
    Usually, widgets whose name only contains the given ``widgetName`` \n\
    are only listed at the end of the information text. If this parameter is set \n\
    to ``1`` (default: ``0``), the full information for all these widgets is printed \n\
    as well. \n\
\n\
Returns \n\
------- \n\
info : dict \n\
    This dictionary is only returned, if ``dictionary`` is set to ``1``. Else ``None`` \n\
    is returned. The dictionary contains relevant information about the desired ``widgetName``.");
PyObject* PythonItom::PyWidgetHelp(PyObject* /*pSelf*/, PyObject* pArgs, PyObject *pKwds)
{
    return PyWidgetOrFilterHelp(true, pArgs, pKwds);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginLoaded_doc,"pluginLoaded(pluginName) -> bool \n\
\n\
Checks if a certain plugin could be successfully loaded.\n\
\n\
This method checks if a specified plugin is loaded and returns ``True`` if \n\
this is the case, otherwise ``False``. \n\
\n\
Parameters \n\
----------- \n\
pluginName :  str \n\
    The name of a specified plugin as usually displayed in the plugin window.\n\
\n\
Returns \n\
------- \n\
bool \n\
    ``True``, if the plugin has been loaded and can be used, else ``False``.");
PyObject* PythonItom::PyPluginLoaded(PyObject* /*pSelf*/, PyObject* pArgs)
{
    const char* pluginName = NULL;
    ito::RetVal retval = ito::retOk;
    
    if (!PyArg_ParseTuple(pArgs, "s", &pluginName))
    {
        return NULL;
    }

    ito::AddInManager *AIM = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    if (!AIM)
    {
        PyErr_SetString(PyExc_RuntimeError, "No addin-manager found");
        return NULL;
    }

    int ptype, pnum, pversion;
    QString pauthor, pdescription, pdetaileddescription, plicense, pabout, ptypestr;

    retval = AIM->getPluginInfo(pluginName, ptype, pnum, pversion, ptypestr, pauthor, pdescription, pdetaileddescription, plicense, pabout);

    if (retval.containsWarningOrError())
    {
        Py_RETURN_FALSE;
    }

    Py_RETURN_TRUE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotLoaded_doc,"plotLoaded(plotName) -> bool \n\
\n\
Checks if a certain plot widget is available and loaded.\n\
\n\
This method checks if a specified plot widget is available and loaded and \n\
returns ``True`` in case of success, otherwise ``False``. \n\
\n\
Parameters \n\
----------- \n\
plotName :  str \n\
    The name of a specified plot widget as displayed in the itom property dialog. \n\
\n\
Returns \n\
------- \n\
bool \n\
    ``True``, if the plot has been loaded and can be used, else ``False``.");
PyObject* PythonItom::PyPlotLoaded(PyObject* /*pSelf*/, PyObject* pArgs)
{
    const char* plotName = NULL;
    ito::RetVal retval = ito::retOk;
    
    if (!PyArg_ParseTuple(pArgs, "s", &plotName))
    {
        return NULL;
    }

    ito::DesignerWidgetOrganizer *dwo = qobject_cast<ito::DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    if (!dwo)
    {
        PyErr_SetString(PyExc_RuntimeError, "No ui-manager found");
        return NULL;
    }

    QList<ito::FigurePlugin> plugins = dwo->getPossibleFigureClasses(0, 0, 0);

    foreach (const FigurePlugin &f, plugins)
    {
        if (QString::compare(f.classname, QLatin1String(plotName), Qt::CaseInsensitive) == 0)
        {
            Py_RETURN_TRUE;
        }
    }

    Py_RETURN_FALSE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotHelp_doc,"plotHelp(plotName = \"\", dictionary = False) -> Optional[Union[List[str], dict]] \n\
\n\
Generates an online help for a desired plot class.\n\
\n\
The output of this method depend on the content of the argument ``plotName``: \n\
\n\
* If it is empty or a star (``*``), a list of all available and loaded plot classes is print to \n\
  the command line (``dictionary=False``) or returned as ``List[str]``. \n\
* If it is a valid plot class name, all relevant information of this plot widget \n\
  (Qt designer plugin), like supported data types, all properties, signals or slots... \n\
  are printed to the command line or returned as nested dictionary structure. \n\
\n\
Parameters \n\
----------- \n\
plotName : str \n\
    See the description above. This value can either be an empty string or a star (``*``) \n\
    or the name of a plot designer plugin class. \n\
dictionary : bool, optional \n\
    if ``True``, this methods returns its output either as :class:`list` of :class:`str` or \n\
    a :class:`dict` with information like slots, signals and properties of the desired plot \n\
    classes (default: ``False``). \n\
\n\
Returns \n\
------- \n\
None or list of str or dict \n\
    Returns ``None``, a list of available plot class names or a nested dictionary with various \n\
    information about the plot class (depending on the arguments).");
PyObject* PythonItom::PyPlotHelp(PyObject* /*pSelf*/, PyObject* pArgs, PyObject *pKwds)
{
    const char *kwlist[] = {"plotName", "dictionary", NULL};
    const char* plotName = NULL;

#if PY_VERSION_HEX < 0x03030000
    unsigned char retDict = 0;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "|sb", const_cast<char**>(kwlist), &plotName, &retDict))
    {
        return NULL;
    }
#else //only python 3.3 or higher has the 'p' (bool, int) type string
    int retDict = 0; //this must be int, not bool!!! (else crash)

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "|sp", const_cast<char**>(kwlist), &plotName, &retDict))
    {
        return NULL;
    }
#endif

    ito::RetVal retval;
    int pluginNum = -1;
    int plugtype = -1;
    int version = -1;
    QString pTypeString;
    QString pAuthor;
    QString pDescription;
    QString pDetailDescription;
    QString pLicense;
    QString pAbout;
    PyObject *result = NULL;
    PyObject *item = NULL;    
    PyObject *itemsDict = NULL;
    QString plotName_ = (plotName != NULL) ? QLatin1String(plotName) : QString("");

    ito::DesignerWidgetOrganizer *dwo = qobject_cast<ito::DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    if (!dwo)
    {
        PyErr_SetString(PyExc_RuntimeError, "no ui-manager found");
        return NULL;
    }
    else if(plotName == NULL || strlen(plotName) == 0 || (strlen(plotName) == 1 && plotName[0] == '*'))
    {
        bool found = false;
        QList<ito::FigurePlugin> plugins = dwo->getPossibleFigureClasses(0, 0, 0);

        if (!retDict)
        {
            std::cout << "Available plots\n----------------------------------------\n";
        }

        result = PyTuple_New(plugins.size());
        int i = 0;
        foreach (const FigurePlugin &fig, plugins)
        {
            if (retDict)
            {
                item = PythonQtConversion::QStringToPyObject(fig.classname); //new ref
                PyTuple_SetItem(result, i, item); //steals a reference
            }
            else
            {
                std::cout << "#" << i <<"\t" << fig.classname.toLatin1().data() << "\n";
            }
            i += 1;
        }     

        if (i == 0 && !retDict)
        {
            std::cout << "No plot plugins found\n";
        }
    }
    else
    {
        /* if className is 1D, 2D, 2.5D or 3D, the default from the respective categories is used:*/
        if (plotName_.compare("1d", Qt::CaseInsensitive) == 0)
        {
            plotName_ = dwo->getFigureClass("DObjStaticLine", "", retval);
        }
        else if (plotName_.compare("2d", Qt::CaseInsensitive) == 0)
        {
            plotName_ = dwo->getFigureClass("DObjStaticImage", "", retval);
        }
        else if (plotName_.compare("2.5d", Qt::CaseInsensitive) == 0)
        {
            plotName_ = dwo->getFigureClass("PerspectivePlot", "", retval);
        }

        bool found = false;
        QList<ito::FigurePlugin> plugins = dwo->getPossibleFigureClasses(0, 0, 0);

        FigurePlugin fig;
        foreach (fig, plugins)
        {
            if (QString::compare(fig.classname, plotName_, Qt::CaseInsensitive) == 0)
            {
                found = true;
                break;
            }
        }

        if (!found)
        {
            PyErr_SetString(PyExc_RuntimeError, "plotName not found");
            return NULL;
        }

        result = PyDict_New();
        if (!retDict) std::cout << "\n";

        if (retDict)
        {
            item = PythonQtConversion::QStringToPyObject(fig.classname); //new ref
            PyDict_SetItemString(result, "name", item);
            Py_DECREF(item);
        }
        else
        {
            std::cout << "NAME:         " << fig.classname.toLatin1().data() << "\n";
        }

        QStringList sl;
        sl = dwo->getPlotInputTypes(fig.plotDataTypes);

        if (retDict)
        {
            item = PythonQtConversion::QStringToPyObject(sl.join(", ")); //new ref
            PyDict_SetItemString(result, "inputtype", item);
            Py_DECREF(item);
        }

        else
        {
            std::cout << "INPUT TYPE:   " << sl.join(", ").toLatin1().data() << "\n";
        }

        sl.clear();
        sl = dwo->getPlotDataFormats(fig.plotDataFormats);
        if (retDict)
        {
            item = PythonQtConversion::QStringToPyObject(sl.join(", ")); //new ref
            PyDict_SetItemString(result, "dataformats", item);
            Py_DECREF(item);
        }

        else
        {
            std::cout << "DATA FORMATS: " << sl.join(", ").toLatin1().data() << "\n";
        }
            
        sl.clear();
        sl = dwo->getPlotFeatures(fig.plotFeatures);
        if (retDict)
        {
            item = PythonQtConversion::QStringToPyObject(sl.join(", ")); //new ref
            PyDict_SetItemString(result, "features", item);
            Py_DECREF(item);
        }

        else
        {
            std::cout << "FEATURES:     " << sl.join(", ").toLatin1().data() << "\n";
        }

        sl.clear();
        sl = dwo->getPlotType(fig.plotFeatures);
        if (retDict)
        {
            item = PythonQtConversion::QStringToPyObject(sl.join(", ")); //new ref
            PyDict_SetItemString(result, "type", item);
            Py_DECREF(item);
        }

        else
        {
            std::cout << "TYPE:         " << sl.join(", ").toLatin1().data() << "\n";
        }

        if (fig.factory)
        {
            //qDebug() << "create instance\n";
            ito::AbstractItomDesignerPlugin *fac = (ito::AbstractItomDesignerPlugin*)(fig.factory->instance());

            int version = fac->getVersion();
            QString versionStr = QString("%1.%2.%3").arg(MAJORVERSION(version)).arg(MINORVERSION(version)).arg(PATCHVERSION(version));
            
            if (retDict)
            {
                item = PythonQtConversion::QStringToPyObject(versionStr); //new ref
                PyDict_SetItemString(result, "version", item);
                Py_DECREF(item);
            }
            else
            {
                std::cout << "VERSION:      " << versionStr.toLatin1().data() << "\n";
            }
            

            if (retDict)
            {
                item = PythonQtConversion::QStringToPyObject(fac->getAuthor()); //new ref
                PyDict_SetItemString(result, "author", item);
                Py_DECREF(item);
            }
            else
            {
                std::cout << "AUTHOR:       " << fac->getAuthor().toLatin1().data() << "\n";
            }

            if (retDict)
            {
                item = PythonQtConversion::QStringToPyObject(fac->getDescription()); //new ref
                PyDict_SetItemString(result, "description", item);
                Py_DECREF(item);
            }
            else
            {
                std::cout << "INFO:         " << fac->getDescription().toLatin1().data() << "\n";
            }

            if (retDict)
            {
                item = PythonQtConversion::QStringToPyObject(fac->getLicenseInfo()); //new ref
                PyDict_SetItemString(result, "license", item);
                Py_DECREF(item);
            }
            else
            {
                std::cout << "LICENSE:      " << fac->getLicenseInfo().toLatin1().data() << "\n";
            }

            if (retDict)
            {
                item = PythonQtConversion::QStringToPyObject(fac->getAboutInfo()); //new ref
                PyDict_SetItemString(result, "about", item);
                Py_DECREF(item);
            }
            else
            {
                std::cout << "ABOUT:        " << fac->getAboutInfo().toLatin1().data() << "\n";
            }
                
            if (retDict)
            {
                item = PythonQtConversion::QStringToPyObject(fac->getDetailDescription()); //new ref
                PyDict_SetItemString(result, "detaildescription", item);
                Py_DECREF(item);
            }
            else
            {
                std::cout << "\nDETAILS:\n" << fac->getDetailDescription().toLatin1().data() << "\n";
            }  

            UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
            ito::UiOrganizer::ClassInfoContainerList objInfo;
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QMetaObject::invokeMethod(uiOrg, "getObjectInfo", Q_ARG(const QString&, fig.classname), Q_ARG(bool, true), Q_ARG(ito::UiOrganizer::ClassInfoContainerList*, &objInfo), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
            locker.getSemaphore()->wait(-1);
            retval += locker.getSemaphore()->returnValue;

            ito::UiOrganizer::ClassInfoContainerList::Iterator mapIter;

            if (retDict)
            {
                itemsDict = PyDict_New();
            }
            else
            {
                std::cout << "\nCLASSINFO:\n";
            }

            for (mapIter = objInfo.begin(); mapIter != objInfo.end(); ++mapIter)
            {
                if (mapIter->m_type == ito::ClassInfoContainer::TypeClassInfo)
                {
                    if (retDict)
                    {
                        item = PythonQtConversion::QStringToPyObject(mapIter->m_description); //new ref
                        PyDict_SetItemString(itemsDict, mapIter->m_name.toLatin1().data(), item);
                        Py_DECREF(item);
                    }
                    else
                    {
                        std::cout << mapIter->m_name.toLatin1().data() << " : " << mapIter->m_shortDescription.toLatin1().data() << "\n";
                    }
                }
            }

            if (retDict)
            {
                PyDict_SetItemString(result, "classinfo", itemsDict);
                Py_DECREF(itemsDict);
                itemsDict = PyDict_New();
            }
            else
            {
                std::cout << "\nPROPERTIES:\n";
            }
            
            for (mapIter = objInfo.begin(); mapIter != objInfo.end(); ++mapIter)
            {
                if (mapIter->m_type == ito::ClassInfoContainer::TypeProperty)
                {
                    if (retDict)
                    {
                        item = PythonQtConversion::QStringToPyObject(mapIter->m_description); //new ref
                        PyDict_SetItemString(itemsDict, mapIter->m_name.toLatin1().data(), item);
                        Py_DECREF(item);
                    }
                    else
                    {
                        std::cout << mapIter->m_shortDescription.toLatin1().data() << "\n";
                    }
                }
            }

            if (retDict)
            {
                PyDict_SetItemString(result, "properties", itemsDict);
                Py_DECREF(itemsDict);
                itemsDict = PyDict_New();
            }
            else
            {
                std::cout << "\nSIGNALS:\n";            
            }

            for (mapIter = objInfo.begin(); mapIter != objInfo.end(); ++mapIter)
            {
                if (mapIter->m_type == ito::ClassInfoContainer::TypeSignal)
                {
                    if (retDict)
                    {
                        item = PythonQtConversion::QStringToPyObject(mapIter->m_description); //new ref
                        PyDict_SetItemString(itemsDict, mapIter->m_name.toLatin1().data(), item);
                        Py_DECREF(item);
                    }
                    else
                    {
                        std::cout << mapIter->m_shortDescription.toLatin1().data() << "\n";
                    }
                }
            }

            if (retDict)
            {
                PyDict_SetItemString(result, "signals", itemsDict);
                Py_DECREF(itemsDict);
                itemsDict = PyDict_New();
            }
            else
            {
                std::cout << "\nSLOTS:\n";  
            }

            for (mapIter = objInfo.begin(); mapIter != objInfo.end(); ++mapIter)
            {
                if (mapIter->m_type == ito::ClassInfoContainer::TypeSlot)
                {
                    if (retDict)
                    {
                        item = PythonQtConversion::QStringToPyObject(mapIter->m_description); //new ref
                        PyDict_SetItemString(itemsDict, mapIter->m_name.toLatin1().data(), item);
                        Py_DECREF(item);
                    }
                    else
                    {
                        std::cout << mapIter->m_shortDescription.toLatin1().data() << "\n";
                    }
                }
            }

            if (retDict)
            {
                PyDict_SetItemString(result, "slots", itemsDict);
                Py_DECREF(itemsDict);
                itemsDict = PyDict_New();
            }
            else
            {
                std::cout << "\nINHERITANCE:\n";
            }
            for (mapIter = objInfo.begin(); mapIter != objInfo.end(); ++mapIter)
            {
                if (mapIter->m_type == ito::ClassInfoContainer::TypeInheritance)
                {
                    if (retDict)
                    {
                        item = PythonQtConversion::QStringToPyObject(mapIter->m_description); //new ref
                        PyDict_SetItemString(itemsDict, mapIter->m_name.toLatin1().data(), item);
                        Py_DECREF(item);
                    }
                    else
                    {
                        std::cout << mapIter->m_name.toLatin1().data() << "\n";
                    }
                }
            }

            if (retDict)
            {
                PyDict_SetItemString(result, "inheritances", itemsDict);
                Py_DECREF(itemsDict);
            }
        }
        else
        {
        
        }
    }

    if(result)
    {
        if (retDict > 0)
        {
            return result;
        }
        else
        {
            Py_DECREF(result);
            Py_RETURN_NONE;
        }
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginHelp_doc, "pluginHelp(pluginName, dictionary = False) -> Optional[dict] \n\
\n\
Generates an online help for the specific plugin.\n\
\n\
Information about an itom plugin library (actuator, dataIO or algorithm), gathered \n\
by this method, are the name of the plugin, its version, its type, contained filters \n\
(in case of an algorithm) or the description and initialization parameters (otherwise). \n\
\n\
Parameters \n\
----------- \n\
pluginName : str \n\
    is the fullname of a plugin library as specified in the plugin toolbox.\n\
dictionary : bool, optional \n\
    if ``True``, this method returns a :class:`dict` with all gathered information \n\
    about the plugin. Else (default), this information is printed to the command line. \n\
\n\
Returns \n\
------- \n\
None or dict \n\
    Returns None or a dict depending on the value of parameter ``dictionary``.");
PyObject* PythonItom::PyPluginHelp(PyObject* /*pSelf*/, PyObject* pArgs, PyObject *pKwds)
{
    const char *kwlist[] = {"pluginName", "dictionary", NULL};
    const char* pluginName = NULL;
#if PY_VERSION_HEX < 0x03030000
    unsigned char retDict = 0;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "s|b", const_cast<char**>(kwlist), &pluginName, &retDict))
    {
        return NULL;
    }
#else //only python 3.3 or higher has the 'p' (bool, int) type string
    int retDict = 0; //this must be int, not bool!!! (else crash)

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "s|p", const_cast<char**>(kwlist), &pluginName, &retDict))
    {
        return NULL;
    }
#endif

    QVector<ito::Param> *paramsMand = NULL;
    QVector<ito::Param> *paramsOpt = NULL;

    ito::RetVal retval = ito::retOk;
    int pluginNum = -1;
    int plugtype = -1;
    int version = -1;
    QString pTypeString;
    QString pAuthor;
    QString pDescription;
    QString pDetailDescription;
    QString pLicense;
    QString pAbout;
    PyObject *result = NULL;
    PyObject *resultmand = NULL;
    PyObject *resultopt = NULL;
    PyObject *item = NULL;    


    ito::AddInManager *AIM = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    if (!AIM)
    {
        PyErr_SetString(PyExc_RuntimeError, "No addin-manager found");
        return NULL;
    }

    retval = AIM->getPluginInfo(pluginName, plugtype, pluginNum, version, pTypeString, pAuthor, pDescription, pDetailDescription, pLicense, pAbout);
    if (retval.containsWarningOrError())
    {
        PythonCommon::setReturnValueMessage(retval, pluginName, PythonCommon::loadPlugin);
        return NULL;
    }
    else
    {
        result = PyDict_New();

        if (!retDict) std::cout << "\n";

        if (pluginName)
        {
            if (retDict)
            {
                item = PythonQtConversion::QByteArrayToPyUnicodeSecure(pluginName); //new ref
                PyDict_SetItemString(result, "name", item);
                Py_DECREF(item);
            }
            else
            {
                std::cout << "NAME:\t " << pluginName << "\n";
            }
        }

        if (retDict)
        {
            item = PythonQtConversion::QStringToPyObject(pTypeString); //new ref
            PyDict_SetItemString(result, "type", item);
            Py_DECREF(item);
        }
        else
        {
            std::cout << "TYPE:\t " << pTypeString.toLatin1().data() << "\n";
        }

        QString versionStr = QString("%1.%2.%3").arg(MAJORVERSION(version)).arg(MINORVERSION(version)).arg(PATCHVERSION(version));
        
        if (retDict)
        {
            item = PythonQtConversion::QStringToPyObject(versionStr); //new ref
            PyDict_SetItemString(result, "version", item);
            Py_DECREF(item);
        }
        else
        {
            std::cout << "VERSION:\t " << versionStr.toLatin1().data() << "\n";
        }

        if (retDict)
        {
            item = PythonQtConversion::QStringToPyObject(pAuthor); //new ref
            PyDict_SetItemString(result, "author", item);
            Py_DECREF(item);
        }
        else
        {
            std::cout << "AUTHOR:\t " << pAuthor.toLatin1().data() << "\n";
        }

        if (retDict)
        {
            item = PythonQtConversion::QStringToPyObject(pDescription); //new ref
            PyDict_SetItemString(result, "description", item);
            Py_DECREF(item);
        }
        else
        {
            std::cout << "INFO:\t\t " << pDescription.toLatin1().data() << "\n";
        }

        if (retDict)
        {
            item = PythonQtConversion::QStringToPyObject(pLicense); //new ref
            PyDict_SetItemString(result, "license", item);
            Py_DECREF(item);
        }
        else
        {
            std::cout << "LICENSE:\t\t " << pLicense.toLatin1().data() << "\n";
        }

        if (retDict)
        {
            item = PythonQtConversion::QStringToPyObject(pAbout); //new ref
            PyDict_SetItemString(result, "about", item);
            Py_DECREF(item);
        }
        else
        {
            std::cout << "ABOUT:\t\t " << pAbout.toLatin1().data() << "\n";
        }
                
        if (retDict)
        {
            item = PythonQtConversion::QStringToPyObject(pDetailDescription); //new ref
            PyDict_SetItemString(result, "detaildescription", item);
            Py_DECREF(item);
        }
        else
        {
            std::cout << "\nDETAILS:\n" << pDetailDescription.toLatin1().data() << "\n";
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

                PythonCommon::setReturnValueMessage(retval, pluginName, PythonCommon::loadPlugin);
                return NULL;
            }

            if (retDict == 0)
            {
                std::cout << "\nINITIALISATION PARAMETERS:\n";
            }

            if (paramsMand)
            {
                if ((*paramsMand).size())
                {
                    if (retDict)
                    {
                        resultmand = printOutParams(paramsMand, false, true, -1, false);
                        PyDict_SetItemString(result, "Mandatory Parameters", resultmand);
                        Py_DECREF(resultmand);
                    }
                    else
                    {
                        std::cout << "\n Mandatory parameters:\n";
                        resultmand = printOutParams(paramsMand, false, true, -1);
                        Py_DECREF(resultmand);
                    }
                  
                }
                else if (!retDict)
                {
                    std::cout << "  Initialisation function has no mandatory parameters \n";
                }
            }
            else if (!retDict)
            {
                   std::cout << "  Initialisation function has no mandatory parameters \n";
            }

            if (paramsOpt)
            {
                if ((*paramsOpt).size())
                {
                    if (retDict)
                    {
                        resultopt = printOutParams(paramsOpt, false, true, -1, false);
                        PyDict_SetItemString(result, "Optional Parameters", resultopt);
                        Py_DECREF(resultopt);
                    }
                    else
                    {
                        std::cout << "\n Optional parameters:\n";
                        resultopt = printOutParams(paramsOpt, false, true, -1);
                        Py_DECREF(resultopt);
                    }                    
                }
                else if (!retDict)
                {
                    std::cout << "  Initialisation function has no optional parameters \n";
                }
            }
            else if (!retDict)
            {
                    std::cout << "  Initialisation function has no optional parameters \n";
            }

            if (!retDict)
            {
                std::cout << "\n";
                std::cout << "\nFor more information use the member functions 'getParamListInfo()' and 'getExecFuncInfo()'\n\n";
            }
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
                    if (!retDict)
                    {
                        std::cout << "\nThis is the container for following filters:\n";
                    }
                    QStringList keyList = funcList.keys();
                    keyList.sort();

                    if (retDict)
                    {
                        PyObject *algorithmlist = PyDict_New();
                        for (int algos = 0; algos < keyList.size(); algos++)
                        {
                            item = PythonQtConversion::QByteArrayToPyUnicodeSecure(keyList.value(algos).toLatin1());
                            PyDict_SetItemString(algorithmlist, keyList.value(algos).toLatin1().data(), item);
                            Py_DECREF(item);
                        }
                        PyDict_SetItemString(result, "filter", algorithmlist);
                        Py_DECREF(algorithmlist);
                    }
                    else
                    {
                        for (int algos = 0; algos < keyList.size(); algos++)
                        {
                            std::cout << "> " << algos << "  " << keyList.value(algos).toLatin1().data() << "\n";
                        }

                        std::cout << "\nFor more information use 'filterHelp(\"filterName\")'\n\n";
                    }
                }
                else if (retDict)
                {
                    Py_INCREF(Py_None);
                    PyDict_SetItemString(result, "filter", Py_None);
                }

                QHash<QString, ito::AddInAlgo::AlgoWidgetDef *> widgetList;
                algoInst->getAlgoWidgetList(widgetList);

                if (widgetList.size() > 0)
                {
                    if (!retDict)
                    {
                        std::cout << "\nThis is the container for following widgets:\n";
                    }

                    QStringList keyList = widgetList.keys();
                    keyList.sort();

                    if (retDict)
                    {
                        PyObject *widgetlist = PyDict_New();
                        for (int widgets = 0; widgets < keyList.size(); widgets++)
                        {
                            item = PythonQtConversion::QByteArrayToPyUnicodeSecure(keyList.value(widgets).toLatin1());
                            PyDict_SetItemString(widgetlist, keyList.value(widgets).toLatin1().data(), item);
                            Py_DECREF(item);
                        }
                        PyDict_SetItemString(result, "widgets", widgetlist);
                        Py_DECREF(widgetlist);
                    }
                    else
                    {
                        for (int widgets = 0; widgets < keyList.size(); widgets++)
                        {
                            std::cout << "> " << widgets << "  " << keyList.value(widgets).toLatin1().data() << "\n";
                        }
                        std::cout << "\nFor more information use 'widgetHelp(\"widgetName\")'\n";
                    }
                }
                else
                {
                    Py_INCREF(Py_None);
                    PyDict_SetItemString(result, "widgets", Py_None);
                }
            }
        }
        break;
    }

    Py_DECREF(noneText);

    if (retDict > 0)
    {
        return result;
    }
    else
    {
        Py_DECREF(result);
        Py_RETURN_NONE;
    }
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyAboutInfo_doc,"aboutInfo(pluginName) -> str \n\
\n\
Returns the `about` information for the given plugin as string.\n\
\n\
Parameters \n\
----------- \n\
pluginName : str \n\
    is the name of a plugin library as specified in the plugin toolbox.\n\
\n\
Returns \n\
------- \n\
str \n\
    Returns a string containing the about information. \n\
\n\
Raises \n\
------- \n\
RuntimeError \n\
    if ``pluginName`` is an unknown plugin.");
PyObject* PythonItom::PyAboutInfo(PyObject* /*pSelf*/, PyObject* pArgs, PyObject *pKwds)
{
    const char *kwlist[] = { "pluginName", NULL };
    const char* pluginName = NULL;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "s", const_cast<char**>(kwlist), &pluginName))
    {
        return NULL;
    }

    ito::AddInManager *AIM = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());

    if (!AIM)
    {
        PyErr_SetString(PyExc_RuntimeError, "No addin-manager found");
        return NULL;
    }

    QString version;
    ito::RetVal retval = AIM->getAboutInfo(pluginName, version);

    if (!retval.containsError())
    {
        return PythonQtConversion::QStringToPyObject(version);
    }

    // transform retval to exception
    PythonCommon::transformRetValToPyException(retval);
    return NULL;
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyItomVersion_doc,"version(dictionary = False, addPluginInfo = False) -> Optional[dict] \n\
\n\
Retrieves, prints out or returns complete version information of itom (and optionally plugins). \n\
\n\
Parameters \n\
----------- \n\
dictionary : bool, optional \n\
    If ``True``, all information is returned as nested :class:`dict`. \n\
    Otherwise (default), this information is printed to the command line. \n\
addPluginInfo : bool, optional \n\
    If ``True``, version information about all loaded plugins are added, too. \n\
    Default: ``False``. \n\
\n\
Returns \n\
------- \n\
None or dict \n\
    version information as nested dict, if ``dictionary`` is ``True``.");
PyObject* PythonItom::PyItomVersion(PyObject* /*pSelf*/, PyObject* pArgs, PyObject* pKwds)
{
    bool returnDict = false;
    bool addPluginInfo = false;
    const char *kwlist[] = { "dictionary", "addPluginInfo", NULL };

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "|bb", const_cast<char**>(kwlist), &returnDict, &addPluginInfo))
    {
        return NULL;
    }

    PyObject* myDic = PyDict_New(); // new ref
    PyObject* myTempDic = PyDict_New(); // new ref
    PyObject* key = NULL;
    PyObject* value = NULL;

    QMap<QString, QString> versionMap = ito::getItomVersionMap();
    QMapIterator<QString, QString> i(versionMap);

    while (i.hasNext()) 
    {
        i.next();

        key = PythonQtConversion::QStringToPyObject(i.key()); // new ref
        value = PythonQtConversion::QStringToPyObject(i.value()); // new ref
        PyDict_SetItem(myTempDic, key, value); // does not steal refs from key or value
        Py_DECREF(key);
        Py_DECREF(value);
    }

    PyDict_SetItemString(myDic, "itom", myTempDic); // does not steal ref from myTempDic
    Py_XDECREF(myTempDic);

    if (addPluginInfo)
    {
        PyObject* myTempDic = PyDict_New();
        char buf[7] = {0};
        ito::AddInManager *aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
        ito::AddInInterfaceBase  *curAddInInterface = NULL;

        if (aim != NULL)
        {
            PyObject* info = NULL;
            PyObject* license = NULL;

            for (int i = 0; i < aim->getTotalNumAddIns(); i++)
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
                    sprintf_s(buf, 7, "%i.%i.%i", first, middle, last);
                    value = PyUnicode_FromString(buf);

                    PyDict_SetItemString(info, "version", value);
                    PyDict_SetItemString(info, "license", license);

                    PyDict_SetItem(myTempDic, key, info);

                    Py_DECREF(key);
                    Py_DECREF(value);
                    Py_DECREF(license);
                    Py_XDECREF(info);
                }
            }
        }

        PyDict_SetItemString(myDic, "plugins", myTempDic);
        Py_XDECREF(myTempDic);
    }

    if (returnDict)
    {
        return myDic;
    }
    else
    {

        PyObject* myKeys = PyDict_Keys(myDic); // new ref
        Py_ssize_t size = PyList_Size(myKeys);
        bool check = true;

        for (Py_ssize_t i = 0; i < size; i++)
        {
            PyObject* currentKey = PyList_GET_ITEM(myKeys, i);
            QString key = PythonQtConversion::PyObjGetString(currentKey, true, check);

            if (!check)
            {
                continue;
            }

            if (i > 0)
            {
                std::cout << "\n";
            }

            std::cout << "---- " << key.toLatin1().toUpper().data() << " ----\n";

            PyObject* currentDict = PyDict_GetItem(myDic, currentKey); // borrowed
            PyObject* mySubKeys = PyDict_Keys(currentDict); // new ref
            int longestKeySize = 0;
            int maxLengthCol1 = 32;
            const int maxLengthCol2 = 120 - 32;
            int temp;
            PyObject *currentSubKey = NULL;

            for (Py_ssize_t m = 0; m < PyList_Size(mySubKeys); ++m)
            {      
                currentSubKey = PyList_GET_ITEM(mySubKeys, m);
                temp = PyUnicode_GET_SIZE(currentSubKey);
                longestKeySize = std::max(temp, longestKeySize);
            }

            maxLengthCol1 = std::min(longestKeySize + 2, maxLengthCol1); // +2 due to colon and space

            for (Py_ssize_t m = 0; m < PyList_Size(mySubKeys); ++m)
            {
                currentSubKey = PyList_GET_ITEM(mySubKeys, m);
                QString subKey = PythonQtConversion::PyObjGetString(currentSubKey, true, check);

                if (!check)
                {
                    continue;
                }

                if (subKey.size() > maxLengthCol1 - 2) // -2 due to colon and space
                {
                    subKey = subKey.left(maxLengthCol1 - 5) + "...";
                }

                subKey = subKey.append(": ").leftJustified(maxLengthCol1, ' ');

                QString subVal;
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

                if (subVal.size() > maxLengthCol2)
                {
                    QStringList parts = subVal.split(" ");
                    QStringList lines;
                    QString current;

                    for (int idx = 0; idx < parts.size(); ++idx)
                    {
                        if (parts[idx].size() + current.size() + 1 <= maxLengthCol2)
                        {
                            current += " " + parts[idx];
                        }
                        else
                        {
                            if (current != "")
                            {
                                lines.append(current.mid(1));
                            }

                            current = " " + parts[idx];
                        }
                    }

                    if (current != "")
                    {
                        lines.append(current.mid(1));
                    }

                    if (lines.size() > 0)
                    {
                        std::cout << subKey.toLatin1().data() << lines[0].toLatin1().data() << "\n";
                        QByteArray empty = QByteArray(maxLengthCol1, ' ');

                        for (int lineIdx = 1; lineIdx < lines.size(); ++lineIdx)
                        {
                            std::cout 
                                << empty.constData()
                                << lines[lineIdx].toLatin1().data() << "\n";
                        }
                    }

                }
                else
                {
                    std::cout << subKey.toLatin1().data() << subVal.toLatin1().data() << "\n";
                }
            }

            Py_XDECREF(mySubKeys);
            
        }

        
        Py_DECREF(myKeys);

        Py_DECREF(myDic);
        Py_RETURN_NONE;
    }

}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyAddButton_doc,"addButton(toolbarName, buttonName, code, icon = \"\", argtuple = []) -> int \n\
\n\
Adds a button to a toolbar in the main window of itom. \n\
\n\
This function adds a button to a toolbar in the main window of itom. If the button is \n\
pressed the given code, function or method is executed. If the toolbar specified by \n\
``toolbarName`` does not exist, it is created. The button will display an optional \n\
icon, or - if not given / not loadable - the ``buttonName`` is displayed as text. \n\
\n\
Itom comes with basic icons addressable by ``:/../iconname.png``, e.g.\n\
``:/gui/icons/close.png``. These natively available icons are listed in the icon \n\
browser in the menu **edit >> icon browser** of any script window. Furthermore you \n\
can give a relative or absolute path to any allowed icon file (the preferred file \n\
format is png). \n\
\n\
For more information see also the section :ref:`toolbar-addtoolbar` of the documentation. \n\
\n\
Parameters \n\
----------- \n\
toolbarName : str \n\
    The name of the toolbar.\n\
buttonName : str \n\
    The name and identifier of the button to create.\n\
code : str or callable \n\
    The code or callable to be executed if the button is pressed.\n\
icon : str, optional \n\
    The filename of an icon file. This can also be relative to the application \n\
    directory of **itom**.\n\
argtuple : tuple, optional \n\
    Arguments, which will be passed to the method (in order to avoid cyclic \n\
    references try to only use basic element types). \n\
\n\
Returns \n\
------- \n\
handle : int \n\
    handle to the newly created button (pass it to :meth:`removeButton` to delete \n\
    exactly this button) \n\
\n\
Raises \n\
------- \n\
RuntimeError \n\
    if the main window is not available \n\
\n\
See Also \n\
--------- \n\
removeButton");
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
    RetVal retValue(retOk);
    ito::FuncWeakRef *funcWeakRef = NULL;

    if (!PyArg_ParseTupleAndKeywords(pArgs, kwds, "ssO|sO!", const_cast<char**>(kwlist), &toolbarName, &name, &code, &icon, &PyTuple_Type, &argtuple))
    {
        PyErr_Clear();

        if (!PyArg_ParseTupleAndKeywords(pArgs, kwds, "ssO|sO!", const_cast<char**>(kwlist), &toolbarName, &name, &code, &icon, &PyList_Type, &argtuple))
        {
            return NULL;
        }
    }

    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine
    qkey = QString(name);
    qname = QString(name);
    qicon = QString(icon);

    if (qkey == "")
    {
        retValue += RetVal(retError,0,QObject::tr("Button must have a valid name.").toLatin1().data());
    }
    else
    {
        funcWeakRef = hashButtonOrMenuCode(code, argtuple, retValue, qcode);
    }

    // this is the handle to the newly created button, this can be used to delete the button 
    // afterwards (it corresponds to the pointer address of the corresponding QAction, casted to size_t)
    QSharedPointer<size_t> buttonHandle(new size_t); 

    if (!retValue.containsError())
    {
        QObject *mainWindow = AppManagement::getMainWindow();
        if (mainWindow)
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QMetaObject::invokeMethod(
                mainWindow, 
                "addToolbarButton", 
                Q_ARG(QString, toolbarName), 
                Q_ARG(QString, qname), 
                Q_ARG(QString, qicon), 
                Q_ARG(QString, qcode), 
                Q_ARG(QSharedPointer<size_t>, buttonHandle), 
                Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

            if (!locker->wait(2000))
            {
                unhashButtonOrMenuCode(funcWeakRef); //if the function is already hashed, release it.
                PyErr_SetString(PyExc_RuntimeError, "Timeout while waiting for button being added.");
                return NULL;
            }
            else
            {
                retValue += locker->returnValue;
                if (funcWeakRef)
                {
                    funcWeakRef->setHandle(*buttonHandle);
                }                   
            }
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "Main window not available. Button cannot be added.");
            return NULL;
        }
    }

    if (!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    return PyLong_FromSize_t(*buttonHandle);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyRemoveButton_doc,"removeButton(handle) -> None \\\n\
removeButton(toolbarName, buttonName = '') -> None \n\
\n\
Removes a button from a given toolbar in the itom main window. \n\
\n\
This method removes an existing button from a toolbar in the main window of \n\
**itom**. This button must have been created by the method :meth:`addButton` before. \n\
If the toolbar is empty after the removal, it is finally deleted. \n\
\n\
A button can be identified by two different ways: \n\
\n\
1. Either pass the ``handle`` of the button, as returned by :meth:`addButton`. \n\
   This can also be used, if multiple buttons should have the same name. \n\
2. Identify the button by its ``toolbarName`` and ``buttonName``. If more than \n\
   one button is available in the toolbar with the given ``buttonName``, all \n\
   matched buttons are removed. \n\
\n\
For more information see also the section :ref:`toolbar-addtoolbar` of the documentation. \n\
\n\
Parameters \n\
----------- \n\
handle : int \n\
    The handle returned by :meth:`addButton`. \n\
toolbarName : str \n\
    The name of the toolbar.\n\
buttonName : str \n\
    The name (str, identifier) of the button to remove (only necessary, \n\
    if ``toolbarName`` is given instead of ``handle``.\n\
\n\
Raises \n\
------- \n\
RuntimeError \n\
    if the main window is not available or the addressed button could not be found. \n\
\n\
See Also \n\
--------- \n\
addButton");
PyObject* PythonItom::PyRemoveButton(PyObject* /*pSelf*/, PyObject* pArgs)
{
    const char* toolbarName = NULL;
    const char* buttonName = NULL;
    unsigned int buttonHandle;
    bool callByNames = true;

    if (! PyArg_ParseTuple(pArgs, "ss", &toolbarName, &buttonName))
    {
        PyErr_Clear();
        callByNames = false;
        if (!PyArg_ParseTuple(pArgs, "I", &buttonHandle))
        {
            PyErr_SetString(
                PyExc_TypeError, 
                "Wrong length or type of arguments. Type help(removeButton) for more information.");
            return NULL;
        }
    }

    QObject *mainWindow = AppManagement::getMainWindow();
    if (mainWindow)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QSharedPointer<size_t> buttonHandle_(new size_t);
        *buttonHandle_ = (size_t)NULL;

        if (callByNames)
        {
            QMetaObject::invokeMethod(
                mainWindow,
                "removeToolbarButton", 
                Q_ARG(QString, toolbarName), 
                Q_ARG(QString, buttonName), 
                Q_ARG(QSharedPointer<size_t>, buttonHandle_), 
                Q_ARG(bool, false), 
                Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        }
        else
        {
            QMetaObject::invokeMethod(
                mainWindow, 
                "removeToolbarButton", 
                Q_ARG(size_t, buttonHandle), 
                Q_ARG(bool, false), 
                Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        }

        if (!locker->wait(2000))
        {
            PyErr_SetString(PyExc_RuntimeError, "Timeout while waiting for button being removed.");
            return NULL;
        }
        else
        {
            if (!PythonCommon::transformRetValToPyException(locker->returnValue)) 
            {
                return NULL;
            }

            if (callByNames)
            {
                buttonHandle = *buttonHandle_;
            }

            ito::RetVal retval = PythonItom::unhashButtonOrMenuCode(buttonHandle);
            if (!PythonCommon::transformRetValToPyException(retval)) 
            {
                return NULL;
            }
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "Main window not available. Button cannot be removed.");
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyAddMenu_doc,"addMenu(type, key, name = "", code = \"\", icon = \"\", argtuple = []) -> int \n\
\n\
This function adds an element to the main window menu bar. \n\
\n\
The root element of every menu list must be of type :attr:`~itom.MENU`. Such a \n\
:attr:`~itom.MENU` element can contain sub-elements. These sub-elements can be either \n\
another :attr:`~itom.MENU`, a :attr:`~itom.SEPARATOR` or a :attr:`~itom.BUTTON`. Only \n\
the :attr:`~itom.BUTTON` itself triggers a signal, which then executes the code, given \n\
by a string or a reference to a callable python method or function. Remember, that this \n\
reference is only stored as a weak pointer. \n\
\n\
If you want to directly add a sub-element, you can give a slash-separated string as ``key`` \n\
argument. Every component of this string then represents the menu element in its specific \n\
level. Only the element in the last can be something else than of type \n\
:attr:`~itom.MENU`.\n\
\n\
Itom comes with basic icons addressable by ``:/../iconname.png``, e.g.\n\
``:/gui/icons/close.png``. These natively available icons are listed in the icon \n\
browser in the menu **edit >> icon browser** of any script window. Furthermore you \n\
can give a relative or absolute path to any allowed icon file (the preferred file \n\
format is png). \n\
\n\
For more information see also the section :ref:`toolbar-createmenu` of the documentation. \n\
\n\
Parameters \n\
----------- \n\
type : int \n\
    The type of the menu-element (:attr:`~itom.BUTTON` : 0 [default], \n\
    :attr:`~itom.SEPARATOR` : 1, :attr:`~itom.MENU` : 2). Use the corresponding \n\
    constans in module :mod:`itom`.\n\
key : str \n\
    A slash-separated string where every sub-element is the key-name for the menu-element \n\
    in the specific level.\n\
name : str, optional \n\
    The text of the menu-element. If it is an empty string, the last component of the \n\
    slash separated ``key`` is used as name. For instance if key is equal to ``item1/item2`` \n\
    the name will be ``item2``. \n\
code : str or callable, optional \n\
    The code to be executed if menu element is pressed.\n\
icon : str, optional \n\
    The filename of an icon-file. This can also be relative to the application directory of \n\
    **itom**.\n\
argtuple : tuple, optional \n\
    Arguments, which will be passed to method (in order to avoid cyclic references try \n\
    to only use basic element types).\n\
\n\
Returns \n\
------- \n\
handle : int \n\
    Handle to the recently added leaf node (action, separator or menu item). Use this \n\
    handle to delete the item including its child items (for type 'menu'). \n\
\n\
Raises \n\
------- \n\
RuntimeError \n\
    if the main window is not available or the given button could not be found. \n\
\n\
See Also \n\
--------- \n\
removeMenu");
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
    RetVal retValue(retOk);
    ito::FuncWeakRef *funcWeakRef = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "is|sOsO!", const_cast<char**>(kwlist), &type, &key, &name, &code, &icon, &PyTuple_Type, &argtuple))
    {
        PyErr_Clear();
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "is|sOsO!", const_cast<char**>(kwlist), &type, &key, &name, &code, &icon, &PyList_Type, &argtuple))
        {
            return NULL;
        }
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
        retValue += RetVal(retError,0,QObject::tr("Menu element must have a valid key.").toLatin1().data());
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
                    retValue += RetVal(
                        retError,
                        0,
                        QObject::tr("For menu elements of type 'BUTTON' any type of code "
                                    "(String or callable method or function) must be indicated.").toLatin1().data());
                }
                else
                {
                    funcWeakRef = hashButtonOrMenuCode(code, argtuple, retValue, qcode);
                }
            break;
            }
        case 1: //SEPARATOR
            {
                bool ok;
                qcode = code ? PythonQtConversion::PyObjGetString(code, true, ok) : "";
                if (ok && qcode != "")
                {
                    retValue += RetVal(retWarning,0,QObject::tr("A menu element of type 'separator' can not execute some code. Code argument is ignored.").toLatin1().data());
                    qcode = "";
                }
                else if (!ok && code != NULL && code != Py_None)
                {
                    retValue += RetVal(retWarning,0,QObject::tr("A menu element of type 'separator' can not execute any function or method. Code argument is ignored.").toLatin1().data());
                }
            break;
            }
        case 2: //MENU
            {
                bool ok;
                qcode = code ? PythonQtConversion::PyObjGetString(code, true, ok) : "";
                if (ok && qcode != "")
                {
                    retValue += RetVal(retWarning,0,QObject::tr("A menu element of type 'menu' can not execute some code. Code argument is ignored.").toLatin1().data());
                    qcode = "";
                }
                else if (!ok && code != NULL && code != Py_None)
                {
                    retValue += RetVal(retWarning,0,QObject::tr("A menu element of type 'menu' can not execute any function or method. Code argument is ignored.").toLatin1().data());
                }
            break;
            }
        }
    }

    QSharedPointer<size_t> menuHandle(new size_t);

    if (!retValue.containsError())
    {
        QObject *mainWindow = AppManagement::getMainWindow();
        if (mainWindow)
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QMetaObject::invokeMethod(
                mainWindow, 
                "addMenuElement", 
                Q_ARG(int, type), 
                Q_ARG(QString, qkey), 
                Q_ARG(QString, qname), 
                Q_ARG(QString, qcode), 
                Q_ARG(QString, qicon), 
                Q_ARG(QSharedPointer<size_t>, menuHandle), 
                Q_ARG(bool, false), 
                Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

            if (!locker->wait(2000))
            {
                unhashButtonOrMenuCode(funcWeakRef);
                PyErr_SetString(PyExc_RuntimeError, "Timeout while waiting that menu is added.");
                return NULL;
            }
            else
            {
                retValue += locker->returnValue;
                if (retValue.containsError())
                {
                    unhashButtonOrMenuCode(funcWeakRef);
                }
                else if (funcWeakRef)
                {
                    funcWeakRef->setHandle(*menuHandle);
                }
            }
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "Main window not available. Menu could not be added.");
            return NULL;
        }
    }

    if (!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    return PyLong_FromSize_t(*menuHandle);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyRemoveMenu_doc,"removeMenu(key) -> None \\\n\
removeMenu(menuHandle) -> None \n\
\n\
Remove a menu element with the given key or handle. \n\
\n\
This function remove a menu element with the given ``key`` or ``menuHandle``. \n\
key is a slash separated list. The sub-components then \n\
lead the way to the final element, which should be removed. \n\
\n\
Alternatively, it is possible to pass the handle obtained from :meth:`addMenu`. \n\
\n\
For more information see also the section :ref:`toolbar-createmenu` of the \n\
documentation.\n\
\n\
Parameters \n\
----------- \n\
key : str\n\
    The key (can be a slash-separated list) of the menu entry to remove. If it \n\
    is a slash-separated list, the menu entry is searched down the path, \n\
    indicated by the components of the list respectively. \n\
    If the desired menu item has further child items, they are removed, too. \n\
menuHandle : int \n\
    The handle of the menu entry that should be removed (including its \n\
    possible child items). This handle is usually returned by :meth:`addMenu`.\n\
\n\
Raises \n\
------- \n\
RuntimeError \n\
    if the main window is not available or the given button could not be found. \n\
\n\
See Also \n\
--------- \n\
addMenu");
PyObject* PythonItom::PyRemoveMenu(PyObject* /*pSelf*/, PyObject* args, PyObject *kwds)
{
    const char* keyName = NULL;
    unsigned int menuHandle;
    const char *kwlist[] = {"key", NULL};
    const char *kwlist2[] = {"menuHandle", NULL};
    bool keyNotHandle = true;
    QString qkey;
    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", const_cast<char**>(kwlist), &keyName))
    {
        PyErr_Clear();
        keyNotHandle = false;

        if (!PyArg_ParseTupleAndKeywords(args, kwds, "I", const_cast<char**>(kwlist2), &menuHandle))
        {
            PyErr_SetString(PyExc_TypeError, "Wrong length or type of arguments. Type help(removeMenu) for more information.");
            return NULL;
        }
    }
    else
    {
        qkey = QString(keyName);
        if (qkey == "")
        {
            PyErr_SetString(PyExc_KeyError, "The given key name must not be empty.");
            return NULL;
        }
    }

    QObject *mainWindow = AppManagement::getMainWindow();
    if (mainWindow)
    {
        QSharedPointer<QVector<size_t> > removedMenuHandles(new QVector<size_t>() );
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

        if (keyNotHandle)
        {
            QMetaObject::invokeMethod(
                mainWindow, 
                "removeMenuElement", 
                Q_ARG(QString, qkey), 
                Q_ARG(QSharedPointer<QVector<size_t> >, removedMenuHandles), 
                Q_ARG(bool, false), 
                Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        }
        else
        {
            QMetaObject::invokeMethod(
                mainWindow, 
                "removeMenuElement", 
                Q_ARG(size_t, menuHandle), 
                Q_ARG(QSharedPointer<QVector<size_t> >, removedMenuHandles), 
                Q_ARG(bool, false), 
                Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        }

        if (!locker->wait(2000))
        {
            PyErr_SetString(PyExc_RuntimeError, "Timeout while waiting that menu is removed.");
            return NULL;
        }
        else
        {
            for (int i = 0; i < removedMenuHandles->size(); ++i)
            {
                unhashButtonOrMenuCode(removedMenuHandles->at(i));
            }

            if (!PythonCommon::transformRetValToPyException(locker->returnValue)) 
            {
                return NULL;
            }
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "Main window not available. Menu could not be removed.");
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDumpButtonsAndMenus_doc, "dumpButtonsAndMenus() -> dict \n\
\n\
Gets all user-defined toolbars, menus and its buttons. \n\
\n\
Returns \n\
-------- \n\
dict \n\
    Dictionary with two top-level entries:: \n\
        \n\
        {'toolbars': {}, 'menus': []} \n\
    \n\
    ``toolbars`` contains a dict of all customized toolbars, where each \n\
    item contains all buttons (actions) of this toolbar. ``menus`` contains \n\
    a list of nested dictionaries for each top level menu.");
/*static*/ PyObject* PythonItom::PyDumpMenusAndButtons(PyObject* pSelf)
{
	QObject *mainWindow = AppManagement::getMainWindow();
	if (mainWindow)
	{
		QSharedPointer<QString > dump(new QString());
		ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

		QMetaObject::invokeMethod(
            mainWindow, 
            "dumpToolbarsAndButtons", 
            Q_ARG(QSharedPointer<QString>, dump), 
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

		if (!locker->wait(2000))
		{
			PyErr_SetString(PyExc_RuntimeError, "Timeout.");
			return NULL;
		}
		else
		{
            // this is a little bit an unconvenient way to parse a python-like string.
            // The string dump is parsed by the python interpreter and represents a 
            // dictionary. This dictionary is then returned.
			PyObject *globals = PyDict_New();
			QString totalString = QString("# coding=iso-8859-15 \n\ntext = %1").arg(*dump);
			PyObject *result = PyRun_String(totalString.toLatin1().data(), Py_single_input, globals, NULL);
			Py_XDECREF(result);
			PyObject *text = PyDict_GetItemString(globals, "text"); //borrowed
			Py_XINCREF(text);
			Py_DECREF(globals);
			return text;
		}
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Main window not available. Dump could not be generated.");
		return NULL;
	}
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyCheckSignals_doc, "checkSignals() -> int \n\
\n\
Verifies if a Python interrupt request is currently queued. \n\
\n\
Returns \n\
-------- \n\
int \n\
    Returns 1 if an interrupt is currently queued, else 0.");
/*static */PyObject* PythonItom::PyCheckSignals(PyObject* /*pSelf*/)
{
    int result = PythonEngine::isInterruptQueued() ? 1 : 0;
    return Py_BuildValue("i", result);
    //Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyProcessEvents_doc, "processEvents() \n\
\n\
This method processes posted events for the Python thread. \n\
\n\
Please use this method with care.");
/*static */PyObject* PythonItom::PyProcessEvents(PyObject* /*pSelf*/)
{
    QCoreApplication::processEvents(QEventLoop::AllEvents);
    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine

    if (pyEngine)
    {
        QCoreApplication::sendPostedEvents(pyEngine, 0);
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyGetDebugger_doc, "getDebugger() -> itoDebugger.itoDebugger \n\
\n\
Returns the ``itoDebugger`` object of this itom session. \n\
\n\
It is usually not recommended and necessary to use this method or the returned \n\
debugger. This method is available for development and debugging purposes. \n\
\n\
Returns \n\
------- \n\
debugger : itoDebugger.itoDebugger \n\
    is the debugger instance of this itom session.");
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

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyGCStartTracking_doc, "gcStartTracking() \n\
\n\
Starts a monitoring session for objects in the garbage collector. \n\
\n\
This method makes a snapshot of all objects currently guarded by \n\
the garbage collector (:mod:`gc`). Before this, ``gc.collect()`` \n\
was called to clear all unnecessary objects. \n\
\n\
Later, call :meth:`gcEndTracking` to get a print out of the \n\
differences between the snapshot at the end and the beginning \n\
of the tracking. \n\
\n\
This methods are usually available for development purposes. \n\
\n\
See Also \n\
-------- \n\
gcEndTracking");
/*static */PyObject* PythonItom::PyGCStartTracking(PyObject * /*pSelf*/)
{
    PyObject *gc = PyImport_AddModule("gc"); //borrowed ref
    PyObject *gc_collect = NULL;
    PyObject *obj_list = NULL;
    PyObject *t = NULL;
    bool ok;
    if (gc)
    {
        gc_collect = PyObject_CallMethod(gc, "collect",""); //new reference
        Py_XDECREF(gc_collect);
        obj_list = PyObject_CallMethod(gc, "get_objects", ""); //new reference
        if (!obj_list)
        {
            PyErr_SetString(PyExc_RuntimeError, "Call to gc.get_objects() failed");
            return NULL;
        }

        m_gcTrackerList.clear();

        for (Py_ssize_t i = 0; i < PyList_Size(obj_list); i++)
        {
            t = PyList_GET_ITEM(obj_list,i); //borrowed
            m_gcTrackerList[(size_t)t] = QString("%1 [%2]").arg(t->ob_type->tp_name).arg(PythonQtConversion::PyObjGetString(t, false, ok)); //t->ob_type->tp_name;
        }

        Py_DECREF(obj_list);
        std::cout << m_gcTrackerList.count() << " elements tracked" << std::endl;
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "Module gc could not be imported");
        return NULL;
    }
    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyGCEndTracking_doc, "gcEndTracking() \n\
\n\
Finishes a monitoring session for objects in the garbage collector. \n\
\n\
This method makes a snapshot of all objects currently guarded by \n\
the garbage collector (:mod:`gc`) and compares the list of objects\n\
with the one collected during the last call of :meth:`gcStartTracking`. \n\
\n\
The difference of both lists of printed to the command line. \n\
\n\
This methods are usually available for development purposes. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
     if :meth:`gcStartTracking` was not called before. \n\
\n\
See Also \n\
-------- \n\
gcStartTracking");
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
        PyErr_SetString(PyExc_RuntimeError, "Tracker has not been started. Call gcStartTracking() first.");
        return NULL;
    }

    if (gc)
    {
        gc_collect = PyObject_CallMethod(gc, "collect",""); //new reference
        Py_XDECREF(gc_collect);
        obj_list = PyObject_CallMethod(gc, "get_objects", ""); //new reference
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
                std::cout << "New Element. Addr:" << (size_t)t << " Type: " << temp.toLatin1().data() << "\n" << std::endl;
            }
        }

         QHashIterator<size_t, QString> i(m_gcTrackerList);
         while (i.hasNext()) 
         {
             i.next();
             std::cout << "Del Element. Addr:" << i.key() << " Type: " << i.value().toLatin1().data() << "\n" << std::endl;
         }

         m_gcTrackerList.clear();


        Py_DECREF(obj_list);
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "Module gc could not be imported");
        return NULL;
    }
    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(autoReloader_doc,"autoReloader(enabled, checkFileExec = True, checkCmdExec = True, checkFctExec = False) \n\
\n\
dis-/enables the module to automatically reload changed modules. \n\
\n\
Use this method to enable or disable (and configure) a tool that automatically tries to \n\
reload imported modules and their submodules if they have changed since the last run. \n\
\n\
Returns \n\
------- \n\
enable : bool \n\
    The auto-reload tool is loaded if it is enabled for the first time. If it is disabled, \n\
    it does not check changes of any imported modules. \n\
checkFileExec : bool \n\
    If ``True`` (default) and auto-reload enabled, a check for modifications is executed \n\
    whenever a script is executed \n\
checkCmdExec : bool \n\
    If ``True`` (default) and auto-reload enabled, a check for modifications is executed \n\
    whenever a command in the command line is executed \n\
checkFctExec : bool \n\
    If ``True`` and auto-reload enabled, a check for modifications is executed whenever a \n\
    function or method is run (e.g. by an event or button click) (default: ``False``)\n\
\n\
Notes \n\
------- \n\
This tool is inspired by and based on the IPython extension `autoreload`. \n\
\n\
Reloading Python modules in a reliable way is in general difficult, \n\
and unexpected things may occur. ``autoReloader`` tries to work around \n\
common pitfalls by replacing function code objects and parts of \n\
classes previously in the module with new versions. This makes the \n\
following things to work: \n\
 \n\
- Functions and classes imported via 'from xxx import foo' are upgraded \n\
  to new versions when 'xxx' is reloaded. \n\
\n\
- Methods and properties of classes are upgraded on reload, so that \n\
  calling 'c.foo()' on an object 'c' created before the reload causes \n\
  the new code for 'foo' to be executed. \n\
 \n\
Some of the known remaining caveats are: \n\
 \n\
- Replacing code objects does not always succeed: changing a @property \n\
  in a class to an ordinary method or a method to a member variable \n\
  can cause problems (but in old objects only). \n\
 \n\
- Functions that are removed (eg. via monkey-patching) from a module \n\
  before it is reloaded are not upgraded. \n\
 \n\
- C extension modules cannot be reloaded, and so cannot be autoreloaded.");
/*static*/ PyObject* PythonItom::PyAutoReloader(PyObject* pSelf, PyObject *args, PyObject *kwds)
{
    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine
    if (pyEngine)
    {
        const char *kwlist[] = {"filterName", "dictionary", "furtherInfos", NULL};

        int enabled = 0;
        int checkFile = 1;
        int checkCmd = 1;
        int checkFct = 0;

        if (!PyArg_ParseTupleAndKeywords(args, kwds,"i|iii", const_cast<char**>(kwlist), &enabled, &checkFile, &checkCmd, &checkFct))
        {
            return NULL;
        }

        pyEngine->setAutoReloader(enabled, checkFile, checkCmd, checkFct);
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(getScreenInfo_doc,"getScreenInfo() -> Dict[str, obj] \n\
\n\
Returns dictionary with information about all available screens. \n\
\n\
This method returns a dictionary with information about the current screen \n\
configuration of this computer. \n\
\n\
Returns \n\
------- \n\
dict \n\
    dictionary with the following content is returned: \n\
    \n\
    * screenCount (int): number of available screens \n\
    * primaryScreen (int): index (0-based) of primary screen \n\
    * geometry (tuple): tuple with dictionaries for each screen containing data for \n\
      width (w), height (h) and its top-left-position (x, y)");
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

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pySaveMatlabMat_doc,"saveMatlabMat(filename, values, matrixName = \"matrix\") \n\
\n\
Save strings, numbers, arrays or combinations into a Matlab mat file. \n\
\n\
Save one or multiple objects (strings, numbers, arrays, :class:`dataObject`, \n\
:class:`numpy.ndarray`...) to a Matlab *mat* file. There are the following \n\
possibilites for saving, depending on the type of ``values``: \n\
\n\
* ``values`` is a :class:`dict`: All values in the dictionary are stored under their \n\
  corresponding key. \n\
* If ``values`` contains one item only, it is saved under the given ``matrixName``. \n\
* If ``value`` is a :class:`list` or :class:`tuple` of objects, ``matrixName`` must \n\
  either be a sequence with the same length than ``value``. Then, each item in ``values`` \n\
  is stored with the respective name in ``matrixName``. Or ``matrixName`` can be omitted. \n\
  Then, the items are stored under the self-incremented keys ``matrix1``, ``matrix2``, ... \n\
\n\
Parameters \n\
----------- \n\
filename : str \n\
    Filename under which the file should be saved (.mat will be appended if not available)\n\
values : dict or list or tuple or obj \n\
    The value(s) to be stored. Can be either a single object (number, string, \n\
    :class:`dataObject`, :class:`numpy.ndarray` among others, or a :class:`list`, \n\
    :class:`tuple` or :class:`dict` of these single objects. \n\
matrixName : str or list or tuple, optional \n\
    If ``values`` is a single value, this parameter must be one single :class:`str`. \n\
    Else if ``values`` is a sequence it must be a sequence of strings with the same \n\
    length or it can be omitted. If ``values`` is a dictionary, this argument is ignored. \n\
\n\
Raises \n\
------ \n\
ImportError \n\
     if :mod:`scipy` and its module :mod:`scipy.io` could not be imported. \n\
\n\
Notes \n\
----- \n\
This method requires the package :mod:`scipy` and its module :mod:`scipy.io`. \n\
\n\
See Also \n\
-------- \n\
loadMatlabMat");
PyObject * PythonItom::PySaveMatlabMat(PyObject * /*pSelf*/, PyObject *pArgs)
{
    // if any arguments are changed in this method, consider to also change 
    // PythonEngine::saveMatlabVariables and PythonEngine::saveMatlabSingleParam.

    PyObject* scipyIoModule = PyImport_ImportModule("scipy.io"); // new reference

    if (scipyIoModule == NULL)
    {
        PyErr_SetString(PyExc_ImportError, "Scipy-module and scipy.io-module could not be loaded.");
        return NULL;
    }

    //Arguments must be: filename -> string,
    //                   dict or sequence, whose elements will be put into a dict with default-name-sets or
    //                      single-py-object, which will be put into a dict with default-name

    PyObject *element = NULL;
    PyObject *saveDict = NULL;
    PyObject *tempItem = NULL;
    PyObject *filename = NULL;
    PyObject *matrixNames = NULL;
    const char *matrixName = "matrix";
    const char *tempName = NULL;
    char *key = NULL;
    PyObject *matrixNamesTuple = NULL;
    PyObject *matlabData = NULL;
    PyObject *matrixNamesItem = NULL;


    if (!PyArg_ParseTuple(pArgs, "UO|O", &filename, &element, &matrixNames))
    {
        Py_XDECREF(scipyIoModule);
        return NULL;
    }

    if (element == Py_None)
    {
        Py_XDECREF(scipyIoModule);
        PyErr_SetString(PyExc_ValueError, "Python element must not be None");
        return NULL;
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
                    PyErr_SetString(PyExc_TypeError, "if matrix name is indicated, it must be one unicode string");
                    return NULL;
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
                    PyErr_SetString(PyExc_TypeError, "if matrix name is indicated, it must be a sequence of unicode strings (same length than elements)");
                    return NULL;
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
                            PyErr_SetString(PyExc_TypeError, "each element of matrix names sequence must be a unicode object");
                            return NULL;
                        }
                        Py_XDECREF(matrixNamesItem);
                    }
                }
            }

            saveDict = PyDict_New();
            int sizeIter = 32; //max buffer length for number in key "matrix%i" with i being number
            int keyLength;

            for (Py_ssize_t i = 0; i < PySequence_Size(element); i++)
            {
                tempItem = PySequence_GetItem(element, i); //new reference

                if (tempName == matrixName)
                {
                    keyLength = strlen(tempName) + sizeIter + 1;
                    key = (char*)calloc(keyLength, sizeof(char));
                    sprintf_s(key, keyLength, "%s%i", matrixName, ((int)i + 1));
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

	PyObject *nameobj = PyUnicode_FromString("savemat");
	PyObject *fiveobj = PyUnicode_FromString("5");
	PyObject *rowobj = PyUnicode_FromString("row");
    PyObject *res = PyObject_CallMethodObjArgs(
        scipyIoModule, nameobj, filename, saveDict, 
        Py_True, fiveobj, Py_True, Py_False, rowobj, NULL); //new reference

    Py_XDECREF(nameobj);
    Py_XDECREF(fiveobj);
    Py_XDECREF(rowobj);
    Py_XDECREF(saveDict);
    Py_XDECREF(scipyIoModule);

	if (res == NULL)
    {
        return NULL;
    }
	else
	{
        Py_DECREF(res);
		Py_RETURN_NONE;
	}
}

//-------------------------------------------------------------------------------------
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

    if (element && Py_TYPE(element) == &PythonDataObject::PyDataObjectType)
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
        PyErr_SetString(PyExc_ValueError, "Element is NULL");
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

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyLoadMatlabMat_doc,"loadMatlabMat(filename) -> dict \n\
\n\
Loads Matlab mat-file by using :mod:`scipy` methods and returns the loaded dictionary. \n\
\n\
Parameters \n\
----------- \n\
filename : str \n\
    Filename from which the data will be imported (.mat will be added if not available)\n\
\n\
Returns \n\
------- \n\
mat : dict \n\
    dictionary with content of file \n\
\n\
Raises \n\
------ \n\
ImportError \n\
     if :mod:`scipy` and its module :mod:`scipy.io` could not be imported. \n\
\n\
Notes \n\
----- \n\
This method requires the package :mod:`scipy` and its module :mod:`scipy.io`. \n\
\n\
See Also \n\
--------- \n\
saveMatlabMat");
PyObject * PythonItom::PyLoadMatlabMat(PyObject * /*pSelf*/, PyObject *pArgs)
{
    PyObject* scipyIoModule = PyImport_ImportModule("scipy.io"); // new reference
    PyObject* resultLoadMat = NULL;

    if (scipyIoModule == NULL)
    {
        PyErr_SetString(PyExc_ImportError, "Scipy-module and scipy.io-module could not be loaded.");
        return NULL;
    }

    //Arguments must be: filename -> string

    PyObject *filename = NULL; //borrowed reference

    if (!PyArg_ParseTuple(pArgs, "U", &filename))
    {
        Py_XDECREF(scipyIoModule);
        return NULL;
    }

    PyObject *kwdDict = PyDict_New();
    PyObject *argTuple = PyTuple_New(1);
    //PyTuple_SetItem(argTuple, 0, PyUnicode_FromString(filename));
	Py_INCREF(filename);
    PyTuple_SetItem(argTuple, 0, filename); //steals a reference
    PyDict_SetItemString(kwdDict, "squeeze_me",Py_True);
	PyObject *loadmatobj = PyUnicode_FromString("loadmat");
    PyObject *callable = PyObject_GetAttr(scipyIoModule, loadmatobj);
	Py_DECREF(loadmatobj);
    resultLoadMat = PyObject_Call(callable, argTuple, kwdDict);
    Py_DECREF(kwdDict);
    Py_DECREF(argTuple);

    if (resultLoadMat)
    {
        //parse every element of dictionary and check if it is a numpy.ndarray. If so, transforms it to c-style contiguous form
        if (PyDict_Check(resultLoadMat))
        {
            PyObject *key = NULL;
            PyObject *value = NULL;
            Py_ssize_t pos = 0;
			PyObject *itomMetaInfoObj = PyUnicode_FromString("itomMetaInformation");
			PyObject *importMatlabMatAsDataObjectObj = PyUnicode_FromString("importMatlabMatAsDataObject");

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
								if (PyDict_Contains(descr->fields, itomMetaInfoObj))
                                {
                                    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
                                    if (pyEngine)
                                    {
										PyObject *result = PyObject_CallMethodObjArgs(pyEngine->itomFunctions, importMatlabMatAsDataObjectObj, value, NULL); //new reference

                                        if (result == NULL || PyErr_Occurred())
                                        {
                                            Py_XDECREF(result);
                                            Py_XDECREF(scipyIoModule);
											Py_XDECREF(itomMetaInfoObj);
											Py_XDECREF(importMatlabMatAsDataObjectObj);
                                            PyErr_PrintEx(0);
                                            PyErr_SetString(PyExc_RuntimeError, "Error while parsing imported dataObject or npDataObject.");
                                            return NULL;
                                        }
                                        PyDict_SetItem(resultLoadMat, key, result);
                                        Py_XDECREF(result);
                                    }
                                    else
                                    {
										Py_XDECREF(scipyIoModule);
										Py_XDECREF(itomMetaInfoObj);
										Py_XDECREF(importMatlabMatAsDataObjectObj);
										PyErr_SetString(PyExc_RuntimeError, "Python Engine not available");
                                        return NULL;
                                    }
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

			Py_XDECREF(itomMetaInfoObj);
			Py_XDECREF(importMatlabMatAsDataObjectObj);
        }
    }

    Py_XDECREF(scipyIoModule);

    return resultLoadMat;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFilter_doc,"filter(name : str, *args, _observer = None, **kwds) -> obj \n\
\n\
Invokes a filter (or algorithm) function from an algorithm-plugin. \n\
\n\
This function is used to invoke itom filter-functions or algorithms, declared within \n\
itom-algorithm plugins. The parameters (arguments) depends on the specific filter \n\
function. Call :meth:`filterHelp` to get a list of available filter functions.\n\
\n\
Pass all mandatory or optional arguments of the filter as positional or keyword-based \n\
parameters. Some filters, that implement the additional observer interface, can accept \n\
another :class:`progressObserver` object, that allows monitoring the progress of the \n\
filter and / or interrupting the execution. If such an observer is given, you have to \n\
pass it as keyword-based argument ``_observer``!. \n\
\n\
During the execution of the filter, the python GIL (general interpreter lock) is \n\
released (e.g. for further asynchronous processes. \n\
\n\
Parameters \n\
----------- \n\
name : str \n\
    The name of the filter\n\
*args : obj \n\
    positional arguments for the specific filter-method \n\
_observer : progressObserver, optional \n\
    if the called filter implements the extended interface with progress and status \n\
    information, an optional :class:`progressObserver` object can be given (only as \n\
    keyword-based parameter) which is then used as observer for the current progress of \n\
    the filter execution. It is then also possible to interrupt the execution earlier \n\
    (depending on the implementation of the filter). The observer object is \n\
    reset before passed to the called filter function (using the slot \n\
    :meth:`~progressObserver.reset`). \n\
**kwds : obj \n\
    keyword-based arguments for the specific filter-method. The argument name \n\
    ``_observer`` is reserved for special use. \n\
\n\
Returns \n\
-------- \n\
out : obj \n\
    The returned values depend on the definition of each filter. In general it is a \n\
    tuple of all output parameters that are defined by the filter function.\n\
\n\
See Also \n\
--------- \n\
filterHelp");
PyObject * PythonItom::PyFilter(PyObject * /*pSelf*/, PyObject *pArgs, PyObject *pKwds)
{
    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine

    int length = PyTuple_Size(pArgs);
    ito::RetVal ret = ito::retOk;

    if (length == 0)
    {
        PyErr_SetString(PyExc_ValueError, "No filter specified");
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
        PyErr_SetString(PyExc_TypeError, "First argument must be the filter name! Wrong argument type!");
        return NULL;
    }

    ito::AddInManager *aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    const QHash<QString, ito::AddInAlgo::FilterDef *>* flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->constFind(key);

    if (cfit == flist->constEnd())
    {
        PyErr_SetString(PyExc_ValueError, "Unknown filter, please check typing!");
        return NULL;
    }

    ito::AddInAlgo::FilterDef* fFunc = cfit.value();
    ito::AddInAlgo::FilterDefExt* fFuncExt = dynamic_cast<ito::AddInAlgo::FilterDefExt*>(fFunc);
    
    QVector<ito::ParamBase> paramsMandBase, paramsOptBase, paramsOutBase;

    const ito::FilterParams* filterParams = aim->getHashedFilterParams(fFunc->m_paramFunc);
    if (filterParams == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Parameters of filter could not be found.");
        return NULL;
    }

    PyObject *positionalArgs = PyTuple_GetSlice(pArgs, 1, PyTuple_Size(pArgs)); //new reference
    PyObject *kwdsArgs = NULL;

    //check if pKwds contain the special argument name 'statusObserver' and if so obtain its value,
    //make a copy of pKwds without this argument and use this to parse the remaining parameters
    PyObject *statusObserverName = PyUnicode_FromString("_observer"); //new reference
    PyObject *statusObserver = pKwds ? PyDict_GetItem(pKwds, statusObserverName) : NULL; //NULL, if it does not contain, else: borrowed reference

    if (statusObserver)
    {
        kwdsArgs = PyDict_Copy(pKwds); //new reference
        PyDict_DelItem(kwdsArgs, statusObserverName);
    }
    else
    {
        kwdsArgs = pKwds;
        Py_XINCREF(kwdsArgs);
    }

    Py_XDECREF(statusObserverName);
    statusObserverName = NULL;

    if (statusObserver)
    {
        if (!PyProgressObserver_Check(statusObserver))
        {
            Py_XDECREF(positionalArgs);
            Py_XDECREF(kwdsArgs);
            kwdsArgs = NULL;
            positionalArgs = NULL;
            PyErr_SetString(PyExc_RuntimeError, "Keyword-based parameter '_observer' must be of type itom.progressObserver");
            return NULL;
        }
        else if (fFuncExt == NULL)
        {
            if (PyErr_WarnEx(PyExc_RuntimeWarning,
                "Parameter '_observer' is given, but the called filter does not implement the extended interface with additional status information", 1) == -1) //exception is raised instead of warning (depending on user defined warning levels)
            {
                Py_XDECREF(positionalArgs);
                Py_XDECREF(kwdsArgs);
                kwdsArgs = NULL;
                positionalArgs = NULL;
                return NULL;
            }
        }
    }

    // parses python-parameters with respect to the default values given py (*it).paramsMand 
    // and (*it).paramsOpt and returns default-initialized ParamBase-Vectors paramsMand and paramsOpt.
    ret += parseInitParams(&(filterParams->paramsMand), &(filterParams->paramsOpt), positionalArgs, kwdsArgs, paramsMandBase, paramsOptBase);

    //makes deep copy from default-output parameters (*it).paramsOut and returns it in paramsOut (ParamBase-Vector)
    ret += copyParamVector(&(filterParams->paramsOut), paramsOutBase);

    Py_XDECREF(positionalArgs);
    Py_XDECREF(kwdsArgs);
    kwdsArgs = NULL;
    positionalArgs = NULL;

    if (ret.containsError())
    {
        PyErr_SetString(PyExc_RuntimeError, QObject::tr("Error while parsing parameters.").toLatin1().data());
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS //from here, python can do something else... (we assume that the filter might be a longer operation)

    QSharedPointer<ito::FunctionCancellationAndObserver> observer;

    if (fFuncExt)
    {
        if (statusObserver)
        {
            observer = *(((PythonProgressObserver::PyProgressObserver*)statusObserver)->progressObserver);
        }

        if (observer.isNull())
        {
            observer = QSharedPointer<ito::FunctionCancellationAndObserver>(new ito::FunctionCancellationAndObserver());
        }

        if (pyEngine)
        {
            pyEngine->addFunctionCancellationAndObserver(observer.toWeakRef());
        }
    }

    try
    {
        if (fFuncExt)
        {
            observer->reset();
            ret = (*(fFuncExt->m_filterFuncExt))(&paramsMandBase, &paramsOptBase, &paramsOutBase, observer);
        }
        else
        {
            ret = (*(fFunc->m_filterFunc))(&paramsMandBase, &paramsOptBase, &paramsOutBase);
        }
    }
    catch (cv::Exception &exc)
    {
        const char* errorStr = cvErrorStr(exc.code);

        ret += ito::RetVal::format(ito::retError, 0, QObject::tr("OpenCV Error: %s (%s) in %s, file %s, line %d").toLatin1().data(),
            errorStr, exc.err.c_str(), exc.func.size() > 0 ?
            exc.func.c_str() : QObject::tr("Unknown function").toLatin1().data(), exc.file.c_str(), exc.line);
        //see also cv::setBreakOnError(true) -> then cv::error(...) forces an access to 0x0000 (throws access error, the debugger stops and you can debug it)

        //use this to raise an access error that forces the IDE to break in this line (replaces cv::setBreakOnError(true)).
#if defined _DEBUG
        static volatile int* p = 0; //if your debugger stops in this line, another exception has been raised and you have now the chance to see your callstack for debugging.
        *p = 0;
#endif
    }
    catch(std::exception &exc)
    {
        if (exc.what())
        {
            ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The exception '%s' has been thrown").toLatin1().data(), exc.what());
        }
        else
        {
            ret += ito::RetVal(ito::retError, 0, QObject::tr("The exception '<unknown>' has been thrown").toLatin1().data());
        }
#if defined _DEBUG
        static volatile int* p = 0; //if your debugger stops in this line, another exception has been raised and you have now the chance to see your callstack for debugging.
        *p = 0;
#endif
    }
    catch (...)
    {
        ret += ito::RetVal(ito::retError, 0, QObject::tr("An unspecified exception has been thrown").toLatin1().data());
#if defined _DEBUG
        static volatile int* p = 0; //if your debugger stops in this line, another exception has been raised and you have now the chance to see your callstack for debugging.
        *p = 0;
#endif
    }

    if (fFuncExt && pyEngine)
    {
        pyEngine->removeFunctionCancellationAndObserver(observer.data());
    }

    if (observer && \
        observer->isCancelled() && \
        observer->cancellationReason() == ito::FunctionCancellationAndObserver::ReasonKeyboardInterrupt)
    {
        ret = ito::retOk; //ignore the error message, since Python will raise a keyboardInterrupt though
    }
    
    Py_END_ALLOW_THREADS //now we want to get back the GIL from Python

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
            return PythonParamConversion::ParamBaseToPyObject(paramsOutBase[0]); //new ref
        }
        else
        {
            //parse output vector to PyObject-Tuple
            PyObject* out = PyTuple_New(paramsOutBase.size());
            PyObject* temp;
            Py_ssize_t i = 0;
            bool error = false;

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
                    error = true;
                    break;
                }
            }

            if (error)
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

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pySaveDataObject_doc,"saveDataObject(filename, dataObject, tagsAsBinary = False) \n\
\n\
Saves a dataObject to the harddrive in a xml-based file format (ido). \n\
\n\
This method writes a :class:`dataObject` into the file specified by ``filename``. \n\
The data is stored in a binary format within a xml-based structure. \n\
All string-tags of the dataObject are encoded in order to avoid xml-errors, \n\
the value of numerical tags are either converted to strings with 15 significant digits \n\
(>32bit) or stored as base64 encoded values. \n\
\n\
Parameters \n\
----------- \n\
filename : str \n\
    absolute or relative file path to the destination file (.ido will be added if \n\
    no valid suffix is given)\n\
dataObject : dataObject \n\
    The `n`-dimensional dataObject to be serialized to the file.\n\
tagsAsBinary : bool, optional \n\
    If ``True`` all number tags are stored as base64 encoded number values in the `ido` \n\
    file. Else (default), they are stored as readable strings. \n\
\n\
Notes \n\
----- \n\
Tagnames which contains special characters might lead to XML-conflics. \n\
\n\
See Also \n\
--------- \n\
loadDataObject");
PyObject* PythonItom::PySaveDataObject(PyObject* /*pSelf*/, PyObject* pArgs, PyObject* pKwds)
{
    ito::RetVal ret(ito::retOk);

    const char *kwlist[] = {"filename", "dataObject", "tagsAsBinary", NULL};
    const char* folderfilename;
    PyObject *pyDataObject = NULL;
    int tagsAsBinary = 0;
    bool tagAsBin = false; // defaults metaData as string (false)

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "sO!|i", const_cast<char**>(kwlist), &folderfilename, &PythonDataObject::PyDataObjectType, &pyDataObject, &tagsAsBinary))
    {
        return NULL;
    }

    PythonDataObject::PyDataObject* elem = (PythonDataObject::PyDataObject*)pyDataObject;
    tagAsBin = tagsAsBinary > 0; 

    ret += ito::saveDOBJ2XML(elem->dataObject, folderfilename, false, tagAsBin);

    if (!PythonCommon::setReturnValueMessage(ret, "saveDataObject", PythonCommon::runFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyLoadDataObject_doc,"loadDataObject(filename, dataObject, doNotAppendIDO = False) \n\
\n\
Loads a dataObject from an IDO file. \n\
\n\
This function reads a `dataObject` from the file specified by filename. \n\
MetaData saveType (string, binary) are extracted from the file and restored within the object.\n\
\n\
Parameters \n\
----------- \n\
filename : str \n\
    absolute or relative ido file path to the target file \n\
dataObject : dataObject \n\
    an allocated, e.g. empty :class:`dataObject`, that is filled with the loaded \n\
    object afterwards. \n\
doNotAppendIDO : bool, optional \n\
    If ``True`` (default: ``False``), the file suffix **ido** is appended to ``filename``. \n\
\n\
Notes \n\
----- \n\
\n\
The value of string tags must be encoded to avoid XML-conflics.\n\
Tag names which contains special characters might lead to XML-conflics.");
PyObject* PythonItom::PyLoadDataObject(PyObject* /*pSelf*/, PyObject* pArgs, PyObject* pKwds)
{
    ito::RetVal ret(ito::retOk);

    const char *kwlist[] = {"filename", "dataObject", "doNotAppendIDO", NULL};
    const char* folderfilename;
    PyObject *pyDataObject = NULL;
    int  pyBool = 0;
    bool appendEnding;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "sO!|i", const_cast<char**>(kwlist), &folderfilename, &PythonDataObject::PyDataObjectType, &pyDataObject, &pyBool))
    {
        return NULL;
    }

    PythonDataObject::PyDataObject* elem = (PythonDataObject::PyDataObject*)pyDataObject;

    appendEnding = (pyBool > 0);

    ret += ito::loadXML2DOBJ(elem->dataObject, folderfilename, false, appendEnding);

    if (!PythonCommon::setReturnValueMessage(ret, "loadDataObject", PythonCommon::runFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}


//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pySetCentralWidgetsSizes_doc, "setCentralWidgetsSizes(sizes) \n\
\n\
Sets the sizes of the central widgets of itom (including command line) from top to bottom. \n\
\n\
This method can be important if at least one widget has been added from :class:`itom.ui`, \n\
type :attr:`ui.TYPECENTRALWIDGET`. These user defined widgets are then added on top \n\
of the central area of itom and stacked above the command line. The list of sizes \n\
indicates the desired heights of all widgets in the center in pixel (from top to bottom). \n\
\n\
If the list contains too much items, all extra values are ignored. If the list contains \n\
too few values, the result is undefined, but the program will still be well-behaved. \n\
\n\
The overall size of the central area will not be affected. Instead, any additional/missing \n\
space is distributed amongst the widgets according to the relative weight of the sizes. \n\
\n\
If you speciy a size of 0, the widget will be invisible and can be made visible again \n\
using this method or by increasing its size again with the mouse. \n\
\n\
Parameters \n\
----------- \n\
sizes : sequence of int \n\
    Sizes in pixel for each central widget from top to bottom (including the command line). \n\
");
PyObject* PythonItom::PySetCentralWidgetsSizes(PyObject* /*pSelf*/, PyObject* pArgs, PyObject* pKwds)
{
	const char *kwlist[] = { "sizes", NULL };
	PyObject *sizes = NULL;

	if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "O", const_cast<char**>(kwlist), &sizes))
	{
		return NULL;
	}

	bool ok;
	QVector<int> sizes_ = PythonQtConversion::PyObjGetIntArray(sizes, false, ok);

	if (!ok)
	{
		PyErr_Format(PyExc_TypeError, "The argument 'sizes' must be a sequence of integers");
		return NULL;
	}

	QObject *mainWindow = AppManagement::getMainWindow();
	if (mainWindow)
	{
		QMetaObject::invokeMethod(mainWindow, "setCentralWidgetsSizes", Q_ARG(QVector<int>, sizes_));
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "Main window is not available");
		return NULL;
	}

	Py_RETURN_NONE;
}



//-------------------------------------------------------------------------------------
PyDoc_STRVAR(getDefaultScaleableUnits_doc,"getDefaultScaleableUnits() -> List[str] \n\
\n\
Gets a list with the strings of the standard scalable units. \n\
\n\
The unit strings returned as a list by this method can be transformed into each \n\
other using :meth:`scaleValueAndUnit`. \n\
\n\
Returns \n\
------- \n\
units : list of str \n\
    List with strings containing all scaleable units \n\
\n\
See Also \n\
-------- \n\
scaleValueAndUnit");
PyObject* PythonItom::getDefaultScaleableUnits(PyObject * /*pSelf*/)
{
    PyObject *myList = PyList_New(0);
	PyObject *temp = PyUnicode_FromString("mm");
    PyList_Append(myList, temp);
	Py_DECREF(temp);
	temp = PyUnicode_FromString("m");
    PyList_Append(myList, temp);
	Py_DECREF(temp);
	temp = PyUnicode_FromString("V");
	PyList_Append(myList, temp);
	Py_DECREF(temp);
	temp = PyUnicode_FromString("s");
	PyList_Append(myList, temp);
	Py_DECREF(temp);
	temp = PyUnicode_FromString("g");
	PyList_Append(myList, temp);
	Py_DECREF(temp);
	temp = PyUnicode_FromString("cd");
	PyList_Append(myList, temp);
	Py_DECREF(temp);
	temp = PyUnicode_FromString("A");
	PyList_Append(myList, temp);
	Py_DECREF(temp);
	temp = PyUnicode_FromString("%");
	PyList_Append(myList, temp);
	Py_DECREF(temp);

    return myList;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(scaleValueAndUnit_doc, "scaleValueAndUnit(scaleableUnits, value, valueUnit) -> Tuple[float, str] \n\
\n\
Rescales a ``value`` and its unit to the next matching SI unit. \n\
\n\
At first, it is checked if the given ''valueUnit'' is contained in the list \n\
of the base units ``scaleableUnits``. If this is the case, \n\
the given ``value`` is scaled such that the returned value is greater or equal \n\
than 1. The scaled value and the new unit is returned then. \n\
\n\
Use the method :meth:`getDefaultScaleableUnits` to obtain a suitable list of SI \\n\
base units. \n\
\n\
Parameters \n\
----------- \n\
scaleableUnits : list of str \n\
    A list of str with all base units that should be considered for scaling. \n\
    If the given ''valueUnit'' is not contained in this list of base units, \n\
    no scaling is done and the returned values are equal to ``[value, valueUnit]``. \n\
value : float \n\
    The value to be scaled\n\
valueUnit : str \n\
    The value unit to be scaled\n\
\n\
Returns \n\
------- \n\
tuple \n\
    The returned tuple has the format ``[newValue, newUnit]``, where ``newValue`` is \n\
    a float and ``newUnit`` is a string. \n\
\n\
Examples \n\
--------- \n\
>>> baseUnits = getDefaultScaleableUnits() \n\
>>> print(scaleValueAndUnit(baseUnits, 0.001, 'm')) \n\
[1, 'mm']");
PyObject* PythonItom::scaleValueAndUnit(PyObject * /*pSelf*/, PyObject *pArgs, PyObject *pKwds)
{
    QStringList myQlist;

    const char *kwlist[] = {"scaleableUnits", "value", "valueUnit", NULL};
    int length = PyTuple_Size(pArgs);
    double value = 0.0;
    double valueOut = 0.0;
    PyObject* unitObj = NULL;
    PyObject *myList = NULL;
    QString unitIn("");
    QString unitOut;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "O!dO", const_cast<char**>(kwlist), &PyList_Type, &myList, &value, &unitObj))
    {
        return NULL;
    }

    for (int i = 0; i < PyList_Size(myList); i++)
    {
        myQlist.append(PythonQtConversion::PyObjGetString(PyList_GetItem(myList, i)));
    }

    if (unitObj)
    {
        bool ok;
        unitIn.append(PythonQtConversion::PyObjGetString(unitObj, true, ok));
        if (!ok)
        {
            PyErr_SetString(PyExc_TypeError, "valueUnit must be a string.");
            return NULL;
        }
    }

    ito::RetVal ret = ito::formatDoubleWithUnit(myQlist, unitIn, value, valueOut, unitOut);

    return Py_BuildValue("dN", valueOut, PythonQtConversion::QStringToPyObject(unitOut)); //"N" -> Py_BuildValue steals reference from QStringToPyObject
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(getAppPath_doc,"getAppPath() -> str\n\
\n\
Returns the absolute path of the base directory of this application.\n\
\n\
The returned value is independent of the current working directory. \n\
\n\
Returns \n\
------- \n\
path : str\n\
    absolute path of this application's base directory");
PyObject* PythonItom::getAppPath(PyObject* /*pSelf*/)
{
    return PythonQtConversion::QStringToPyObject(QDir::cleanPath(QCoreApplication::applicationDirPath()));
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(getQtToolPath_doc, "getQtToolPath(toolname) -> str \n\
\n\
Gets the absolute path of the given Qt tool \n\
\n\
Parameters \n\
----------- \n\
toolname : str \n\
    The filename of the tool that should be searched \n\
    (e.g. ``qcollectiongenerator``; suffix is not required)\n\
\n\
Returns \n\
------- \n\
path : str \n\
    Absolute path to the given Qt tool. \n\
\n\
Raises \n\
------- \n\
FileExistsError \n\
    if the given toolname could not be found");
PyObject* PythonItom::getQtToolPath(PyObject* /*pSelf*/, PyObject* pArgs)
{
    char *toolname = NULL;
    if (!PyArg_ParseTuple(pArgs, "s", &toolname))
    {
        return NULL;
    }

    bool found;
    QString name = ProcessOrganizer::getAbsQtToolPath(QLatin1String(toolname), &found);

    if (found)
    {
        return PythonQtConversion::QStringToPyObject(name);
    }
    else
    {
        PyErr_SetString(PyExc_FileExistsError, "Absolute path to desired binary could not be found.");
        return NULL;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(getCurrentPath_doc,"getCurrentPath() -> str \n\
\n\
Returns the absolute path of the current working directory.\n\
\n\
The current working directory is also displayed on the right side \n\
of the status bar of the main window of itom. \n\
\n\
Returns \n\
------- \n\
str\n\
    the absolute path of the current working directory \n\
\n\
See Also \n\
--------- \n\
setCurrentPath");
PyObject* PythonItom::getCurrentPath(PyObject* /*pSelf*/)
{
    return PythonQtConversion::QStringToPyObject(QDir::cleanPath(QDir::currentPath()));
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(setCurrentPath_doc,"setCurrentPath(newPath) -> bool \n\
\n\
Set current working directory to a new absolute path. \n\
\n\
sets the absolute path of the current working directory to 'newPath'. \n\
The current working directory is the base directory for all subsequent relative \n\
pathes of icon-files, script-files, ui-files, relative import statements... \n\
\n\
The current directory is always indicated in the right corner of the status \n\
bar of the main window. \n\
\n\
Parameters \n\
----------- \n\
newPath : str \n\
    The new path for the current working directory.\n\
\n\
Returns \n\
------- \n\
success : bool \n\
    ``True`` in case of success else ``False``. \n\
\n\
See Also \n\
--------- \n\
getCurrentPath");
PyObject* PythonItom::setCurrentPath(PyObject* /*pSelf*/, PyObject* pArgs)
{
    PyObject *pyObj = NULL;
    if (!PyArg_ParseTuple(pArgs, "O", &pyObj))
    {
        PyErr_SetString(PyExc_RuntimeError, "Method requires a string as argument");
        return NULL;
    }

    bool ok;
    QString path;
    path = PythonQtConversion::PyObjGetString(pyObj,true,ok);
    if (ok == false)
    {
        PyErr_SetString(PyExc_RuntimeError, "NewPath parameter could not be interpreted as string.");
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

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyCompressData_doc, "compressData(text) -> bytes \n\
\n\
Compresses a given string text, using zlib. \n\
\n\
The compression is done using the zlib library and the command \n\
`qCompress` of the Qt framework. \n\
\n\
Parameters \n\
----------- \n\
text : str \n\
    The string that should be compressed. \n\
level : int \n\
    The compression level: -1 selects the default compression level of `zlib`, else \n\
    a level in the range [0, 9]. \n\
\n\
Returns \n\
------- \n\
compressed_text : bytes \n\
    The compressed version of ``text``. \n\
\n\
See Also \n\
--------- \n\
uncompressData");
/*static*/ PyObject* PythonItom::compressData(PyObject* pSelf, PyObject* pArgs)
{
    int level = -1;
    const char *data = NULL;
    int dataLength = 0;

    if (!PyArg_ParseTuple(pArgs, "s#|i", &data, &dataLength, &level))
    {
        return NULL;
    }

    if ( level < -1 || level > 9)
    {
        PyErr_SetString(PyExc_RuntimeError, "Compression level must be -1 (default: level 6) or between 0 and 9");
        return NULL;
    }

    QByteArray uncompressed(data, dataLength);
    QByteArray compressed = qCompress(uncompressed, level);
    return PyBytes_FromStringAndSize(compressed.data(), compressed.size());
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUncompressData_doc, "uncompressData(compressed_text) -> bytes \n\
\n\
Uncompresses a given compressed text, using zlib. \n\
\n\
The uncompression is done using the zlib library and the command \n\
`qUncompress` of the Qt framework. \n\
\n\
Parameters \n\
----------- \n\
compressed_text : bytes \n\
    The compressed bytes string. \n\
\n\
Returns \n\
------- \n\
uncompressed_text : bytes \n\
    The uncompressed ``compressed_text``. \n\
\n\
See Also \n\
--------- \n\
compressData");
/*static*/ PyObject* PythonItom::uncompressData(PyObject* pSelf, PyObject* pArgs)
{
    PyObject *byteObj = NULL;

    if (!PyArg_ParseTuple(pArgs, "O!", &PyBytes_Type, &byteObj))
    {
        return NULL;
    }

    QByteArray compressed(PyBytes_AsString(byteObj), PyBytes_Size(byteObj));
    QByteArray uncompressed = qUncompress(compressed);
    return PyBytes_FromStringAndSize(uncompressed.data(), uncompressed.size());
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyRegisterResources_doc, "registerResource(rccFileName, mapRoot = \"\") -> bool \n\
\n\
Registers a resource file with the given rccFileName. \n\
\n\
This method opens a given Qt rcc resource file and registers its content at the location \n\
in the resource tree specified by ``mapRoot``. This ``mapRoot`` must usually be a slash separated \n\
path, starting with a slash. \n\\n\
\n\
To generate a rcc file, create an index of all files, that should be added to the resource file, \n\
in a qrc file and uses the rcc binary from Qt to compile the rcc file. \n\
\n\
This method is new in itom > 4.0.0. \n\
\n\
Parameters \n\
----------- \n\
rccFileName : str\n\
    filepath to the rcc resource file \n\
mapRoot : str, optional \n\
    root key, where the resources should be registered below (default: empty string) \n\
\n\
Returns \n\
---------- \n\
bool \n\
    ``True`` if the file could be successfully opened, else ``False``.\n\
\n\
See Also \n\
--------- \n\
unregisterResource");
/*static*/ PyObject* PythonItom::PyRegisterResource(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds)
{
    const char *kwlist[] = { "rccFileName", "mapRoot", NULL };
    
    const char* rccFileName = NULL;
    const char* mapRoot = NULL;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "s|s", const_cast<char**>(kwlist), &rccFileName, &mapRoot))
    {
        return NULL;
    }

    QString _rcc = QLatin1String(rccFileName);
    QString _mapRoot = mapRoot == NULL ? QString() : QLatin1String(mapRoot);

    bool retVal = QResource::registerResource(_rcc, _mapRoot);
    
    if (retVal)
    {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUnregisterResources_doc, "unregisterResource(rccFileName, mapRoot = \"\") -> bool \n\
\n\
Unregisters the resource with the given rccFileName. \n\
\n\
This method tries to unload all resources in the given rcc resource file from the location \n\
in the resource tree specified by ``mapRoot``. The ``mapRoot`` must usually be a slash separated \n\
path, starting with a slash. \n\
\n\
This method is new in itom > 4.0.0. \n\
\n\
Parameters \n\
----------- \n\
rccFileName : str\n\
    filepath to the rcc resource file \n\
mapRoot : str, optional \n\
    root key, where the resources should be unloaded from (default: empty string). \n\
\n\
Returns \n\
---------- \n\
bool \n\
    ``True`` if the file could be successfully opened, else ``False``.\n\
\n\
See Also \n\
--------- \n\
registerResource");
/*static*/ PyObject* PythonItom::PyUnregisterResource(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds)
{
    const char *kwlist[] = { "rccFileName", "mapRoot", NULL };

    const char* rccFileName = NULL;
    const char* mapRoot = NULL;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "s|s", const_cast<char**>(kwlist), &rccFileName, &mapRoot))
    {
        return NULL;
    }

    QString _rcc = QLatin1String(rccFileName);
    QString _mapRoot = mapRoot == NULL ? QString() : QLatin1String(mapRoot);

    bool retVal = QResource::unregisterResource(_rcc, _mapRoot);

    if (retVal)
    {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(setApplicationCursor_doc,"setApplicationCursor(cursorIndex = -1) \n\
\n\
Changes the itom cursor or restores the previously set cursor if -1. \n\
\n\
This methods changes the overall cursor icon of itom where ``cursorIndex`` \n\
corresponds to the Qt enumeration ``Qt::CursorShape``. e.g.:\n\
\n\
    * 0: Arrow \n\
    * 2: Cross Cursor \n\
    * 3: Wait Curson \n\
    * 13: Pointing Hand Cursor \n\
    * 14: Forbidden Cursor \n\
    * 16: Busy Cursor \n\
\n\
Every change of the cursor is put on a stack. The previous cursor type is \n\
restored, if ``cursorIndex`` is set to ``-1``. \n\
\n\
Parameters \n\
----------- \n\
cursorIndex : int, optional\n\
    The cursor enumeration value of the desired cursor shape (``Qt::CursorShape``) \n\
    or ``-1`` if the previous cursor should be restored (default)");
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

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyLoadIDC_doc,"loadIDC(filename) -> dict \n\
\n\
loads a pickled idc-file and returns the content as dictionary. \n\
\n\
This methods loads the given idc file using the method :meth:`pickle.load` from the \n\
Python buildin module :mod:`pickle` and returns the loaded dictionary. \n\
\n\
Parameters \n\
----------- \n\
filename : str \n\
    Filename to the `idc`-file, that should be loaded. Can be an absolute \n\
    path, or relative with respect to the current working directory. \n\
\n\
Returns \n\
-------- \n\
content : dict \n\
    dictionary with loaded content. \n\
\n\
See Also \n\
--------- \n\
pickle.load, saveIDC");
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
            PyObject *dict = PyDict_New(); //new reference
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
            return PyErr_Format(PyExc_RuntimeError, "The file '%s' does not exist", info.absoluteFilePath().toUtf8().data());
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "Python Engine not available");
        return NULL;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pySaveIDC_doc,"saveIDC(filename, dict, overwriteIfExists = True) \n\
\n\
Saves the given dictionary as pickled idc-file.\n\
\n\
This method saves the given dictionary ``dict`` as pickled idc-file using the method \n\
:meth:`pickle.dump` from the builtin module :mod:`pickle`.\n\
The file will be saved with the pickle protocol version 3 (default for Python 3).\n\
\n\
Parameters \n\
----------- \n\
filename : str \n\
    Filename of the destination `idc` file. Can be an absolute filename \n\
    or relative with respect to the current working directory. \n\
dict : dict \n\
    dictionary which should be pickled. All values in the dictionary must be able \n\
    to be pickled (e.g. all Python base types, dataObjects, numpy.ndarrays...). \n\
overwriteIfExists : bool, optional \n\
    If ``True``, an existing file will be overwritten. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if the file cannot be overwritten or if it exists, but ``overwriteIfExists`` \n\
    is ``False``. \n\
\n\
See Also \n\
--------- \n\
pickle.dump, loadIDC");
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
            RetVal retval = pyEngine->pickleDictionary(dict, filename);

            if (!PythonCommon::transformRetValToPyException(retval))
            {
                return NULL;
            }

            Py_RETURN_NONE;
        }
        else
        {
            return PyErr_Format(PyExc_RuntimeError, "The file '%s' cannot be overwritten", info.absoluteFilePath().toUtf8().data());
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "Python Engine not available");
        return NULL;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyCheckIsAdmin_doc,"userIsAdmin() -> bool \n\
\n\
Returns ``True`` if the current user has administrator rights.\n\
\n\
For more information about the user management of itom, see :ref:`gui-user-management`. \n\
\n\
Returns \n\
------- \n\
bool \n\
    ``True`` if current user has administrator rights, otherwise ``False``.\n\
\n\
See Also \n\
-------- \n\
userIsUser, userIsDeveloper, userGetInfo");
PyObject* PythonItom::userCheckIsAdmin(PyObject* /*pSelf*/)
{
    UserOrganizer* userOrg = ito::UserOrganizer::getInstance();
    if (userOrg == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "userOrganizer not available");
        return NULL;
    }

    if (userOrg->getCurrentUserRole() == ito::userRoleAdministrator)
    {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyCheckIsDeveloper_doc, "userIsDeveloper() -> bool \n\
\n\
Returns ``True`` if the current user has developer rights.\n\
\n\
This method only returns ``True``, if the current user has developer rights, not if \n\
he has higher rights, like adminstrator. \n\
For more information about the user management of itom, see :ref:`gui-user-management`. \n\
\n\
Returns \n\
------- \n\
bool \n\
    ``True`` if current user has developer rights, otherwise ``False``. \n\
\n\
See Also \n\
-------- \n\
userIsUser, userIsAdministrator, userGetInfo");
PyObject* PythonItom::userCheckIsDeveloper(PyObject* /*pSelf*/)
{
    UserOrganizer* userOrg = ito::UserOrganizer::getInstance();
    if (userOrg == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "userOrganizer not available");
        return NULL;
    }

    if (userOrg->getCurrentUserRole() == ito::userRoleDeveloper)
    {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyCheckIsUser_doc, "userIsUser() -> bool \n\
\n\
Returns ``True`` if the current user has user rights.\n\
\n\
This method only returns ``True``, if the current user has user rights, not if \n\
he has higher rights, like developer or adminstrator. \n\
For more information about the user management of itom, see :ref:`gui-user-management`. \n\
\n\
Returns \n\
------- \n\
bool \n\
    ``True`` if current user has user rights, otherwise ``False``.\n\
\n\
See Also \n\
-------- \n\
userIsDeveloper, userIsAdministrator, userGetInfo");
PyObject* PythonItom::userCheckIsUser(PyObject* /*pSelf*/)
{
    UserOrganizer* userOrg = ito::UserOrganizer::getInstance();
    if (userOrg == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "userOrganizer not available");
        return NULL;
    }

    if (userOrg->getCurrentUserRole() == ito::userRoleBasic)
    {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyGetUserInfo_doc,"userGetInfo() -> Dict[str, str] \n\
\n\
Returns a dictionary with relevant information about the current user. \n\
\n\
Returns \n\
------- \n\
dict \n\
    dictionary with the following content is returned: \n\
    \n\
    * Name: The name of the current user \n\
    * Type: The user right type of the current user [user, administrator, developer] \n\
    * ID: The user ID \n\
    * File: The location and name of the corresponding setting file (ini).");
PyObject* PythonItom::userGetUserInfo(PyObject* /*pSelf*/)
{

    UserOrganizer* userOrg = ito::UserOrganizer::getInstance();
    if (userOrg == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "userOrganizer not available");
        return NULL;
    }

    PyObject* returnDict = PyDict_New();

    // Name
    //PyObject *item = PyUnicode_FromString(userOrg->getCurrentUserName().toLatin1().data());
    PyObject *item = PyUnicode_DecodeLatin1(userOrg->getCurrentUserName().toLatin1().data(), userOrg->getCurrentUserName().length(), NULL);
    PyDict_SetItemString(returnDict, "Name", item);
    Py_DECREF(item);
    
    // Type
    switch(userOrg->getCurrentUserRole())
    {
        case ito::userRoleBasic:
            item = PyUnicode_FromString("user");
        break;
        case ito::userRoleAdministrator:
            item = PyUnicode_FromString("administrator");
        break;
        case ito::userRoleDeveloper:
            item = PyUnicode_FromString("developer");
        break;
        default:
            item = PyUnicode_FromString("D.A.U.");
    }
    
    PyDict_SetItemString(returnDict, "Type", item);
    Py_DECREF(item); 

    // ID
    item = PyUnicode_DecodeLatin1(userOrg->getCurrentUserId().toLatin1().data(), userOrg->getCurrentUserId().length(), NULL);
    PyDict_SetItemString(returnDict, "ID", item);
    Py_DECREF(item); 

    // FILE
    //item = PyUnicode_FromString(userOrg->getSettingsFile().toLatin1().data());
	QString settingsFile = userOrg->getCurrentUserSettingsFile();
    item = PyUnicode_DecodeLatin1(settingsFile.toLatin1().data(), settingsFile.length(), NULL);
    PyDict_SetItemString(returnDict, "File", item);
    Py_DECREF(item); 

    return returnDict;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyItom_FigureClose_doc,"close(handle) -> None \\\n\
close(all = \"all\") -> None \n\
\n\
Closes a specific or all opened figures. \n\
\n\
This method closes and deletes either one specific figure (if ``handle`` is given \n\
and valid), or all opened figures (if the string argument ``\"all\"`` is given). \n\
All figure can only be closed, if no other figure references this figure (e.g. \n\
line cut of an image plot (2D). \n\
\n\
This method is a redirect of the staticmethod :meth:`figure.close`. \n\
\n\
Parameters \n\
----------- \n\
handle : int \n\
    a valid figure handle, whose reference figure should be closed. \n\
    This figure handle is for instance obtained by the first value of the \n\
    returned tuple of :meth:`plot`, :meth:`plot1`, :meth:`plot2` among others. \n\
all : {\"all\"} \n\
    Pass the string ``\"all\"``  if all closeable opened figures should be closed. \n\
\n\
Notes \n\
------- \n\
If a :class:`figure` instance still keeps a reference to any figure, it is only closed \n\
and will be deleted after that the last referencing instance has been deleted. \n\
\n\
See Also \n\
--------- \n\
figure.close");

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(setPalette_doc,"setPalette(name, colorStops, inverseColor1, inverseColor2, invalidColor) \n\
\n\
Changes a given color palette or creates a new one with the given arguments. \n\
\n\
This methods modifies an existing color palette (if a palette with ``name`` \n\
already exists) or creates a new color palette with the given ``name``. An existing \n\
color palette can only be modified, if it has no write protection, which is the case \n\
for all pre-defined color palettes of itom (see color palette editor in itom property \n\
editor). If any of the optional values are not given, default values (from the ``gray`` \n\
color palette) are used, or, if the color palette ``name`` already exists, \n\
these values are left unchanged\n\
\n\
To obtain the parameters of an existing color palette, that can be used as arguments \n\
of this method, unpack the returned dictionary of :meth:`getPalette`. \n\
\n\
It is also possible to modify or create color palettes in the color palette editor of \n\
the itom property dialog. For more information see :ref:`gui-color-palette-editor`. \n\
\n\
Parameters \n\
----------- \n\
name : str \n\
    Name of the color palette. \n\
colorStops : tuple \n\
    Tuple with all color stops of this color palette. Each item of this tuple is \n\
    another tuple with two values. The first value is the float position of the \n\
    color stop in the range [0.0, 1.0]. The 2nd value is the :class:`rgba32` color \n\
    at this position. Colors between two adjacent color stops are linearly interpolated. \n\
    \n\
    The position of the first color stop has to be 0.0, the one of the last stop 1.0.\n\
    There must be at least two colorStops.\n\
inverseColor1 : rgba, optional \n\
    First inverse color, used for instance for line cuts, markers etc. of a 2D plot. \n\
inverseColor2 : rgba, optional \n\
    second inverse color, used for instance for line cuts, markers etc. of a 2D plot. \n\
invalidColor : rgba, optional \n\
    color used for ``NaN`` or ``Inf`` values in plots. If the invalid color is not given \n\
    and an existing color palette also has no invalid color, the color of the first color \n\
    stop is taken. \n\
\n\
See Also \n\
--------- \n\
getPalette, getPaletteList");
PyObject* PythonItom::PySetPalette(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds)
{
    const char *kwlist[] = {"name", "colorStops", "inverseColor1", "inverseColor2", "invalidColor", NULL};
    char *name = NULL;
    PyObject *colorStops = NULL;
    PyObject *inverseColor1 = NULL;
    PyObject *inverseColor2 = NULL;
    PyObject *invalidColor = NULL;

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "sO!|O!O!O!", const_cast<char**>(kwlist), &name, &PyTuple_Type, &colorStops, \
        &PythonRgba::PyRgbaType, &inverseColor1, &PythonRgba::PyRgbaType, &inverseColor2, &PythonRgba::PyRgbaType, &invalidColor))
    {
        return NULL;
    }

    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    ito::PaletteOrganizer *paletteOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();

    if(paletteOrganizer == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Palette organizer not loaded");
        return NULL;    
    }

    //check if palette already exists
    QSharedPointer<ito::ItomPaletteBase> sharedPalette(new ito::ItomPaletteBase);
    ito::ItomPaletteBase newPalette;

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    QMetaObject::invokeMethod(
        paletteOrganizer, 
        "getColorBarThreaded", 
        Q_ARG(QString,QLatin1String(name)), 
        Q_ARG(QSharedPointer<ito::ItomPaletteBase>, sharedPalette), 
        Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

    if (!locker.getSemaphore()->wait(60000))
    {
        PyErr_SetString(PyExc_RuntimeError, "Timeout while checking if palette already exists.");
        return NULL;
    }

    if (locker.getSemaphore()->returnValue != ito::retOk)
    {
        //new color palette
        //get gray, and use this as default
        QSharedPointer<ito::ItomPaletteBase> sharedPalette2(new ito::ItomPaletteBase);

        ItomSharedSemaphoreLocker locker2(new ItomSharedSemaphore());

        QMetaObject::invokeMethod(
            paletteOrganizer, 
            "getColorBarThreaded", 
            Q_ARG(QString,QLatin1String("gray")),
            Q_ARG(QSharedPointer<ito::ItomPaletteBase>, sharedPalette2), 
            Q_ARG(ItomSharedSemaphore*,locker2.getSemaphore()));

        if (!locker2.getSemaphore()->wait(60000))
        {
            PyErr_SetString(PyExc_RuntimeError, "Timeout while getting default color palette.");
            return NULL;
        }
        else
        {
            newPalette = *sharedPalette2;
            newPalette.removeWriteProtection();
        }
    }
    else
    {
        if (sharedPalette->isWriteProtected())
        {
            return PyErr_Format(
                PyExc_RuntimeError, 
                "The color palette '%s' is readonly and cannot be changed", 
                name);
        }
        else
        {
            newPalette = *sharedPalette;
        }
    }

    newPalette.setName(QLatin1String(name));

    if (inverseColor1)
    {
        ito::Rgba32 rgba = ((ito::PythonRgba::PyRgba*)inverseColor1)->rgba;
        newPalette.setInverseColorOne(QColor(rgba.r, rgba.g, rgba.b,rgba.a));
    }

    if (inverseColor2)
    {
        ito::Rgba32 rgba = ((ito::PythonRgba::PyRgba*)inverseColor2)->rgba;
        newPalette.setInverseColorTwo(QColor(rgba.r, rgba.g, rgba.b,rgba.a));
    }

    if (invalidColor)
    {
        ito::Rgba32 rgba = ((ito::PythonRgba::PyRgba*)invalidColor)->rgba;
        newPalette.setInvalidColor(QColor(rgba.r, rgba.g, rgba.b,rgba.a));
    }

    QVector<QPair<qreal, QColor> > stops;
    int length = PyTuple_Size(colorStops);
    PyObject *subtuple;

    for (int i = 0; i < length; ++i)
    {
        subtuple = PyTuple_GetItem(colorStops, i); //borrowed

        if (PyTuple_Check(subtuple))
        {
            double pos;
            PyObject *color;
            bool found;
            if (!PyArg_ParseTuple(subtuple, "dO!", &pos, &PythonRgba::PyRgbaType, &color))
            {
                return PyErr_Format(
                    PyExc_RuntimeError, 
                    "The %i. item of colorStops must be a tuple with a real value, followed by a itom.rgba color.", 
                    i);
            }
            ito::Rgba32 rgba = ((ito::PythonRgba::PyRgba*)color)->rgba;

            if (i == 0)
            {
                if (qFuzzyCompare(pos, 0.0) == false)
                {
                    return PyErr_Format(PyExc_RuntimeError, "The position of the first color stop must be 0.0.");
                }
                else
                {
                    stops.append( QPair<qreal, QColor>(pos, QColor(rgba.r, rgba.g, rgba.b,rgba.a)) );
                }
            }
            else if (i == length - 1)
            {
                if (qFuzzyCompare(pos, 1.0) == false)
                {
                    return PyErr_Format(PyExc_RuntimeError, "The position of the last color stop must be 1.0.");
                }
                else
                {
                    stops.append( QPair<qreal, QColor>(pos, QColor(rgba.r, rgba.g, rgba.b,rgba.a)) );
                }
            }
            else
            {
                if (pos < 0.0 || pos > 1.0)
                {
                    return PyErr_Format(PyExc_RuntimeError, "The position of any color stop must be in the range [0.0, 1.0].");
                }
                else
                {
                    found = false;

                    for (int j = 1; j < stops.size(); ++j)
                    {
                        if (stops[j].first > pos)
                        {
                            stops.insert(j - 1, QPair<qreal, QColor>(pos, QColor(rgba.r, rgba.g, rgba.b,rgba.a)));
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        stops.append( QPair<qreal, QColor>(pos, QColor(rgba.r, rgba.g, rgba.b,rgba.a)) );
                    }
                }
            }
        }
        else
        {
            return PyErr_Format(PyExc_RuntimeError, "The %i. item of colorStops must be a tuple with two values.", i);
        }
    }

    if (stops.size() < 2)
    {
        return PyErr_Format(PyExc_RuntimeError, "colorStops must consist of at least two color stops.");
    }

    newPalette.setColorStops(stops);

    ItomSharedSemaphoreLocker locker3(new ItomSharedSemaphore());

    QMetaObject::invokeMethod(
        paletteOrganizer, 
        "setColorBarThreaded", 
        Q_ARG(QString,QString(name)), 
        Q_ARG(ito::ItomPaletteBase, newPalette), 
        Q_ARG(ItomSharedSemaphore*,locker3.getSemaphore()));

    if (!locker3.getSemaphore()->wait(60000))
    {
        PyErr_SetString(PyExc_RuntimeError, "Timeout while setting palette");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(locker3.getSemaphore()->returnValue))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(getPalette_doc, "getPalette(name) -> dict \n\
\n\
Returns all relevant data of an existing color palette. \n\
\n\
If a color palette with this ``name`` exists, its relevant data is returned \n\
as dictionary. The values in this dictionary can also be used to call the \n\
method :meth:`setPalette`. \n\
\n\
Parameters \n\
----------- \n\
name : str \n\
    name of the new palette. \n\
\n\
Returns \n\
-------- \n\
palette : dict \n\
    Dictionary with the following entries: \n\
    \n\
    name : str \n\
        name of the color palette. \n\
    colorStops : tuple \n\
        tuple with all color stops, each element is another tuple whose first value is \n\
        the float position of the stop in the range [0.0, 1.0]. The 2nd value is the \n\
        corresponding :class:`rgba` color. The first color stop is always at \n\
        position 0.0, the last one at position 1.0. \n\
    inverseColor1 : rgba \n\
        first inverse color. \n\
    inverseColor2 : rgba \n\
        2nd inverse color. \n\
    invalidColor : rgba \n\
        color used for ``NaN`` or ``Inf`` values. \n\
\n\
Raises \n\
----------- \n\
RuntimeError \n\
    if no color palette with the given name is available. \n\
\n\
See Also \n\
--------- \n\
setPalette, getPaletteList");
PyObject* PythonItom::PyGetPalette(PyObject* pSelf, PyObject* pArgs)
{
    char* name = NULL;
    PyObject *subtuple = NULL;

    unsigned char overwriteIfExists = 1;

    if (!PyArg_ParseTuple(pArgs, "s", &name)) //all borrowed
    {
        return NULL;
    }

    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    ito::PaletteOrganizer *paletteOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();

    if(paletteOrganizer == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Palette organizer not loaded");
        return NULL;    
    }

    QSharedPointer<ito::ItomPaletteBase> sharedPalette(new ito::ItomPaletteBase);

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    QMetaObject::invokeMethod(
        paletteOrganizer, 
        "getColorBarThreaded", 
        Q_ARG(QString,QString(name)), 
        Q_ARG(QSharedPointer<ito::ItomPaletteBase>, sharedPalette), 
        Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

    if (!locker.getSemaphore()->wait(60000))
    {
        PyErr_SetString(PyExc_RuntimeError, "Timeout while getting palette");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }

    PyObject *colorStops = PyTuple_New(sharedPalette->getNumColorStops());

    for(int elem = 0; elem < sharedPalette->getNumColorStops(); elem++)
    {
        subtuple = PyTuple_New(2); //new ref
        ito::PythonRgba::PyRgba* rgb = \
            (ito::PythonRgba::PyRgba*)ito::PythonRgba::PyRgba_new(
                &ito::PythonRgba::PyRgbaType, NULL, NULL ); //new ref
        const QColor &c = sharedPalette->getColor(elem);
        rgb->rgba.a = c.alpha();
        rgb->rgba.r = c.red();
        rgb->rgba.g = c.green();
        rgb->rgba.b = c.blue();
        PyTuple_SetItem(subtuple, 0, PyFloat_FromDouble(sharedPalette->getPos(elem))); //steals a reference
        PyTuple_SetItem(subtuple, 1, (PyObject*)rgb); //steals a reference
        PyTuple_SetItem(colorStops, elem, subtuple); //steals a reference
    }

    PyObject *dict = PyDict_New();
    PyObject *name_ = PythonQtConversion::QByteArrayToPyUnicodeSecure(sharedPalette->getName().toLatin1());
    PyDict_SetItemString(dict, "name", name_);
    Py_DECREF(name_);

    PyDict_SetItemString(dict, "colorStops", colorStops);
    Py_DECREF(colorStops);

    ito::PythonRgba::PyRgba* inverseColor1 = \
        (ito::PythonRgba::PyRgba*)ito::PythonRgba::PyRgba_new(
            &ito::PythonRgba::PyRgbaType, NULL, NULL ); //new ref
    const QColor &invColor1 = sharedPalette->getInverseColorOne();
    inverseColor1->rgba.a = invColor1.alpha();
    inverseColor1->rgba.r = invColor1.red();
    inverseColor1->rgba.g = invColor1.green();
    inverseColor1->rgba.b = invColor1.blue();
    PyDict_SetItemString(dict, "inverseColor1", (PyObject*)inverseColor1);
    Py_DECREF(inverseColor1);

    ito::PythonRgba::PyRgba* inverseColor2 = \
        (ito::PythonRgba::PyRgba*)ito::PythonRgba::PyRgba_new(
            &ito::PythonRgba::PyRgbaType, NULL, NULL ); //new ref
    const QColor &invColor2 = sharedPalette->getInverseColorTwo();
    inverseColor2->rgba.a = invColor2.alpha();
    inverseColor2->rgba.r = invColor2.red();
    inverseColor2->rgba.g = invColor2.green();
    inverseColor2->rgba.b = invColor2.blue();
    PyDict_SetItemString(dict, "inverseColor2", (PyObject*)inverseColor2);
    Py_DECREF(inverseColor2);

    ito::PythonRgba::PyRgba* invalidColor = \
        (ito::PythonRgba::PyRgba*)ito::PythonRgba::PyRgba_new(
            &ito::PythonRgba::PyRgbaType, NULL, NULL ); //new ref
    const QColor &invColor = sharedPalette->getInvalidColor();
    invalidColor->rgba.a = invColor.alpha();
    invalidColor->rgba.r = invColor.red();
    invalidColor->rgba.g = invColor.green();
    invalidColor->rgba.b = invColor.blue();
    PyDict_SetItemString(dict, "invalidColor", (PyObject*)invalidColor);
    Py_DECREF(invalidColor);

    return dict;
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(getPaletteList_doc,"getPaletteList(type = 0) -> Tuple[str] \n\
\n\
Returns a tuple with the names of all currently available color palettes. \n\
\n\
Parameters \n\
----------- \n\
type : int, optional \n\
    Unused parameter. \n\
\n\
Returns \n\
------- \n\
tuple of str \n\
    Tuple with the names of all available color palettes. \n\
\n\
See Also \n\
--------- \n\
setPalette, getPalette");
PyObject* PythonItom::PyGetPaletteList(PyObject* pSelf, PyObject* pArgs)
{
    int typefilter = 0;
    PyObject *tuple = NULL;

    unsigned char overwriteIfExists = 1;

    if (!PyArg_ParseTuple(pArgs, "|i", &typefilter)) //all borrowed
    {
        return NULL;
    }

    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    ito::PaletteOrganizer *paletteOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();

    if(paletteOrganizer == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Palette organizer not loaded");
        return NULL;    
    }

    QSharedPointer<QStringList> sharedPalettes(new QStringList);

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    QMetaObject::invokeMethod(
        paletteOrganizer, 
        "getColorBarListThreaded", 
        Q_ARG(QSharedPointer<QStringList>, sharedPalettes), 
        Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

    if (!locker.getSemaphore()->wait(60000))
    {
        PyErr_SetString(PyExc_RuntimeError, "Timeout while getting palette names");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }

    tuple = PyTuple_New(sharedPalettes->size());

    for(int elem = 0; elem < sharedPalettes->size(); elem++)
    {
        PyObject *item;
        item = PythonQtConversion::QByteArrayToPyUnicodeSecure((*sharedPalettes)[elem].toLatin1());
        PyTuple_SetItem(tuple, elem, item);
    }

    return tuple;
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyClearAll_doc, "clearAll() \n\
\n\
Clears all variables in the global workspace. \n\
\n\
This method clears all variables in the global workspace, that have been \n\
added after the startup process of itom. This only affects variables, that \n\
are also displayed in the workspace toolbox. Variables, like methods, functions, \n\
classes etc. are filtered out, and will therefore not be deleted.\n\
\n\
Variables, that have been created by any startup script will also not be deleted.");
PyObject* PythonItom::PyClearAll(PyObject* pSelf)
{
    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    pyEngine->pythonClearAll();
    Py_RETURN_NONE;
}
//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //
//                                                                                                              //
//          PYTHON MODULES - - - PYTHON TYPES - - - PYTHON MODULES                                              //
//                                                                                                              //
//  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //  //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//-------------------------------------------------------------------------------------
PyMethodDef PythonItom::PythonMethodItom[] = {
    // "Python name", C Ffunction Code, Argument Flags, __doc__ description
    {"scriptEditor", (PyCFunction)PythonItom::PyOpenEmptyScriptEditor, METH_NOARGS, pyOpenEmptyScriptEditor_doc},
    {"newScript", (PyCFunction)PythonItom::PyNewScript, METH_NOARGS, pyNewScript_doc},
    {"openScript", (PyCFunction)PythonItom::PyOpenScript, METH_VARARGS, pyOpenScript_doc},
	{"showHelpViewer", (PyCFunction)PythonItom::PyShowHelpViewer, METH_VARARGS, pyShowHelpViewer_doc },
    {"plot", (PyCFunction)PythonItom::PyPlotImage, METH_VARARGS | METH_KEYWORDS, pyPlotImage_doc},
    {"plot1", (PyCFunction)PythonItom::PyPlot1d, METH_VARARGS | METH_KEYWORDS, pyPlot1d_doc},
    {"plot2", (PyCFunction)PythonItom::PyPlot2d, METH_VARARGS | METH_KEYWORDS, pyPlot2d_doc},
    {"plot25", (PyCFunction)PythonItom::PyPlot25d, METH_VARARGS | METH_KEYWORDS, pyPlot25d_doc},
    {"liveImage", (PyCFunction)PythonItom::PyLiveImage, METH_VARARGS | METH_KEYWORDS, pyLiveImage_doc},
    {"close", (PyCFunction)PythonFigure::PyFigure_close, METH_VARARGS, pyItom_FigureClose_doc}, /*class static figure.close(...)*/
    {"filter", (PyCFunction)PythonItom::PyFilter, METH_VARARGS | METH_KEYWORDS, pyFilter_doc},
    {"filterHelp", (PyCFunction)PythonItom::PyFilterHelp, METH_VARARGS | METH_KEYWORDS, pyFilterHelp_doc},
    {"widgetHelp", (PyCFunction)PythonItom::PyWidgetHelp, METH_VARARGS | METH_KEYWORDS, pyWidgetHelp_doc},
    {"pluginHelp", (PyCFunction)PythonItom::PyPluginHelp, METH_VARARGS | METH_KEYWORDS, pyPluginHelp_doc},
    {"aboutInfo", (PyCFunction)PythonItom::PyAboutInfo, METH_VARARGS | METH_KEYWORDS, pyAboutInfo_doc},
    {"pluginLoaded", (PyCFunction)PythonItom::PyPluginLoaded, METH_VARARGS, pyPluginLoaded_doc},
    {"plotHelp", (PyCFunction)PythonItom::PyPlotHelp, METH_VARARGS | METH_KEYWORDS, pyPlotHelp_doc},
    {"plotLoaded", (PyCFunction)PythonItom::PyPlotLoaded, METH_VARARGS, pyPlotLoaded_doc},
    {"version", (PyCFunction)PythonItom::PyItomVersion, METH_VARARGS | METH_KEYWORDS, pyItomVersion_doc},
    {"saveDataObject", (PyCFunction)PythonItom::PySaveDataObject, METH_VARARGS | METH_KEYWORDS, pySaveDataObject_doc},
    {"loadDataObject", (PyCFunction)PythonItom::PyLoadDataObject, METH_VARARGS | METH_KEYWORDS, pyLoadDataObject_doc},
	{"setCentralWidgetsSizes", (PyCFunction)PythonItom::PySetCentralWidgetsSizes, METH_VARARGS | METH_KEYWORDS, pySetCentralWidgetsSizes_doc},
    {"addButton", (PyCFunction)PythonItom::PyAddButton, METH_VARARGS | METH_KEYWORDS, pyAddButton_doc},
    {"removeButton", (PyCFunction)PythonItom::PyRemoveButton, METH_VARARGS, pyRemoveButton_doc},
    {"addMenu", (PyCFunction)PythonItom::PyAddMenu, METH_VARARGS | METH_KEYWORDS, pyAddMenu_doc},
    {"removeMenu", (PyCFunction)PythonItom::PyRemoveMenu, METH_VARARGS | METH_KEYWORDS, pyRemoveMenu_doc},
	{"dumpButtonsAndMenus", (PyCFunction)PythonItom::PyDumpMenusAndButtons, METH_NOARGS, pyDumpButtonsAndMenus_doc},
    {"saveMatlabMat", (PyCFunction)PythonItom::PySaveMatlabMat, METH_VARARGS, pySaveMatlabMat_doc},
    {"loadMatlabMat", (PyCFunction)PythonItom::PyLoadMatlabMat, METH_VARARGS, pyLoadMatlabMat_doc},
    {"scaleValueAndUnit", (PyCFunction)PythonItom::scaleValueAndUnit, METH_VARARGS | METH_KEYWORDS, scaleValueAndUnit_doc},
    {"getDefaultScaleableUnits", (PyCFunction)PythonItom::getDefaultScaleableUnits, METH_NOARGS, getDefaultScaleableUnits_doc},
    {"getAppPath", (PyCFunction)PythonItom::getAppPath, METH_NOARGS, getAppPath_doc},
    {"getQtToolPath", (PyCFunction)PythonItom::getQtToolPath, METH_VARARGS, getQtToolPath_doc },
    {"getCurrentPath", (PyCFunction)PythonItom::getCurrentPath, METH_NOARGS, getCurrentPath_doc},
    {"setCurrentPath", (PyCFunction)PythonItom::setCurrentPath, METH_VARARGS, setCurrentPath_doc},
    {"checkSignals", (PyCFunction)PythonItom::PyCheckSignals, METH_NOARGS, pyCheckSignals_doc},
    {"processEvents", (PyCFunction)PythonItom::PyProcessEvents, METH_NOARGS, pyProcessEvents_doc},
    {"getDebugger", (PyCFunction)PythonItom::PyGetDebugger, METH_NOARGS, pyGetDebugger_doc},
    {"gcStartTracking", (PyCFunction)PythonItom::PyGCStartTracking, METH_NOARGS, pyGCStartTracking_doc},
    {"gcEndTracking", (PyCFunction)PythonItom::PyGCEndTracking, METH_NOARGS, pyGCEndTracking_doc},
    {"getScreenInfo", (PyCFunction)PythonItom::PyGetScreenInfo, METH_NOARGS, getScreenInfo_doc},
    {"setApplicationCursor", (PyCFunction)PythonItom::setApplicationCursor, METH_VARARGS, setApplicationCursor_doc},
    {"loadIDC", (PyCFunction)PythonItom::PyLoadIDC, METH_VARARGS | METH_KEYWORDS, pyLoadIDC_doc},
    {"saveIDC", (PyCFunction)PythonItom::PySaveIDC, METH_VARARGS | METH_KEYWORDS, pySaveIDC_doc},
    {"compressData", (PyCFunction)PythonItom::compressData, METH_VARARGS, pyCompressData_doc},
    {"uncompressData", (PyCFunction)PythonItom::uncompressData, METH_VARARGS, pyUncompressData_doc},
    {"userIsAdmin", (PyCFunction)PythonItom::userCheckIsAdmin, METH_NOARGS, pyCheckIsAdmin_doc},
    {"userIsDeveloper", (PyCFunction)PythonItom::userCheckIsDeveloper, METH_NOARGS, pyCheckIsDeveloper_doc},
    {"userIsUser", (PyCFunction)PythonItom::userCheckIsUser, METH_NOARGS, pyCheckIsUser_doc},
    {"userGetInfo", (PyCFunction)PythonItom::userGetUserInfo, METH_NOARGS, pyGetUserInfo_doc},
    {"autoReloader", (PyCFunction)PythonItom::PyAutoReloader, METH_VARARGS | METH_KEYWORDS, autoReloader_doc},
    {"clc", (PyCFunction)PythonItom::PyClearCommandLine, METH_NOARGS, pyClearCommandLine_doc},
    {"getPalette", (PyCFunction)PythonItom::PyGetPalette, METH_VARARGS, getPalette_doc},
    {"setPalette", (PyCFunction)PythonItom::PySetPalette, METH_VARARGS | METH_KEYWORDS, setPalette_doc},
    {"getPaletteList", (PyCFunction)PythonItom::PyGetPaletteList, METH_VARARGS, getPaletteList_doc},
    {"clearAll", (PyCFunction)PythonItom::PyClearAll, METH_NOARGS, pyClearAll_doc},
    {"registerResource", (PyCFunction)PythonItom::PyRegisterResource, METH_VARARGS | METH_KEYWORDS, pyRegisterResources_doc},
    {"unregisterResource", (PyCFunction)PythonItom::PyUnregisterResource, METH_VARARGS | METH_KEYWORDS, pyUnregisterResources_doc},
    {NULL, NULL, 0, NULL}
};

PyModuleDef PythonItom::PythonModuleItom = {
    PyModuleDef_HEAD_INIT, "itom", NULL, -1, PythonItom::PythonMethodItom,
    NULL, NULL, NULL, NULL
};

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
/*static*/ ito::FuncWeakRef* PythonItom::hashButtonOrMenuCode(PyObject *code, PyObject *argtuple, ito::RetVal &retval, QString &codeString)
{
    ito::FuncWeakRef *ref = NULL;

    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine
    
    if (pyEngine)
    {
        if (!code)
        {
            retval += ito::RetVal(ito::retError, 0, QObject::tr("No code is given").toLatin1().data());
        }
        else if (PyMethod_Check(code) || PyFunction_Check(code))
        {
            
            PyObject *arguments = PyTuple_New(1);
            Py_INCREF(code);
            PyTuple_SetItem(arguments,0,code); //steals ref
            ito::PythonProxy::PyProxy *proxy = (ito::PythonProxy::PyProxy*)PyObject_CallObject((PyObject *) &PythonProxy::PyProxyType, arguments); //new ref
            Py_DECREF(arguments);
                        
            if (proxy)
            {
                if (argtuple)
                {
                    if (PyTuple_Check(argtuple))
                    {
                        Py_INCREF(argtuple);
                    }
                    else if(PyList_Check(argtuple)) //list
                    {
                        argtuple = PyList_AsTuple(argtuple); //returns new reference
                    }
                    else
                    {
                        retval += RetVal(retError, 0, QObject::tr("Given argument must be of type tuple or list.").toLatin1().data());
                        Py_DECREF(proxy);
                    }
                }

                if (!retval.containsError())
                {
                    size_t funcID = (++(pyEngine->m_pyFuncWeakRefAutoInc));
                    codeString = PythonEngine::fctHashPrefix + QString::number(funcID);

                    ref = &(pyEngine->m_pyFuncWeakRefHashes.insert(funcID, ito::FuncWeakRef(proxy, argtuple))).value();

                    Py_DECREF(proxy);
                    Py_XDECREF(argtuple);
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Proxy object of function or method could not be created.").toLatin1().data());
            }
        }
        else
        {
            bool ok;
            codeString = PythonQtConversion::PyObjGetString(code, true, ok);
            if (!ok)
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Code is no function or method call and no executable code string").toLatin1().data());
            }
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, QObject::tr("Python engine is not available").toLatin1().data());
    }

    return ref;
}

//-------------------------------------------------------------------------------------
//this method removes a possible proxy object to a python method or function including its possible argument tuple
//from the hash list that has been added there for guarding functions or objects related to toolbar buttons or menu entries.
/*static*/ ito::RetVal PythonItom::unhashButtonOrMenuCode(const size_t &handle)
{
    ito::RetVal retval;

    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine
    
    if (pyEngine)
    {
        //check if hashValue is in m_pyFuncWeakRefHashes and delete it and all hashValues which start with the given hashValue (hence its childs)
        QHash<size_t, FuncWeakRef >::iterator it = pyEngine->m_pyFuncWeakRefHashes.begin();

        while(it != pyEngine->m_pyFuncWeakRefHashes.end())
        {
            if (it->getHandle() == handle)
            {
                it = pyEngine->m_pyFuncWeakRefHashes.erase(it);
            }
            else
            {
                ++it;
            }
        }

        return ito::retOk;
    }

    return ito::RetVal(ito::retError, 0, QObject::tr("Python engine is not available").toLatin1().data());
}

//-------------------------------------------------------------------------------------
/*static*/ ito::RetVal PythonItom::unhashButtonOrMenuCode(const ito::FuncWeakRef *funcWeakRef)
{
    ito::RetVal retval;

    if (!funcWeakRef) return retval;

    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine
    
    if (pyEngine)
    {
        //check if hashValue is in m_pyFuncWeakRefHashes and delete it and all hashValues which start with the given hashValue (hence its childs)
        QHash<size_t, FuncWeakRef >::iterator it = pyEngine->m_pyFuncWeakRefHashes.begin();

        while(it != pyEngine->m_pyFuncWeakRefHashes.end())
        {
            if (&(*it) == funcWeakRef)
            {
                it = pyEngine->m_pyFuncWeakRefHashes.erase(it);
            }
            else
            {
                ++it;
            }
        }

        return ito::retOk;
    }

    return ito::RetVal(ito::retError, 0, QObject::tr("Python engine is not available").toLatin1().data());
}

} //end namespace ito

//-------------------------------------------------------------------------------------
