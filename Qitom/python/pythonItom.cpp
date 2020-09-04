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
PyDoc_STRVAR(pyOpenEmptyScriptEditor_doc,"scriptEditor() -> opens new, empty script editor window (undocked)");
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyNewScript_doc, "newScript() -> opens an empty, new script in the current script window.\n\
\n\
Creates a new itom script in the latest opened editor window.");
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyOpenScript_doc,"openScript(filename) -> open the given script in current script window.\n\
\n\
Open the python script indicated by *filename* in a new tab in the current, latest opened editor window. \n\
Filename can be either a string with a relative or absolute filename to the script to open or any object \n\
with a `__file__` attribute. This attribute is then read and used as path. \n\
\n\
The relative filename is relative with respect to the current directory. \n\
\n\
Parameters \n\
----------- \n\
filename : {str} or {obj} \n\
    Relative or absolute filename to a python script that is then opened (in the current editor window). Alternatively an object with a `__file__` attribute is allowed.");
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyShowHelpViewer_doc, "showHelpViewer(collectionFile = '') -> open the user documentation in the help viewer.\n\
\n\
The user documentation is shown in an external help viewer. Optionally, it is possible to load a user-defined collection file \n\
in this help viewer.\n\
\n\
Parameters \n\
----------- \n\
collectionFile : {str} \n\
	If given, the indicated collectionFile will be loaded in the help viewer. Per default, the user documentation is loaded (pass an empty string or nothing).");
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

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonItom::PyClearCommandLine(PyObject *pSelf)
{
    PythonEngine *pyEngine = PythonEngine::instance; //works since pythonItom is friend with pythonEngine
    if (pyEngine)
    {
        emit pyEngine->clearCommandLine();
    }

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotImage_doc,"plot(data, className = '', properties = {}) -> plots a dataObject, pointCloud or polygonMesh in a new figure \n\
\n\
Plots an existing dataObject, pointCloud or polygonMesh in a dockable, not blocking window. \n\
The style of the plot depends on the object dimensions.\n\
\n\
If no 'className' is given, the type of the plot is chosen depending on the type and the size \n\
of the object. The defaults for several plot classes can be adjusted in the property dialog of itom. \n\
\n\
You can also set a class name of your preferred plot plugin (see also property dialog of itom). \n\
If your preffered plot is not able to display the given object, a warning is returned and the default \n\
plot type is used again. For dataObjects, it is also possible to simply set 'className' to '1D', '2D' \n\
or '2.5D' in order to choose the default plot type depending on these aliases. For pointCloud and \n\
polygonMesh only the alias '2.5D' is valid. \n\
\n\
Every plot has several properties that can be configured in the Qt Designer (if the plot is embedded in a GUI), \n\
or by the property toolbox in the plot itself or by using the info() method of the corresponding itom.uiItem instance. \n\
\n\
Use the 'properties' argument to pass a dictionary with properties you want to set. \n\
\n\
Parameters \n\
----------- \n\
data : {DataObject, PointCloud, PolygonMesh} \n\
    Is the data object, point cloud or polygonal mesh, that will be plotted.\n\
className : {str}, optional \n\
    class name of desired plot (if not indicated or if the className can not be found, the default plot will be used (see application settings)) \n\
	Depending on the object, you can also use '1D', '2D' or '2.5D' for displaying the object in the default plot of \n\
	the indicated categories. If nothing is given, the plot category is guessed from 'data'.\n\
properties : {dict}, optional \n\
    optional dictionary of properties that will be directly applied to the plot widget. \n\
\n\
Returns \n\
-------- \n\
index : {int} \n\
    This index is the figure index of the plot figure that is opened by this command. Use *figure(index)* to get a reference to the \n\
    figure window of this plot. The plot can be closed by 'close(index)'. \n\
plotHandle: {plotItem} \n\
    Handle of the plot. This handle is used to control the properties of the plot, connect to its signals or call slots of the plot. \n\
\n\
See Also \n\
---------- \n\
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlot1d_doc, "plot1(data, xData = None, className = '', properties = {}) -> plots a dataObject as an 1d plot in a new figure \n\
\n\
Plots an existing dataObject in a dockable, not blocking window. \n\
\n\
If a xData is given, the plot uses this vector for the values of the x axis of the plot.\n\
\n\
The plot type of this function is '1D'.\n\
\n\
Every plot has several properties that can be configured in the Qt Designer (if the plot is embedded in a GUI), \n\
or by the property toolbox in the plot itself or by using the info() method of the corresponding itom.uiItem instance. \n\
\n\
Use the 'properties' argument to pass a dictionary with properties you want to set. \n\
\n\
Parameters \n\
----------- \n\
data : {DataObject} \n\
    Is the data object whose region of interest will be plotted.\n\
xData : {DataObject}, optional \n\
    Is the data object whose values are used for the axis.\n\
className : {str}, optional \n\
    class name of the desired 1D plot (if not indicated default plot will be used, see application settings) \n\
properties : {dict}, optional \n\
    optional dictionary of properties that will be directly applied to the plot widget. \n\
\n\
Returns \n\
-------- \n\
index : {int} \n\
    This index is the figure index of the plot figure that is opened by this command. Use *figure(index)* \n\
    to get a reference to the figure window of this plot. The plot can be closed by 'close(index)'. \n\
plotHandle: {plotItem} \n\
    Handle of the plot. This handle is used to control the properties of the plot, connect to its signals or call slots of the plot. \n\
\n\
See Also \n\
---------- \n\
liveImage, plotItem, plot, plot2, plot25");

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlot2d_doc, "plot2(data, properties = {}) -> plots a dataObject in a new figure \n\
\n\
Plots an existing dataObject in a dockable, not blocking window. \n\
The style of the plot depends on the object dimensions.\n\
\n\
The plot type of this function is '2D'.\n\
\n\
Every plot has several properties that can be configured in the Qt Designer (if the plot is embedded in a GUI), \n\
or by the property toolbox in the plot itself or by using the info() method of the corresponding itom.uiItem instance. \n\
\n\
Use the 'properties' argument to pass a dictionary with properties you want to set to a certain value. \n\
\n\
Parameters \n\
----------- \n\
data : {DataObject} \n\
    Is the data object whose region of interest will be plotted.\n\
className : {str}, optional \n\
    class name of the desired 2D plot (if not indicated default plot will be used, see application settings) \n\
properties : {dict}, optional \n\
    optional dictionary of properties that will be directly applied to the plot widget. \n\
\n\
Returns \n\
-------- \n\
index : {int} \n\
    This index is the figure index of the plot figure that is opened by this command. Use *figure(index)* to get a \n\
    reference to the figure window of this plot. The plot can be closed by 'close(index)'. \n\
plotHandle: {plotItem} \n\
    Handle of the plot. This handle is used to control the properties of the plot, connect to its signals or call slots of the plot. \n\
\n\
See Also \n\
---------- \n\
liveImage, plotItem, plot, plot1, plot25");

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlot25d_doc, "plot25(data, className = '', properties = {}) -> plots a dataObject, pointCloud or polygonMesh in a new figure \n\
\n\
Plots an existing dataObject, pointCloud or polygonMesh in a dockable, not blocking window. \n\
The style of the plot depends on the object dimensions.\n\
\n\
The plot type of this function is '2.5D'.\n\
\n\
Every plot has several properties that can be configured in the Qt Designer (if the plot is embedded in a GUI), \n\
or by the property toolbox in the plot itself or by using the info() method of the corresponding itom.uiItem instance. \n\
\n\
Use the 'properties' argument to pass a dictionary with properties you want to set to a certain value. \n\
\n\
Parameters \n\
----------- \n\
data : {DataObject, PointCloud, PolygonMesh} \n\
    Is the data object whose region of interest will be plotted.\n\
className : {str}, optional \n\
    class name of the desired 2.5D plot (if not indicated default plot will be used, see application settings) \n\
properties : {dict}, optional \n\
    optional dictionary of properties that will be directly applied to the plot widget. \n\
\n\
Returns \n\
-------- \n\
index : {int} \n\
    This index is the figure index of the plot figure that is opened by this command. Use *figure(index)* to get a \n\
    reference to the figure window of this plot. The plot can be closed by 'close(index)'. \n\
plotHandle: {plotItem} \n\
    Handle of the plot. This handle is used to control the properties of the plot, connect to its signals or call slots of the plot. \n\
\n\
See Also \n\
---------- \n\
liveImage, plotItem, plot, plot1, plot2");

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyLiveImage_doc,"liveImage(cam, className = '', properties = {}) -> show a camera live image in a new figure\n\
\n\
Creates a plot-image (2D) and automatically grabs images into this window.\n\
This function is not blocking.\n\
\n\
If no 'className' is given, the type of the plot is chosen depending on the type and the size \n\
of the object. The defaults for several plot classes can be adjusted in the property dialog of itom. \n\
\n\
You can also set a class name of your preferred plot plugin (see also property dialog of itom). \n\
If your preferred plot is not able to display the given object, a warning is returned and the default \n\
plot type is used again. For dataObjects, it is also possible to simply set 'className' to '1D' or '2D' \n\
in order to choose the default plot type depending on these aliases. \n\
\n\
Every plot has several properties that can be configured in the Qt Designer (if the plot is embedded in a GUI), \n\
or by the property toolbox in the plot itself or by using the info() method of the corresponding itom.uiItem instance. \n\
\n\
Use the 'properties' argument to pass a dictionary with properties you want to set to a certain value. \n\
\n\
Parameters \n\
----------- \n\
cam : {dataIO-Instance} \n\
    Camera grabber device from which images are acquired.\n\
className : {str}, optional \n\
    class name of desired plot (if not indicated or if the className can not be found, the default plot will be used (see application settings) \n\
properties : {dict}, optional \n\
    optional dictionary of properties that will be directly applied to the plot widget. \n\
\n\
Returns \n\
------- \n\
index : {int} \n\
    This index is the figure index of the plot figure that is opened by this command. Use *figure(index)* to get a reference to the figure window of this live image plot. The plot can be closed by 'close(index)'. \n\
plotHandle: {plotItem} \n\
    Handle of the live image plot. This handle is used to control the properties of the plot, connect to its signals or call slots of the plot. \n\
\n\
See Also \n\
--------- \n\
plot, plotItem");
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

//----------------------------------------------------------------------------------------------------------------------------------
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

    ito::RetVal retval = 0;
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

        for (int n = 0; n < keyList.size(); n++)    // get the longest name in this list
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
//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFilterHelp_doc, "filterHelp(filterName = '', dictionary = 0, furtherInfos = 0) -> generates an online help for the given filter(s). \n\
\n\
This method prints information about one specific filter (algorithm) or a list of filters to the console output. If one specific filter, defined \
in an algorithm plugin can be found that case-sensitively fits the given filterName its full documentation is printed. Else, a list of filters \
is printed whose name contains the given filterName.\n\
\n\
Parameters \n\
----------- \n\
filterName : {str}, optional \n\
    is the fullname or a part of any filter-name which should be displayed. \n\
    If filterName is empty or no filter matches filterName (case sensitive) a list with all suitable filters is given. \n\
dictionary : {dict}, optional \n\
    if dictionary == 1, a dictionary with all relevant components of the filter's documentation is returned and nothing is printed to the command line [default: 0] \n\
furtherInfos : {int}, optional \n\
    Usually, filters or algorithms whose name only contains the given filterName are only listed at the end of the information text. \n\
    If this parameter is set to 1 [default: 0], the full information for all these filters is printed as well. \n\
\n\
Returns \n\
------- \n\
out : {None or dict} \n\
    In its default parameterization this method returns None. Depending on the parameter dictionary it is also possible that this method \
    returns a dictionary with the single components of the information text.");

PyObject* PythonItom::PyFilterHelp(PyObject* /*pSelf*/, PyObject* pArgs, PyObject *pKwds)
{
    return PyWidgetOrFilterHelp(false, pArgs, pKwds);
}

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyWidgetHelp_doc,"widgetHelp(filterName = '', dictionary = 0, furtherInfos = 0) -> generates an online help for the given widget(s). \n\
\n\
This method prints information about one specific widget or a list of widgets to the console output. If one specific widget, defined \
in an algorithm plugin can be found that case-sensitively fits the given widgetName its full documentation is printed. Else, a list of widgets \
is printed whose name contains the given widgetName.\n\
\n\
Parameters \n\
----------- \n\
widgetName : {str}, optional \n\
    is the fullname or a part of any widget-name which should be displayed. \n\
    If widgetName is empty or no widget matches widgetName (case sensitive) a list with all suitable widgets is given. \n\
dictionary : {dict}, optional \n\
    if dictionary == 1, a dictionary with all relevant components of the widget's documentation is returned and nothing is printed to the command line [default: 0] \n\
furtherInfos : {int}, optional \n\
    Usually, widgets whose name only contains the given widgetName are only listed at the end of the information text. \n\
    If this parameter is set to 1 [default: 0], the full information for all these widgets is printed as well. \n\
\n\
Returns \n\
------- \n\
out : {None or dict} \n\
    In its default parameterization this method returns None. Depending on the parameter dictionary it is also possible that this method \
    returns a dictionary with the single components of the information text.");
PyObject* PythonItom::PyWidgetHelp(PyObject* /*pSelf*/, PyObject* pArgs, PyObject *pKwds)
{
    return PyWidgetOrFilterHelp(true, pArgs, pKwds);
}

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginLoaded_doc,"pluginLoaded(pluginName) -> check if a certain plugin could be successfully loaded.\n\
\n\
Checks if a specified plugin is loaded and returns the result as a boolean expression. \n\
\n\
Parameters \n\
----------- \n\
pluginName :  {str} \n\
    The name of a specified plugin as usually displayed in the plugin window.\n\
\n\
Returns \n\
------- \n\
result : {bool} \n\
    True, if the plugin has been loaded and can be used, else False.");
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

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotLoaded_doc,"plotLoaded(plotName) -> check if a certain plot widget is loaded.\n\
\n\
Checks if a specified plot widget is loaded and returns the result as a boolean expression. \n\
\n\
Parameters \n\
----------- \n\
pluginName :  {str} \n\
    The name of a specified plot widget as displayed in the preferences window.\n\
\n\
Returns \n\
------- \n\
result : {bool} \n\
    True, if the plot has been loaded and can be used, else False.");
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

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotHelp_doc,"plotHelp(plotName = '', dictionary = False) -> generates an online help for the specified plot.\n\
Gets (also print to console) the available slots / properties of the plot specified by plotName (str, as specified in the properties window).\n\
\n\
Parameters \n\
----------- \n\
plotName : {str} \n\
    is the fullname of a plot as specified in the properties window (case insensitive).\n\
    if nothing, '*' or an empty string is given, a list of available widgets is returned.\n\
dictionary : {bool}, optional \n\
    if True, this methods returns a dict with plot slots and properties and does not print anything to the console (default: False)\n\
\n\
Returns \n\
------- \n\
out : {None or dict} \n\
    Returns None or a dict depending on the value of parameter `dictionary`.");
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
            PyErr_SetString(PyExc_RuntimeError, "figure not found");
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

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginHelp_doc,"pluginHelp(pluginName = '', dictionary = False) -> generates an online help for the specified plugin.\n\
                              Gets (also print to console) the initialisation parameters of the plugin specified pluginName (str, as specified in the plugin window).\n\
If `dictionary == True`, a dict with all plugin parameters is returned and nothing is printed to the console.\n\
\n\
Parameters \n\
----------- \n\
pluginName : {str} \n\
    is the fullname of a plugin as specified in the plugin window.\n\
dictionary : {bool}, optional \n\
    if `dictionary == True`, function returns a dict with plugin parameters and does not print anything to the console (default: False)\n\
\n\
Returns \n\
------- \n\
out : {None or dict} \n\
    Returns None or a dict depending on the value of parameter `dictionary`.");
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

    ito::RetVal retval = 0;
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
//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyAboutInfo_doc,"aboutInfo(pluginName) -> returns the about information for the specified plugin.\n\
\n\
Parameters \n\
----------- \n\
pluginName : {str} \n\
    is the fullname of a plugin as specified in the plugin window.\n\
\n\
Returns \n\
------- \n\
out : {None or dict} \n\
    Returns a string containing the about information.");
PyObject* PythonItom::PyAboutInfo(PyObject* /*pSelf*/, PyObject* pArgs)
{
    const char* pluginName = NULL;

    if (!PyArg_ParseTuple(pArgs, "s", &pluginName))
    {
        return NULL;
    }

    QVector<ito::Param> *paramsMand = NULL;

    ito::RetVal retval = ito::retOk;

    QString version;
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

    retval = AIM->getAboutInfo(pluginName, version);
    if (!retval.containsError())
    {
        return PythonQtConversion::QStringToPyObject(version);
    }
    return NULL;
}
//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyITOMVersion_doc,"version(returnDict = False, addPluginInfos = False) -> retrieve complete information about itom version numbers\n\
\n\
Parameters \n\
----------- \n\
toggle-output : {bool}, optional\n\
    default = false\n\
    if true, output will be written to a dictionary else to console.\n\
addPluginInfos : {bool}, optional \n\
    default = false\n\
    if true, add informations about plugin versions.\n\
\n\
Returns \n\
------- \n\
None (display outPut) or PyDictionary with version information.\n\
\n\
Notes \n\
----- \n\
\n\
Retrieve complete version information of itom and if specified version information of loaded plugins\n\
and print it either to the console or to a PyDictionary.");
PyObject* PythonItom::PyITOMVersion(PyObject* /*pSelf*/, PyObject* pArgs)
{
    bool toggleOut = false;
    bool addPlugIns = false;

    if (!PyArg_ParseTuple(pArgs, "|bb", &toggleOut, &addPlugIns))
    {
        return NULL;
    }

    PyObject* myDic = PyDict_New();
    PyObject* myTempDic = PyDict_New();
    PyObject* key = NULL;
    PyObject* value = NULL;

    QMap<QString, QString> versionMap = ito::getItomVersionMap();
    QMapIterator<QString, QString> i(versionMap);

    while (i.hasNext()) 
    {
        i.next();

        key = PythonQtConversion::QStringToPyObject(i.key());
        value = PythonQtConversion::QStringToPyObject(i.value());
        PyDict_SetItem(myTempDic, key, value);

        Py_DECREF(key);
        Py_DECREF(value);
    }

    PyDict_SetItemString(myDic, "itom", myTempDic);
    Py_XDECREF(myTempDic);

    if (addPlugIns)
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

    if (toggleOut)
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

            std::cout << key.toLatin1().toUpper().data() << ":\n";

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

                std::cout << subKey.toLatin1().data() <<"\t" << subVal.toLatin1().data() << "\n";

            }

            Py_XDECREF(mySubKeys);
            
        }

        
        Py_DECREF(myKeys);

        Py_DECREF(myDic);
        Py_RETURN_NONE;
    }

}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyAddButton_doc,"addButton(toolbarName, buttonName, code, icon = '', argtuple = []) -> adds a button to a toolbar in the main window \n\
\n\
This function adds a button to a toolbar in the main window. If the button is pressed the given code, function or method is executed. \n\
If the toolbar specified by 'toolbarName' does not exist, it is created. The button will show the optional icon, or if not given or not \n\
loadable, 'buttonName' is displayed as text. \n\
\n\
itom comes with basic icons addressable by ':/../iconname.png', e.g. ':/gui/icons/close.png'. These natively available icons are listed \n\
in the icon-browser in the menu 'edit >> iconbrowser' of any script window. Furthermore you can give a relative or absolute path to \n\
any allowed icon file (the preferred file format is png). \n\
\n\
Parameters \n\
----------- \n\
toolbarName : {str} \n\
    The name of the toolbar.\n\
buttonName : {str} \n\
    The name and identifier of the button to create.\n\
code : {str, method, function}\n\
    The code to be executed if the button is pressed.\n\
icon : {str}, optional \n\
    The filename of an icon-file. This can also be relative to the application directory of 'itom'.\n\
argtuple : {tuple}, optional \n\
    Arguments, which will be passed to the method (in order to avoid cyclic references try to only use basic element types). \n\
\n\
Returns \n\
------- \n\
handle : {int} \n\
    handle to the newly created button (pass it to removeButton to delete exactly this button) \n\
\n\
Raises \n\
------- \n\
RuntimeError : \n\
    if the main window is not available \n\
\n\
See Also \n\
--------- \n\
removeButton()");
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
        //PyErr_SetString(PyExc_TypeError, "wrong length or type of arguments. Type help(addMenu) for more information.");
        //return NULL;
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

    QSharedPointer<size_t> buttonHandle(new size_t); //this is the handle to the newly created button, this can be used to delete the button afterwards (it corresponds to the pointer address of the corresponding QAction, casted to size_t)

    if (!retValue.containsError())
    {
        QObject *mainWindow = AppManagement::getMainWindow();
        if (mainWindow)
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QMetaObject::invokeMethod(mainWindow, "addToolbarButton", Q_ARG(QString, toolbarName), Q_ARG(QString, qname), Q_ARG(QString, qicon), Q_ARG(QString, qcode), Q_ARG(QSharedPointer<size_t>, buttonHandle), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyRemoveButton_doc,"removeButton(handle | toolbarName, buttonName = '') -> removes a button from a given toolbar. \n\
\n\
This method removes an existing button from a toolbar in the main window of 'itom'. This button must have been \n\
created using `addButton`. If the toolbar is empty after the removal, it is finally deleted. \n\
\n\
Pass either the 'handle' parameter of both 'toolbarName' and 'buttonName'. It is more precise to use the handle in order to exactly \n\
delete the button that has been created by a call to `addButton`. Using the names of the toolbar and the button always delete any \n\
button that has been created using this data. \n\
\n\
Parameters \n\
----------- \n\
handle : {int} \n\
    The handle returned by addButton(). \n\
toolbarName : {str} \n\
    The name of the toolbar.\n\
buttonName : {str} \n\
    The name (str, identifier) of the button to remove (only necessary, if toolbarName is given instead of handle).\n\
\n\
Raises \n\
------- \n\
RuntimeError : \n\
    if the main window is not available or the given button could not be found. \n\
\n\
See Also \n\
--------- \n\
addButton()");
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
            PyErr_SetString(PyExc_TypeError, "Wrong length or type of arguments. Type help(removeButton) for more information.");
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
            QMetaObject::invokeMethod(mainWindow, "removeToolbarButton", Q_ARG(QString, toolbarName), Q_ARG(QString, buttonName), Q_ARG(QSharedPointer<size_t>, buttonHandle_), Q_ARG(bool, false), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        }
        else
        {
            QMetaObject::invokeMethod(mainWindow, "removeToolbarButton", Q_ARG(size_t, buttonHandle), Q_ARG(bool, false), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyAddMenu_doc,"addMenu(type, key, name = <last_section_of_key>, code = '', icon = '', argtuple = []) -> adds an element to the menu bar of itom. \n\
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
itom comes with basic icons addressable by ':/../iconname.png', e.g. ':/gui/icons/close.png'. These natively available icons are listed \n\
in the icon-browser in the menu 'edit >> iconbrowser' of any script window. Furthermore you can give a relative or absolute path to \n\
any allowed icon file (the preferred file format is png). \n\
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
Returns \n\
------- \n\
handle : {int} \n\
    Handle to the recently added leaf node (action, separator or menu item). Use this handle to delete the item including its child items (for type 'menu'). \n\
\n\
Raises \n\
------- \n\
RuntimeError : \n\
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
                    retValue += RetVal(retError,0,QObject::tr("For menu elements of type 'BUTTON' any type of code (String or callable method or function) must be indicated.").toLatin1().data());
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
            QMetaObject::invokeMethod(mainWindow, "addMenuElement", Q_ARG(int, type), Q_ARG(QString, qkey), Q_ARG(QString, qname), Q_ARG(QString, qcode), Q_ARG(QString, qicon), Q_ARG(QSharedPointer<size_t>, menuHandle), Q_ARG(bool, false), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyRemoveMenu_doc,"removeMenu(key | menuHandle) -> remove a menu element with the given key or handle. \n\
\n\
This function remove a menu element with the given key or menuHandle. \n\
key is a slash separated list. The sub-components then \n\
lead the way to the final element, which should be removed. \n\
\n\
Alternatively, it is possible to pass the handle obtained by `addMenu`. \n\
\n\
Parameters \n\
----------- \n\
key : {str}, optional\n\
    The name (str, identifier) of the menu entry to remove.\n\
handle : {int}, optional \n\
    The handle of the menu entry that should be removed (including its possible child items). \n\
\n\
Raises \n\
------- \n\
RuntimeError : \n\
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
            QMetaObject::invokeMethod(mainWindow, "removeMenuElement", Q_ARG(QString, qkey), Q_ARG(QSharedPointer<QVector<size_t> >, removedMenuHandles), Q_ARG(bool, false), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        }
        else
        {
            QMetaObject::invokeMethod(mainWindow, "removeMenuElement", Q_ARG(size_t, menuHandle), Q_ARG(QSharedPointer<QVector<size_t> >, removedMenuHandles), Q_ARG(bool, false), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDumpButtonsAndMenus_doc, "dumpButtonsAndMenus() -> returns a dictionary with the set of user-defined toolbars, buttons, menus and actions.");
/*static*/ PyObject* PythonItom::PyDumpMenusAndButtons(PyObject* pSelf)
{
	QObject *mainWindow = AppManagement::getMainWindow();
	if (mainWindow)
	{
		QSharedPointer<QString > dump(new QString());
		ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

		QMetaObject::invokeMethod(mainWindow, "dumpToolbarsAndButtons", Q_ARG(QSharedPointer<QString>, dump), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

		if (!locker->wait(2000))
		{
			PyErr_SetString(PyExc_RuntimeError, "Timeout.");
			return NULL;
		}
		else
		{
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

//----------------------------------------------------------------------------------------------------------------------------------
/*static */PyObject* PythonItom::PyCheckSignals(PyObject* /*pSelf*/)
{
    int result = PythonEngine::isInterruptQueued() ? 1 : 0; //PyErr_CheckSignals();
    return Py_BuildValue("i", result);
    //Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(autoReloader_doc,"autoReloader(enabled, checkFileExec = True, checkCmdExec = True, checkFctExec = False) -> dis-/enables the module to automatically reload changed modules \n\
\n\
Use this method to enable or disable (and configure) a tool that automatically tries to reload imported modules and their submodules if they have changed \n\
since the last run. \n\
\n\
Returns \n\
------- \n\
enable : {bool} \n\
    The auto-reload tool is loaded if it is enabled for the first time. If it is disabled, \n\
    it does not check changes of any imported modules. \n\
checkFileExec : {bool} \n\
    If True (default) and auto-reload enabled, a check for modifications is executed whenever a script is executed \n\
checkCmdExec : {bool} \n\
    If True (default) and auto-reload enabled, a check for modifications is executed whenever a command in the command line is executed \n\
checkFctExec : {bool} \n\
    If True and auto-reload enabled, a check for modifications is executed whenever a function or method is run (e.g. by an event or button click) (default: False)\n\
\n\
Notes \n\
------- \n\
This tool is inspired by and based on the IPython extension 'autoreload'. \n\
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(getScreenInfo_doc,"getScreenInfo() -> returns dictionary with information about all available screens. \n\
\n\
This method returns a dictionary with information about the current screen configuration of this computer. \n\
\n\
Returns \n\
------- \n\
screenInfo : {dict} \n\
    dictionary with the following content is returned: \n\
    \n\
    * screenCount (int): number of available screens \n\
    * primaryScreen (int): index (0-based) of primary screen \n\
    * geometry (tuple): tuple with dictionaries for each screen containing data for width (w), height (h) and its top-left-position (x, y)");
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
PyDoc_STRVAR(pySaveMatlabMat_doc,"saveMatlabMat(filename, values, matrixName = 'matrix') -> save strings, numbers, arrays or combinations into a Matlab mat file. \n\
\n\
Save one or multiple objects (strings, numbers, arrays, `dataObject`, `numpy.ndarray`...) to a Matlab *mat* file. \n\
There are the following possibilites for saving: \n\
\n\
* One given value is saved under one given 'matrixName' or 'matrix' if 'matrixName' is not given. \n\
* A list or tuple of objects is given. If no 'matrixName' is given, the items get the names 'matrix1', 'matrix2'... Else, 'matrixName' must be a sequence of value names with the same length than 'values'. \n\
* A dictionary is given, such that each value is stored under its corresponding key. \n\
\n\
Parameters \n\
----------- \n\
filename : {str} \n\
    Filename under which the file should be saved (.mat will be appended if not available)\n\
values : {dictionary, list, tuple, variant} \n\
    single value, dictionary, list or tuple with elements of type number, string, array (dataObject, numpy.ndarray...)\n\
matrix-name : {str, list, tuple}, optional \n\
    if 'values' is a single value, this parameter must be one single str, if 'values' is a sequence it must be a sequence of strings with the same length, if 'values' is a dictionary this argument is ignored. \n\
\n\
See Also \n\
---------- \n\
loadMatlabMat");
PyObject * PythonItom::PySaveMatlabMat(PyObject * /*pSelf*/, PyObject *pArgs)
{
    //if any arguments are changed in this method, consider to also change PythonEngine::saveMatlabVariables and PythonEngine::saveMatlabSingleParam.

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
    PyObject *res = PyObject_CallMethodObjArgs(scipyIoModule, nameobj, filename, saveDict, Py_True, fiveobj, Py_True, Py_False, rowobj, NULL); //new reference
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

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyLoadMatlabMat_doc,"loadMatlabMat(filename) -> loads Matlab mat-file by using scipy methods and returns the loaded dictionary. \n\
\n\
This function loads matlab mat-file by using scipy methods and returns the loaded dictionary. \n\
\n\
Parameters \n\
----------- \n\
filename : {str} \n\
    Filename from which the data will be imported (.mat will be added if not available)\n\
\n\
Returns \n\
------- \n\
mat : {dict} \n\
    dictionary with content of file \n\
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

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFilter_doc,"filter(name : str, *args, **kwds, _observer : progressObserver = None) -> invoke a filter (or algorithm) function from an algorithm-plugin. \n\
\n\
This function is used to invoke itom filter-functions or algorithms, declared within itom-algorithm plugins.\n\
The parameters (arguments) depends on the specific filter function (see filterHelp(name)),\n\
By filterHelp() a list of available filter functions is retrieved. \n\
\n\
Parameters \n\
----------- \n\
name : {str} \n\
    The name of the filter\n\
*args : {variant} \n\
    positional arguments for the specific filter-method \n\
**kwds : {variant} \n\
    keyword-based arguments for the specific filter-method. The argument name 'observer' is reserved for special use. \n\
_observer : {progressObserver, optional} \n\
    if the called filter implements the extended interface with progress and status information, an optional itom.progressObserver \n\
    object can be given (only as keyword-based parameter) which is then used as observer for the current progress of the filter \n\
    execution. It is then also possible to interrupt the execution earlier (depending on the implementation of the filter). \n\
    The observer object is reset() before passed to the called filter function (using the slot reset()). \n\
\n\
Returns \n\
-------- \n\
out : {variant} \n\
    The returned values depend on the definition of each filter. In general it is a tuple of all output parameters that are defined by the filter function.\n\
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

    //parses python-parameters with respect to the default values given py (*it).paramsMand and (*it).paramsOpt and returns default-initialized ParamBase-Vectors paramsMand and paramsOpt.
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

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pySaveDataObject_doc,"saveDataObject(filename, dataObject, tagsAsBinary = False) -> save a dataObject to harddrive in a xml-based file format. \n\
\n\
This method writes a `dataObject` into the file specified by 'filename'. The data is stored in a binary format within a xml-based structure. \n\
All string-tags of the dataObject are encoded in order to avoid xml-errors, the value of numerical tags are converted to string with \n\
15 significant digits (>32bit, tagsAsBinary = False [default]) or in a binary format (tagsAsBinary = True). \n\
\n\
Parameters \n\
----------- \n\
filename : {str} \n\
    Filename and Path of the destination (.ido will be added if no .*-ending is available)\n\
dataObject : {DataObject} \n\
    An allocated dataObject of n-Dimensions.\n\
tagsAsBinary : {bool}, optional \n\
    Optional tag to toggle if numeric-tags should be saved (metaData) as binary or by default as string.\n\
\n\
Notes \n\
----- \n\
Tagnames which contains special characters leads to XML-conflics. \n\
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
    int pyBool = 0;
    bool tagAsBin = false; // defaults metaData as string (false)

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "sO!|i", const_cast<char**>(kwlist), &folderfilename, &PythonDataObject::PyDataObjectType, &pyDataObject, &pyBool))
    {
        return NULL;
    }

    PythonDataObject::PyDataObject* elem = (PythonDataObject::PyDataObject*)pyDataObject;
    tagAsBin = pyBool > 0; 

    ret += ito::saveDOBJ2XML(elem->dataObject, folderfilename, false, tagAsBin);

    if (!PythonCommon::setReturnValueMessage(ret, "saveDataObject", PythonCommon::runFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyLoadDataObject_doc,"loadDataObject(filename, dataObject, doNotAppendIDO = False) -> load a dataObject from the harddrive. \n\
\n\
This function reads a `dataObject` from the file specified by filename. \n\
MetaData saveType (string, binary) are extracted from the file and restored within the object.\n\
\n\
Parameters \n\
----------- \n\
filename : {str} \n\
    Filename and Path of the destination (.ido will be added if not available)\n\
dataObject : {`dataObject`} \n\
    A pre-allocated `dataObject` (empty dataObject is allowed).\n\
doNotAppendIDO : {bool}, optional \n\
    False[default]: file suffix *.ido* will not be appended to filename, True: it will be added.\n\
\n\
Notes \n\
----- \n\
\n\
The value of string-Tags must be encoded to avoid XML-conflics.\n\
Tagnames which contains special characters leads to XML-conflics.");
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


//---------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pySetCentralWidgetsSizes_doc, "setCentralWidgetsSizes(sizes) -> set the sizes of the central widgets of itom (including command line) from top to bottom. \n\
\n\
This method can be important if at least one widget has been added from itom.ui, type ui.TYPECENTRALWIDGET. \n\
These user defined widgets are then added on top of the central area of itom and stacked above the command line. \n\
The list of sizes indicates the desired heights of all widgets in the center in pixel (from top to bottom). \n\
\n\
If the list contains too much items, all extra values are ignored. If the list contains too few values, the result \n\
is undefined, but the program will still be well-behaved. \n\
\n\
The overall size of the central area will not be affected. Instead, any additional/missing space is distributed amongst the \n\
widgets according to the relative weight of the sizes. \n\
\n\
If you speciy a size of 0, the widget will be invisible and can be made visible again using this method or by increasing its \n\
size again with the mouse. \n\
\n\
Parameters \n\
----------- \n\
sizes : {seq. of int} \n\
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



//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(getDefaultScaleableUnits_doc,"getDefaultScaleableUnits() -> Get a list with the strings of the standard scalable units. \n\
\n\
The unit strings returned as a list by this method can be transformed into each other using `scaleValueAndUnit`. \n\
\n\
Returns \n\
------- \n\
units : {list} \n\
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(scaleValueAndUnit_doc,"ScaleValueAndUnit(scaleableUnits, value, valueUnit) -> Scale a value and its unit and returns [value, 'Unit'] \n\
\n\
Parameters \n\
----------- \n\
scaleableUnits : {PyList of Strings} \n\
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
Rescale a value with SI-unit (e.g. 0.01 mm to 10 micrometer). Used together with itom.getDefaultScaleableUnits()");
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(getAppPath_doc,"getAppPath() -> returns absolute path of application base directory.\n\
\n\
This function returns the absolute path of application base directory.\n\
The return value is independent of the current working directory. \n\
\n\
Returns \n\
------- \n\
path : {str}\n\
    absolute path of this application's base directory");
PyObject* PythonItom::getAppPath(PyObject* /*pSelf*/)
{
    return PythonQtConversion::QStringToPyObject(QDir::cleanPath(QCoreApplication::applicationDirPath()));
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(getQtToolPath_doc, "getQtToolPath(toolname) -> get the absolute path of the Qt tool \n\
\n\
Parameters \n\
----------- \n\
toolname : {str} \n\
    The filename of the tool that should be searched (e.g. qcollectiongenerator; suffix is not required)\n\
\n\
Returns \n\
------- \n\
path : {str} \n\
    Absolute path to the Qt tool \n\
\n\
Raises \n\
------- \n\
FileExistsError : \n\
    if the given toolname could not be found \n\
");
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(getCurrentPath_doc,"getCurrentPath() -> returns absolute path of current working directory.\n\
\n\
Returns \n\
------- \n\
Path : {str}\n\
    absolute path of current working directory \n\
\n\
See Also \n\
---------- \n\
setCurrentPath");
PyObject* PythonItom::getCurrentPath(PyObject* /*pSelf*/)
{
    return PythonQtConversion::QStringToPyObject(QDir::cleanPath(QDir::currentPath()));
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(setCurrentPath_doc,"setCurrentPath(newPath) -> set current working directory to given absolute newPath \n\
\n\
sets the absolute path of the current working directory to 'newPath'. The current working directory is the base \n\
directory for all subsequent relative pathes of icon-files, script-files, ui-files, relative import statements... \n\
\n\
The current directory is always indicated in the right corner of the status bar of the main window. \n\
\n\
Parameters \n\
----------- \n\
newPath : {str} \n\
    The new working path of this application\n\
\n\
Returns \n\
------- \n\
success : {bool} \n\
    True in case of success else False \n\
\n\
See Also \n\
--------- \n\
getCurrentPath()");
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

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyRegisterResources_doc, "registerResource(rccFileName: str, mapRoot: str = "") -> Registers the resource with the given rccFileName. \n\
\n\
This method opens a given Qt rcc resource file and registers its content at the location \n\
in the resource tree specified by mapRoot. The mapRoot must usually be a slash separated \n\
path, starting with a slash. \n\\n\
\n\
To generate a rcc file, create an index of all files, that should be added to the resource file, \n\
in a qrc file and uses the rcc binary from Qt to compile the rcc file. \n\
\n\
This method is new in itom > 4.0.0. \n\
\n\
Parameters \n\
----------- \n\
rccFileName : {str}\n\
    filepath to the rcc resource file \n\
mapRoot : {str}, optional \n\
    root key, where the resources should be registered below (default: empty string) \n\
\n\
Returns \n\
---------- \n\
True if the file could be successfully opened, else False.\n\
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUnregisterResources_doc, "unregisterResource(rccFileName: str, mapRoot: str = "") -> Unregisters the resource with the given rccFileName. \n\
\n\
This method tries to unload all resources in the given rcc resource file from the location \n\
in the resource tree specified by `mapRoot`. The mapRoot must usually be a slash separated \n\
path, starting with a slash. \n\
\n\
\n\
This method is new in itom > 4.0.0. \n\
\n\
Parameters \n\
----------- \n\
rccFileName : {str}\n\
    filepath to the rcc resource file \n\
mapRoot : {str}, optional \n\
    root key, where the resources should be unloaded from (default: empty string). \n\
\n\
Returns \n\
---------- \n\
True if the file could be successfully opened, else False.\n\
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(setApplicationCursor_doc,"setApplicationCursor(cursorIndex = -1) -> changes the itom cursor or restores the previously set cursor if -1 \n\
\n\
This methods changes the overall cursor icon of itom where cursorIndex corresponds to the Qt enumeration Qt::CursorShape. e.g.:\n\
\n\
    * 0: Arrow \n\
    * 2: Cross Cursor \n\
    * 3: Wait Curson \n\
    * 13: Pointing Hand Cursor \n\
    * 14: Forbidden Cursor \n\
    * 16: Busy Cursor \n\
\n\
Parameters \n\
----------- \n\
cursorIndex : {int} optional\n\
    The cursor enumeration value of the desired cursor shape (Qt::CursorShape) or -1 if the previous cursor should be restored (default)");
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyLoadIDC_doc,"loadIDC(filename) -> load a pickled idc-file and return the content as dictionary\n\
\n\
This methods loads the given idc-file using the method `pickle.load` from the python-buildin module `pickle` and returns the loaded dictionary.\n\
\n\
Parameters \n\
----------- \n\
filename : {String} \n\
    absolute filename or filename relative to the current directory. \n\
\n\
Returns \n\
-------- \n\
content : {dict} \n\
    dictionary with loaded content \n\
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pySaveIDC_doc,"saveIDC(filename, dict, overwriteIfExists = True) -> saves the given dictionary as pickled idc-file.\n\
\n\
This method saves the given dictionary as pickled idc-file using the method dump from the builtin module pickle.\n\
The file will be saved with the pickle protocol version 3 (default for Python 3).\n\
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyCheckIsAdmin_doc,"userIsAdmin() -> return True if USER has administrator status.\n\
\n\
This method returns a boolean expression. If the USER defined by the user managment has administrator status it is true, in other cases it is False. \n\
\n\
Returns \n\
------- \n\
isRequestedType : {boolean} \n\
    Boolean return value \n\
    \n\
");
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
//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyCheckIsDeveloper_doc,"userIsDeveloper() -> return True if USER has developer status.\n\
\n\
This method returns a boolean expression. If the USER defined by the user managment has developer status it is true, in other cases it is False. \n\
\n\
Returns \n\
------- \n\
isRequestedType : {boolean} \n\
    Boolean return value \n\
    \n\
");
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
//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyCheckIsUser_doc,"userIsUser() -> return True if USER has only user status.\n\
\n\
This method returns a boolean expression. If the USER defined by the user managment has only user status it is true, in other cases it is False. \n\
\n\
Returns \n\
------- \n\
isRequestedType : {boolean} \n\
    Boolean return value \n\
    \n\
");
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyGetUserInfo_doc,"userGetInfo() -> return a dictionary with the current user management information.\n\
\n\
This method returns a dictionary which contains the current user concerning system configuration. \n\
\n\
Returns \n\
------- \n\
isUser : {dict} \n\
    dictionary with the following content is returned: \n\
    \n\
    * Name (string): The name of the current user \n\
    * Type (string): The user type as string [user, administrator, developer] \n\
    * ID (string): The user ID as a string \n\
    * File (string): The location and name of the corresponding initialization file.");
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyItom_FigureClose_doc,"close(handle|'all') -> method to close any specific or all open figures (unless any figure-instance still keeps track of them)\n\
\n\
This method closes and deletes any specific figure (given by handle) or all opened figures. This method always calls the static method \n\
`figure.close`.\n\
\n\
Parameters \n\
----------- \n\
handle : {`dataIO`, str} \n\
    any figure handle (>0) or 'all' in order to close all opened figures \n\
\n\
Notes \n\
------- \n\
If any instance of class 'figure' still keeps a reference to any figure, it is only closed and will be deleted after that the last referencing instance has been deleted. \n\
\n\
See Also \n\
--------- \n\
figure.close");

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(setPalette_doc,"setPalette(name, entries) -> set the palette for color bars defined by name.\n\
\n\
This methods sets a palette defined by entries within the palette organizer. If the palette does not exist, a new one is created.\n\
If the palette already exists and is not write protected, the palette is overwritten. If any of the optional values \n\
is not given, default values (from the 'gray' color palette) are used or, if the color palette already exists, these values are left unchanged\n\
\n\
If a palette is available in terms of a dictionary, returned by itom.getPalette, the use the star-operator, to unpack this\n\
dictionary as keyword-arguments, used as parameters for this method.\n\
\n\
Parameters \n\
----------- \n\
name : {str} \n\
    name of palette \n\
colorStops : {tuple} \n\
    tuple with all color stops, each element is another tuple whose first value is the float position [0.0,1.0]\n\
    and the 2nd value is the color (itom.rgba32). The position of the first color stop has to be 0.0, the one of the last stop 1.0.\n\
    There must be at least two colorStops.\n\
inverseColor1 : {itom.rgba32}, optional \n\
    first defined inverse color \n\
inverseColor2 : {itom.rgba32}, optional \n\
    2nd defined inverse color \n\
invalidColor : {itom.rgba32}, optional \n\
    color used for NaN or Inf values \n\
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

    QMetaObject::invokeMethod(paletteOrganizer, "getColorBarThreaded", Q_ARG(QString,QLatin1String(name)), Q_ARG(QSharedPointer<ito::ItomPaletteBase>, sharedPalette), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

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

        QMetaObject::invokeMethod(paletteOrganizer, "getColorBarThreaded", Q_ARG(QString,QLatin1String("gray")), Q_ARG(QSharedPointer<ito::ItomPaletteBase>, sharedPalette2), Q_ARG(ItomSharedSemaphore*,locker2.getSemaphore()));

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
            return PyErr_Format(PyExc_RuntimeError, "The color palette '%s' is readonly and cannot be changed", name);
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
                return PyErr_Format(PyExc_RuntimeError, "The %i. item of colorStops must be a tuple with a real value, followed by a color of type itom.rgba32.", i);
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

    QMetaObject::invokeMethod(paletteOrganizer, "setColorBarThreaded", Q_ARG(QString,QString(name)), Q_ARG(ito::ItomPaletteBase, newPalette), Q_ARG(ItomSharedSemaphore*,locker3.getSemaphore()));

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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(getPalette_doc,"getPalette(name) -> get the color palette used in color bars defined by name.\n\
\n\
If a color palette with this name is given, a tuple is returned whose length corresponds \n\
to the number of color stops. Every item is a tuple with two elements, where the first element \n\
denotes the normalized index position of the color stop [0,1] and the second element is of type itom.rgba32 \n\
and indicates the color at the stop position. \n\
\n\
Parameters \n\
----------- \n\
name : {string} \n\
    name of the new palette. \n\
\n\
Returns \n\
-------- \n\
palette : {dict} \n\
    Dictionary with the following entries: \n\
    \n\
    name : {str} \n\
        name of palette \n\
    colorStops : {tuple} \n\
        tuple with all color stops, each element is another tuple whose first value is the float position [0.0,1.0]  and the 2nd value is the color (itom.rgba32) \n\
    inverseColor1 : {itom.rgba32} \n\
        first defined inverse color \n\
    inverseColor2 : {itom.rgba32} \n\
        2nd defined inverse color \n\
    invalidColor : {itom.rgba32} \n\
        color used for NaN or Inf values \n\
\n\
Raises \n\
----------- \n\
RuntimeError : \n\
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

    QMetaObject::invokeMethod(paletteOrganizer, "getColorBarThreaded", Q_ARG(QString,QString(name)), Q_ARG(QSharedPointer<ito::ItomPaletteBase>, sharedPalette), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

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
        ito::PythonRgba::PyRgba* rgb = (ito::PythonRgba::PyRgba*)ito::PythonRgba::PyRgba_new(&ito::PythonRgba::PyRgbaType, NULL, NULL ); //new ref
        rgb->rgba.a = sharedPalette->getColor(elem).alpha();
        rgb->rgba.r = sharedPalette->getColor(elem).red();
        rgb->rgba.g = sharedPalette->getColor(elem).green();
        rgb->rgba.b = sharedPalette->getColor(elem).blue();
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

    ito::PythonRgba::PyRgba* inverseColor1 = (ito::PythonRgba::PyRgba*)ito::PythonRgba::PyRgba_new(&ito::PythonRgba::PyRgbaType, NULL, NULL ); //new ref
    inverseColor1->rgba.a = sharedPalette->getInverseColorOne().alpha();
    inverseColor1->rgba.r = sharedPalette->getInverseColorOne().red();
    inverseColor1->rgba.g = sharedPalette->getInverseColorOne().green();
    inverseColor1->rgba.b = sharedPalette->getInverseColorOne().blue();
    PyDict_SetItemString(dict, "inverseColor1", (PyObject*)inverseColor1);
    Py_DECREF(inverseColor1);

    ito::PythonRgba::PyRgba* inverseColor2 = (ito::PythonRgba::PyRgba*)ito::PythonRgba::PyRgba_new(&ito::PythonRgba::PyRgbaType, NULL, NULL ); //new ref
    inverseColor2->rgba.a = sharedPalette->getInverseColorTwo().alpha();
    inverseColor2->rgba.r = sharedPalette->getInverseColorTwo().red();
    inverseColor2->rgba.g = sharedPalette->getInverseColorTwo().green();
    inverseColor2->rgba.b = sharedPalette->getInverseColorTwo().blue();
    PyDict_SetItemString(dict, "inverseColor2", (PyObject*)inverseColor2);
    Py_DECREF(inverseColor2);

    ito::PythonRgba::PyRgba* invalidColor = (ito::PythonRgba::PyRgba*)ito::PythonRgba::PyRgba_new(&ito::PythonRgba::PyRgbaType, NULL, NULL ); //new ref
    invalidColor->rgba.a = sharedPalette->getInvalidColor().alpha();
    invalidColor->rgba.r = sharedPalette->getInvalidColor().red();
    invalidColor->rgba.g = sharedPalette->getInvalidColor().green();
    invalidColor->rgba.b = sharedPalette->getInvalidColor().blue();
    PyDict_SetItemString(dict, "invalidColor", (PyObject*)invalidColor);
    Py_DECREF(invalidColor);

    return dict;
}
//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(getPaletteList_doc,"getPaletteList(typefilter = 0) -> returns a tuple of all currently available names of color palettes.\n\
\n\
\n\
Parameters \n\
----------- \n\
typefilter : {int} \n\
    currently unused parameter \n\
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

    QMetaObject::invokeMethod(paletteOrganizer, "getColorBarListThreaded", Q_ARG(int,typefilter), Q_ARG(QSharedPointer<QStringList>, sharedPalettes), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

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
//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonItom::PyClearAll(PyObject* pSelf)
{
    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    pyEngine->pythonClearAll();
    Py_RETURN_NONE;
}
//----------------------------------------------------------------------------------------------------------------------------------
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
	{"showHelpViewer", (PyCFunction)PythonItom::PyShowHelpViewer, METH_VARARGS, pyShowHelpViewer_doc },
    {"plot", (PyCFunction)PythonItom::PyPlotImage, METH_VARARGS | METH_KEYWORDS, pyPlotImage_doc},
    {"plot1", (PyCFunction)PythonItom::PyPlot1d, METH_VARARGS | METH_KEYWORDS, pyPlot1d_doc},
    {"plot2", (PyCFunction)PythonItom::PyPlot2d, METH_VARARGS | METH_KEYWORDS, pyPlot2d_doc},
    {"plot25", (PyCFunction)PythonItom::PyPlot25d, METH_VARARGS | METH_KEYWORDS, pyPlot2d_doc},
    {"liveImage", (PyCFunction)PythonItom::PyLiveImage, METH_VARARGS | METH_KEYWORDS, pyLiveImage_doc},
    {"close", (PyCFunction)PythonFigure::PyFigure_close, METH_VARARGS, pyItom_FigureClose_doc}, /*class static figure.close(...)*/
    {"filter", (PyCFunction)PythonItom::PyFilter, METH_VARARGS | METH_KEYWORDS, pyFilter_doc},
    {"filterHelp", (PyCFunction)PythonItom::PyFilterHelp, METH_VARARGS | METH_KEYWORDS, pyFilterHelp_doc},
    {"widgetHelp", (PyCFunction)PythonItom::PyWidgetHelp, METH_VARARGS | METH_KEYWORDS, pyWidgetHelp_doc},
    {"pluginHelp", (PyCFunction)PythonItom::PyPluginHelp, METH_VARARGS | METH_KEYWORDS, pyPluginHelp_doc},
    {"aboutInfo", (PyCFunction)PythonItom::PyAboutInfo, METH_VARARGS, pyAboutInfo_doc},
    {"pluginLoaded", (PyCFunction)PythonItom::PyPluginLoaded, METH_VARARGS, pyPluginLoaded_doc},
    {"plotHelp", (PyCFunction)PythonItom::PyPlotHelp, METH_VARARGS | METH_KEYWORDS, pyPlotHelp_doc},
    {"plotLoaded", (PyCFunction)PythonItom::PyPlotLoaded, METH_VARARGS, pyPlotLoaded_doc},
    {"version", (PyCFunction)PythonItom::PyITOMVersion, METH_VARARGS, pyITOMVersion_doc},
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
    {"checkSignals", (PyCFunction)PythonItom::PyCheckSignals, METH_NOARGS, NULL},
    {"processEvents", (PyCFunction)PythonItom::PyProcessEvents, METH_NOARGS, NULL},
    {"getDebugger", (PyCFunction)PythonItom::PyGetDebugger, METH_NOARGS, "getDebugger() -> returns new reference to debugger instance"},
    {"gcStartTracking", (PyCFunction)PythonItom::PyGCStartTracking, METH_NOARGS, "gcStartTracking() -> stores the current object list of the garbage collector."},
    {"gcEndTracking", (PyCFunction)PythonItom::PyGCEndTracking, METH_NOARGS, "gcEndTracking() -> compares the current object list of the garbage collector with the recently saved list."},
    //{"getGlobalDict", (PyCFunction)PythonItom::PyGetGlobalDict, METH_NOARGS, "getGlobalDict() -> returns borrowed reference to global dictionary of itom python instance"},
    {"getScreenInfo", (PyCFunction)PythonItom::PyGetScreenInfo, METH_NOARGS, getScreenInfo_doc},
    {"setApplicationCursor", (PyCFunction)PythonItom::setApplicationCursor, METH_VARARGS, setApplicationCursor_doc},
    {"loadIDC", (PyCFunction)PythonItom::PyLoadIDC, METH_VARARGS | METH_KEYWORDS, pyLoadIDC_doc},
    {"saveIDC", (PyCFunction)PythonItom::PySaveIDC, METH_VARARGS | METH_KEYWORDS, pySaveIDC_doc},
    {"compressData", (PyCFunction)PythonItom::compressData, METH_VARARGS, "compressData(str) -> compresses the given string using the method qCompress"},
    {"uncompressData", (PyCFunction)PythonItom::uncompressData, METH_VARARGS, "uncompressData(str) -> uncompresses the given string using the method qUncompress"},
    {"userIsAdmin", (PyCFunction)PythonItom::userCheckIsAdmin, METH_NOARGS, pyCheckIsAdmin_doc},
    {"userIsDeveloper", (PyCFunction)PythonItom::userCheckIsDeveloper, METH_NOARGS, pyCheckIsDeveloper_doc},
    {"userIsUser", (PyCFunction)PythonItom::userCheckIsUser, METH_NOARGS, pyCheckIsUser_doc},
    {"userGetInfo", (PyCFunction)PythonItom::userGetUserInfo, METH_NOARGS, pyGetUserInfo_doc},
    {"autoReloader", (PyCFunction)PythonItom::PyAutoReloader, METH_VARARGS | METH_KEYWORDS, autoReloader_doc},
    {"clc", (PyCFunction)PythonItom::PyClearCommandLine, METH_NOARGS, "clc() -> clears the itom command line (if available)"},
    {"getPalette", (PyCFunction)PythonItom::PyGetPalette, METH_VARARGS, getPalette_doc},
    {"setPalette", (PyCFunction)PythonItom::PySetPalette, METH_VARARGS | METH_KEYWORDS, setPalette_doc},
    {"getPaletteList", (PyCFunction)PythonItom::PyGetPaletteList, METH_VARARGS, getPaletteList_doc},
    {"clearAll", (PyCFunction)PythonItom::PyClearAll, METH_NOARGS, "clears all variables in workspace (holds variables created by any startup script)"},
    {"registerResource", (PyCFunction)PythonItom::PyRegisterResource, METH_VARARGS | METH_KEYWORDS, pyRegisterResources_doc},
    {"unregisterResource", (PyCFunction)PythonItom::PyUnregisterResource, METH_VARARGS | METH_KEYWORDS, pyUnregisterResources_doc},
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

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
