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
#include "../organizer/userOrganizer.h"

#include <qdir.h>
#include <qcoreapplication.h>
#include <qdesktopwidget.h>

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
        //check if argument is a PyObject with a __file__ argument
        PyObject *obj = NULL;
        if (!PyArg_ParseTuple(pArgs, "O", &obj))
        {
            return NULL;
        }
        else if (PyObject_HasAttrString(obj, "__file__"))
        {
            PyObject *__file__ = PyObject_GetAttrString(obj, "__file__");
            bool ok;
            QString f = PythonQtConversion::PyObjGetString(__file__,true,ok);
            if (ok)
            {
                filename2 = f.toLatin1();
                filename = filename2.data(); //be carefull, filename is borrowed from filename2
            }
            else
            {
                return PyErr_Format(PyExc_ValueError, "__file__ attribute of given argument could not be parsed as string.");
            }
        }
        else
        { 
            return PyErr_Format(PyExc_ValueError, "argument is no filename string and no other object that has a __file__ attribute.");
        }
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
    class name of desired plot (if not indicated or if the className can not be found, the default plot will be used (see application settings)");
PyObject* PythonItom::PyPlotImage(PyObject * /*pSelf*/, PyObject *pArgs, PyObject *pKwds)
{
    const char *kwlist[] = {"data", "className", NULL};
    PyObject *data = NULL;
//    int areaIndex = 0;
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
PyDoc_STRVAR(pyLiveImage_doc,"liveImage(cam, [className]) -> show a camera live image in a newly created figure\n\
\n\
Creates a plot-image (2D) and automatically grabs images into this window.\n\
This function is not blocking.\n\
\n\
Parameters \n\
----------- \n\
cam : {dataIO-Instance} \n\
    Camera grabber device from which images are acquired.\n\
className : {str}, optional \n\
    class name of desired plot (if not indicated or if the className can not be found, the default plot will be used (see application settings)");
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
                    //(*(wFunc->m_paramFunc))(&paramsMand, &paramsOpt);

                    if (!retDict)
                    {
                        std::cout << "DESCRIPTION    " << wFunc->m_description.toLatin1().data() << "\n";
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
                    filterParams = AIM->getHashedFilterParams(fFunc->m_paramFunc);
                    //(*(fFunc->m_paramFunc))(&paramsMand, &paramsOpt);

                    if (!retDict)
                    {
                        std::cout << "DESCRIPTION    " << fFunc->m_description.toLatin1().data() << "\n";
                    }
                    else
                    {
                        item = PythonQtConversion::QByteArrayToPyUnicodeSecure(fFunc->m_description.toLatin1());
                        PyDict_SetItemString(resulttemp, "description", item);
                        Py_DECREF(item);
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
                            resultmand = PrntOutParams(&(filterParams->paramsMand), false, true, -1);
                            Py_DECREF(resultmand);
                        }
                        else
                        {
                            resultmand = PrntOutParams(&(filterParams->paramsMand), false, true, -1, false);
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
                            resultopt = ito::PrntOutParams(&(filterParams->paramsOpt), false, true, -1);
                            Py_DECREF(resultopt);
                        }
                        else
                        {
                            resultopt = ito::PrntOutParams(&(filterParams->paramsOpt), false, true, -1, false);
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
                            resultopt = ito::PrntOutParams(&(filterParams->paramsOut), false, true, -1);
                            Py_DECREF(resultopt);
                        }
                        else
                        {
                            resultopt = ito::PrntOutParams(&(filterParams->paramsOut), false, true, -1, false);
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
                                resultmand = PrntOutParams(&(filterParams->paramsMand), false, true, -1);
                                Py_DECREF(resultmand);
                            }
                            else
                            {
                                resultmand = PrntOutParams(&(filterParams->paramsMand), false, true, -1, false);
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
                                resultopt = PrntOutParams(&(filterParams->paramsOpt), false, true, -1);
                                Py_DECREF(resultopt);
                            }
                            else
                            {
                                resultopt = PrntOutParams(&(filterParams->paramsOpt), false, true, -1, false);
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
                                resultopt = PrntOutParams(&(filterParams->paramsOut), false, true, -1);
                                Py_DECREF(resultopt);
                            }
                            else
                            {
                                resultopt = PrntOutParams(&(filterParams->paramsOut), false, true, -1, false);
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
                continue;
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
PyDoc_STRVAR(pyFilterHelp_doc, "filterHelp([filterName, dictionary = 0, furtherInfos = 0]) -> generates an online help for the given filter(s). \n\
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
PyDoc_STRVAR(pyWidgetHelp_doc,"widgetHelp([widgetName, dictionary = 0, furtherInfos = 0]) -> generates an online help for the given widget(s). \n\
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

    ito::AddInManager *AIM = ito::AddInManager::getInstance();
    if (!AIM)
    {
        return PyErr_Format(PyExc_RuntimeError, "no addin-manager found");
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
PyDoc_STRVAR(pyPluginHelp_doc,"pluginHelp(pluginName [, dictionary = False]) -> generates an online help for the specified plugin.\n\
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


    ito::AddInManager *AIM = ito::AddInManager::getInstance();
    if (!AIM)
    {
        return PyErr_Format(PyExc_RuntimeError, "no addin-manager found");
    }

    retval = AIM->getPluginInfo(pluginName, plugtype, pluginNum, version, pTypeString, pAuthor, pDescription, pDetailDescription, pLicense, pAbout);
    if (retval.containsWarningOrError())
    {
        if (retval.hasErrorMessage())
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

                if (retval.hasErrorMessage())
                {
                    return PyErr_Format(PyExc_RuntimeError, "Could not load plugin: %s with error message: \n%s\n", pluginName, retval.errorMessage());
                }
                else
                {
                    return PyErr_Format(PyExc_RuntimeError, "Could not load plugin: %s\n", pluginName);
                }
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
                        resultmand = PrntOutParams(paramsMand, false, true, -1, false);
                        PyDict_SetItemString(result, "Mandatory Parameters", resultmand);
                        Py_DECREF(resultmand);
                    }
                    else
                    {
                        std::cout << "\n Mandatory parameters:\n";
                        resultmand = PrntOutParams(paramsMand, false, true, -1);
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
                        resultopt = PrntOutParams(paramsOpt, false, true, -1, false);
                        PyDict_SetItemString(result, "Optional Parameters", resultopt);
                        Py_DECREF(resultopt);
                    }
                    else
                    {
                        std::cout << "\n Optional parameters:\n";
                        resultopt = PrntOutParams(paramsOpt, false, true, -1);
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
Retrieve complete version information of itom and if specified version information of loaded plugins\n\
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
PyDoc_STRVAR(pyAddButton_doc,"addButton(toolbarName, buttonName, code [, icon, argtuple]) -> adds a button to a toolbar in the main window \n\
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
        retValue += RetVal(retError,0,QObject::tr("Button must have a valid name.").toLatin1().data());
    }
    else
    {
        if (!code)
        {
            retValue += RetVal(retError,0,QObject::tr("Any type of code (String or callable method or function) must be indicated.").toLatin1().data());
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
                    retValue += RetVal(retError,0,QObject::tr("The given button name is already associated to a python method or function. The button can not be created.").toLatin1().data());
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
                        retValue += RetVal(retError,0,QObject::tr("Could not create a itom.proxy-object  of the given callable method or function.").toLatin1().data());
                    }
                }
            }
            else
            {
                retValue += RetVal(retError,0,QObject::tr("The code parameter must either be a python code snippet or a callable method or function object.").toLatin1().data());
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
This method removes an existing button from a toolbar in the main window of 'itom'. This button must have been \n\
created using `addButton`. If the toolbar is empty after the removal, it is finally deleted. \n\
\n\
Parameters \n\
----------- \n\
toolbarName : {str} \n\
    The name of the toolbar.\n\
buttonName : {str} \n\
    The name (str, identifier) of the button to remove.\n\
\n\
See Also \n\
--------- \n\
addButton()");
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
PyDoc_STRVAR(pyAddMenu_doc,"addMenu(type, key [, name, code, icon, argtuple]) -> adds an element to the menu bar of itom. \n\
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
            else if (!ok) //check whether code is a method or function
            {
                if (PyMethod_Check(code) || PyFunction_Check(code))
                {
                    //create hash-string
                    qkey2 = qkey + "_" +  QString::number(pyEngine->m_pyFuncWeakRefHashesAutoInc++);
                    qcode = ":::itomfcthash:::" + qkey2;
                    if (pyEngine->m_pyFuncWeakRefHashes.contains(qkey2))
                    {
                        retValue += RetVal(retError,0,QObject::tr("The given key is already associated to a python method or function. The menu element can not be created.").toLatin1().data());
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
                            retValue += RetVal(retError,0,QObject::tr("Could not create a itom.proxy-object  of the given callable method or function.").toLatin1().data());
                        }
                    }
                }
                else
                {
                    retValue += RetVal(retError,0,QObject::tr("The code parameter must either be a python code snippet or a callable method or function object.").toLatin1().data());
                }
            }
            break;
            }
        case 1: //SEPARATOR
            {
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
This function remove a menu element with the given key. \n\
key is a slash separated list. The sub-components then \n\
lead the way to the final element, which should be removed. \n\
\n\
Parameters \n\
----------- \n\
key : {str} \n\
    The name (str, identifier) of the menu entry to remove.\n\
\n\
See Also \n\
--------- \n\
addMenu");
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
            m_gcTrackerList[(size_t)t] = QString("%1 [%2]").arg(t->ob_type->tp_name).arg(PythonQtConversion::PyObjGetString(t, false, ok)); //t->ob_type->tp_name;
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
        PyErr_SetString(PyExc_RuntimeError,"module gc could not be imported");
        return NULL;
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
PyDoc_STRVAR(pySaveMatlabMat_doc,"saveMatlabMat(filename, values[, matrixName = 'matrix']) -> save strings, numbers, arrays or combinations into a Matlab mat file. \n\
\n\
Save one or multiple objects (strings, numbers, arrays, `dataObject`, `numpy.ndarray`, `npDataObject`...) to a Matlab *mat* file. \n\
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
    single value, dictionary, list or tuple with elements of type number, string, array (dataObject, numpy.ndarray, npDataObject...)\n\
matrix-name : {str, list, tuple}, optional \n\
    if 'values' is a single value, this parameter must be one single str, if 'values' is a sequence it must be a sequence of strings with the same length, if 'values' is a dictionary this argument is ignored. \n\
\n\
See Also \n\
---------- \n\
loadMatlabMat");
PyObject * PythonItom::PySaveMatlabMat(PyObject * /*pSelf*/, PyObject *pArgs)
{
    PyObject* scipyIoModule = PyImport_ImportModule("scipy.io"); // new reference

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
        return NULL;
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
                    snprintf(key, strlen(tempName) + sizeIter, "%s%i", matrixName, ((size_t)i + 1));
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
    //if (!PyObject_CallMethodObjArgs(scipyIoModule, PyUnicode_FromString("savemat"), PyUnicode_FromString(filename), saveDict, Py_True, PyUnicode_FromString("5"), Py_True, Py_False, PyUnicode_FromString("row"), NULL))
    if (!PyObject_CallMethodObjArgs(scipyIoModule, PyUnicode_FromString("savemat"), PyUnicode_DecodeLatin1(filename, strlen(filename), NULL), saveDict, Py_True, PyUnicode_FromString("5"), Py_True, Py_False, PyUnicode_FromString("row"), NULL))
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
        return PyErr_Format(PyExc_ImportError, "scipy-module and scipy.io-module could not be loaded.");
    }

    //Arguments must be: filename -> string

    const char *filename = NULL;


    if (!PyArg_ParseTuple(pArgs, "s", &filename))
    {
        Py_XDECREF(scipyIoModule);
        return NULL;
    }

    PyObject *kwdDict = PyDict_New();
    PyObject *argTuple = PyTuple_New(1);
    //PyTuple_SetItem(argTuple, 0, PyUnicode_FromString(filename));
    PyTuple_SetItem(argTuple, 0, PyUnicode_DecodeLatin1(filename, strlen(filename), NULL));
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
PyDoc_STRVAR(pyFilter_doc,"filter(name [, furtherParameters, ...]) -> invoke a filter (or algorithm) function from an algorithm-plugin. \n\
\n\
This function is used to invoke itom filter-functions or algorithms, declared within itom-algorithm plugins.\n\
The parameters (arguments) depends on the specific filter function (see filterHelp(name)),\n\
By filterHelp() a list of available filter functions is retrieved. \n\
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
out : {variant} \n\
    The returned values depend on the definition of each filter. In general it is a tuple of all output parameters that are defined by the filter function.\n\
\n\
See Also \n\
--------- \n\
filterHelp");
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

    try
    {
        ret = (*(fFunc->m_filterFunc))(&paramsMandBase, &paramsOptBase, &paramsOutBase);
    }
    catch (cv::Exception exc)
    {
        const char* errorStr = cvErrorStr(exc.code);

        ret += ito::RetVal::format(ito::retError,0,"OpenCV Error: %s (%s) in %s, file %s, line %d",
            errorStr, exc.err.c_str(), exc.func.size() > 0 ?
            exc.func.c_str() : "unknown function", exc.file.c_str(), exc.line );
        //see also cv::setBreakOnError(true) -> then cv::error(...) forces an access to 0x0000 (throws access error, the debugger stops and you can debug it)

        //use this to raise an access error that forces the IDE to break in this line (replaces cv::setBreakOnError(true)).
#if defined _DEBUG
        static volatile int* p = 0; //if your debugger stops in this line, another exception has been raised and you have now the chance to see your callstack for debugging.
        *p = 0;
#endif
    }
    catch(std::exception exc)
    {
        if (exc.what())
        {
            ret += ito::RetVal::format(ito::retError,0,"The exception '%s' has been thrown", exc.what()); 
        }
        else
        {
            ret += ito::RetVal(ito::retError,0,"The exception '<unknown>' has been thrown"); 
        }
#if defined _DEBUG
        static volatile int* p = 0; //if your debugger stops in this line, another exception has been raised and you have now the chance to see your callstack for debugging.
        *p = 0;
#endif
    }
    catch (...)
    {
        ret += ito::RetVal(ito::retError,0,"An unspecified exception has been thrown");  
#if defined _DEBUG
        static volatile int* p = 0; //if your debugger stops in this line, another exception has been raised and you have now the chance to see your callstack for debugging.
        *p = 0;
#endif
    }
    

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
PyDoc_STRVAR(pySaveDataObject_doc,"saveDataObject(filename, dataObject [, tagsAsBinary = False]) -> save a dataObject to harddrive in a xml-based file format. \n\
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
    Optional tag to toogle if numeric-tags should be saved (metaData) as binary or by default as string.\n\
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

    if (ret.containsError())
    {
        if (ret.hasErrorMessage())
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
PyDoc_STRVAR(pyLoadDataObject_doc,"loadDataObject(filename, dataObject [, doNotAppendIDO]) -> load a dataObject from the harddrive. \n\
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

    if (ret.containsError())
    {
        if (ret.hasErrorMessage())
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
    /*
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

    */

    PyObject * errText = PyUnicode_DecodeLatin1("Hällo Wörl", 10, NULL);
    PyErr_Format(PyExc_RuntimeError, "%U", errText);
    Py_DECREF(errText);
    return NULL;
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
    const char* unitString = NULL;
    PyObject *myList = NULL;
    QString unitIn("");
    QString unitOut("");

    if (!PyArg_ParseTupleAndKeywords(pArgs, pKwds, "O!ds", const_cast<char**>(kwlist), &PyList_Type, &myList, &value, &unitString))
    {
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

    return Py_BuildValue("ds", valueOut, unitOut.toLatin1().data());
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
        return PyErr_Format(PyExc_RuntimeError, "compression level must be -1 (default: level 6) or between 0 and 9");
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
//    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
//    if (pyEngine)
//    {
//        PyObject *dict = pyEngine->getMainDictionary();
//        if (dict)
//        {
//            Py_INCREF(dict);
//            return dict;
//        }
//        PyErr_Format(PyExc_RuntimeError, "The global dictionary is not available.");
//        return NULL;
//    }
//    else
//    {
//        PyErr_Format(PyExc_RuntimeError, "Python Engine is not available.");
//        return NULL;
//    }
//}




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
            return PyErr_Format(PyExc_RuntimeError, "The file '%s' does not exist", info.absoluteFilePath().toLatin1().data());
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
This method saves the given dictionary as pickled idc-file using the method dump from the builtin module pickle.\n\
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
                Py_DECREF(dict);
                return NULL;
            }

            Py_RETURN_NONE;
        }
        else
        {
            return PyErr_Format(PyExc_RuntimeError, "The file '%s' cannot be overwritten", info.absoluteFilePath().toLatin1().data());
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
        return PyErr_Format(PyExc_RuntimeError, "userOrganizer not available");
    }

    if (userOrg->getUserRole() == ito::userTypeAdministrator)
    {
        return Py_True;
    }

    return Py_False;
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
        return PyErr_Format(PyExc_RuntimeError, "userOrganizer not available");
    }

    if (userOrg->getUserRole() == ito::userTypeDeveloper)
    {
        return Py_True;
    }

    return Py_False;
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
        return PyErr_Format(PyExc_RuntimeError, "userOrganizer not available");
    }

    if (userOrg->getUserRole() == ito::userTypeBasic)
    {
        return Py_True;
    }

    return Py_False;
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
        return PyErr_Format(PyExc_RuntimeError, "userOrganizer not available");
    }

    PyObject* returnDict = PyDict_New();

    // Name
    //PyObject *item = PyUnicode_FromString(userOrg->getUserName().toLatin1().data());
    PyObject *item = PyUnicode_DecodeLatin1(userOrg->getUserName().toLatin1().data(), userOrg->getUserName().length(), NULL);
    PyDict_SetItemString(returnDict, "Name", item);
    Py_DECREF(item);
    
    // Type
    switch(userOrg->getUserRole())
    {
        case ito::userTypeBasic:
            item = PyUnicode_FromString("user");
        break;
        case ito::userTypeAdministrator:
            item = PyUnicode_FromString("administrator");
        break;
        case ito::userTypeDeveloper:
            item = PyUnicode_FromString("developer");
        break;
        default:
            item = PyUnicode_FromString("D.A.U.");
    }
    
    PyDict_SetItemString(returnDict, "Type", item);
    Py_DECREF(item); 

    // ID
    //item = PyUnicode_FromString(userOrg->getUserID().toLatin1().data());
    item = PyUnicode_DecodeLatin1(userOrg->getUserID().toLatin1().data(), userOrg->getUserID().length(), NULL);
    PyDict_SetItemString(returnDict, "ID", item);
    Py_DECREF(item); 

    // FILE
    //item = PyUnicode_FromString(userOrg->getSettingsFile().toLatin1().data());
    item = PyUnicode_DecodeLatin1(userOrg->getSettingsFile().toLatin1().data(), userOrg->getSettingsFile().length(), NULL);
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
    {"filterHelp", (PyCFunction)PythonItom::PyFilterHelp, METH_VARARGS | METH_KEYWORDS, pyFilterHelp_doc},
    {"widgetHelp", (PyCFunction)PythonItom::PyWidgetHelp, METH_VARARGS | METH_KEYWORDS, pyWidgetHelp_doc},
    {"pluginHelp", (PyCFunction)PythonItom::PyPluginHelp, METH_VARARGS | METH_KEYWORDS, pyPluginHelp_doc},
    {"pluginLoaded", (PyCFunction)PythonItom::PyPluginLoaded, METH_VARARGS, pyPluginLoaded_doc},
    {"version", (PyCFunction)PythonItom::PyITOMVersion, METH_VARARGS, pyITOMVersion_doc},
    {"saveDataObject", (PyCFunction)PythonItom::PySaveDataObject, METH_VARARGS | METH_KEYWORDS, pySaveDataObject_doc},
    {"loadDataObject", (PyCFunction)PythonItom::PyLoadDataObject, METH_VARARGS | METH_KEYWORDS, pyLoadDataObject_doc},
    {"addButton", (PyCFunction)PythonItom::PyAddButton, METH_VARARGS | METH_KEYWORDS, pyAddButton_doc},
    {"removeButton", (PyCFunction)PythonItom::PyRemoveButton, METH_VARARGS, pyRemoveButton_doc},
    {"addMenu", (PyCFunction)PythonItom::PyAddMenu, METH_VARARGS | METH_KEYWORDS, pyAddMenu_doc},
    {"removeMenu", (PyCFunction)PythonItom::PyRemoveMenu, METH_VARARGS | METH_KEYWORDS, pyRemoveMenu_doc},
    {"saveMatlabMat", (PyCFunction)PythonItom::PySaveMatlabMat, METH_VARARGS, pySaveMatlabMat_doc},
    {"loadMatlabMat", (PyCFunction)PythonItom::PyLoadMatlabMat, METH_VARARGS, pyLoadMatlabMat_doc},
    {"scaleValueAndUnit", (PyCFunction)PythonItom::scaleValueAndUnit, METH_VARARGS | METH_KEYWORDS, scaleValueAndUnit_doc},
    {"getDefaultScaleableUnits", (PyCFunction)PythonItom::getDefaultScaleableUnits, METH_NOARGS, getDefaultScaleableUnits_doc},
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
    {"compressData", (PyCFunction)PythonItom::compressData, METH_VARARGS, "compresses the given string using the method qCompress"},
    {"uncompressData", (PyCFunction)PythonItom::uncompressData, METH_VARARGS, "uncompresses the given string using the method qUncompress"},
    {"userIsAdmin", (PyCFunction)PythonItom::userCheckIsAdmin, METH_NOARGS, pyCheckIsAdmin_doc},
    {"userIsDeveloper", (PyCFunction)PythonItom::userCheckIsDeveloper, METH_NOARGS, pyCheckIsDeveloper_doc},
    {"userIsUser", (PyCFunction)PythonItom::userCheckIsUser, METH_NOARGS, pyCheckIsUser_doc},
    {"userGetInfo", (PyCFunction)PythonItom::userGetUserInfo, METH_NOARGS, pyGetUserInfo_doc},
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
