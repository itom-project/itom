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

#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API
#endif
#include "pythonEngine.h"

#include "node.h"

#include "../global.h"
#include "pythonNpDataObject.h"
#include "pythonStream.h"

#include "pythonItom.h"
#include "pythonUi.h"
#include "pythonFigure.h"
#include "pythonUiTimer.h"
#include "pythonPlugins.h"
#include "pythonPCL.h"
#include "pythonProxy.h"
#include "pythonPlotItem.h"
#include "pythontParamConversion.h"
#include "pythonRegion.h"
#include "pythonRgba.h"

#include "../organizer/addInManager.h"

#include <qobject.h>
#include <qcoreapplication.h>

#include <qstringlist.h>
#include <qdir.h>
#include <qdesktopwidget.h>

#include <qsettings.h>
#include <AppManagement.h>

#include <qsharedpointer.h>
#include <qtimer.h>
#include <qtextcodec.h>

#include <QtCore/qmath.h>

#if ITOM_PYTHONMATLAB == 1
#include "pythonMatlab.h"
#endif

#if linux
#include <langinfo.h>
#endif

namespace ito
{

QMutex PythonEngine::instatiated;
QMutex PythonEngine::instancePtrProtection;
PythonEngine* PythonEngine::instance = NULL;

using namespace ito;

//----------------------------------------------------------------------------------------------------------------------------------
//public
const PythonEngine *PythonEngine::getInstance()
{
    QMutexLocker locker(&PythonEngine::instancePtrProtection);
    return const_cast<PythonEngine*>(PythonEngine::instance);
}

//----------------------------------------------------------------------------------------------------------------------------------
//private
PythonEngine *PythonEngine::getInstanceInternal()
{
    QMutexLocker locker(&PythonEngine::instancePtrProtection);
    return PythonEngine::instance;
}

//----------------------------------------------------------------------------------------------------------------------------------
PythonEngine::PythonEngine() :
    started(false),
    m_pDesktopWidget(NULL),
    pythonState(pyStateIdle),
    bpModel(NULL),
    mainModule(NULL),
    mainDictionary(NULL),
    localDictionary(NULL),
    globalDictionary(NULL),
    itomDbgModule(NULL),
    itomDbgInstance(NULL),
    itomModule(NULL),
    itomFunctions(NULL),
    m_pyModGC(NULL),
    m_pyModSyntaxCheck(NULL),
    m_pyFuncWeakRefHashesAutoInc(0),
    m_executeInternalPythonCodeInDebugMode(false),
    dictUnicode(NULL)
{
    qRegisterMetaType<tPythonDbgCmd>("tPythonDbgCmd");
    qRegisterMetaType<tPythonTransitions>("tPythonTransitions");
    qRegisterMetaType<BreakPointItem>("BreakPointItem");
    qRegisterMetaType<ItomSharedSemaphore*>("ItomSharedSemaphore*");
    qRegisterMetaType<DataObject*>("DataObject*");
    qRegisterMetaType<ito::DataObject>("ito::DataObject");
    qRegisterMetaType<ito::DataObject*>("ito::DataObject*");
    qRegisterMetaType<ito::DataObject>("ito::DataObject");
    qRegisterMetaType<QVariant*>("QVariant*");
    qRegisterMetaType<StringMap>("StringMap");
    qRegisterMetaType<AddInDataIO*>("AddInDataIO*");
    qRegisterMetaType<QVariantMap>("QVariantMap");
    qRegisterMetaType<AddInBase*>("ito::AddInBase*");
    qRegisterMetaType<QSharedPointer<QVariant> >("QSharedPointer<QVariant>");
    qRegisterMetaType<QSharedPointer<unsigned int> >("QSharedPointer<unsigned int>");
    qRegisterMetaType<QSharedPointer<int> >("QSharedPointer<int>");
    qRegisterMetaType<IntList >("IntList");
    qRegisterMetaType<QStringList >("QStringList");
    qRegisterMetaType<QSharedPointer<double> >("QSharedPointer<double>");
    qRegisterMetaType<QSharedPointer<bool> >("QSharedPointer<bool>");
    qRegisterMetaType<QSharedPointer<char> >("QSharedPointer<char>");
    qRegisterMetaType<QSharedPointer<QString> >("QSharedPointer<QString>");
    qRegisterMetaType<QSharedPointer<QByteArray> >("QSharedPointer<QByteArray>");
    qRegisterMetaType<QSharedPointer<QStringList> >("QSharedPointer<QStringList>");
    qRegisterMetaType<QSharedPointer<QVariantMap> >("QSharedPointer<QVariantMap>");
    qRegisterMetaType<QSharedPointer<QObject*> >("QSharedPointer<QObject*>");
    qRegisterMetaType<QPointer<QObject> >("QPointer<QObject>");
    qRegisterMetaType<QSharedPointer<IntList> >("QSharedPointer<IntList>"); 
    qRegisterMetaType<QSharedPointer<IntVector> >("QSharedPointer<QVector<int>>"); //if the string is QVector<int> and not IntList (which is the same), Q_ARG(QShared...<QVector<int>>) can be submitted and not QShared..<QIntVector>
    qRegisterMetaType<PyObject*>("PyObject*");
    qRegisterMetaType<QSharedPointer<MethodDescriptionList> >("QSharedPointer<MethodDescriptionList>");
    qRegisterMetaType<QSharedPointer<FctCallParamContainer> >("QSharedPointer<FctCallParamContainer>");
    qRegisterMetaType<QVector<SharedParamPointer> >("QVector<QSharedPointer<ito::Param>>"); 
    qRegisterMetaType<QSharedPointer<ParamVector> >("QSharedPointer<QVector<ito::Param>>");
    qRegisterMetaType<QVector<SharedParamBasePointer> >("QVector<SharedParamBasePointer>"); 
    qRegisterMetaType<QSharedPointer<SharedParamBasePointerVector> >("QSharedPointer<SharedParamBasePointerVector>");
    qRegisterMetaType<QSharedPointer<ParamBaseVector> >("QSharedPointer<QVector<ito::ParamBase>>");
    qRegisterMetaType<QSharedPointer<ito::DataObject> >("QSharedPointer<ito::DataObject>");
    qRegisterMetaType<QPointer<ito::AddInDataIO> >("QPointer<ito::AddInDataIO>");
    qRegisterMetaType<QPointer<ito::AddInActuator> >("QPointer<ito::AddInActuator>");
    qRegisterMetaType<QSharedPointer< QSharedPointer< unsigned int > > >("QSharedPointer<QSharedPointer<unsigned int>>");
#if ITOM_POINTCLOUDLIBRARY > 0    
    qRegisterMetaType<ito::PCLPointCloud >("ito::PCLPointCloud");
    qRegisterMetaType<ito::PCLPolygonMesh >("ito::PCLPolygonMesh");
    qRegisterMetaType<ito::PCLPoint >("ito::PCLPoint");
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
    qRegisterMetaType<ito::PyWorkspaceContainer*>("PyWorkspaceContainer*");
    qRegisterMetaType<ito::PyWorkspaceItem*>("PyWorkspaceItem*");
    qRegisterMetaType<ito::PythonQObjectMarshal>("ito::PythonQObjectMarshal");
    qRegisterMetaType<Qt::CursorShape>("Qt::CursorShape");

    bpModel = new ito::BreakPointModel();
    bpModel->restoreState(); //get breakPoints from last session

    m_pDesktopWidget = new QDesktopWidget(); //must be in constructor, since the constructor is executed in main-thread

    QMutexLocker locker(&PythonEngine::instancePtrProtection);
    PythonEngine::instance = const_cast<PythonEngine*>(this);
    locker.unlock();

    connect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(propertiesChanged()));

}

//----------------------------------------------------------------------------------------------------------------------------------
PythonEngine::~PythonEngine()
{
    if (started)
    {
        pythonShutdown();
    }

    bpModel->saveState(); //save current set of breakPoints to settings file
    DELETE_AND_SET_NULL(bpModel);

    DELETE_AND_SET_NULL(m_pDesktopWidget);

    QMutexLocker locker(&PythonEngine::instancePtrProtection);
    PythonEngine::instance = NULL;
    locker.unlock();

}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonSetup(ito::RetVal *retValue)
{
    PyObject *itomDbgClass = NULL;
    PyObject *itomDbgDict = NULL;
//    bool numpyAvailable = true;

    qDebug() << "python in thread: " << QThread::currentThreadId ();

    RetVal tretVal(retOk);
    if (!started)
    {
        if (PythonEngine::instatiated.tryLock(5000))
        {
            dictUnicode = PyUnicode_FromString("__dict__");

            PyImport_AppendInittab("itom", &PythonItom::PyInitItom);                //!< add all static, known function calls to python-module itom

            PyImport_AppendInittab("itomDbgWrapper", &PythonEngine::PyInitItomDbg);  //!< add all static, known function calls to python-module itomdbg

            Py_Initialize();                                                        //!< must be called after any PyImport_AppendInittab-call
//            PyEval_InitThreads();                                                   //!< prepare Python multithreading

            itomModule = PyImport_ImportModule("itom");
            m_pyModGC = PyImport_ImportModule("gc"); //new reference

            pythonAddBuiltinMethods();
            mainModule = PyImport_AddModule("__main__"); // reference to the module __main__ , where code above has been evaluated

            if (mainModule)
            {
                mainDictionary = PyModule_GetDict(mainModule); //borrowed
            }

            setGlobalDictionary(mainDictionary);   // reference to string-list of available methods, member-variables... of module.
            setLocalDictionary(NULL);

            emitPythonDictionary(true, true, getGlobalDictionary(), getLocalDictionary());

            if (_import_array() < 0)
            {
//                numpyAvailable = false;
                PyErr_Print();
                PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import. Please verify that you have numpy 1.6 or higher installed.");
                (*retValue) += RetVal(retError, 0, "numpy.core.multiarray failed to import. Please verify that you have numpy 1.6 or higher installed.\n");
                return;
            }

            //!< start python-type pythonStream, in order to redirect stdout and stderr to std::cout and std::cerr (possibly redirected to qDebugStream)
            if (PyType_Ready(&PyStream::PythonStreamType) >= 0)
            {
                Py_INCREF(&PyStream::PythonStreamType);
                PyModule_AddObject(itomModule, "pythonStream", (PyObject *)&PyStream::PythonStreamType);
            }

            // ck moved this here from below import numpy to print out early errors like missing numpy
            if  ((tretVal = runString("import sys")) != ito::retOk)
                (*retValue) += ito::RetVal(ito::retError, 0, tr("error importing sys in start python engine\n").toLatin1().data());
            if ((tretVal = runString("import itom")) != ito::retOk)
                (*retValue) += ito::RetVal(ito::retError, 0, tr("error importing itom in start python engine\n").toLatin1().data());
            if ((tretVal = runString("sys.stdout = itom.pythonStream(1)")) != ito::retOk)
                (*retValue) += ito::RetVal(ito::retError, 0, tr("error redirecting stdout in start python engine\n").toLatin1().data());
            if ((tretVal = runString("sys.stderr = itom.pythonStream(2)")) != ito::retOk)
                (*retValue) += ito::RetVal(ito::retError, 0, tr("error redirecting stderr in start python engine\n").toLatin1().data());


            static wchar_t *wargv = L"";
            PySys_SetArgv(1, &wargv);

            ito::PythonDataObject::PyDataObjectType.tp_base = 0;
            ito::PythonDataObject::PyDataObjectType.tp_free = PyObject_Free;
            ito::PythonDataObject::PyDataObjectType.tp_alloc = PyType_GenericAlloc;
            if (PyType_Ready(&ito::PythonDataObject::PyDataObjectType) >= 0)
            {
                Py_INCREF(&ito::PythonDataObject::PyDataObjectType);
                PyModule_AddObject(itomModule, "dataObject", (PyObject *)&ito::PythonDataObject::PyDataObjectType);
            }


            ito::PythonDataObject::PyDataObjectIterType.tp_base =0;
            ito::PythonDataObject::PyDataObjectIterType.tp_free = PyObject_Free;
            ito::PythonDataObject::PyDataObjectIterType.tp_alloc = PyType_GenericAlloc;
            if (PyType_Ready(&ito::PythonDataObject::PyDataObjectIterType) >= 0)
            {
                Py_INCREF(&ito::PythonDataObject::PyDataObjectIterType);
                //PyModule_AddObject(itomModule, "dataObjectIter", (PyObject *)&PythonDataObject::PyDataObjectIterType);
            }

            if (PyType_Ready(&PythonPlugins::PyActuatorPluginType) >= 0)
            {
                Py_INCREF(&PythonPlugins::PyActuatorPluginType);
                PyModule_AddObject(itomModule, "actuator", (PyObject *)&PythonPlugins::PyActuatorPluginType);
            }

// pending for deletion
/*
            if (PyType_Ready(&PythonPlugins::PyActuatorAxisType) >= 0)
            {
                Py_INCREF(&PythonPlugins::PyActuatorAxisType);
                PyModule_AddObject(itomModule, "axis", (PyObject *)&PythonPlugins::PyActuatorAxisType);
            }
*/

            if (PyType_Ready(&PythonPlugins::PyDataIOPluginType) >= 0)
            {
                Py_INCREF(&PythonPlugins::PyDataIOPluginType);
                PythonPlugins::PyDataIOPlugin_addTpDict(PythonPlugins::PyDataIOPluginType.tp_dict);
                PyModule_AddObject(itomModule, "dataIO", (PyObject *)&PythonPlugins::PyDataIOPluginType);
            }

#if 0 //algo plugins do not exist as instances, they only contain static methods, callable by itom.filter
            if (PyType_Ready(&PythonPlugins::PyAlgoPluginType) >= 0)
            {
                Py_INCREF(&PythonPlugins::PyAlgoPluginType);
                PyModule_AddObject(itomModule, "algo", (PyObject *)&PythonPlugins::PyAlgoPluginType);
            }
#endif

//#if ITOM_NPDATAOBJECT //right now, npDataObject exists but raises a python exception if ITOM_NPDATAOBJECT is not defined
            PythonNpDataObject::PyNpDataObjectType.tp_base = &PyArray_Type;
            PythonNpDataObject::PyNpDataObjectType.tp_free = PyObject_Free;
            PythonNpDataObject::PyNpDataObjectType.tp_alloc = PyType_GenericAlloc;
            if (PyType_Ready(&PythonNpDataObject::PyNpDataObjectType) >= 0)
            {
                Py_INCREF(&PythonNpDataObject::PyNpDataObjectType);
                PyModule_AddObject(itomModule, "npDataObject", (PyObject *)&PythonNpDataObject::PyNpDataObjectType);
            }
//#endif //ITOM_NPDATAOBJECT

            if (PyType_Ready(&PythonTimer::PyTimerType) >= 0)
            {
                Py_INCREF(&PythonTimer::PyTimerType);
                PyModule_AddObject(itomModule, "timer", (PyObject *)&PythonTimer::PyTimerType);
            }

            if (PyType_Ready(&PythonUi::PyUiItemType) >= 0)
            {
                Py_INCREF(&PythonUi::PyUiItemType);
                PythonUi::PyUiItem_addTpDict(PythonUi::PyUiItemType.tp_dict);
                PyModule_AddObject(itomModule, "uiItem", (PyObject *)&PythonUi::PyUiItemType);
            }

            PythonUi::PyUiType.tp_base = &PythonUi::PyUiItemType; //Ui is derived from UiItem
            if (PyType_Ready(&PythonUi::PyUiType) >= 0)
            {
                Py_INCREF(&PythonUi::PyUiType);
                PythonUi::PyUi_addTpDict(PythonUi::PyUiType.tp_dict);
                PyModule_AddObject(itomModule, "ui", (PyObject *)&PythonUi::PyUiType);
            }

            PythonFigure::PyFigureType.tp_base = &PythonUi::PyUiItemType; //Figure is derived from UiItem
            if (PyType_Ready(&PythonFigure::PyFigureType) >= 0)
            {
                Py_INCREF(&PythonFigure::PyFigureType);
                PythonFigure::PyFigure_addTpDict(PythonFigure::PyFigureType.tp_dict);
                PyModule_AddObject(itomModule, "figure", (PyObject *)&PythonFigure::PyFigureType);
            }

            PythonPlotItem::PyPlotItemType.tp_base = &PythonUi::PyUiItemType; //PlotItem is derived from UiItem
            if (PyType_Ready(&PythonPlotItem::PyPlotItemType) >= 0)
            {
                Py_INCREF(&PythonPlotItem::PyPlotItemType);
                PythonPlotItem::PyPlotItem_addTpDict(PythonPlotItem::PyPlotItemType.tp_dict);
                PyModule_AddObject(itomModule, "plotItem", (PyObject *)&PythonPlotItem::PyPlotItemType);
            }

            if (PyType_Ready(&PythonProxy::PyProxyType) >= 0)
            {
                Py_INCREF(&PythonProxy::PyProxyType);
                PythonProxy::PyProxy_addTpDict(PythonProxy::PyProxyType.tp_dict);
                PyModule_AddObject(itomModule, "proxy", (PyObject *)&PythonProxy::PyProxyType);
            }

            if (PyType_Ready(&PythonRegion::PyRegionType) >= 0)
            {
                Py_INCREF(&PythonRegion::PyRegionType);
                PythonRegion::PyRegion_addTpDict(PythonRegion::PyRegionType.tp_dict);
                PyModule_AddObject(itomModule, "region", (PyObject *)&PythonRegion::PyRegionType);
            }

            if (PyType_Ready(&PythonRgba::PyRgbaType) >= 0)
            {
                Py_INCREF(&PythonRgba::PyRgbaType);
                //PythonRgba::PyRegion_addTpDict(PythonRegion::PyRgbaType.tp_dict);
                PyModule_AddObject(itomModule, "rgba", (PyObject *)&PythonRgba::PyRgbaType);
            }

#if ITOM_POINTCLOUDLIBRARY > 0
            if (PyType_Ready(&PythonPCL::PyPointType) >= 0)
            {
                Py_INCREF(&PythonPCL::PyPointType);
                PythonPCL::PyPoint_addTpDict(PythonPCL::PyPointType.tp_dict);
                PyModule_AddObject(itomModule, "point", (PyObject *)&PythonPCL::PyPointType);
            }

            if (PyType_Ready(&PythonPCL::PyPointCloudType) >= 0)
            {
                Py_INCREF(&PythonPCL::PyPointCloudType);
                PythonPCL::PyPointCloud_addTpDict(PythonPCL::PyPointCloudType.tp_dict);
                PyModule_AddObject(itomModule, "pointCloud", (PyObject *)&PythonPCL::PyPointCloudType);
            }

            if (PyType_Ready(&PythonPCL::PyPolygonMeshType) >= 0)
            {
                Py_INCREF(&PythonPCL::PyPolygonMeshType);
                PythonPCL::PyPolygonMesh_addTpDict(PythonPCL::PyPolygonMeshType.tp_dict);
                PyModule_AddObject(itomModule, "polygonMesh", (PyObject *)&PythonPCL::PyPolygonMeshType);
            }
#endif //#if ITOM_POINTCLOUDLIBRARY > 0

#if ITOM_PYTHONMATLAB == 1
            PyImport_AppendInittab("matlab", &PyInit_matlab);
#endif

            //try to add folder "itom-package" to sys.path
            PyObject *syspath = PySys_GetObject("path"); //borrowed reference
            if (syspath)
            {
                if (PyList_Check(syspath))
                {
                    //path to application folder
                    QDir appPath = QDir::cleanPath(QCoreApplication::applicationDirPath());
                    if (appPath.exists())
                    {
                        PyList_Append(syspath, PythonQtConversion::QStringToPyObject(appPath.absolutePath()));
                    }

                    //path to site-packages folder
                    if (appPath.cd("itom-packages"))
                    {
                        PyList_Append(syspath, PythonQtConversion::QStringToPyObject(appPath.absolutePath()));
                    }
                    else
                    {
                        std::cerr << "folder itom-packages could not be found" << std::endl;
                    }
                }
                else
                {
                    std::cerr << "sys.path is no list" <<std::endl;
                }
            }
            else
            {
                std::cerr << "could not get variable sys.path" <<std::endl;
                PyErr_Print();
            }

            //PyImport_AppendInittab("itomDbgWrapper",&PythonEngine::PyInitItomDbg); //!< add all static, known function calls to python-module itomDbgWrapper
            //try to add the module 'frosted' for syntax check
            m_pyModSyntaxCheck = PyImport_ImportModule("itomSyntaxCheck"); //new reference
            if (m_pyModSyntaxCheck == NULL)
            {
                PyErr_Clear();
            }

            // import itoFunctions
            itomFunctions = PyImport_ImportModule("itoFunctions"); // new reference
            if (itomFunctions == NULL)
            {
                (*retValue) += ito::RetVal(ito::retError, 0, tr("the module itoFunctions could not be loaded.").toLatin1().data());
                std::cerr << "the module itoFunctions could not be loaded." << std::endl;
                PyErr_Print();
                PyErr_Clear();
            }

            //!< import itoDbgBase3
            itomDbgModule = PyImport_ImportModule("itoDebugger"); // new reference
            if (itomDbgModule == NULL)
            {
                (*retValue) += ito::RetVal(ito::retError, 0, tr("the module itoDebugger could not be loaded.").toLatin1().data());
                std::cerr << "the module itoDebugger could not be loaded." <<std::endl;
                PyErr_Print();
            }
            else
            {
                itomDbgDict = PyModule_GetDict(itomDbgModule); //!< borrowed reference
                itomDbgClass = PyDict_GetItemString(itomDbgDict, "itoDebugger"); // borrowed reference
                itomDbgDict = NULL;
                if (itomDbgClass == NULL)
                {
                    (*retValue) += ito::RetVal(ito::retError, 0, tr("the module itoDebugger could not be loaded.").toLatin1().data());
                    PyErr_Print();
                    //printPythonError(PySys_GetObject("stderr"));
                }
                else
                {
                    itomDbgInstance = PyObject_CallObject(itomDbgClass, NULL); //!< http://bytes.com/topic/python/answers/649229-_pyobject_new-pyobject_init-pyinstance_new-etc, new reference
                }
            }

            (*retValue) += stringEncodingChanged();

            runString("from itom import *");

            // Setup for 
            PyObject *itomDir = PyObject_Dir(itomModule); //new ref
            if (itomDir && PyList_Check(itomDir))
            {
                Py_ssize_t len = PyList_GET_SIZE(itomDir);

                
                m_itomMemberClasses.clear();

                for (Py_ssize_t l = 0; l < len; ++l)
                {
                    PyObject *dirItem = PyList_GET_ITEM(itomDir, l); //borrowed ref
                    bool ok;
                    QString string = PythonQtConversion::PyObjGetString(dirItem, false, ok);

                    if (ok)
                    {
                        if (!string.startsWith("__"))
                        {
                            m_itomMemberClasses.append(string + ",");
                        }
                    }
                }
                m_itomMemberClasses.remove(m_itomMemberClasses.length()-1, 1);
            }
               
            Py_XDECREF(itomDir);

            started = true;
        }
        else
        {
            (*retValue) += ito::RetVal(ito::retError, 2, tr("deadlock in python setup.").toLatin1().data());
        }
    }

    return;
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");

    m_includeItom = settings.value("syntaxIncludeItom", true).toBool();

    settings.endGroup();
}

void PythonEngine::propertiesChanged()
{
    readSettings();
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::scanAndRunAutostartFolder(QString currentDirAfterScan /*= QString()*/)
{
    //store current directory
    QString currentDir;
    if (currentDirAfterScan.isEmpty())
    {
        currentDir = QDir::current().canonicalPath();
    }
    else
    {
        currentDir = currentDirAfterScan;
    }
    QStringList files;
    QStringList absoluteFilePaths;
    QDir folder;

    //scan autostart-folder of itom-packages folder an execute every py-file
    folder = QDir::cleanPath(QCoreApplication::applicationDirPath());
    if (folder.cd("itom-packages"))
    {
        if (folder.cd("autostart"))
        {
            folder.setNameFilters(QStringList("*.py"));
            folder.setFilter(QDir::Files | QDir::NoDotAndDotDot);
            files = folder.entryList();
            foreach (const QString &fileName, files)
            {
                absoluteFilePaths.append(folder.absoluteFilePath(fileName));
            }

            if (absoluteFilePaths.size() > 0)
            {
                pythonRunFile(absoluteFilePaths.join(";"));
            }
        }
    }

    //reset current directory if any autostart scripts have been loaded or if the currentDirAfterScan
    //string is given, since then, the programmer wants this method to always set the currentDirAfterScan
    //at the end.
    if (absoluteFilePaths.count() > 0 || !currentDirAfterScan.isEmpty())
    {
        QDir::setCurrent(currentDir);
        emit pythonCurrentDirChanged();
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::pythonShutdown(ItomSharedSemaphore *aimWait)
{
    ItomSharedSemaphoreLocker locker(aimWait);

    RetVal retValue(retOk);
    if (started)
    {
		//delete all remaining weak references in m_pyFuncWeakRefHashes (if available)
		QHash<QString, QPair<PyObject*,PyObject*> >::iterator it = m_pyFuncWeakRefHashes.begin();
		while(it != m_pyFuncWeakRefHashes.end())
		{
			Py_XDECREF(it->first);
			Py_XDECREF(it->second);
			it= m_pyFuncWeakRefHashes.erase(it);
		}

        Py_XDECREF(itomDbgInstance);
        itomDbgInstance = NULL;
        Py_XDECREF(itomDbgModule);
        itomDbgModule = NULL;

        Py_XDECREF(itomModule);
        itomModule = NULL;

        Py_XDECREF(itomFunctions);
        itomFunctions = NULL;

        Py_XDECREF(m_pyModSyntaxCheck);
        m_pyModSyntaxCheck = NULL;

        Py_XDECREF(m_pyModGC);
        m_pyModGC = NULL;

        if (Py_IsInitialized())
        {
            if (PyErr_Occurred())
            {
                PyErr_Print();
            }
            PyErr_Clear();
            Py_Finalize();
        }
        else
        {
            retValue += RetVal(retError, 1, tr("Python not initialized").toLatin1().data());
        }

        Py_XDECREF(dictUnicode);

        //delete[] PythonAdditionalModuleITOM; //!< must be alive until the end of the python session!!! (http://coding.derkeiler.com/Archive/Python/comp.lang.python/2007-01/msg01036.html)

        mainModule = NULL;
        mainDictionary = NULL;
        localDictionary = NULL;
        globalDictionary = NULL;

        PythonEngine::instatiated.unlock();

        started = false;
    }

    if (aimWait)
    {
        aimWait->returnValue = retValue;
        aimWait->release();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::pythonAddBuiltinMethods()
{
    //nach: http://code.activestate.com/recipes/54352-defining-python-class-methods-in-c/



    ////!< insert all dynamic function calls to PythonAdditionalModuleITOM, which must be "alive" until Py_Finalize()
    //int numberOfDynamicElements = 1;
    //PythonAdditionalModuleITOM = new PyMethodDef[numberOfDynamicElements];
    //PythonAdditionalModuleITOM[0].ml_doc = NULL;
    //PythonAdditionalModuleITOM[0].ml_flags = METH_VARARGS;
    //PythonAdditionalModuleITOM[0].ml_meth = PythonEngine::pythonInterfaceWrapper;
    //PythonAdditionalModuleITOM[0].ml_name = "general";


    //addMethodToModule(&PythonAdditionalModuleITOM[0]);

    //!< insert all dynamic function calls to PythonAdditionalModuleITOM, which must be "alive" until Py_Finalize()
    /*int numberOfDynamicElements = 1;
    PythonAdditionalModuleITOM = new PyMethodDef[numberOfDynamicElements];
    PythonAdditionalModuleITOM[0].ml_doc = NULL;
    PythonAdditionalModuleITOM[0].ml_flags = METH_VARARGS;
    PythonAdditionalModuleITOM[0].ml_meth = PythonEngine::PyNullMethod;
    PythonAdditionalModuleITOM[0].ml_name = "smoothingFilter";


    addMethodToModule(&PythonAdditionalModuleITOM[0]);*/



    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::addMethodToModule(PyMethodDef *def)
{
    //nach: http://code.activestate.com/recipes/54352-defining-python-class-methods-in-c/

    PyObject *moduleDict = NULL; //!< module-dictionary (borrowed)
    PyObject *func = NULL; //!< function object for new dynamic function call (new reference)

    moduleDict = PyModule_GetDict(itomModule);
    func = PyCFunction_NewEx(def , PyBytes_FromString(def->ml_name) , PyBytes_FromString("itom"));
    PyDict_SetItemString(moduleDict , def->ml_name , func);
    //PyDict_SetItemString(moduleDict, def->ml_name, Py_None);


    Py_XDECREF(func);
    func = NULL;

    moduleDict = NULL;

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::delMethodFromModule(const char* ml_name)
{
    RetVal retValue = RetVal(retOk);

    //nach: http://code.activestate.com/recipes/54352-defining-python-class-methods-in-c/
    PyObject *moduleDict = NULL; //!< module-dictionary (borrowed)

    moduleDict = PyModule_GetDict(itomModule);

    if (PyDict_DelItemString(moduleDict, ml_name))
    {
        retValue += RetVal(retError, 1, tr("method name not found in builtin itom").toLatin1().data());
    }

    moduleDict = NULL;

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::stringEncodingChanged()
{
    ito::RetVal retval;

    enum unicodeEncodings { utf_16, utf_16_LE, utf_16_BE, utf_32, utf_32_BE, utf_32_LE, other };
    PythonQtConversion::unicodeEncodings encodingType = PythonQtConversion::other;
    QByteArray encodingName = "";
    bool found = false;
//    QList<QByteArray> qtCodecNames = QTextCodec::codecForCStrings()->aliases();
//    qtCodecNames.append(QTextCodec::codecForCStrings()->name());
    //QList<QByteArray> qtCodecNames = QTextCodec::availableCodecs();
	QTextCodec *codec = NULL;
	QByteArray curQtCodec;

#if linux
    // google says this should work on linux ... didn't test it
    codec = QTextCodec::codecForName(nl_langinfo(CODESET));
#else
    codec = QTextCodec::codecForLocale();
#endif

	if (codec)
	{
		QList<QByteArray> aliases;
#ifndef linux
		if (codec->name() == "System" || codec->name() == "system")
		{
			aliases << "ISO-8859-1"; //with Qt4 and Windows, the default codec is called System and is then mapped to ISO-8859-1
		}
#endif
		aliases << codec->name() << codec->aliases();
		foreach(const QByteArray &qtCodecName, aliases)
		{
			//check the following default codecs (mbcs is not supported by Qt, since not in the table http://www.iana.org/assignments/character-sets/character-sets.xml)
			if (qtCodecName == "UTF-8")
			{
				encodingType = PythonQtConversion::utf_8;
				encodingName = "utf_8";
			}
			else if (qtCodecName == "ISO-8859-1" || qtCodecName == "latin1")
			{
				encodingType = PythonQtConversion::latin_1;
				encodingName = "latin_1";
			}
			else if (qtCodecName == "US-ASCII")
			{
				encodingType = PythonQtConversion::ascii;
				encodingName = "ascii";
			}
			else if (qtCodecName == "UTF-16")
			{
				encodingType = PythonQtConversion::utf_16;
				encodingName = "utf_16";
			}
			else if (qtCodecName == "UTF-16LE")
			{
				encodingType = PythonQtConversion::utf_16_LE;
				encodingName = "utf_16_le";
			}
			else if (qtCodecName == "UTF-16BE")
			{
				encodingType = PythonQtConversion::utf_16_BE;
				encodingName = "utf_16_be";
			}
			else if (qtCodecName == "UTF-32")
			{
				encodingType = PythonQtConversion::utf_32;
				encodingName = "utf_32";
			}
			//else if (qtCodecNames.contains("UTF-32BE"))
			else if (qtCodecName == "UTF-32BE")
			{
				encodingType = PythonQtConversion::utf_32_BE;
				encodingName = "utf_32_be";
			}
			//else if (qtCodecNames.contains("UTF-32LE"))
			else if (qtCodecName == "UTF-32LE")
			{
				encodingType = PythonQtConversion::utf_32_LE;
				encodingName = "utf_32_le";
			}

			if (encodingType != PythonQtConversion::other)
			{
				break;
			}
		}

		if (encodingType == PythonQtConversion::other)
		{
			encodingType = PythonQtConversion::other;
			found = false;

			foreach (const QByteArray &ba, aliases)
			{
				if (PyCodec_KnownEncoding(ba.data()))
				{
					encodingName = ba;
					found = true;
					break;
				}
			}
        
			if (!found)
			{
				if(codec->name().isEmpty())
				{
					retval += RetVal(ito::retWarning, 0, tr("Qt text encoding not compatible to python. Python encoding is set to latin 1").toLatin1().data());
				}
				else
				{
					retval += RetVal(ito::retWarning, 0, tr("Qt text encoding %1 not compatible to python. Python encoding is set to latin 1").arg(codec->name().data()).toLatin1().data());
				}
					
				encodingType = PythonQtConversion::latin_1;
				encodingName = "latin_1";
			}
		}
	}
	else
	{
		retval += ito::RetVal(ito::retWarning,0,"default text codec could not be obtained. Latin1 is used");
		encodingType = PythonQtConversion::latin_1;
		encodingName = "latin_1";
	}
	
	PythonQtConversion::textEncoding = encodingType;
	PythonQtConversion::textEncodingName = encodingName;

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<int> PythonEngine::parseAndSplitCommandInMainComponents(const char *str, QByteArray &encoding) const
{
    //see http://docs.python.org/devguide/compiler.html
    _node *n = PyParser_SimpleParseString(str, Py_file_input); 
    _node *n2 = n;
    if (n==NULL)
    {
        return QList<int>();
    }

    if (TYPE(n) == 335) //encoding declaration, this is one level higher
    {
        n2 = CHILD(n,0);
        encoding = n->n_str;
    }
    else
    {
        encoding = QByteArray();
    }

    QList<int> ret;
    _node *temp;
    for (int i = 0 ; i < NCH(n2) ; i++)
    {
        temp = CHILD(n2,i);
        if (TYPE(temp) != 4 && TYPE(temp) != 0) //include of graminit.h leads to error if included in header-file, type 0 and 4 seems to be empty line and end of file or something else
        {
            ret.append(temp->n_lineno);
        }
    }

    PyNode_Free(n);
    return ret;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::runString(const QString &command)
{
    RetVal retValue = RetVal(retOk);

    PyObject *mainDict = getGlobalDictionary();
    PyObject *localDict = getLocalDictionary();
    PyObject *result = NULL;

    if (mainDict == NULL)
    {
        std::cerr << "main dictionary is empty. python probably not started" << std::endl;
        retValue += RetVal(retError, 1, tr("main dictionary is empty").toLatin1().data());
    }
    else
    {
        //input to PyRun_String must be UTF8
        if (command.contains('\n')) //multi-line commands must have the Py_file_input flag
        {
            result = PyRun_String(command.toUtf8().data(), Py_file_input /*Py_single_input*/ , mainDict, localDict); //Py_file_input is used such that multi-line commands (separated by \n) are evaluated
        }
        else //this command is a single line command, then Py_single_input must be set, such that the output of any command is printed in the next line, else this output is supressed (if no print command is executed)
        {
            result = PyRun_String(command.toUtf8().data(), Py_single_input, mainDict , localDict); //Py_file_input is used such that multi-line commands (separated by \n) are evaluated

        }

        if (result == NULL)
        {
                if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_SystemExit))
                {
                    std::cerr << "wish to exit (not possible yet)" << std::endl;
                    retValue += RetVal(retError, 2, tr("exiting desired.").toLatin1().data());
                }
                else
                {
                    PyErr_Print();
                    retValue += RetVal(retError, 2, tr("error while evaluating python string.").toLatin1().data());
                }
                PyErr_Clear();

        }


        Py_XDECREF(result);
    }

    return retValue;
}

////----------------------------------------------------------------------------------------------------------------------------------
///**
//* checks syntax of python-file.
//* @param pythonFileName Name of Python-File
//*/
//RetVal PythonEngine::syntaxCheck(char* pythonFileName)
//{
//    RetVal retValue = RetVal(retOk);
//
//    PyObject* result = NULL;
//
//    if (itomDbgInstance == NULL)
//    {
//        return RetVal(retError);
//    }
//    else
//    {
//        result = PyObject_CallMethod(itomDbgInstance, "compileScript", "s", pythonFileName);
//
//        if (result == NULL) //!< syntax error
//        {
//            PyErr_Print();
//
//            retValue += RetVal(retError);
//            //printPythonError(std::cout);
//        }
//    }
//
//    return retValue;
//}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::runPyFile(const QString &pythonFileName)
{
    PyObject* result = NULL;
    PyObject* compile = NULL;
    RetVal retValue = RetVal(retOk);

    int method = 2; //1: direct, 2: by itomDebugger.py (sets the system path to folder of executed file)

    QString desiredPath = QFileInfo(pythonFileName).canonicalPath();
    QString currentDir = QDir::current().canonicalPath();

    if (desiredPath != currentDir)
    {
        QDir::setCurrent(desiredPath);
        emit pythonCurrentDirChanged();
    }

    if (method == 1)
    {
        //direct call
        QFile data(pythonFileName);
        if (data.exists() == false)
        {
            retValue += RetVal(retError, 0, tr("file does not exist").toLatin1().data());
        }
        else
        {
            if (data.open(QFile::ReadOnly))
            {
                QTextStream stream(&data);
                QByteArray fileContent = stream.readAll().toLatin1();
                QByteArray filename = data.fileName().toLatin1();
                data.close();

                compile = Py_CompileString(fileContent.data(), filename.data(), Py_file_input);
                if (compile == NULL)
                {
                    if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_SystemExit))
                    {
                        std::cerr << "wish to exit (not possible yet)" << std::endl;
                        retValue += RetVal(retError);
                    }
                    else
                    {
                        PyObject *oldTBLimit = PySys_GetObject("tracebacklimit");

                        modifyTracebackDepth(2, true);
                        PyErr_Print();

                        if (oldTBLimit != NULL)
                        {
                            PySys_SetObject("tracebacklimit", oldTBLimit);
                        }
                        retValue += RetVal(retError);
                        //printPythonError(PySys_GetObject("stderr"));
                    }
                    PyErr_Clear();
                }
                else
                {
                    result = PyEval_EvalCode(compile, mainDictionary, NULL);

                    if (result == NULL)
                    {
                        if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_SystemExit))
                        {
                            std::cerr << "wish to exit (not possible yet)" << std::endl;
                            retValue += RetVal(retError);
                        }
                        else
                        {
                            //PyObject *oldTBLimit = PySys_GetObject("tracebacklimit");
                            //modifyTracebackDepth(2, true);
                            PyErr_Print();

                            /*if (oldTBLimit != NULL)
                            {
                                PySys_SetObject("tracebacklimit", oldTBLimit);
                            }*/
                            retValue += RetVal(retError);
                            //printPythonError(PySys_GetObject("stderr"));
                        }
                        PyErr_Clear();
                    }

                    Py_XDECREF(result);
                    Py_XDECREF(compile);
                }
            }
            else
            {
                retValue += RetVal(retError, 0, tr("file could not be opened in readonly-mode").toLatin1().data());
            }
        }
    }
    else if (method == 2)
    {

        if (itomDbgInstance == NULL)
        {
            return RetVal(retError);
        }
        else
        {
            result = PyObject_CallMethod(itomDbgInstance, "runScript", "s", pythonFileName.toUtf8().data()); //"s" requires UTF8 encoded char*

            if (result == NULL)
            {
                if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_SystemExit))
                {
                    std::cerr << "wish to exit (not possible yet)" << std::endl;
                    retValue += RetVal(retError);
                }
                else
                {
                    PyObject *oldTBLimit = PySys_GetObject("tracebacklimit");

                    modifyTracebackDepth(2, true);
                    PyErr_Print();

                    if (oldTBLimit != NULL)
                    {
                        PySys_SetObject("tracebacklimit", oldTBLimit);
                    }
                    retValue += RetVal(retError);
                    //printPythonError(PySys_GetObject("stderr"));
                }
                PyErr_Clear();
            }

            Py_XDECREF(result);
        }
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::debugFunction(PyObject *callable, PyObject *argTuple)
{
    PyObject* result = NULL;
    RetVal retValue = RetVal(retOk);

    if (itomDbgInstance == NULL)
    {
        return RetVal(retError);
    }
    else
    {
        //!< first, clear all existing breakpoints
        result = PyObject_CallMethod(itomDbgInstance, "clear_all_breaks", "");
        if (result == NULL)
        {
            std::cerr << tr("Error while clearing all breakpoints in itoDebugger.").toLatin1().data() << "\n" << std::endl;
            printPythonErrorWithoutTraceback(); //traceback is sense-less, since the traceback is in itoDebugger.py only!
            return RetVal(retError);
        }

        //!< submit all breakpoints
        QList<BreakPointItem> bp = bpModel->getBreakpoints();
        QList<BreakPointItem>::iterator it;
        int pyBpNumber;
        QModelIndex modelIndex;
        int row = 0;
        for (it = bp.begin() ; it != bp.end() ; ++it)
        {
            if ((*it).pythonDbgBpNumber==-1)
            {
                retValue += pythonAddBreakpoint((*it).filename, (*it).lineno, (*it).enabled, (*it).temporary, (*it).condition, (*it).ignoreCount, pyBpNumber);
                bpModel->setPyBpNumber(*it,pyBpNumber);

                if (retValue.containsError())
                {
                    return retValue;
                }
            }
            row++;
        }

        //!< setup connections for live-changes in breakpoints
        setupBreakPointDebugConnections();

        result = PyObject_CallMethod(itomDbgInstance, "debugFunction", "OO", callable, argTuple);

        clearDbgCmdLoop();

        if (result == NULL) //!< syntax error
        {
            if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_SystemExit))
            {
                std::cerr << "wish to exit (not possible yet)" << std::endl;
                retValue += RetVal(retError);
            }
            else
            {
                PyObject *oldTBLimit = PySys_GetObject("tracebacklimit");

                modifyTracebackDepth(3, true);
                PyErr_Print();

                if (oldTBLimit != NULL)
                {
                    PySys_SetObject("tracebacklimit", oldTBLimit);
                }
                retValue += RetVal(retError);
            }
            PyErr_Clear();
        }

        setGlobalDictionary();
        setLocalDictionary(NULL);

        //!< disconnect connections for live-changes in breakpoints
        shutdownBreakPointDebugConnections();
        bpModel->resetAllPyBpNumbers();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::runFunction(PyObject *callable, PyObject *argTuple)
{
    RetVal retValue = RetVal(retOk);

    PyObject *ret = PyObject_CallObject(callable, argTuple);
    if (ret == NULL)
    {
        PyErr_Print();
        retValue += RetVal(retError);
    }

    Py_XDECREF(ret);
    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::debugFile(const QString &pythonFileName)
{
    PyObject* result = NULL;
    RetVal retValue = RetVal(retOk);

    QString desiredPath = QFileInfo(pythonFileName).canonicalPath();
    QString currentDir = QDir::current().canonicalPath();

    if (desiredPath != currentDir)
    {
        QDir::setCurrent(desiredPath);
        emit pythonCurrentDirChanged();
    }

    if (itomDbgInstance == NULL)
    {
        return RetVal(retError);
    }
    else
    {
        //!< first, clear all existing breakpoints
        result = PyObject_CallMethod(itomDbgInstance, "clear_all_breaks", "");
        if (result == NULL)
        {
            std::cerr << tr("Error while clearing all breakpoints in itoDebugger.").toLatin1().data() << "\n" << std::endl;
            printPythonErrorWithoutTraceback(); //traceback is sense-less, since the traceback is in itoDebugger.py only!
            return RetVal(retError);
        }

        //!< submit all breakpoints
        QList<BreakPointItem> bp = bpModel->getBreakpoints();
        QList<BreakPointItem>::iterator it;
        int pyBpNumber;
        QModelIndex modelIndex;
        int row = 0;
        for (it = bp.begin() ; it != bp.end() ; ++it)
        {
            if ((*it).pythonDbgBpNumber==-1)
            {
                retValue += pythonAddBreakpoint((*it).filename, (*it).lineno, (*it).enabled, (*it).temporary, (*it).condition, (*it).ignoreCount, pyBpNumber);
                bpModel->setPyBpNumber(*it,pyBpNumber);

                if (retValue.containsError()) //error occurred, but already printed
                {
                    return retValue;
                }
            }
            row++;
        }

        //!< setup connections for live-changes in breakpoints
        setupBreakPointDebugConnections();

        result = PyObject_CallMethod(itomDbgInstance, "debugScript", "s", pythonFileName.toUtf8().data()); //"s" requires utf-8 encoded string

        clearDbgCmdLoop();

        if (result == NULL) //!< syntax error
        {
            if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_SystemExit))
            {
                std::cerr << "wish to exit (not possible yet)" << std::endl;
                retValue += RetVal(retError);
            }
            else
            {
                PyObject *oldTBLimit = PySys_GetObject("tracebacklimit");

                modifyTracebackDepth(3, true);
                PyErr_Print();

                if (oldTBLimit != NULL)
                {
                    PySys_SetObject("tracebacklimit", oldTBLimit);
                }
                retValue += RetVal(retError);
            }
            PyErr_Clear();
        }

        setGlobalDictionary();
        setLocalDictionary(NULL);

        //!< disconnect connections for live-changes in breakpoints
        shutdownBreakPointDebugConnections();
        bpModel->resetAllPyBpNumbers();
    }

    return retValue;
}


//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::debugString(const QString &command)
{
    PyObject* result = NULL;
    RetVal retValue = RetVal(retOk);

    if (itomDbgInstance == NULL)
    {
        return RetVal(retError);
    }
    else
    {
        //!< first, clear all existing breakpoints
        result = PyObject_CallMethod(itomDbgInstance, "clear_all_breaks", "");
        if (result == NULL)
        {
            std::cerr << tr("Error while clearing all breakpoints in itoDebugger.").toLatin1().data() << "\n" << std::endl;
            printPythonErrorWithoutTraceback(); //traceback is sense-less, since the traceback is in itoDebugger.py only!
            return RetVal(retError);
        }

        //!< submit all breakpoints
        QList<BreakPointItem> bp = bpModel->getBreakpoints();
        QList<BreakPointItem>::iterator it;
        int pyBpNumber;
        QModelIndex modelIndex;
        int row = 0;
        for (it = bp.begin() ; it != bp.end() ; ++it)
        {
            if ((*it).pythonDbgBpNumber==-1)
            {
                retValue += pythonAddBreakpoint((*it).filename, (*it).lineno, (*it).enabled, (*it).temporary, (*it).condition, (*it).ignoreCount, pyBpNumber);
                bpModel->setPyBpNumber(*it,pyBpNumber);

                if (retValue.containsError())
                {
                    return retValue;
                }
            }
            row++;
        }

        //!< setup connections for live-changes in breakpoints
        setupBreakPointDebugConnections();

        result = PyObject_CallMethod(itomDbgInstance, "debugString", "s", command.toUtf8().data()); //command must be UTF8

        clearDbgCmdLoop();

        if (result == NULL) //!< syntax error
        {
            if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_SystemExit))
            {
                std::cerr << "wish to exit (not possible yet)" << std::endl;
                retValue += RetVal(retError);
            }
            else
            {
                PyObject *oldTBLimit = PySys_GetObject("tracebacklimit");

                modifyTracebackDepth(3, true);
                PyErr_Print();

                if (oldTBLimit != NULL)
                {
                    PySys_SetObject("tracebacklimit", oldTBLimit);
                }
                retValue += RetVal(retError);
            }
            PyErr_Clear();
        }

        setGlobalDictionary();
        setLocalDictionary(NULL);

        //!< disconnect connections for live-changes in breakpoints
        shutdownBreakPointDebugConnections();
        bpModel->resetAllPyBpNumbers();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonSyntaxCheck(const QString &code, QPointer<QObject> sender)
{
    if (m_pyModSyntaxCheck)
    {
        QString firstLine;
        if (m_includeItom)
        {
            firstLine = "from itom import " + m_itomMemberClasses + "\n" + code;
        }
        else
        {
            firstLine = code;
        }
        PyObject *result = PyObject_CallMethod(m_pyModSyntaxCheck, "check", "s", firstLine.toUtf8().data());

        if (result && PyList_Check(result) && PyList_Size(result) >= 2)
        {
            QString unexpectedErrors;
            QString flakes;

            bool ok;
            unexpectedErrors = PythonQtConversion::PyObjGetString( PyList_GetItem(result,0), false, ok);
            if (!ok)
            {
                unexpectedErrors = "<<error>>";
            }

            flakes = PythonQtConversion::PyObjGetString( PyList_GetItem(result,1), false, ok);
            if (!ok)
            {
                flakes = "<<error>>";
            }
            else
            {   
                if (m_includeItom)
                {   // if itom is automatically included, this block is correcting the line numbers
                    QStringList sFlakes = flakes.split("\n");
                    if (sFlakes.length() > 0)
                    {
                        while (sFlakes.at(0).startsWith("code:1:"))
                        {
                            sFlakes.removeFirst();
                            if (sFlakes.length() == 0)
                            {
                                break;
                            }
                        }
                        for (int i = 0; i < sFlakes.length(); ++i)
                        {
                            QRegExp reg("(code:)(\\d+)");
                            reg.indexIn(sFlakes[i]);
                            int line = reg.cap(2).toInt() - 1;
                            sFlakes[i].replace(QRegExp("code:\\d+:"), "code:"+QString::number(line)+":");
                        }
                        flakes = sFlakes.join("\n");
                    }
                }   // if not, no correction is nessesary
            }
            QObject *s = sender.data();
            if (s)
            {
                QMetaObject::invokeMethod(s, "syntaxCheckResult", Q_ARG(QString, unexpectedErrors), Q_ARG(QString, flakes));
            }
        }
        else if (!result)
        {
            std::cerr << "Error when calling the syntax check module of python\n" << std::endl;
            PyErr_Print();
        }

        Py_XDECREF(result);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::pythonAddBreakpoint(const QString &filename, const int lineno, const bool enabled, const bool temporary, const QString &condition, const int ignoreCount, int &pyBpNumber)
{
    PyObject *result = NULL;

    pyBpNumber = -1;

    if (itomDbgInstance == NULL)
    {
        return RetVal(retError);
    }
    else
    {
        PyObject *PyEnabled = enabled ? Py_True : Py_False;
        PyObject *PyTemporary = temporary ? Py_True : Py_False;

        if (condition == "")
        {
            result = PyObject_CallMethod(itomDbgInstance, "addNewBreakPoint", "siOOOi", filename.toUtf8().data(), lineno+1, PyEnabled, PyTemporary, Py_None, ignoreCount);
        }
        else
        {
            result = PyObject_CallMethod(itomDbgInstance, "addNewBreakPoint", "siOOsi", filename.toUtf8().data(), lineno+1, PyEnabled, PyTemporary, condition.toLatin1().data(), ignoreCount);
        }

        if (result == NULL)
        {
            Py_XDECREF(result);
            std::cerr << tr("Error while transmitting breakpoints to itoDebugger.").toLatin1().data() << "\n" << std::endl;
            printPythonErrorWithoutTraceback(); //traceback is sense-less, since the traceback is in itoDebugger.py only!
            return RetVal(retError);
        }
        else
        {
            long retNumber = PyLong_AsLong(result);
            if (retNumber == -1)
            {
                pyBpNumber = -1;
                Py_XDECREF(result);
                return RetVal(retError);
            }
            else
            {
                //!> retNumber is new pyBpNumber, must now be added to BreakPointModel
                pyBpNumber = static_cast<int>(retNumber);
            }
        }
    }
    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::pythonEditBreakpoint(const int pyBpNumber, const QString &filename, const int lineno, const bool enabled, const bool temporary, const QString &condition, const int ignoreCount)
{
    PyObject *result = NULL;
    if (itomDbgInstance == NULL)
    {
        return RetVal(retError);
    }
    else if (pyBpNumber >= 0)
    {
        PyObject *PyEnabled = enabled ? Py_True : Py_False;
        PyObject *PyTemporary = temporary ? Py_True : Py_False;

        if (condition == "")
        {
            result = PyObject_CallMethod(itomDbgInstance, "editBreakPoint", "isiOOOi", pyBpNumber, filename.toUtf8().data(), lineno+1, PyEnabled, PyTemporary, Py_None, ignoreCount);
        }
        else
        {
            result = PyObject_CallMethod(itomDbgInstance, "editBreakPoint", "isiOOsi", pyBpNumber, filename.toUtf8().data(), lineno+1, PyEnabled, PyTemporary, condition.toLatin1().data(), ignoreCount);
        }

        if (result == NULL)
        {
            Py_XDECREF(result);
            std::cerr << tr("Error while editing breakpoint in itoDebugger.").toLatin1().data() << "\n" << std::endl;
            printPythonErrorWithoutTraceback(); //traceback is sense-less, since the traceback is in itoDebugger.py only!
            return RetVal(retError);
        }
        else
        {
            long retNumber = PyLong_AsLong(result);
            if (retNumber == -1)
            {
                Py_XDECREF(result);
                return RetVal(retError);
            }
        }
    }
    else
    {
        qDebug() << "Breakpoint in file " << filename << ", line " << lineno << " can not be edited since it could not be registered in python (maybe an commented or blank line)";
    }

    Py_XDECREF(result);
    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::pythonDeleteBreakpoint(const int pyBpNumber)
{
    PyObject *result = NULL;
    if (itomDbgInstance == NULL)
    {
        return RetVal(retError);
    }
    else if (pyBpNumber >= 0)
    {
        result = PyObject_CallMethod(itomDbgInstance, "clearBreakPoint", "i", pyBpNumber);
        if (result == NULL)
        {
            Py_XDECREF(result);
            std::cerr << tr("Error while clearing breakpoint in itoDebugger.").toLatin1().data() << "\n" << std::endl;
            printPythonErrorWithoutTraceback(); //traceback is sense-less, since the traceback is in itoDebugger.py only!
            return RetVal(retError);
        }
        else
        {
            long retNumber = PyLong_AsLong(result);
            if (retNumber == -1)
            {
                Py_XDECREF(result);
                return RetVal(retError);
            }

        }
    }
    else
    {
        qDebug() << "Breakpoint could not be deleted. Its python-internal bp-nr is invalid (maybe an commented or blank line).";
    }

    Py_XDECREF(result);
    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::modifyTracebackDepth(int NrOfLevelsToPopAtFront, bool /*showTraceback*/)
{
    if (PyErr_Occurred())
    {
        PyObject* pyErrType = NULL;
        PyObject* pyErrValue = NULL;
        PyObject* pyErrTrace = NULL;
        PyErr_Fetch(&pyErrType, &pyErrValue, &pyErrTrace);

        PyTracebackObject* tb = (PyTracebackObject*)pyErrTrace;

        int depth=0;
//        int line;

        while(tb != NULL)
        {
            depth++;
//            line = tb->tb_lineno;
            tb = tb->tb_next;
        }

        if (depth - NrOfLevelsToPopAtFront > 0 && NrOfLevelsToPopAtFront > -1)
        {
            PySys_SetObject("tracebacklimit", Py_BuildValue("i",depth - NrOfLevelsToPopAtFront));
            PyErr_Restore(pyErrType, pyErrValue, pyErrTrace);
        }
        else if (depth - NrOfLevelsToPopAtFront <= 0 && NrOfLevelsToPopAtFront > -1)
        {
            //PyException_SetTraceback(pyErrValue,Py_None);
            PyErr_Restore(pyErrType, pyErrValue, NULL);
        }

        return RetVal(retOk);
    }
    else
    {
        return RetVal(retError);
    }

}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::checkForPyExceptions()
{
    ito::RetVal retval;

    if (PyErr_Occurred())
    {
        PyObject* pyErrType = NULL;
        PyObject* pyErrValue = NULL;
        PyObject* pyErrTrace = NULL;
//        PyObject* pyErrSubValue = NULL;
        QString errType;
        QString errText;
        QString errLine;

        PyErr_Fetch(&pyErrType, &pyErrValue, &pyErrTrace); //new references
        PyErr_NormalizeException(&pyErrType, &pyErrValue, &pyErrTrace);

        errType = PythonQtConversion::PyObjGetString( pyErrType );
        errText = PythonQtConversion::PyObjGetString( pyErrValue );

        retval += ito::RetVal::format(ito::retError, 0, "%s (%s)", errText.toLatin1().data(), errType.toLatin1().data());

        Py_XDECREF(pyErrTrace);
        Py_DECREF(pyErrType);
        Py_DECREF(pyErrValue);

        PyErr_Clear();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::printPythonErrorWithoutTraceback()
{
    PyObject *exception, *v, *tb;
    tb = NULL;

    if (PyErr_Occurred())
    {
        PyErr_Fetch(&exception, &v, &tb);
        if (exception == NULL)
        {
            return;
        }

        PyErr_NormalizeException(&exception, &v, &tb);
        Py_XDECREF(tb);
        tb = Py_None;
        Py_INCREF(tb);
        PyException_SetTraceback(v, tb);
        if (exception == NULL)
        {
            return;
        }
        PyErr_Display(exception, v, tb);
        Py_XDECREF(exception);
        Py_XDECREF(v);
        Py_XDECREF(tb);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::setGlobalDictionary(PyObject* globalDict)
{
    dictChangeMutex.lock();
    if (globalDict != NULL)
    {
        globalDictionary = globalDict;
    }
    else
    {
        globalDictionary = mainDictionary;
    }
    /*else if (mainModule != NULL)
    {
        globalDictionary = PyModule_GetDict(mainModule);
    }
    else
    {
        globalDictionary = NULL;
    }*/
    dictChangeMutex.unlock();
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::setLocalDictionary(PyObject* localDict)
{
    dictChangeMutex.lock();
    localDictionary = localDict;
    dictChangeMutex.unlock();
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonRunString(QString cmd)
{
    QByteArray ba(cmd.toLatin1());
    if (ba.trimmed().startsWith("#"))
    {
        ba.prepend("pass"); //a single command line leads to an error while execution
    }
    //ba.replace("\\n",QByteArray(1,'\n')); //replace \n by ascii(10) in order to realize multi-line evaluations

    switch (pythonState)
    {
    case pyStateIdle:
        pythonStateTransition(pyTransBeginRun);
        runString(ba.data());
        emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
        pythonStateTransition(pyTransEndRun);
        break;
    case pyStateRunning:
    case pyStateDebugging:
        // no command execution allowed if running or debugging without being in waiting mode
        qDebug() << "it is now allowed to run a python string in mode pyStateRunning or pyStateDebugging";
        break;
    case pyStateDebuggingWaiting:
        pythonStateTransition(pyTransDebugExecCmdBegin);
        runString(ba.data());
        emitPythonDictionary(true, true, getGlobalDictionary(), getLocalDictionary());
        pythonStateTransition(pyTransDebugExecCmdEnd);
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonRunFile(QString filename)
{
    QStringList list;

    switch (pythonState)
    {
    case pyStateIdle:
        pythonStateTransition(pyTransBeginRun);
        list = filename.split(";");
        foreach (const QString &filenameTemp, list)
        {
            if (filenameTemp != "")
            {
                runPyFile(filenameTemp);
            }
        }
        emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
        pythonStateTransition(pyTransEndRun);
        break;
    case pyStateRunning:
    case pyStateDebugging:
    case pyStateDebuggingWaiting:
    case pyStateDebuggingWaitingButBusy:
        // no command execution allowed if running or debugging without being in waiting mode
        qDebug() << "it is now allowed to run a python file in mode pyStateRunning, pyStateDebugging, pyStateDebuggingWaiting or pyStateDebuggingWaitingButBusy";
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonDebugFile(QString filename)
{
    switch (pythonState)
    {
    case pyStateIdle:
        pythonStateTransition(pyTransBeginDebug);
        debugFile(filename);
        emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
        pythonStateTransition(pyTransEndDebug);
        break;
    case pyStateRunning:
    case pyStateDebugging:
    case pyStateDebuggingWaiting:
    case pyStateDebuggingWaitingButBusy:
        // no command execution allowed if running or debugging without being in waiting mode
        qDebug() << "it is now allowed to debug a python file in mode pyStateRunning, pyStateDebugging, pyStateDebuggingWaiting or pyStateDebuggingWaitingButBusy";
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonDebugString(QString cmd)
{
    QByteArray ba = cmd.toLatin1();
    if (ba.trimmed().startsWith("#"))
    {
        ba.prepend("pass"); //a single comment while cause an error in python
    }

    switch (pythonState)
    {
    case pyStateIdle:
        pythonStateTransition(pyTransBeginDebug);
        debugString(cmd.toLatin1().data());
        emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
        pythonStateTransition(pyTransEndDebug);
        break;
    case pyStateRunning:
    case pyStateDebugging:
    case pyStateDebuggingWaiting:
    case pyStateDebuggingWaitingButBusy:
        // no command execution allowed if running or debugging without being in waiting mode
        qDebug() << "it is now allowed to debug a python string in mode pyStateRunning, pyStateDebugging, pyStateDebuggingWaiting or pyStateDebuggingWaitingButBusy";
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonExecStringFromCommandLine(QString cmd)
{
    //QByteArray ba(cmd.toLatin1());
    if (cmd.trimmed().startsWith("#"))
    {
        cmd.prepend("pass"); //a single command line leads to an error while execution
    }
    //ba.replace("\\n",QByteArray(1,'\n')); //replace \n by ascii(10) in order to realize multi-line evaluations

    switch (pythonState)
    {
    case pyStateIdle:

        if (m_executeInternalPythonCodeInDebugMode)
        {
            pythonStateTransition(pyTransBeginDebug);
            debugString(cmd);
            emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
            pythonStateTransition(pyTransEndDebug);
        }
        else
        {
            pythonStateTransition(pyTransBeginRun);
            runString(cmd);
            emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
            pythonStateTransition(pyTransEndRun);
        }
        break;
    case pyStateRunning:
    case pyStateDebugging:
        // no command execution allowed if running or debugging without being in waiting mode
        std::cerr << "it is now allowed to run a python string in mode pyStateRunning or pyStateDebugging\n" << std::endl;
        break;
    case pyStateDebuggingWaiting:
        pythonStateTransition(pyTransDebugExecCmdBegin);
        runString(cmd);
        emitPythonDictionary(true, true, getGlobalDictionary(), getLocalDictionary());
        pythonStateTransition(pyTransDebugExecCmdEnd);
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonDebugFunction(PyObject *callable, PyObject *argTuple)
{
    switch (pythonState)
    {
    case pyStateIdle:
        pythonStateTransition(pyTransBeginDebug);
        debugFunction(callable,argTuple);
        emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
        pythonStateTransition(pyTransEndDebug);
        break;
    case pyStateRunning:
    case pyStateDebugging:
        // no command execution allowed if running or debugging without being in waiting mode
        std::cerr << "it is now allowed to debug a function or python string in mode pyStateRunning or pyStateDebugging\n" << std::endl;
        break;
    case pyStateDebuggingWaiting:
    case pyStateDebuggingWaitingButBusy:
        pythonStateTransition(pyTransDebugExecCmdBegin);
        std::cout << "Function will be executed instead of debugged since another debug session is currently running.\n" << std::endl;
        pythonRunFunction(callable, argTuple);
        pythonStateTransition(pyTransDebugExecCmdEnd);
        // no command execution allowed if running or debugging without being in waiting mode
        //qDebug() << "it is now allowed to debug a python function or method in mode pyStateRunning, pyStateDebugging, pyStateDebuggingWaiting or pyStateDebuggingWaitingButBusy";
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//do not execute this method from another thread, only execute it within python-thread since this method is not thread safe
void PythonEngine::pythonRunFunction(PyObject *callable, PyObject *argTuple)
{
    switch (pythonState)
    {
    case pyStateIdle:
        pythonStateTransition(pyTransBeginRun);
        runFunction(callable, argTuple);
        emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
        pythonStateTransition(pyTransEndRun);
        break;
    case pyStateRunning:
    case pyStateDebugging:
    case pyStateDebuggingWaitingButBusy: //functions (from signal-calls) can be executed whenever another python method is executed (only possible if another method executing python code is calling processEvents. processEvents stops until this "runFunction" has been terminated
        runFunction(callable, argTuple);
        emitPythonDictionary(true, true, getGlobalDictionary(), getLocalDictionary());
        break;
    case pyStateDebuggingWaiting:
        pythonStateTransition(pyTransDebugExecCmdBegin);
        runFunction(callable, argTuple);
        emitPythonDictionary(true, true, getGlobalDictionary(), getLocalDictionary());
        pythonStateTransition(pyTransDebugExecCmdEnd);
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonRunStringOrFunction(QString cmdOrFctHash)
{
    static QString prefix = ":::itomfcthash:::";
    QString hashValue;

    if (cmdOrFctHash.startsWith(prefix))
    {
        hashValue = cmdOrFctHash.mid(prefix.length());
        QHash<QString, QPair<PyObject*,PyObject*> >::iterator it = this->m_pyFuncWeakRefHashes.find(hashValue);
        if (it != m_pyFuncWeakRefHashes.end())
        {
            PyObject *callable = it->first; //PyWeakref_GetObject(*it); //borrowed reference
            PyObject *argTuple = it->second;
            if (argTuple)
            {
                Py_INCREF(argTuple);
            }
            else
            {
                argTuple = PyTuple_New(0); //new ref
            }

            if (callable)
            {
                Py_INCREF(callable);
                pythonRunFunction(callable, argTuple);
                Py_XDECREF(callable);
            }
            else
            {
                std::cerr << "The method associated with the key '" << hashValue.toLatin1().data() << "' does not exist any more\n" << std::endl;
            }
            Py_XDECREF(argTuple);    
        }
        else
        {
            std::cerr << "No action associated with key '" << hashValue.toLatin1().data() << "' could be found in internal hash table\n" << std::endl;
        }
        
    }
    else
    {
        pythonRunString(cmdOrFctHash);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonDebugStringOrFunction(QString cmdOrFctHash)
{
    static QString prefix = ":::itomfcthash:::";
    QString hashValue;

    if (cmdOrFctHash.startsWith(prefix))
    {
        hashValue = cmdOrFctHash.mid(prefix.length());
        QHash<QString, QPair<PyObject*,PyObject*> >::iterator it = this->m_pyFuncWeakRefHashes.find(hashValue);
        if (it != m_pyFuncWeakRefHashes.end())
        {
            PyObject *callable = it->first; //PyWeakref_GetObject(*it); //borrowed reference
            PyObject *argTuple = it->second;
            if (argTuple)
            {
                Py_INCREF(argTuple);
            }
            else
            {
                argTuple = PyTuple_New(0); //new ref
            }
            if (callable)
            {
                Py_INCREF(callable);
                pythonDebugFunction(callable, argTuple);
            }
            else
            {
                std::cerr << "The method associated with the key " << hashValue.toLatin1().data() << " does not exist any more\n" << std::endl;
            }
            Py_XDECREF(argTuple);
            Py_XDECREF(callable);
        }
        else
        {
            std::cerr << "No action associated with key '" << hashValue.toLatin1().data() << "' could be found in internal hash table\n" << std::endl;
        }
        
    }
    else
    {
        pythonDebugString(cmdOrFctHash);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::pythonStateTransition(tPythonTransitions transition)
{
    RetVal retValue(retOk);
    pythonStateChangeMutex.lock();

    switch (pythonState)
    {
    case pyStateIdle:
        if (transition == pyTransBeginRun)
        {
            pythonState = pyStateRunning;
            emit(pythonStateChanged(transition));
        }
        else if (transition == pyTransBeginDebug)
        {
            pythonState = pyStateDebugging;
            emit(pythonStateChanged(transition));
        }
        else
        {
            retValue += RetVal(retError);
        }
        break;
    case pyStateRunning:
        if (transition == pyTransEndRun)
        {
            pythonState = pyStateIdle;
            emit(pythonStateChanged(transition));
        }
        else
        {
            retValue += RetVal(retError);
        }
        break;
    case pyStateDebugging:
        if (transition == pyTransEndDebug)
        {
            pythonState = pyStateIdle;
            emit(pythonStateChanged(transition));
        }
        else if (transition == pyTransDebugWaiting)
        {
            pythonState = pyStateDebuggingWaiting;
            emit(pythonStateChanged(transition));
        }
        else
        {
            retValue += RetVal(retError);
        }
        break;
    case pyStateDebuggingWaiting:
        if (transition == pyTransEndDebug)
        {
            pythonState = pyStateIdle;
            emit(pythonStateChanged(transition));
        }
        else if (transition == pyTransDebugContinue)
        {
            pythonState = pyStateDebugging;
            emit(pythonStateChanged(transition));
        }
        else if (transition == pyTransDebugExecCmdBegin)
        {
            pythonState = pyStateDebuggingWaitingButBusy;
            emit(pythonStateChanged(transition));
        }
        else
        {
            retValue += RetVal(retError);
        }
        break;
    case pyStateDebuggingWaitingButBusy:
        if (transition == pyTransEndDebug)
        {
            pythonState = pyStateIdle;
            emit(pythonStateChanged(transition));
        }
        else if (transition == pyTransDebugExecCmdEnd)
        {
            pythonState = pyStateDebuggingWaiting;
            emit(pythonStateChanged(transition));
        }
        else
        {
            retValue += RetVal(retError);
        }
        break;
    }

    pythonStateChangeMutex.unlock();

    return retValue;
}

////----------------------------------------------------------------------------------------------------------------------------------
//void PythonEngine::setDbgCmd(tPythonDbgCmd dbgCmd)
//{
//    dbgCmdMutex.lock();
//    debugCommand = dbgCmd;
//    dbgCmdMutex.unlock();
//}
//
////----------------------------------------------------------------------------------------------------------------------------------
//void PythonEngine::resetDbgCmd()
//{
//    dbgCmdMutex.lock();
//    debugCommand = pyDbgNone;
//    dbgCmdMutex.unlock();
//
//}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::enqueueDbgCmd(ito::tPythonDbgCmd dbgCmd)
{
    if (dbgCmd != pyDbgNone)
    {
        dbgCmdMutex.lock();
        debugCommandQueue.enqueue(dbgCmd); //if you don't want, that shortcuts are collected in a queue and handled one after the other one, then only enqueue the new command if the queue is empty
        dbgCmdMutex.unlock();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::tPythonDbgCmd PythonEngine::dequeueDbgCmd()
{
    tPythonDbgCmd cmd = pyDbgNone;
    dbgCmdMutex.lock();
    if (debugCommandQueue.length()>0) 
    {
        cmd = debugCommandQueue.dequeue();
    }
    dbgCmdMutex.unlock();

    return cmd;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool PythonEngine::DbgCommandsAvailable()
{
    bool ret;
    dbgCmdMutex.lock();
    ret = debugCommandQueue.length()>0;
    dbgCmdMutex.unlock();
    return ret;
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::clearDbgCmdLoop()
{
    dbgCmdMutex.lock();
    debugCommandQueue.clear();
    dbgCmdMutex.unlock();
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::breakPointAdded(BreakPointItem bp, int row)
{
    int pyBpNumber;
    pythonAddBreakpoint(bp.filename, bp.lineno, bp.enabled, bp.temporary, bp.condition, bp.ignoreCount, pyBpNumber);
    bpModel->setPyBpNumber(bp, pyBpNumber);
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::breakPointDeleted(QString /*filename*/, int /*lineNo*/, int pyBpNumber)
{
    pythonDeleteBreakpoint(pyBpNumber);
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::breakPointChanged(BreakPointItem /*oldBp*/, ito::BreakPointItem newBp)
{
    pythonEditBreakpoint(newBp.pythonDbgBpNumber, newBp.filename, newBp.lineno, newBp.enabled, newBp.temporary, newBp.condition, newBp.ignoreCount);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::setupBreakPointDebugConnections()
{
    connect(bpModel, SIGNAL(breakPointAdded(BreakPointItem,int)), this, SLOT(breakPointAdded(BreakPointItem,int)));
    connect(bpModel, SIGNAL(breakPointDeleted(QString,int,int)), this, SLOT(breakPointDeleted(QString,int,int)));
    connect(bpModel, SIGNAL(breakPointChanged(BreakPointItem,BreakPointItem)), this, SLOT(breakPointChanged(BreakPointItem, BreakPointItem)));
    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::shutdownBreakPointDebugConnections()
{
    disconnect(bpModel, SIGNAL(breakPointAdded(BreakPointItem,int)), this, SLOT(breakPointAdded(BreakPointItem,int)));
    disconnect(bpModel, SIGNAL(breakPointDeleted(QString,int,int)), this, SLOT(breakPointDeleted(QString,int,int)));
    disconnect(bpModel, SIGNAL(breakPointChanged(BreakPointItem,BreakPointItem)), this, SLOT(breakPointChanged(BreakPointItem, BreakPointItem)));
    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::registerWorkspaceContainer(PyWorkspaceContainer *container, bool registerNotUnregister, bool globalNotLocal)
{
    if (!container) return;

    if (registerNotUnregister)
    {
        if (globalNotLocal && m_mainWorkspaceContainer.contains(container) == false)
        {
            connect(container,SIGNAL(getChildNodes(PyWorkspaceContainer*,QString)),this,SLOT(workspaceGetChildNode(PyWorkspaceContainer*,QString)));
            m_mainWorkspaceContainer.insert(container);
        }
        else if (!globalNotLocal && m_localWorkspaceContainer.contains(container) == false)
        {
            connect(container,SIGNAL(getChildNodes(PyWorkspaceContainer*,QString)),this,SLOT(workspaceGetChildNode(PyWorkspaceContainer*,QString)));
            m_localWorkspaceContainer.insert(container);
        }

        emitPythonDictionary(true, true, getGlobalDictionary(), getLocalDictionary());
    }
    else
    {
        if (globalNotLocal)
        {
            disconnect(container,SIGNAL(getChildNodes(PyWorkspaceContainer*,QString)),this,SLOT(workspaceGetChildNode(PyWorkspaceContainer*,QString)));
            m_mainWorkspaceContainer.remove(container);
        }
        else
        {
            disconnect(container,SIGNAL(getChildNodes(PyWorkspaceContainer*,QString)),this,SLOT(workspaceGetChildNode(PyWorkspaceContainer*,QString)));
            m_localWorkspaceContainer.remove(container);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonEngine::getPyObjectByFullName(bool globalNotLocal, QStringList &fullName)
{
    PyObject *obj = NULL;
    QStringList items = fullName; //.split(".");
    int i=0;
    float f=0.0;
    PyObject *tempObj = NULL;
    bool ok;
    if (items.count() > 0 && items[0] == "") items.removeFirst();

    if (globalNotLocal)
    {
        obj = getGlobalDictionary();
    }
    else
    {
        obj = getLocalDictionary();
    }

    while(items.count() > 0 && obj)
    {
        if (PyDict_Check(obj))
        {
            tempObj = PyDict_GetItemString(obj, items[0].toLatin1().data());
            if (tempObj == NULL) //maybe key is a number
            {
                i = items[0].toInt(&ok);
                if (ok)
                {
                    tempObj = PyDict_GetItem(obj, PyLong_FromLong(i));
                }
                if (!ok || obj == NULL)
                {
                    f = items[0].toFloat(&ok); //here, often, a rounding problem occurres... (this could not be fixed until now)
                    if (ok)
                    {
                        tempObj = PyDict_GetItem(obj, PyFloat_FromDouble(f));
                    }
                }
            }
            obj = tempObj;
        }
        else if (PyList_Check(obj))
        {
            i = items[0].toInt(&ok);
            if (!ok || i<0 || i>=PyList_Size(obj)) return NULL; //error
            obj = PyList_GET_ITEM(obj,i);
        }
        else if (PyTuple_Check(obj))
        {
            i = items[0].toInt(&ok);
            if (!ok || i<0 || i>=PyTuple_Size(obj)) return NULL; //error
            obj = PyTuple_GET_ITEM(obj,i);
        }
        else if (PyObject_HasAttr(obj, dictUnicode))
        {
            PyObject *temp = PyObject_GetAttr(obj, dictUnicode);
            if (temp)
            {
                tempObj = PyDict_GetItemString(temp, items[0].toLatin1().data());
                if (tempObj == NULL) //maybe key is a number
                {
                    i = items[0].toInt(&ok);
                    if (ok)
                    {
                        tempObj = PyDict_GetItem(temp, PyLong_FromLong(i));
                    }
                    if (!ok || obj == NULL)
                    {
                        f = items[0].toFloat(&ok); //here, often, a rounding problem occurres... (this could not be fixed until now)
                        if (ok)
                        {
                            tempObj = PyDict_GetItem(temp, PyFloat_FromDouble(f));
                        }
                    }
                }
                obj = tempObj;
                Py_DECREF(temp);
            }
            else
            {
                return NULL;
            }
        }
        else
        {
            return NULL; //error
        }
        items.removeFirst();
    }

    return obj;
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::workspaceGetChildNode(PyWorkspaceContainer *container, QString fullNameParentItem)
{
    QStringList itemNameSplit = fullNameParentItem.split(container->getDelimiter());
    PyObject *obj = getPyObjectByFullName(container->isGlobalWorkspace(), itemNameSplit);
    
    if (obj)
    {
        container->loadDictionary(obj, fullNameParentItem);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::workspaceGetValueInformation(PyWorkspaceContainer *container, QString fullItemName, QSharedPointer<QString> extendedValue, ItomSharedSemaphore *semaphore)
{
    QStringList itemNameSplit = fullItemName.split(container->getDelimiter());
    PyObject *obj = getPyObjectByFullName(container->isGlobalWorkspace(), itemNameSplit);

    if (obj == NULL)
    {
        *extendedValue = "";
    }
    else
    {
        PyObject *repr = PyObject_Repr(obj);
        if (repr == NULL)
        {
            *extendedValue = "unknown";
        }
        else if (PyUnicode_Check(repr))
        {
            bool ok = false;
            *extendedValue = PythonQtConversion::PyObjGetString(repr,false,ok);
            if (ok == false)
            {
                *extendedValue = "unknown";
            }
            Py_XDECREF(repr);
        
        }
        else
        {
            *extendedValue = "unknown";
            Py_XDECREF(repr);
        }
    }

    if (semaphore)
    {
        semaphore->release();
        semaphore->deleteSemaphore();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::emitPythonDictionary(bool emitGlobal, bool emitLocal, PyObject* globalDict, PyObject* localDict)
{
    //qDebug() << "python emitPythonDictionary. Thread: " << QThread::currentThreadId ();
    if (emitGlobal && m_mainWorkspaceContainer.count() > 0)
    {
        if (globalDict != NULL)
        {
            foreach (ito::PyWorkspaceContainer* cont, m_mainWorkspaceContainer)
            {
                cont->m_accessMutex.lock();
                cont->loadDictionary(globalDict,"");
                cont->m_accessMutex.unlock();
            }
        }
        else
        {
            foreach (ito::PyWorkspaceContainer* cont, m_mainWorkspaceContainer)
            {
                cont->m_accessMutex.lock();
                cont->clear();
                cont->m_accessMutex.unlock();
            }
        }
    }

    if (emitLocal && m_localWorkspaceContainer.count() > 0)
    {
        if (localDict != NULL)
        {
            foreach (ito::PyWorkspaceContainer* cont, m_localWorkspaceContainer)
            {
                cont->m_accessMutex.lock();
                cont->loadDictionary(localDict,"");
                cont->m_accessMutex.unlock();
            }
        }
        else
        {
            foreach (ito::PyWorkspaceContainer* cont, m_localWorkspaceContainer)
            {
                cont->m_accessMutex.lock();
                cont->clear();
                cont->m_accessMutex.unlock();
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonDebugCommand(tPythonDbgCmd cmd)
{
    enqueueDbgCmd(cmd);
};

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonGenericSlot(PyObject* callable, PyObject *argumentTuple)
{
    if (argumentTuple != NULL && !PyTuple_Check(argumentTuple))
    {
        std::cout << "argumentTuple of pythonGenericSlot is no tuple" << std::endl;
        return;
    }

    Py_ssize_t argumentLength = argumentTuple == NULL ? 0 : PyTuple_Size(argumentTuple);
    int numPythonArgs = -1;

    if (PyFunction_Check(callable))
    {
        PyObject* o = callable;
        PyFunctionObject* func = (PyFunctionObject*)o;
        PyCodeObject* code = (PyCodeObject*)func->func_code;
        if (!(code->co_flags & 0x04))
        {
            numPythonArgs = code->co_argcount;
        }
        else
        {
            // variable numbers of arguments allowed
        }
    }
    else if (PyMethod_Check(callable))
    {
        PyObject* o = callable;
        PyMethodObject* method = (PyMethodObject*)o;
        if (PyFunction_Check(method->im_func))
        {
            PyFunctionObject* func = (PyFunctionObject*)method->im_func;
            PyCodeObject* code = (PyCodeObject*)func->func_code;
            if (!(code->co_flags & 0x04))
            {
                numPythonArgs = code->co_argcount - 1; // we subtract one because the first is "self"
            }
            else
            {
            // variable numbers of arguments allowed
            }
        }
    }

    if (numPythonArgs != -1 && numPythonArgs != argumentLength)
    {
        std::cout << "number of arguments does not fit to requested number of arguments by python callable method" << std::endl;
        return;
    }

    PyObject* result = NULL;
    PyErr_Clear();
    result = PyObject_CallObject(callable, argumentTuple);
    if (result == NULL)
    {
        PyErr_Print();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonInterruptExecution() const
{
//    PyGILState_STATE gstate;
//    gstate = PyGILState_Ensure();

    /* Perform Python actions here. */
    //PyErr_SetString(PyExc_KeyboardInterrupt, "User Interrupt");
    PyErr_SetInterrupt();
    /* evaluate result or handle exception */

    /* Release the thread. No Python API allowed beyond this point. */
//    PyGILState_Release(gstate);

    qDebug("PyErr_SetInterrupt() in pythonThread");
};

////----------------------------------------------------------------------------------------------------------------------------------
//PyObject* PythonEngine::checkForTimeoutHelper(ItomSharedSemaphore* semaphore, int timeout, PyObject *retValueOk)
//{
//    if (semaphore && semaphore->wait(timeout))
//    {
//        ItomSharedSemaphore::deleteSemaphore(semaphore);
//        semaphore = NULL;
//        Py_INCREF(retValueOk);
//        return retValueOk;
//    }
//    else
//    {
//        ItomSharedSemaphore::deleteSemaphore(semaphore);
//        semaphore = NULL;
//        if (PyErr_CheckSignals() == -1) //!< check if key interrupt occured
//        {
//            return PyErr_Occurred();
//        }
//        PyErr_SetString(PyExc_RuntimeError, "timeout in ItomSharedSemaphore");
//        return NULL;
//    }
//}

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
PyObject* PythonEngine::PyDbgCommandLoop(PyObject * /*pSelf*/, PyObject *pArgs)
{

    PyObject *self = NULL;
    PyObject *frame = NULL;
    PyObject *frame2 = NULL;
    PyObject* temp;
    PyObject* temp2;
    PyObject* globalDict = NULL;
    PyObject* localDict = NULL;

    tPythonDbgCmd recentDbgCmd = pyDbgNone;

    PythonEngine* pyEngine = PythonEngine::getInstanceInternal();

    long lineno;
    bool ok;
    QString filename;
    Py_INCREF(pArgs);

    if (!PyArg_ParseTuple(pArgs, "OO", &self, &frame))
    {
        Py_XDECREF(pArgs);
        return Py_None;
    }

    temp = PyObject_GetAttrString(frame, "f_lineno");
    lineno = PyLong_AsLong(temp);
    Py_XDECREF(temp);
    temp = PyObject_GetAttrString(frame, "f_code");
    temp2 = PyObject_GetAttrString(temp, "co_filename");
    filename = PythonQtConversion::PyObjGetString(temp2,false,ok);
    Py_XDECREF(temp2);    

    QStringList stack_files;
    IntList     stack_lines;
    QStringList stack_methods;

    stack_files.append(filename);
    stack_lines.append(lineno);

    temp2 = PyObject_GetAttrString(temp, "co_name");
    stack_methods.append(PythonQtConversion::PyObjGetString(temp2,false,ok));
    Py_XDECREF(temp2);

    Py_XDECREF(temp);

    frame2 = PyObject_GetAttrString(frame, "f_back");

    while(frame2 != NULL && frame2 != Py_None)
    {
        temp = PyObject_GetAttrString(frame2, "f_lineno");
        stack_lines.append(PyLong_AsLong(temp));
        Py_XDECREF(temp);
        
        temp = PyObject_GetAttrString(frame2, "f_code");
        temp2 = PyObject_GetAttrString(temp, "co_filename");
        stack_files.append(PythonQtConversion::PyObjGetString(temp2,false,ok));
        Py_XDECREF(temp2);
        temp2 = PyObject_GetAttrString(temp, "co_name");
        stack_methods.append(PythonQtConversion::PyObjGetString(temp2,false,ok));
        Py_XDECREF(temp);
        Py_XDECREF(temp2);

        Py_XDECREF(frame2);
        frame2 = PyObject_GetAttrString(frame2, "f_back");
    }

    Py_XDECREF(frame2);

    emit pyEngine->updateCallStack(stack_files, stack_lines, stack_methods);


    //qDebug() << "Debug stop in file: " << filename << " at line " << lineno;

    ////!< prepare for waiting loop
    //pyEngine->resetDbgCmd();

    pyEngine->pythonStateTransition(pyTransDebugWaiting);
    if (filename != "" && filename.contains("<") == false && filename.contains(">") == false)
    {
        QFileInfo info(filename);
        if (info.exists())
        {
            emit(pyEngine->pythonDebugPositionChanged(info.canonicalFilePath() , lineno));
        }
    }

    globalDict = PyObject_GetAttrString(frame, "f_globals"); //new ref
    localDict = PyObject_GetAttrString(frame, "f_locals"); //new ref

    pyEngine->setLocalDictionary(localDict);
    pyEngine->setGlobalDictionary(globalDict);

    if (filename == "<string>") //indicates that an exception has been thrown while debugging, then let the debugger run and finish in order that the exception is printed out
    {
        if (!PyObject_CallMethod(self, "set_continue", ""))
        {
            PyErr_Print();
        }
    }
    else //proceed the normal debug turnus
    {
        //only actualize workspace if debugger is idle
        if (!pyEngine->DbgCommandsAvailable())
        {
            if (localDict != globalDict)
            {
                pyEngine->emitPythonDictionary(true,true,globalDict,localDict);
            }
            else
            {
                pyEngine->emitPythonDictionary(true,false,globalDict,NULL);
            }
        }

        while(!pyEngine->DbgCommandsAvailable()) //->isValidDbgCmd())
        {
            QCoreApplication::processEvents();
            //QCoreApplication::sendPostedEvents(pyEngine,0);

            if (PyErr_CheckSignals() == -1) //!< check if key interrupt occurred
            {
                pyEngine->clearDbgCmdLoop();
                pyEngine->pythonStateTransition(pyTransDebugContinue);
                Py_XDECREF(pArgs);

                Py_XDECREF(globalDict);
                globalDict = NULL;
                pyEngine->setLocalDictionary(NULL);
                Py_XDECREF(localDict);
                localDict = NULL;
                return PyErr_Occurred();
            }

        }

        recentDbgCmd = pyEngine->dequeueDbgCmd();

        switch (recentDbgCmd)
        {
        case ito::pyDbgStep:
            if (!PyObject_CallMethod(self, "set_step", ""))
            {
                PyErr_Print();
            }
            break;
        case ito::pyDbgContinue:
            if (!PyObject_CallMethod(self, "set_continue", ""))
            {
                PyErr_Print();
            }
            break;
        case ito::pyDbgStepOver:
            if (!PyObject_CallMethod(self, "set_next", "O", frame))
            {
                PyErr_Print();
            }
            break;
        case ito::pyDbgStepOut:
            if (!PyObject_CallMethod(self,"set_return", "O", frame))
            {
                PyErr_Print();
            }
            break;
        case ito::pyDbgQuit:
            if (!PyObject_CallMethod(self,"do_quit", "O", frame)) //!< do_quit instead of set_quit, since one member-variable is set in itoDebugger.py
            {
                PyErr_Print();
            }
            break;
        }
    }

    pyEngine->setGlobalDictionary(NULL); //reset to mainDictionary of itom
    Py_XDECREF(globalDict);
    globalDict = NULL;

    pyEngine->setLocalDictionary(NULL);
    Py_XDECREF(localDict);
    localDict = NULL;

    emit (pyEngine->deleteCallStack());

    //pyEngine->clearDbgCmdLoop();
    //emit(pyThread->pythonDebuggerContinued());
    pyEngine->pythonStateTransition(pyTransDebugContinue);

    Py_XDECREF(pArgs);

    return Py_BuildValue("i", 1);
}


//----------------------------------------------------------------------------------------------------------------------------------
bool PythonEngine::renameVariable(bool globalNotLocal, QString oldKey, QString newKey, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);

    tPythonState oldState = pythonState;
    bool retVal = true;
    PyObject* dict = NULL;
    PyObject* value;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        std::cerr << "it is not allowed to rename a variable in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy\n" << std::endl;
        retVal = false;
    }
    else
    {
        if (pythonState == pyStateIdle)
        {
            pythonStateTransition(pyTransBeginRun);
        }
        else if (pythonState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdBegin);
        }

        if (globalNotLocal)
        {
            dict = getGlobalDictionary();
        }
        else
        {
            dict = getLocalDictionary();
        }

        if (dict == NULL)
        {
            retVal = false;
            std::cerr << "variable " << oldKey.toLatin1().data() << " can not be renamed, since dictionary is not available\n" << std::endl;
        }
        else
        {
            //if (!PyUnicode_IsIdentifier(PyUnicode_FromString(newKey.toLatin1().data())))
            if (!PyUnicode_IsIdentifier(PyUnicode_DecodeLatin1(newKey.toLatin1().data(), newKey.length(), NULL)))
            {
                PyErr_Clear();
                retVal = false;
                std::cerr << "variable name " << newKey.toLatin1().data() << " is invalid.\n" << std::endl;
            }
            else
            {
                if (PyDict_GetItemString(dict, oldKey.toLatin1().data()) == NULL)
                {
                    retVal = false;
                    std::cerr << "variable " << oldKey.toLatin1().data() << " can not be found in dictionary\n" << std::endl;
                }
                else if (PyDict_GetItemString(dict, newKey.toLatin1().data()) != NULL)
                {
                    retVal = false;
                    std::cerr << "variable " << newKey.toLatin1().data() << " already exists in dictionary\n" << std::endl;
                }
                else
                {
                    value = PyDict_GetItemString(dict, oldKey.toLatin1().data());
                    //Py_INCREF(value); //do not increment, since value is already incremented by SetItemString-method.
                    PyDict_SetItemString(dict, newKey.toLatin1().data(), value); //first set new, then delete in order not to loose the reference inbetween
                    PyDict_DelItemString(dict, oldKey.toLatin1().data());
                    

                    if (PyErr_Occurred())
                    {
                        retVal = false;
                        PyErr_Print();
                    }
                }
            }
        }

        if (semaphore != NULL) //release semaphore now, since the following emit command will be a blocking connection, too.
        {
            semaphore->release();
        }

        if (globalNotLocal)
        {
            emitPythonDictionary(true, false, getGlobalDictionary(), NULL);
        }
        else
        {
            emitPythonDictionary(false, true, NULL, getLocalDictionary());
        }

        if (oldState == pyStateIdle)
        {
            pythonStateTransition(pyTransEndRun);
        }
        else if (oldState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdEnd);
        }
    }

    if (semaphore != NULL) semaphore->release();

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool PythonEngine::deleteVariable(bool globalNotLocal, QStringList keys, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);

    tPythonState oldState = pythonState;
    bool retVal = true;
    PyObject* dict = NULL;
    QString key;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        std::cerr << "it is not allowed to delete a variable in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy\n" << std::endl;
        retVal = false;
    }
    else
    {

        if (pythonState == pyStateIdle)
        {
            pythonStateTransition(pyTransBeginRun);
        }
        else if (pythonState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdBegin);
        }

        if (globalNotLocal)
        {
            dict = getGlobalDictionary();
        }
        else
        {
            dict = getLocalDictionary();
        }

        if (dict == NULL)
        {
            retVal = false;
            std::cerr << "variables " << " can not be deleted, since dictionary is not available\n" << std::endl;
        }
        else
        {
            foreach (key, keys)
            {
                PyDict_DelItemString(dict, key.toLatin1().data());

                if (PyErr_Occurred())
                {
                    retVal = false;
                    PyErr_Print();
                    break;
                }
            }

        }

        if (semaphore != NULL) semaphore->release();

        if (globalNotLocal)
        {
            emitPythonDictionary(true, false, getGlobalDictionary(), NULL);
        }
        else
        {
            emitPythonDictionary(false, true, NULL, getLocalDictionary());
        }


        if (oldState == pyStateIdle)
        {
            pythonStateTransition(pyTransEndRun);
        }
        else if (oldState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdEnd);
        }
    }

    if (semaphore != NULL) semaphore->release();

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::saveMatlabVariables(bool globalNotLocal, QString filename, QStringList varNames, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);

    tPythonState oldState = pythonState;
    RetVal retVal;
    PyObject* dict = NULL;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += RetVal(retError,0,"it is not allowed to save a variable in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy");
    }
    else
    {
        if (pythonState == pyStateIdle)
        {
            pythonStateTransition(pyTransBeginRun);
        }
        else if (pythonState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdBegin);
        }

        if (globalNotLocal)
        {
            dict = getGlobalDictionary();
        }
        else
        {
            dict = getLocalDictionary();
        }

        if (dict == NULL)
        {
            retVal += RetVal(retError,0,"variables can not be saved since dictionary is not available");
        }
        else
        {
            //build dictionary, which should be pickled
            PyObject* pyRet;
            PyObject* pArgs = PyTuple_New(3);
            //PyTuple_SetItem(pArgs,0, PyUnicode_FromString(filename.toLatin1().data()));
            PyTuple_SetItem(pArgs,0, PyUnicode_DecodeLatin1(filename.toLatin1().data(), filename.length(), NULL));
            

            PyObject* keyList = PyList_New(0);
            PyObject* valueList = PyList_New(0);
            PyObject* tempElem = NULL;

            for (int i = 0 ; i < varNames.size() ; i++)
            {
                tempElem = PyDict_GetItemString(dict, varNames.at(i).toLatin1().data()); //borrowed

                if (tempElem == NULL)
                {
                    std::cerr << "variable '" << varNames.at(i).toLatin1().data() << "' can not be found in dictionary and will not be exported.\n" << std::endl;
                }
                else
                {
                    
                    //PyList_Append(keyList , PyUnicode_FromString(varNames.at(i).toLatin1().data()));
                    PyList_Append(keyList , PyUnicode_DecodeLatin1(varNames.at(i).toLatin1().data(), varNames.at(i).length(), NULL));
                    PyList_Append(valueList, tempElem);
                }
            }

            PyTuple_SetItem(pArgs,1,valueList);
            PyTuple_SetItem(pArgs,2,keyList);
            pyRet = ito::PythonItom::PySaveMatlabMat(NULL, pArgs);

            retVal += checkForPyExceptions();

            Py_XDECREF(pArgs);
        }


        if (oldState == pyStateIdle)
        {
            pythonStateTransition(pyTransEndRun);
        }
        else if (oldState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdEnd);
        }
    }

    if (semaphore != NULL) 
    {
        semaphore->returnValue = retVal;
        semaphore->release();
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::loadMatlabVariables(bool globalNotLocal, QString filename, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);
    tPythonState oldState = pythonState;
    RetVal retVal;
    PyObject* destinationDict = NULL;
    bool released = false;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += RetVal(retError,0,"it is not allowed to load matlab variables in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy");
    }
    else
    {
        if (pythonState == pyStateIdle)
        {
            pythonStateTransition(pyTransBeginRun);
        }
        else if (pythonState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdBegin);
        }

        if (globalNotLocal)
        {
            destinationDict = getGlobalDictionary();
        }
        else
        {
            destinationDict = getLocalDictionary();
        }

        if (destinationDict == NULL)
        {
            retVal += RetVal(retError,0,"variables can not be load since dictionary is not available");
        }
        else
        {
            //PyObject *pArgs = PyTuple_Pack(1, PyUnicode_FromString(filename.toLatin1().data()));
            PyObject *pArgs = PyTuple_Pack(1, PyUnicode_DecodeLatin1(filename.toLatin1().data(), filename.length(), NULL));
            PyObject *dict = ito::PythonItom::PyLoadMatlabMat(NULL, pArgs);
            Py_DECREF(pArgs);

            retVal += checkForPyExceptions();

            if (dict == NULL || retVal.containsError())
            {
            }
            else
            {
                PyObject *key, *value;
                Py_ssize_t pos = 0;

                while (PyDict_Next(dict, &pos, &key, &value)) //returns borrowed references to key and value.
                {
                    PyDict_SetItem(destinationDict, key, value);
                }
            }

            Py_XDECREF(dict);

            if (semaphore) 
            {
                semaphore->returnValue = retVal;
                semaphore->release();
                released = true;
            }

            if (globalNotLocal)
            {
                emitPythonDictionary(true, false, getGlobalDictionary(), NULL);
            }
            else
            {
                emitPythonDictionary(false, true, NULL, getLocalDictionary());
            }
        }

        if (oldState == pyStateIdle)
        {
            pythonStateTransition(pyTransEndRun);
        }
        else if (oldState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdEnd);
        }
    }

    if (semaphore && !released)
    {
        semaphore->returnValue = retVal;
        semaphore->release();
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::putParamsToWorkspace(bool globalNotLocal, QStringList names, QVector<SharedParamBasePointer > values, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);
    tPythonState oldState = pythonState;
    ito::RetVal retVal;
    PyObject* destinationDict = NULL;
    PyObject* value = NULL;


    bool released = false;

    if (names.size() != values.size())
    {
        retVal += ito::RetVal(ito::retError, 0, tr("The number of names and values must be equal").toLatin1().data());
    }
    else if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += ito::RetVal(ito::retError, 0, tr("It is not allowed to load matlab variables in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy").toLatin1().data());
    }
    else
    {
        if (pythonState == pyStateIdle)
        {
            pythonStateTransition(pyTransBeginRun);
        }
        else if (pythonState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdBegin);
        }

        if (globalNotLocal)
        {
            destinationDict = getGlobalDictionary();
        }
        else
        {
            destinationDict = getLocalDictionary();
        }

        if (destinationDict == NULL)
        {
            retVal = false;
            retVal += ito::RetVal(ito::retError, 0, tr("values cannot be saved since workspace dictionary not available.").toLatin1().data());
        }
        else
        {
            PyObject *existingItem = NULL;

            for (int i=0; i<names.size();i++)
            {
                existingItem = PyDict_GetItemString(destinationDict, names[i].toLatin1().data()); //borrowed ref

                if (existingItem)
                {
                    if (PyFunction_Check(existingItem) || PyCFunction_Check(existingItem))
                    {
                        retVal += ito::RetVal::format(ito::retError, 0, tr("Function '%s' in this workspace can not be overwritten.").toLatin1().data(), names[i].toLatin1().data());
                        break;
                    }
                    else if (PyMethod_Check(existingItem))
                    {
                        retVal += ito::RetVal::format(ito::retError, 0, tr("Method '%s' in this workspace can not be overwritten.").toLatin1().data(), names[i].toLatin1().data());
                        break;
                    }
                    else if (PyType_Check(existingItem))
                    {
                        retVal += ito::RetVal::format(ito::retError, 0, tr("Type or class '%s' in this workspace can not be overwritten.").toLatin1().data(), names[i].toLatin1().data());
                        break;
                    }
                    else if (PyModule_Check(existingItem))
                    {
                        retVal += ito::RetVal::format(ito::retError, 0, tr("Module '%s' in this workspace can not be overwritten.").toLatin1().data(), names[i].toLatin1().data());
                        break;
                    }
                }

                value = PythonParamConversion::ParamBaseToPyObject(*(values[i]));
                if (value == NULL)
                {
                    retVal += ito::RetVal::format(ito::retError, 0, tr("error while transforming value '%s' to PyObject*.").toLatin1().data(), names[i].toLatin1().data());
                }
                else
                {
                    PyDict_SetItemString(destinationDict, names[i].toLatin1().data(), value); //existing is automatically decremented
                    Py_XDECREF(value);
                }
            }

            if (semaphore != NULL) 
            {
                semaphore->returnValue = retVal;
                semaphore->release();
                released = true;
            }

            if (globalNotLocal)
            {
                emitPythonDictionary(true, false, getGlobalDictionary(), NULL);
            }
            else
            {
                emitPythonDictionary(false, true, NULL, getLocalDictionary());
            }
        }

        if (oldState == pyStateIdle)
        {
            pythonStateTransition(pyTransEndRun);
        }
        else if (oldState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdEnd);
        }
    }

    if (semaphore != NULL && !released) 
    {
        semaphore->returnValue = retVal;
        semaphore->release();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::getParamsFromWorkspace(bool globalNotLocal, QStringList names, QVector<int> paramBaseTypes, QSharedPointer<SharedParamBasePointerVector > values, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);
    tPythonState oldState = pythonState;
    ito::RetVal retVal;
    PyObject* sourceDict = NULL;
    PyObject* value = NULL;
    bool released = false;
    QSharedPointer<ito::ParamBase> param;

    values->clear();
    if (paramBaseTypes.size() == 0) paramBaseTypes.fill(0, names.size()); //if types vector is empty, fill it with zeros, such that the type is guessed in PythonParamConversion.

    if (names.size() != paramBaseTypes.size())
    {
        retVal += ito::RetVal(ito::retError, 0, tr("The number of names and types must be equal").toLatin1().data());
    }
    else if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += ito::RetVal(ito::retError, 0, tr("it is not allowed to load matlab variables in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy").toLatin1().data());
    }
    else
    {
        if (pythonState == pyStateIdle)
        {
            pythonStateTransition(pyTransBeginRun);
        }
        else if (pythonState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdBegin);
        }

        if (globalNotLocal)
        {
            sourceDict = getGlobalDictionary();
        }
        else
        {
            sourceDict = getLocalDictionary();
        }

        if (sourceDict == NULL)
        {
            retVal = false;
            retVal += ito::RetVal(ito::retError, 0, tr("values cannot be obtained since workspace dictionary not available.").toLatin1().data());
        }
        else
        {
            for (int i=0; i<names.size();i++)
            {
                value = PyDict_GetItemString (sourceDict, names[i].toLatin1().data()); //borrowed
                if (value == NULL)
                {
                    retVal += ito::RetVal(ito::retError, 0, tr("item '%1' does not exist in workspace.").arg(names[i]).toLatin1().data());
                    break;
                }
                else
                {
                    //non strict conversion, such that numpy-arrays are converted to dataObject, if possible
                    //the value of pyObject is either copied to param, or in case of a pointer-type, a shallow copy of this pointer-type is stored in 
                    //param, and if the param runs out of scope, the special deleter method of QSharedPointer
                    param = PythonParamConversion::PyObjectToParamBase(value, names[i].toLatin1().data(), retVal, paramBaseTypes[i], false);
                    if (!retVal.containsError())
                    {
                        *values << param;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            if (semaphore != NULL) 
            {
                semaphore->returnValue = retVal;
                semaphore->release();
                released = true;
            }

            if (globalNotLocal)
            {
                emitPythonDictionary(true, false, getGlobalDictionary(), NULL);
            }
            else
            {
                emitPythonDictionary(false, true, NULL, getLocalDictionary());
            }
        }

        if (oldState == pyStateIdle)
        {
            pythonStateTransition(pyTransEndRun);
        }
        else if (oldState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdEnd);
        }
    }

    if (semaphore != NULL && !released) 
    {
        semaphore->returnValue = retVal;
        semaphore->release();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::registerAddInInstance(QString varname, ito::AddInBase *instance, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);
    RetVal retVal(retOk);

    tPythonState oldState = pythonState;
    PyObject* dict = NULL;
    PyObject* value = NULL;
    bool globalNotLocal = true; //may also be accessed by parameter, if desired
    QByteArray ba = varname.toLatin1();
    const char* varname2 = ba.data();

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += RetVal(retError, 0, tr("It is not allowed to register an AddIn-instance in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy").toLatin1().data());

        if (semaphore != NULL) //release semaphore now, since the following emit command will be a blocking connection, too.
        {
            semaphore->returnValue = retVal;
            semaphore->release();
        }
    }
    else
    {
        if (pythonState == pyStateIdle)
        {
            pythonStateTransition(pyTransBeginRun);
        }
        else if (pythonState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdBegin);
        }

        if (globalNotLocal)
        {
            dict = getGlobalDictionary();
        }
        else
        {
            dict = getLocalDictionary();
        }

        if (dict == NULL)
        {
            retVal += RetVal(retError, 0, tr("Dictionary is not available").toLatin1().data());
        }
        else
        {
            //if (!PyUnicode_IsIdentifier(PyUnicode_FromString(varname2)))
            if (!PyUnicode_IsIdentifier(PyUnicode_DecodeLatin1(varname2, strlen(varname2), NULL)))
            {
                PyErr_Clear();
                QString ErrStr = tr("variable name '%1' is no valid python variable name.").arg(varname);
                retVal += RetVal(retError, 0, ErrStr.toLatin1().data());
            }
            else
            {
                if (PyDict_GetItemString(dict, varname2) != NULL)
                {
                    QString ErrStr = tr("variable name '%1' already exists in dictionary").arg(varname);
                    retVal += RetVal(retError, 0, ErrStr.toLatin1().data());
                }
                else
                {
                    if (instance->getBasePlugin()->getType() & ito::typeDataIO)
                    {
                        PythonPlugins::PyDataIOPlugin *dataIOPlugin;
                        dataIOPlugin = PyObject_New(PythonPlugins::PyDataIOPlugin, &PythonPlugins::PyDataIOPluginType); //new ref
                        if (dataIOPlugin == NULL)
                        {
                            retVal += RetVal(retError, 0, tr("No instance of python class dataIO could be created").toLatin1().data());
                        }
                        else
                        {
                            dataIOPlugin->base = NULL;
                            instance->getBasePlugin()->incRef(instance);
                            dataIOPlugin->dataIOObj = (ito::AddInDataIO*)instance;
                            value = (PyObject*)dataIOPlugin;
                        }
                    }
                    else if (instance->getBasePlugin()->getType() & ito::typeActuator)
                    {
                        PythonPlugins::PyActuatorPlugin *actuatorPlugin;
                        actuatorPlugin = PyObject_New(PythonPlugins::PyActuatorPlugin, &PythonPlugins::PyActuatorPluginType); //new ref
                        if (actuatorPlugin == NULL)
                        {
                            retVal += RetVal(retError, 0, tr("No instance of python class actuator could be created").toLatin1().data());
                        }
                        else
                        {
                            actuatorPlugin->base = NULL;
                            instance->getBasePlugin()->incRef(instance);
                            actuatorPlugin->actuatorObj = (ito::AddInActuator*)instance;
                            value = (PyObject*)actuatorPlugin;
                        }
                    }
                    else
                    {
                        retVal += RetVal(retError, 0, tr("AddIn must be of type dataIO or actuator").toLatin1().data());
                    }

                    if (!retVal.containsError())
                    {
                        PyDict_SetItemString(dict, varname2, value); //increments reference of value
                        Py_XDECREF(value);
                        if (PyErr_Occurred())
                        {
                            retVal += RetVal(retError, 0, tr("Dictionary is not available").toLatin1().data());
                            PyErr_Print();
                        }
                    }


                }
            }
        }

        if (semaphore != NULL) //release semaphore now, since the following emit command will be a blocking connection, too.
        {
            semaphore->returnValue = retVal;
            semaphore->release();
        }

        if (globalNotLocal)
        {
            emitPythonDictionary(true, false, getGlobalDictionary(), NULL);
        }
        else
        {
            emitPythonDictionary(false, true, NULL, getLocalDictionary());
        }

        if (oldState == pyStateIdle)
        {
            pythonStateTransition(pyTransEndRun);
        }
        else if (oldState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdEnd);
        }
    }


    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::getSysModules(QSharedPointer<QStringList> modNames, QSharedPointer<QStringList> modFilenames, QSharedPointer<IntList> modTypes, ItomSharedSemaphore *semaphore)
{
    RetVal retValue;
    tPythonState oldState = pythonState;
    PyObject *elem;
    bool ok;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retValue += RetVal(retError, 0, tr("it is not allowed to get modules if python is currently executed").toLatin1().data());
    }
    else
    {
        if (pythonState == pyStateIdle)
        {
            pythonStateTransition(pyTransBeginRun);
        }
        else if (pythonState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdBegin);
        }

        //code
        if (itomFunctions == NULL)
        {
            retValue += RetVal(retError, 0, tr("the script itomFunctions.py is not available").toLatin1().data());
        }
        else
        {
            PyObject *result = PyObject_CallMethod(itomFunctions, "getModules", "");

            if (!result)
            {
                retValue += RetVal(retError, 0, tr("error while loading the modules").toLatin1().data());
                PyErr_Print();
            }
            else
            {
                for (Py_ssize_t i = 0; i<PyList_Size(result);i++)
                {
                    elem = PyList_GetItem(result,i); //borrowed
                    modNames->append(PythonQtConversion::PyObjGetString(PyList_GetItem(elem,0),true,ok));
                    modFilenames->append(PythonQtConversion::PyObjGetString(PyList_GetItem(elem,1),true,ok));
                    modTypes->append(PythonQtConversion::PyObjGetInt(PyList_GetItem(elem,2),true,ok));
                }
            }
            Py_XDECREF(result);
        }
        //code

        if (oldState == pyStateIdle)
        {
            pythonStateTransition(pyTransEndRun);
        }
        else if (oldState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdEnd);
        }
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::reloadSysModules(QSharedPointer<QStringList> modNames, ItomSharedSemaphore *semaphore)
{
    RetVal retValue;
    tPythonState oldState = pythonState;
    //PyObject *elem;
    //bool ok;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retValue += RetVal(retError, 0, tr("it is not allowed to get modules if python is currently executed").toLatin1().data());
    }
    else
    {
        if (pythonState == pyStateIdle)
        {
            pythonStateTransition(pyTransBeginRun);
        }
        else if (pythonState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdBegin);
        }

        //code
        if (itomFunctions == NULL)
        {
            retValue += RetVal(retError, 0, tr("the script itomFunctions.py is not available").toLatin1().data());
        }
        else
        {
            PyObject *stringList = PythonQtConversion::QStringListToPyList(*modNames);
            modNames->clear();

            PyObject *result = PyObject_CallMethod(itomFunctions, "reloadModules", "O", stringList);

            if (!result)
            {
                retValue += RetVal(retError, 0, tr("error while reloading the modules").toLatin1().data());
                PyErr_Print();
            }
            else
            {
                
            }
            Py_XDECREF(result);
            Py_XDECREF(stringList);
        }
        //code

        if (oldState == pyStateIdle)
        {
            pythonStateTransition(pyTransEndRun);
        }
        else if (oldState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdEnd);
        }
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::pickleVariables(bool globalNotLocal, QString filename, QStringList varNames, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);

    tPythonState oldState = pythonState;
    RetVal retVal;
    PyObject* dict = NULL;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += ito::RetVal(retError, 0, "it is not allowed to pickle a variable in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy");
    }
    else
    {
        if (pythonState == pyStateIdle)
        {
            pythonStateTransition(pyTransBeginRun);
        }
        else if (pythonState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdBegin);
        }

        if (globalNotLocal)
        {
            dict = getGlobalDictionary();
        }
        else
        {
            dict = getLocalDictionary();
        }

        if (dict == NULL)
        {
            retVal += ito::RetVal(retError, 0, "variables can not be pickled since dictionary is not available");
        }
        else
        {
            //build dictionary, which should be pickled
            PyObject* exportDict = PyDict_New();
            PyObject* tempElem = NULL;

            for (int i = 0 ; i < varNames.size() ; i++)
            {
                tempElem = PyDict_GetItemString(dict, varNames.at(i).toLatin1().data()); //borrowed

                if (tempElem == NULL)
                {
                    std::cerr << "variable '" << varNames.at(i).toLatin1().data() << "' can not be found in dictionary and will not be exported.\n" << std::endl;
                }
                else
                {
                    PyDict_SetItemString(exportDict, varNames.at(i).toLatin1().data(), tempElem); //increments tempElem by itsself
                }
            }

            retVal += pickleDictionary(exportDict, filename);

            PyDict_Clear(exportDict);
            Py_DECREF(exportDict);
            exportDict = NULL;
        }


        if (oldState == pyStateIdle)
        {
            pythonStateTransition(pyTransEndRun);
        }
        else if (oldState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdEnd);
        }
    }

    if (semaphore != NULL)
    {
        semaphore->returnValue = retVal;
        semaphore->release();
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::pickleDictionary(PyObject *dict, QString filename)
{
    RetVal retval;

    if (mainModule == NULL)
    {
        return RetVal(retError, 0, "mainModule is empty or cannot be accessed");
    }

    PyObject* pickleModule = PyImport_AddModule("pickle"); // borrowed reference

    if (pickleModule == NULL)
    {
        retval += checkForPyExceptions();
        return retval;
    }

    PyObject *builtinsModule = PyObject_GetAttrString(mainModule, "__builtins__"); //borrowed

    if (builtinsModule == NULL)
    {
        retval += checkForPyExceptions();
        return retval;
    }

    PyObject* openMethod = PyDict_GetItemString(PyModule_GetDict(builtinsModule), "open"); //borrowed
    //PyObject* fileHandle = PyObject_CallFunction(openMethod, "ss", filename.toLatin1().data(),"wb\0"); //new reference
    
    PyObject* pyMode = PyUnicode_FromString("wb\0");
    PyObject* fileHandle = NULL;

    PyObject* pyFileName = PyUnicode_DecodeLatin1(filename.toLatin1().data(), filename.length(), NULL);
    
    if(pyFileName != NULL)
    {
        fileHandle = PyObject_CallFunctionObjArgs(openMethod, pyFileName, pyMode, NULL);
        Py_DECREF(pyFileName);
    }
    
    if(pyMode) Py_DECREF(pyMode);


    if (fileHandle == NULL)
    {
        retval += checkForPyExceptions();
    }
    else
    {
        PyObject *result = NULL;
        
        try
        {
            result = PyObject_CallMethodObjArgs(pickleModule, PyUnicode_FromString("dump"), dict, fileHandle, NULL);
        }
        catch(std::bad_alloc &/*ba*/)
        {
            retval += RetVal(retError, 0, "No more memory available during pickling.");
        }
        catch(std::exception &exc)
        {
            if (exc.what())
            {
                retval += ito::RetVal::format(ito::retError,0,"The exception '%s' has been thrown during pickling.", exc.what()); 
            }
            else
            {
                retval += ito::RetVal(ito::retError,0,"Pickle error. An unspecified exception has been thrown."); 
            }
        }
        catch (...)
        {
            retval += ito::RetVal(ito::retError,0,"Pickle error. An unspecified exception has been thrown.");  
        }

        if (result == NULL)
        {
            retval += checkForPyExceptions();
        }

        Py_XDECREF(result);

        if (!PyObject_CallMethod(fileHandle, "close", ""))
        {
            retval += checkForPyExceptions();
        }
    }

    Py_XDECREF(fileHandle);

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::unpickleVariables(bool globalNotLocal, QString filename, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);
    tPythonState oldState = pythonState;
    RetVal retVal;
    bool released = false;
    PyObject* destinationDict = NULL;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += RetVal(retError, 0, "it is not allowed to unpickle a data collection in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy");
    }
    else
    {
        if (pythonState == pyStateIdle)
        {
            pythonStateTransition(pyTransBeginRun);
        }
        else if (pythonState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdBegin);
        }

        if (globalNotLocal)
        {
            destinationDict = getGlobalDictionary();
        }
        else
        {
            destinationDict = getLocalDictionary();
        }

        if (destinationDict == NULL)
        {
            retVal += RetVal(retError, 0, "variables can not be unpickled since dictionary is not available");
        }
        else
        {

            retVal += unpickleDictionary(destinationDict, filename, true);

            if (semaphore && !released)
            {
                semaphore->returnValue = retVal;
                semaphore->release();
                released = true;
            }

            if (globalNotLocal)
            {
                emitPythonDictionary(true, false, getGlobalDictionary(), NULL);
            }
            else
            {
                emitPythonDictionary(false, true, NULL, getLocalDictionary());
            }
        }


        if (oldState == pyStateIdle)
        {
            pythonStateTransition(pyTransEndRun);
        }
        else if (oldState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdEnd);
        }
    }

    if (semaphore && !released)
    {
        semaphore->returnValue = retVal;
        semaphore->release();
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::unpickleDictionary(PyObject *destinationDict, QString filename, bool overwrite)
{
    RetVal retval;

    if (mainModule == NULL)
    {
        return RetVal(retError, 0, "mainModule is empty or cannot be accessed");
    }

    PyObject* pickleModule = PyImport_AddModule("pickle"); // borrowed reference

    if (pickleModule == NULL)
    {
        retval += checkForPyExceptions();
        return retval;
    }

    PyObject *builtinsModule = PyObject_GetAttrString(mainModule, "__builtins__"); //borrowed

    if (builtinsModule == NULL)
    {
        retval += checkForPyExceptions();
        return retval;
    }

    PyObject* openMethod = PyDict_GetItemString(PyModule_GetDict(builtinsModule), "open"); //borrowed
    //PyObject* fileHandle = PyObject_CallFunction(openMethod, "ss", filename.toLatin1().data(), "rb\0"); //new reference
    
    PyObject* pyMode = PyUnicode_FromString("rb\0");
    PyObject* fileHandle = NULL;

    PyObject* pyFileName = PyUnicode_DecodeLatin1(filename.toLatin1().data(), filename.length(), NULL);
    
    if(pyFileName != NULL)
    {
        fileHandle = PyObject_CallFunctionObjArgs(openMethod, pyFileName, pyMode, NULL);
        Py_DECREF(pyFileName);
    }
    
    if(pyMode) Py_DECREF(pyMode);

    if (fileHandle == NULL)
    {
        retval += checkForPyExceptions();
    }
    else
    {
        PyObject *unpickledItem = NULL;
        try
        {
            unpickledItem = PyObject_CallMethodObjArgs(pickleModule, PyUnicode_FromString("load"), fileHandle, NULL); //new ref
        }
        catch(std::bad_alloc &/*ba*/)
        {
            retval += RetVal(retError, 0, "No more memory available during unpickling.");
        }
        catch(std::exception &exc)
        {
            if (exc.what())
            {
                retval += ito::RetVal::format(ito::retError,0,"The exception '%s' has been thrown during unpickling.", exc.what()); 
            }
            else
            {
                retval += ito::RetVal(ito::retError,0,"Unpickling error. An unspecified exception has been thrown."); 
            }
        }
        catch (...)
        {
            retval += ito::RetVal(ito::retError,0,"Unpickling error. An unspecified exception has been thrown.");  
        }


        if (unpickledItem == NULL)
        {
            retval += checkForPyExceptions();
        }
        else if (!PyDict_Check(unpickledItem))
        {
            retval += RetVal(retError, 0, "unpickling error. This file contains no dictionary as base element.");
        }
        else
        {
            //try to write every element of unpickledItem-dict to destinationDictionary
            PyObject *key, *value;
            Py_ssize_t pos = 0;

            while (PyDict_Next(unpickledItem, &pos, &key, &value))
            {
                if (PyDict_Contains(destinationDict, key) && overwrite)
                {
                    if (overwrite)
                    {
                        PyDict_DelItem(destinationDict, key);
                        //Py_INCREF(value);
                        PyDict_SetItem(destinationDict, key, value); //value is not stolen by SetItem
                    }
                    else
                    {
                        qDebug() << "variable with key '" << PyUnicode_AS_DATA(key) << "' already exists and must not be overwritten.";
                    }
                }
                else
                {
                    PyDict_SetItem(destinationDict, key, value);
                }
            }
  
        }

        if (!PyObject_CallMethod(fileHandle, "close", ""))
        {
            retval += checkForPyExceptions();
        }

        Py_XDECREF(fileHandle);
        Py_XDECREF(unpickledItem);
    }

    return retval;

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
PyMethodDef PythonEngine::PyMethodItomDbg[] = {
    // "Python name", C Ffunction Code, Argument Flags, __doc__ description
    {"pyDbgCommandLoop", PythonEngine::PyDbgCommandLoop, METH_VARARGS, "will be invoked if debugger stopped at the given filename and line"},
    {NULL, NULL, 0, NULL}
};

PyModuleDef PythonEngine::PyModuleItomDbg = {
    PyModuleDef_HEAD_INIT, "itomDbgWrapper", NULL, -1, PythonEngine::PyMethodItomDbg,
    NULL, NULL, NULL, NULL
};

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonEngine::PyInitItomDbg(void)
{
    return PyModule_Create(&PyModuleItomDbg);
}

} //end namespace ito
