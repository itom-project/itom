/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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
#include "pythonFont.h"
#include "pythonShape.h"
#include "pythonAutoInterval.h"

#include "../organizer/addInManager.h"
#include "common/interval.h"
#include "../helper/sleeper.h"

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

#include "../organizer/paletteOrganizer.h"

#include <qmessagebox.h>

#if ITOM_PYTHONMATLAB == 1
#include "pythonMatlab.h"
#endif

#ifndef WIN32
#include <langinfo.h>
#endif

namespace ito
{

QMutex PythonEngine::instatiated;
QMutex PythonEngine::instancePtrProtection;
PythonEngine* PythonEngine::instance = NULL;
QString PythonEngine::fctHashPrefix = ":::itomfcthash:::";

using namespace ito;

//----------------------------------------------------------------------------------------------------------------------------------
FuncWeakRef::FuncWeakRef() : m_proxyObject(NULL), m_argument(NULL), m_handle(0) 
{
}

//----------------------------------------------------------------------------------------------------------------------------------
FuncWeakRef::FuncWeakRef(PythonProxy::PyProxy *proxyObject, PyObject *argTuple /*= NULL*/) :
    m_proxyObject(proxyObject),
    m_argument(argTuple),
    m_handle(0)
{
    Py_XINCREF(m_proxyObject);
    Py_XINCREF(m_argument);
}

//----------------------------------------------------------------------------------------------------------------------------------
FuncWeakRef::FuncWeakRef(const FuncWeakRef &rhs) :
    m_proxyObject(rhs.m_proxyObject),
    m_argument(rhs.m_argument),
    m_handle(0)
{
    Py_XINCREF(m_proxyObject);
    Py_XINCREF(m_argument);
}

//----------------------------------------------------------------------------------------------------------------------------------
FuncWeakRef::~FuncWeakRef()
{
    Py_XDECREF(m_proxyObject);
    Py_XDECREF(m_argument);
}

//----------------------------------------------------------------------------------------------------------------------------------
FuncWeakRef& FuncWeakRef::operator =(FuncWeakRef rhs)
{
    PythonProxy::PyProxy *t1 = m_proxyObject;
    PyObject *t2 = m_argument;

    m_proxyObject = rhs.m_proxyObject;
    m_argument = rhs.m_argument;
    Py_XINCREF(m_proxyObject);
    Py_XINCREF(m_argument);
    m_handle = rhs.m_handle;
    Py_XDECREF(t1);
    Py_XDECREF(t2);

    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
void FuncWeakRef::setHandle(const size_t &handle)
{
    m_handle = handle;
}

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
    m_started(false),
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
    m_pyFuncWeakRefAutoInc(0),
    m_pyModGC(NULL),
    m_pyModSyntaxCheck(NULL),
    m_executeInternalPythonCodeInDebugMode(false),
    dictUnicode(NULL),
    m_pythonThreadId(0),
    m_includeItomImportString(""),
    m_pUserDefinedPythonHome(NULL)
{
    qRegisterMetaType<tPythonDbgCmd>("tPythonDbgCmd");
    qRegisterMetaType<size_t>("size_t");
    qRegisterMetaType<tPythonTransitions>("tPythonTransitions");
    qRegisterMetaType<BreakPointItem>("BreakPointItem");
    qRegisterMetaType<ItomSharedSemaphore*>("ItomSharedSemaphore*");
    qRegisterMetaType<DataObject*>("DataObject*");
    qRegisterMetaType<ito::DataObject>("ito::DataObject");
    qRegisterMetaType<ito::DataObject*>("ito::DataObject*");
    qRegisterMetaType<ito::AutoInterval>("ito::AutoInterval");
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
    qRegisterMetaType<QSharedPointer<size_t> >("QSharedPointer<size_t>");
    qRegisterMetaType<QSharedPointer<QVector<size_t> > >("QSharedPointer<QVector<size_t> >");
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
    qRegisterMetaType<ito::PCLPointCloud >("ito::PCLPointCloud&");
    qRegisterMetaType<ito::PCLPolygonMesh >("ito::PCLPolygonMesh&");
    qRegisterMetaType<ito::PCLPoint >("ito::PCLPoint");
    qRegisterMetaType<QSharedPointer<ito::PCLPointCloud> >("QSharedPointer<ito::PCLPointCloud>");
    qRegisterMetaType<QSharedPointer<ito::PCLPolygonMesh> >("QSharedPointer<ito::PCLPolygonMesh>");
    qRegisterMetaType<QSharedPointer<ito::PCLPoint> >("QSharedPointer<ito::PCLPoint>");
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
    qRegisterMetaType<ito::PyWorkspaceContainer*>("PyWorkspaceContainer*");
    qRegisterMetaType<ito::PyWorkspaceItem*>("PyWorkspaceItem*");
    qRegisterMetaType<ito::PythonQObjectMarshal>("ito::PythonQObjectMarshal");
    qRegisterMetaType<Qt::CursorShape>("Qt::CursorShape");
    qRegisterMetaType<ito::ItomPaletteBase>("ito::ItomPaletteBase");
    qRegisterMetaType<QSharedPointer<ito::ItomPaletteBase> >("QSharedPointer<ito::ItomPaletteBase>");
    qRegisterMetaType<ito::ItomPlotHandle>("ito::ItomPlotHandle");
    qRegisterMetaType<ito::Shape>("ito::Shape");
    qRegisterMetaType<QVector<ito::Shape> >("QVector<ito::Shape>");
    qRegisterMetaType<QSharedPointer<QVector<ito::Shape> > >("QSharedPointer<QVector<ito::Shape>>");
    qRegisterMetaType<ito::PythonNone>("ito::PythonNone");

    m_autoReload.modAutoReload = NULL;
    m_autoReload.classAutoReload = NULL;
    m_autoReload.checkFctExec = false;
    m_autoReload.checkFileExec = true;
    m_autoReload.checkStringExec = true;
    m_autoReload.enabled = false;

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
    if (m_started)
    {
        pythonShutdown();
    }

    bpModel->saveState(); //save current set of breakPoints to settings file
    DELETE_AND_SET_NULL(bpModel);

    DELETE_AND_SET_NULL(m_pDesktopWidget);

    QMutexLocker locker(&PythonEngine::instancePtrProtection);
    PythonEngine::instance = NULL;
    locker.unlock();

    if (m_pUserDefinedPythonHome)
    {
        PyMem_RawFree(m_pUserDefinedPythonHome);
        m_pUserDefinedPythonHome = NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonSetup(ito::RetVal *retValue)
{
    PyObject *itomDbgClass = NULL;
    PyObject *itomDbgDict = NULL;

    m_pythonThreadId = QThread::currentThreadId ();
    //qDebug() << "python in thread: " << m_pythonThreadId;

	/*set new seed for random generator of OpenCV. 
	This is required to have real random values for any randn or randu command.
	The seed must be set in every thread. This is for the main thread.
	*/
	cv::theRNG().state = (uint64)cv::getCPUTickCount();
	/*seed is set*/

    readSettings();

    RetVal tretVal(retOk);
    if (!m_started)
    {
        if (PythonEngine::instatiated.tryLock(5000))
        {
            QString pythonSubDir = QCoreApplication::applicationDirPath() + QString("/python%1").arg(PY_MAJOR_VERSION);
            //check if an alternative home directory of Python should be set:
            QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
            settings.beginGroup("Python");
            QString pythonHomeFromSettings = settings.value("pyHome", "").toString();
            int pythonDirState = settings.value("pyDirState", -1).toInt();
            if (pythonDirState == -1) //not yet decided
            {
#ifdef WIN32
                if (QDir(pythonSubDir).exists() && \
                    QFileInfo(pythonSubDir + QString("/python%1%2.dll").arg(PY_MAJOR_VERSION).arg(PY_MINOR_VERSION)).exists())
                {
                    pythonDirState = 0; //use pythonXX subdirectory of itom as python home path
                }
                else
                {
                    pythonDirState = 1; //use python default search mechanism for home path (e.g. registry...)
                }
#else
                pythonDirState = 1;
#endif
                settings.setValue("pyDirState", pythonDirState);
            }

            settings.endGroup();

            QString pythonDir = "";
            if (pythonDirState == 0) //use pythonXX subdirectory of itom as python home path
            {
                if (QDir(pythonSubDir).exists())
                {
                    pythonDir = pythonSubDir;
                }
                else
                {
                    (*retValue) += RetVal::format(retError, 0, tr("The itom subdirectory of Python '%s' is not existing.\nPlease change setting in the property dialog of itom.").toLatin1().data(),
                        pythonSubDir.toLatin1().data());
                    return;
                }
            }
            else if (pythonDirState == 2) //user-defined value
            {
                
                if (QDir(pythonHomeFromSettings).exists())
                {
                    pythonDir = pythonHomeFromSettings;
                }
                else
                {
                    (*retValue) += RetVal::format(retError, 0, tr("Settings value Python::pyHome has not been set as Python Home directory since it does not exist:  %s").toLatin1().data(),
                        pythonHomeFromSettings.toLatin1().data());
                    return;
                }
            }

            if (pythonDir != "")
            {
                //the python home path given to Py_SetPythonHome must be persistent for the whole Python session
#if PY_VERSION_HEX < 0x03050000
                m_pUserDefinedPythonHome = (wchar_t*)PyMem_RawMalloc((pythonDir.size() + 10) * sizeof(wchar_t));
                memset(m_pUserDefinedPythonHome, 0, (pythonDir.size() + 10) * sizeof(wchar_t));
                pythonDir.toWCharArray(m_pUserDefinedPythonHome);
#else
                m_pUserDefinedPythonHome = Py_DecodeLocale(pythonDir.toLatin1().data(), NULL);
#endif
                Py_SetPythonHome(m_pUserDefinedPythonHome);
            }

/*            if (pythonHomeDirectory != "")
            {
                if (QDir(pythonHomeDirectory).exists())
                {
                    //the python home path given to Py_SetPythonHome must be persistent for the whole Python session
#if PY_VERSION_HEX < 0x03050000
					m_pUserDefinedPythonHome = (wchar_t*)PyMem_RawMalloc((pythonHomeDirectory.size() + 10) * sizeof(wchar_t));
					memset(m_pUserDefinedPythonHome, 0, (pythonHomeDirectory.size() + 10) * sizeof(wchar_t));
					pythonHomeDirectory.toWCharArray(m_pUserDefinedPythonHome);
#else
                    m_pUserDefinedPythonHome = Py_DecodeLocale(pythonHomeDirectory.toLatin1().data(), NULL);
#endif
                    Py_SetPythonHome(m_pUserDefinedPythonHome);
                }
                else
                {
                    qDebug() << "Settings value Python::pyHome has not been set as Python Home directory since it does not exist: " << pythonHomeDirectory;
                }
            }*/

            //read directory values from Python
            qDebug() << "Py_GetPythonHome:" << QString::fromWCharArray(Py_GetPythonHome());
            qDebug() << "Py_GetPath:" << QString::fromWCharArray(Py_GetPath());
            qDebug() << "Py_GetProgramName:" << QString::fromWCharArray(Py_GetProgramName());

            //check PythonHome to prevent crash upon initialization of Python:
            QString pythonHome = QString::fromWCharArray(Py_GetPythonHome());
#ifdef WIN32
            QStringList pythonPath = QString::fromWCharArray(Py_GetPath()).split(";");
#else
            QStringList pythonPath = QString::fromWCharArray(Py_GetPath()).split(":");
#endif
            QDir pythonHomeDir(pythonHome);
            bool pythonPathValid = false;
            if (!pythonHomeDir.exists() && pythonHome != "")
            {
                (*retValue) += RetVal::format(retError, 0, tr("The home directory of Python is currently set to the non-existing directory '%s'\nPython cannot be started. Please set either the environment variable PYTHONHOME to the base directory of python \nor correct the base directory in the property dialog of itom.").toLatin1().data(), 
                    pythonHomeDir.absolutePath().toLatin1().data());
                return;
            }

            foreach(const QString &path, pythonPath)
            {
                QDir pathDir(path);
                if (pathDir.exists("os.py") || pathDir.exists("os.pyc"))
                {
                    pythonPathValid = true;
                    break;
                }
            }

            if (!pythonPathValid)
            {
                (*retValue) += RetVal::format(retError, 0, tr("The built-in library path of Python could not be found. The current home directory is '%s'\nPython cannot be started. Please set either the environment variable PYTHONHOME to the base directory of python \nor correct the base directory in the preferences dialog of itom.").toLatin1().data(), 
                    pythonHomeDir.absolutePath().toLatin1().data());
                return;
            }

            dictUnicode = PyUnicode_FromString("__dict__");

            PyImport_AppendInittab("itom", &PythonItom::PyInitItom);                //!< add all static, known function calls to python-module itom

            PyImport_AppendInittab("itomDbgWrapper", &PythonEngine::PyInitItomDbg);  //!< add all static, known function calls to python-module itomdbg

#if ITOM_PYTHONMATLAB == 1
            PyImport_AppendInittab("matlab", &PythonMatlab::PyInit_matlab);
#endif

            Py_Initialize();                                                        //!< must be called after any PyImport_AppendInittab-call

            qDebug() << "Py_Initialize done.";

            PyEval_InitThreads();                                                   //!< prepare Python multithreading

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
                PyErr_PrintEx(0);
                PyErr_SetString(PyExc_ImportError, tr("numpy.core.multiarray failed to import. Please verify that you have numpy 1.6 or higher installed.").toLatin1().data());
                (*retValue) += RetVal(retError, 0, tr("numpy.core.multiarray failed to import. Please verify that you have numpy 1.6 or higher installed.\n").toLatin1().data());
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
            //the streams __stdout__ and __stderr__, pointing to the original streams at startup are None, but need to have a valid value for instance when using pip.
            if ((tretVal = runString("sys.stdout = sys.__stdout__ = itom.pythonStream(1)")) != ito::retOk)
                (*retValue) += ito::RetVal(ito::retError, 0, tr("error redirecting stdout in start python engine\n").toLatin1().data());
            if ((tretVal = runString("sys.stderr = sys.__stderr__ = itom.pythonStream(2)")) != ito::retOk)
                (*retValue) += ito::RetVal(ito::retError, 0, tr("error redirecting stderr in start python engine\n").toLatin1().data());
            if ((tretVal = runString("sys.stdin = sys.__stdin__ = itom.pythonStream(3)")) != ito::retOk)
                (*retValue) += ito::RetVal(ito::retError, 0, tr("error redirecting stdin in start python engine\n").toLatin1().data());


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

            if (PyType_Ready(&PythonFont::PyFontType) >= 0)
            {
                Py_INCREF(&PythonFont::PyFontType);
                PythonFont::PyFont_addTpDict(PythonFont::PyFontType.tp_dict);
                PyModule_AddObject(itomModule, "font", (PyObject *)&PythonFont::PyFontType);
            }

            if (PyType_Ready(&PythonShape::PyShapeType) >= 0)
            {
                Py_INCREF(&PythonShape::PyShapeType);
                PythonShape::PyShape_addTpDict(PythonShape::PyShapeType.tp_dict);
                PyModule_AddObject(itomModule, "shape", (PyObject *)&PythonShape::PyShapeType);
            }

            if (PyType_Ready(&PythonRgba::PyRgbaType) >= 0)
            {
                Py_INCREF(&PythonRgba::PyRgbaType);
                //PythonRgba::PyRegion_addTpDict(PythonRegion::PyRgbaType.tp_dict);
                PyModule_AddObject(itomModule, "rgba", (PyObject *)&PythonRgba::PyRgbaType);
            }

            if (PyType_Ready(&PythonAutoInterval::PyAutoIntervalType) >= 0)
            {
                Py_INCREF(&PythonAutoInterval::PyAutoIntervalType);
                //PythonRgba::PyRegion_addTpDict(PythonRegion::PyRgbaType.tp_dict);
                PyModule_AddObject(itomModule, "autoInterval", (PyObject *)&PythonAutoInterval::PyAutoIntervalType);
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

#if defined WIN32
            //on windows, sys.executable returns the path of qitom.exe. The absolute path to python.exe is given by sys.exec_prefix
            PyObject *python_path_prefix = PySys_GetObject("exec_prefix"); //borrowed reference
            if (python_path_prefix)
            {
                bool ok;
                m_pythonExecutable = PythonQtConversion::PyObjGetString(python_path_prefix, true, ok);
                if (ok)
                {
                    QDir pythonPath(m_pythonExecutable);
                    if (pythonPath.exists())
                    {
                        m_pythonExecutable = pythonPath.absoluteFilePath("python.exe");
                    }
                    else
                    {
                        m_pythonExecutable = QString();
                    }
                }
                else
                {
                    m_pythonExecutable = QString();
                }
            }
#elif defined linux
            //on linux, sys.executable returns the absolute path to the python application, even in an embedded mode.
            PyObject *python_executable = PySys_GetObject("executable"); //borrowed reference
            if (python_executable)
            {
                bool ok;
                m_pythonExecutable = PythonQtConversion::PyObjGetString(python_executable, true, ok);
                if (!ok)
                {
                    m_pythonExecutable = QString();
                }
            }
#else //APPLE
            //on apple, sys.executable returns the absolute path to the python application, even in an embedded mode. (TODO: Check this assumption)
            PyObject *python_executable = PySys_GetObject("executable"); //borrowed reference
            if (python_executable)
            {
                bool ok;
                m_pythonExecutable = PythonQtConversion::PyObjGetString(python_executable, true, ok);
                if (!ok)
                {
                    m_pythonExecutable = QString();
                }
            }
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
                PyErr_PrintEx(0);
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
                (*retValue) += ito::RetVal(ito::retError, 0, tr("the module itoFunctions could not be loaded. Make sure that the script itoFunctions.py is available in the itom root directory.").toLatin1().data());
                std::cerr << "the module itoFunctions could not be loaded." << std::endl;
                PyErr_PrintEx(0);
                PyErr_Clear();
            }

            //!< import itoDebugger
            itomDbgModule = PyImport_ImportModule("itoDebugger"); // new reference
            if (itomDbgModule == NULL)
            {
                (*retValue) += ito::RetVal(ito::retError, 0, tr("the module itoDebugger could not be loaded. Make sure that the script itoDebugger.py is available in the itom root directory.").toLatin1().data());
                std::cerr << "the module itoDebugger could not be loaded." <<std::endl;
                PyErr_PrintEx(0);
            }
            else
            {
                itomDbgDict = PyModule_GetDict(itomDbgModule); //!< borrowed reference
                itomDbgClass = PyDict_GetItemString(itomDbgDict, "itoDebugger"); // borrowed reference
                itomDbgDict = NULL;
                if (itomDbgClass == NULL)
                {
                    (*retValue) += ito::RetVal(ito::retError, 0, tr("the class itoDebugger in the module itoDebugger could not be loaded.").toLatin1().data());
                    PyErr_PrintEx(0);
                    //printPythonError(PySys_GetObject("stderr"));
                }
                else
                {
                    itomDbgInstance = PyObject_CallObject(itomDbgClass, NULL); //!< http://bytes.com/topic/python/answers/649229-_pyobject_new-pyobject_init-pyinstance_new-etc, new reference
                }
            }

            //!< import autoReloader (mod only, class will be instatiated if enabled for the first time)
            m_autoReload.modAutoReload = PyImport_ImportModule("autoreload");
            if (m_autoReload.modAutoReload == NULL)
            {
                (*retValue) += ito::RetVal(ito::retError, 0, tr("the module 'autoreload' could not be loaded. Make sure that the script autoreload.py is available in the itom-packages directory.").toLatin1().data());
                std::cerr << "the module 'autoreload' could not be loaded." <<std::endl;
                PyErr_PrintEx(0);
            }

            qDebug() << "itom specific python modules loaded.";

            PyThreadState *pts = PyGILState_GetThisThreadState(); //wichtige Zeile
            PyEval_ReleaseThread(pts);

            (*retValue) += stringEncodingChanged();

            qDebug() << "python exec: from itom import *";

            runString("from itom import *");

            //parse the main components of module itom to generate a string like "from itom import dataObject, dataIO, ... to be prepended to each script before syntax check (only for this case)
            PyGILState_STATE gstate = PyGILState_Ensure();
            
            PyObject *itomDir = PyObject_Dir(itomModule); //new ref
            if (itomDir && PyList_Check(itomDir))
            {
                Py_ssize_t len = PyList_GET_SIZE(itomDir);
                QStringList elements;
                elements.reserve(len);

                for (Py_ssize_t l = 0; l < len; ++l)
                {
                    PyObject *dirItem = PyList_GET_ITEM(itomDir, l); //borrowed ref
                    bool ok;
                    QString string = PythonQtConversion::PyObjGetString(dirItem, false, ok);

                    if (ok)
                    {
                        if (!string.startsWith("__"))
                        {
                            elements.append(string);
                        }
                    }
                }
                
                if (elements.size() > 0)
                {
                    m_includeItomImportString = QString("from itom import %1").arg(elements.join(", "));
                }
            }
               
            Py_XDECREF(itomDir);
            PyGILState_Release(gstate);
            //end parse main components

            m_started = true;
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

    m_includeItomImportBeforeSyntaxCheck = settings.value("syntaxIncludeItom", true).toBool();

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::propertiesChanged()
{
    readSettings();
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonEngine::setPyErrFromException(const std::exception &exc)
{
    const std::exception *p_exc = &exc;
    const cv::Exception *p_cvexc = NULL;

    if ((p_cvexc = dynamic_cast<const cv::Exception*>(p_exc)) != NULL)
    {
        const char* errorStr = cvErrorStr(p_cvexc->code);
        return PyErr_Format(PyExc_RuntimeError, "OpenCV Error: %s (%s) in %s, file %s, line %d",
            errorStr, p_cvexc->err.c_str(), p_cvexc->func.size() > 0 ?
            p_cvexc->func.c_str() : "unknown function", p_cvexc->file.c_str(), p_cvexc->line);
    }
    else
    {
        if (exc.what())
        {
            return PyErr_Format(PyExc_RuntimeError, "The exception '%s' has been thrown", exc.what());
        }
        else
        {
            return PyErr_Format(PyExc_RuntimeError, "The exception '<unknown>' has been thrown"); 
        }
    }
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
    if (m_started)
    {
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        //unload the possibly loaded auto-reload tool
        if (m_autoReload.classAutoReload)
        {
            Py_XDECREF(m_autoReload.classAutoReload);
            m_autoReload.classAutoReload = NULL;
        }
        if (m_autoReload.modAutoReload)
        {
            Py_XDECREF(m_autoReload.modAutoReload);
            m_autoReload.modAutoReload = NULL;
        }
        m_autoReload.checkFctExec = false;
        m_autoReload.checkFileExec = false;
        m_autoReload.checkStringExec = false;

        //delete all remaining weak references in m_pyFuncWeakRefHashes (if available)
        QHash<size_t, FuncWeakRef>::iterator it = m_pyFuncWeakRefHashes.begin();
        while(it != m_pyFuncWeakRefHashes.end())
        {
            it = m_pyFuncWeakRefHashes.erase(it);
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
                PyErr_PrintEx(0);
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

        m_started = false;
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
#ifdef WIN32
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
            else if (qtCodecName == "ISO-8859-1" || qtCodecName == "latin1" || qtCodecName == "cp1252" || qtCodecName == "windows-1252")
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
            qDebug() << "encodingType == PythonQtConversion::other. Try to check if python understands one of the following codecs:" << aliases;

            encodingType = PythonQtConversion::other;
            found = false;

            PyGILState_STATE gstate = PyGILState_Ensure();

            foreach (const QByteArray &ba, aliases)
            {
                if (PyCodec_KnownEncoding(ba.data()))
                {
                    encodingName = ba;
                    found = true;
                    break;
                }
            }

            PyGILState_Release(gstate);
        
            if (!found)
            {
                if (codec->name().isEmpty())
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

    qDebug() << "Set encodings to: " << PythonQtConversion::textEncoding << ": " << PythonQtConversion::textEncodingName;

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<int> PythonEngine::parseAndSplitCommandInMainComponents(const char *str, QByteArray &encoding) const
{
    PyGILState_STATE gstate = PyGILState_Ensure();

    //see http://docs.python.org/devguide/compiler.html
    _node *n = PyParser_SimpleParseString(str, Py_file_input);
    _node *n2 = n;
    if (n == NULL)
    {
        PyGILState_Release(gstate);
        //here: error indicator is set.
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
    
    PyGILState_Release(gstate);

    return ret;
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::setAutoReloader(bool enabled, bool checkFile, bool checkCmd, bool checkFct)
{
    if (m_autoReload.modAutoReload)
    {
        PyGILState_STATE gstate = PyGILState_Ensure();

        if (enabled)
        {
            if (!m_autoReload.classAutoReload)
            {
                PyObject *dictItem = PyDict_GetItemString(PyModule_GetDict(m_autoReload.modAutoReload), "ItomAutoreloader"); // borrowed reference
                if (dictItem == NULL)
                {
                    std::cerr << "The class 'ItomAutoreloader' could not be found" << std::endl;
                    PyErr_PrintEx(0);
                }
                else
                {
                    m_autoReload.classAutoReload = PyObject_CallObject(dictItem, NULL); //!< http://bytes.com/topic/python/answers/649229-_pyobject_new-pyobject_init-pyinstance_new-etc, new reference
                }
            }

            if (m_autoReload.classAutoReload)
            {
                m_autoReload.enabled = true;
                m_autoReload.checkFctExec = checkFct;
                m_autoReload.checkFileExec = checkFile;
                m_autoReload.checkStringExec = checkCmd;

                PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "autoreload", "s", "2");
                if (!result)
                {
                    PyErr_PrintEx(0);
                    m_autoReload.enabled = false;
                }
                Py_XDECREF(result);
            }
            else
            {
                m_autoReload.enabled = false;
            }
        }
        else
        {
            m_autoReload.enabled = false;

            if (m_autoReload.classAutoReload)
            {
                PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "autoreload", "s", "0");
                if (!result)
                {
                    PyErr_PrintEx(0);
                }
                Py_XDECREF(result);
            }
        }

        PyGILState_Release(gstate);
    }
    else
    {
        m_autoReload.enabled = false;
    }

    emit pythonAutoReloadChanged(m_autoReload.enabled, m_autoReload.checkFileExec, m_autoReload.checkStringExec, m_autoReload.checkFctExec);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::autoReloaderCheck()
{
    ito::RetVal retval;

    if (m_autoReload.modAutoReload)
    {
        if (m_autoReload.enabled && m_autoReload.classAutoReload)
        {
            PyGILState_STATE gstate = PyGILState_Ensure();

            PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "autoreload", "");
            if (!result)
            {
                PyErr_PrintEx(0);
            }
            Py_XDECREF(result);
            
            result = PyObject_CallMethod(m_autoReload.classAutoReload, "post_execute_hook", "");
            if (!result)
            {
                PyErr_PrintEx(0);
            }
            Py_XDECREF(result);

            PyGILState_Release(gstate);
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, tr("An automatic reload cannot be executed since auto reloader is not enabled.").toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("An automatic reload cannot be executed since module 'autoreload' could not be loaded.").toLatin1().data());
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::runString(const QString &command)
{
    //command must be a single-line command. A single-line command only means, that it must only consist of one block (e.g. an if-loop including its content is also a single-line command)
    //if it is not single line, Py_single_input below must be replaced.

    RetVal retValue = RetVal(retOk);
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *mainDict = getGlobalDictionary();
    PyObject *localDict = getLocalDictionary();
    PyObject *result = NULL;

    if (mainDict == NULL)
    {
        std::cerr << "main dictionary is empty. python probably not started" << std::endl;
        retValue += RetVal(retError, 1, tr("main dictionary is empty").toLatin1().data());
    }
    else if (PyErr_Occurred() == PyExc_SyntaxError)
    {
        PyErr_PrintEx(0);
        //check if already a syntax error has been raised (come from previous call to parseAndSplitCommandInMainComponents)
        retValue += RetVal(retError, 2, tr("syntax error").toLatin1().data());
        PyErr_Clear();
    }
    else
    {
        m_interruptCounter = 0;
        if (m_autoReload.enabled && m_autoReload.checkStringExec)
        {
            PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "pre_run_cell", "");
            if (!result)
            {
                PyErr_PrintEx(0);
            }
            Py_XDECREF(result);
        }

        try
        {
            //Py_single_input for single-line commands forces the result (if != PyNone) to be immediately printed to the command line, which is a desired behaviour.
            //Py_single_input forces inputs that evaluate to something different than None will be printed.
            result = PyRun_String(command.toUtf8().data(), Py_single_input, mainDict, localDict);

            ////input to PyRun_String must be UTF8
            //if (command.contains('\n')) //multi-line commands must have the Py_file_input flag
            //{
            //    result = PyRun_String(command.toUtf8().data(), Py_single_input, mainDict, localDict); //Py_file_input is used such that multi-line commands (separated by \n) are evaluated
            //}
            //else //this command is a single line command, then Py_single_input must be set, such that the output of any command is printed in the next line, else this output is supressed (if no print command is executed)
            //{
            //    result = PyRun_String(command.toUtf8().data(), Py_single_input /*was in 2015: Py_single_input*/, mainDict , localDict); //Py_file_input is used such that multi-line commands (separated by \n) are evaluated
            //}
        }
        catch(std::exception &exc)
        {
            result = setPyErrFromException(exc);
        }

        if (result == NULL)
        {
            if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_SystemExit))
            {
                std::cerr << "wish to exit (not possible yet)\n" << std::endl;
                retValue += RetVal(retError, 2, tr("exiting desired.").toLatin1().data());
            }
            else
            {
                PyErr_PrintEx(0);
                retValue += RetVal(retError, 2, tr("error while evaluating python string.").toLatin1().data());
            }
            PyErr_Clear();
        }

        if (m_autoReload.enabled && m_autoReload.checkStringExec)
        {
            PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "post_execute_hook", "");
            if (!result)
            {
                PyErr_PrintEx(0);
            }
            Py_XDECREF(result);
        }

        Py_XDECREF(result);
    }

    PyGILState_Release(gstate);

    return retValue;
}

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

    m_interruptCounter = 0;
    PyGILState_STATE gstate = PyGILState_Ensure();

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

                if (m_autoReload.enabled && m_autoReload.checkFileExec)
                {
                    PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "pre_run_cell", "");
                    if (!result)
                    {
                        PyErr_PrintEx(0);
                    }
                    Py_XDECREF(result);
                }

                compile = Py_CompileString(fileContent.data(), filename.data(), Py_file_input);
                if (compile == NULL)
                {
                    if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_SystemExit))
                    {
                        std::cerr << "wish to exit (not possible yet)\n" << std::endl;
                        retValue += RetVal(retError);
                    }
                    else
                    {
                        PyObject *oldTBLimit = PySys_GetObject("tracebacklimit");

                        modifyTracebackDepth(2, true);
                        PyErr_PrintEx(0);

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
                            std::cerr << "wish to exit (not possible yet)\n" << std::endl;
                            retValue += RetVal(retError);
                        }
                        else
                        {
                            PyErr_PrintEx(0);
                            retValue += RetVal(retError);
                        }
                        PyErr_Clear();
                    }

                    Py_XDECREF(result);
                    Py_XDECREF(compile);
                }

                if (m_autoReload.enabled && m_autoReload.checkFileExec)
                {
                    PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "post_execute_hook", "");
                    if (!result)
                    {
                        PyErr_PrintEx(0);
                    }
                    Py_XDECREF(result);
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
            if (m_autoReload.enabled && m_autoReload.checkFileExec)
            {
                PyObject *result2 = PyObject_CallMethod(m_autoReload.classAutoReload, "pre_run_cell", "");
                if (!result2)
                {
                    PyErr_PrintEx(0);
                }
                Py_XDECREF(result2);
            }

            try
            {
                result = PyObject_CallMethod(itomDbgInstance, "runScript", "s", pythonFileName.toUtf8().data()); //"s" requires UTF8 encoded char*
            }
            catch(std::exception &exc)
            {
                result = setPyErrFromException(exc);
            }

            if (result == NULL)
            {
                if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_SystemExit))
                {
                    std::cerr << "wish to exit (not possible yet)\n" << std::endl;
                    retValue += RetVal(retError);
                }
                else
                {
                    PyObject *oldTBLimit = PySys_GetObject("tracebacklimit");

                    modifyTracebackDepth(2, true);
                    PyErr_PrintEx(0);

                    if (oldTBLimit != NULL)
                    {
                        PySys_SetObject("tracebacklimit", oldTBLimit);
                    }
                    retValue += RetVal(retError);
                    //printPythonError(PySys_GetObject("stderr"));
                }
                PyErr_Clear();
            }

            if (m_autoReload.enabled && m_autoReload.checkFileExec)
            {
                PyObject *result2 = PyObject_CallMethod(m_autoReload.classAutoReload, "post_execute_hook", "");
                if (!result2)
                {
                    PyErr_PrintEx(0);
                }
                Py_XDECREF(result2);
            }

            Py_XDECREF(result);
        }
    }

    PyGILState_Release(gstate);

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------

ito::RetVal PythonEngine::runFunction(PyObject *callable, PyObject *argTuple, bool gilExternal /*= false*/)
{
    RetVal retValue = RetVal(retOk);
    m_interruptCounter = 0;
    PyGILState_STATE gstate;
    if (!gilExternal)
    {
        gstate = PyGILState_Ensure();
    }

    if (m_autoReload.enabled && m_autoReload.checkFctExec)
    {
        PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "pre_run_cell", "");
        if (!result)
        {
            PyErr_PrintEx(0);
        }
        Py_XDECREF(result);
    }

    PyObject *ret;
    try
    {
        ret = PyObject_CallObject(callable, argTuple);
    }
    catch(std::exception &exc)
    {
        ret = setPyErrFromException(exc);
    }

    if (ret == NULL)
    {
        PyErr_PrintEx(0);
        retValue += RetVal(retError);
    }

    Py_XDECREF(ret);

    if (m_autoReload.enabled && m_autoReload.checkFctExec)
    {
        PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "post_execute_hook", "");
        if (!result)
        {
            PyErr_PrintEx(0);
        }
        Py_XDECREF(result);
    }

    if (!gilExternal)
    {
        PyGILState_Release(gstate);
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------

ito::RetVal PythonEngine::debugFunction(PyObject *callable, PyObject *argTuple, bool gilExternal /*= false*/)
{
    PyObject* result = NULL;
    RetVal retValue = RetVal(retOk);
    m_interruptCounter = 0;
    if (itomDbgInstance == NULL)
    {
        return RetVal(retError);
    }
    else
    {
        PyGILState_STATE gstate;
        if (!gilExternal)
        {
            gstate = PyGILState_Ensure();
        }

        //!< first, clear all existing breakpoints
        result = PyObject_CallMethod(itomDbgInstance, "clear_all_breaks", "");
        if (result == NULL)
        {
            std::cerr << tr("Error while clearing all breakpoints in itoDebugger.").toLatin1().data() << "\n" << std::endl;
            printPythonErrorWithoutTraceback(); //traceback is sense-less, since the traceback is in itoDebugger.py only!

            if (!gilExternal)
            {
                PyGILState_Release(gstate);
            }

            return RetVal(retError);
        }

        //!< submit all breakpoints
        QList<BreakPointItem> bp = bpModel->getBreakpoints();
        QList<BreakPointItem>::iterator it;
        int pyBpNumber;
        QModelIndex modelIndex;

        ito::RetVal retValueTemp;

        for (it = bp.begin() ; it != bp.end() ; ++it)
        {

            if (it->pythonDbgBpNumber == -1)
            {

                retValueTemp = pythonAddBreakpoint(it->filename, it->lineno, it->enabled, it->temporary, it->condition, it->ignoreCount, pyBpNumber);
                if (retValueTemp == ito::retOk)
                {
                    bpModel->setPyBpNumber(*it,pyBpNumber);
                }
                else
                {
                    bpModel->setPyBpNumber(*it,-1);
                    std::cerr << (retValueTemp.hasErrorMessage() ? retValueTemp.errorMessage() : "unspecified error when adding breakpoint to debugger") << "\n" << std::endl;
                }
            }

        }

        //!< setup connections for live-changes in breakpoints
        setupBreakPointDebugConnections();

        if (m_autoReload.enabled && m_autoReload.checkFctExec)
        {
            PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "pre_run_cell", "");
            if (!result)
            {
                PyErr_PrintEx(0);
            }
            Py_XDECREF(result);
        }

        try
        {
            result = PyObject_CallMethod(itomDbgInstance, "debugFunction", "OO", callable, argTuple);
        }
        catch(std::exception &exc)
        {
            result = setPyErrFromException(exc);
        }

        clearDbgCmdLoop();

        if (result == NULL) //!< syntax error
        {
            if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_SystemExit))
            {
                std::cerr << "wish to exit (not possible yet)\n" << std::endl;
                retValue += RetVal(retError);
            }
            else
            {
                PyObject *oldTBLimit = PySys_GetObject("tracebacklimit");

                modifyTracebackDepth(3, true);
                PyErr_PrintEx(0);

                if (oldTBLimit != NULL)
                {
                    PySys_SetObject("tracebacklimit", oldTBLimit);
                }
                retValue += RetVal(retError);
            }
            PyErr_Clear();
        }

        if (m_autoReload.enabled && m_autoReload.checkFctExec)
        {
            PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "post_execute_hook", "");
            if (!result)
            {
                PyErr_PrintEx(0);
            }
            Py_XDECREF(result);
        }

        setGlobalDictionary();
        setLocalDictionary(NULL);

        //!< disconnect connections for live-changes in breakpoints
        shutdownBreakPointDebugConnections();
        bpModel->resetAllPyBpNumbers();

        if (!gilExternal)
        {
            PyGILState_Release(gstate);
        }
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::debugFile(const QString &pythonFileName)
{
    PyObject* result = NULL;
    RetVal retValue = RetVal(retOk);

    QString desiredPath = QFileInfo(pythonFileName).canonicalPath();
    QString currentDir = QDir::current().canonicalPath();
    m_interruptCounter = 0;
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
        PyGILState_STATE gstate = PyGILState_Ensure();

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
        ito::RetVal retValueTemp;

        for (it = bp.begin() ; it != bp.end() ; ++it)
        {
            if (it->pythonDbgBpNumber == -1)
            {
                retValueTemp = pythonAddBreakpoint(it->filename, it->lineno, it->enabled, it->temporary, it->condition, it->ignoreCount, pyBpNumber);
                if (retValueTemp == ito::retOk)
                {
                    bpModel->setPyBpNumber(*it,pyBpNumber);
                }
                else
                {
                    bpModel->setPyBpNumber(*it,-1);
                    std::cerr << (retValueTemp.hasErrorMessage() ? retValueTemp.errorMessage() : "unspecified error when adding breakpoint to debugger") << "\n" << std::endl;
                }
            }
        }

        //!< setup connections for live-changes in breakpoints
        setupBreakPointDebugConnections();

        if (m_autoReload.enabled && m_autoReload.checkFileExec)
        {
            PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "pre_run_cell", "");
            if (!result)
            {
                PyErr_PrintEx(0);
            }
            Py_XDECREF(result);
        }

        try
        {
            result = PyObject_CallMethod(itomDbgInstance, "debugScript", "s", pythonFileName.toUtf8().data()); //"s" requires utf-8 encoded string
        }
        catch(std::exception &exc)
        {
            result = setPyErrFromException(exc);
        }

        clearDbgCmdLoop();

        if (result == NULL) //!< syntax error
        {
            if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_SystemExit))
            {
                std::cerr << "wish to exit (not possible yet)\n" << std::endl;
                retValue += RetVal(retError);
            }
            else
            {
                PyObject *oldTBLimit = PySys_GetObject("tracebacklimit");

                modifyTracebackDepth(3, true);
                PyErr_PrintEx(0);

                if (oldTBLimit != NULL)
                {
                    PySys_SetObject("tracebacklimit", oldTBLimit);
                }
                retValue += RetVal(retError);
            }
            PyErr_Clear();
        }

        if (m_autoReload.enabled && m_autoReload.checkFileExec)
        {
            PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "post_execute_hook", "");
            if (!result)
            {
                PyErr_PrintEx(0);
            }
            Py_XDECREF(result);
        }

        setGlobalDictionary();
        setLocalDictionary(NULL);

        //!< disconnect connections for live-changes in breakpoints
        shutdownBreakPointDebugConnections();
        bpModel->resetAllPyBpNumbers();

        PyGILState_Release(gstate);
    }

    return retValue;
}


//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::debugString(const QString &command)
{
    //command must be a single-line command. A single-line command only means, that it must only consist of one block (e.g. an if-loop including its content is also a single-line command)
    //if it is not single line, Py_single_input below must be replaced.

    PyObject* result = NULL;
    RetVal retValue = RetVal(retOk);
    m_interruptCounter = 0;
    if (itomDbgInstance == NULL)
    {
        return RetVal(retError);
    }
    else if (PyErr_Occurred() == PyExc_SyntaxError)
    {
        PyErr_PrintEx(0);
        //check if already a syntax error has been raised (come from previous call to parseAndSplitCommandInMainComponents)
        retValue += RetVal(retError, 2, tr("syntax error").toLatin1().data());
        PyErr_Clear();
    }
    else
    {
        PyGILState_STATE gstate = PyGILState_Ensure();

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
        ito::RetVal retValueTemp;

        for (it = bp.begin() ; it != bp.end() ; ++it)
        {
            if (it->pythonDbgBpNumber == -1)
            {
                retValueTemp = pythonAddBreakpoint(it->filename, it->lineno, it->enabled, it->temporary, it->condition, it->ignoreCount, pyBpNumber);
                if (retValueTemp == ito::retOk)
                {
                    bpModel->setPyBpNumber(*it,pyBpNumber);
                }
                else
                {
                    bpModel->setPyBpNumber(*it,-1);
                    std::cerr << (retValueTemp.hasErrorMessage() ? retValueTemp.errorMessage() : "unspecified error when adding breakpoint to debugger") << "\n" << std::endl;
                }
            }
        }

        //!< setup connections for live-changes in breakpoints
        setupBreakPointDebugConnections();

        if (m_autoReload.enabled && m_autoReload.checkStringExec)
        {
            PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "pre_run_cell", "");
            if (!result)
            {
                PyErr_PrintEx(0);
            }
            Py_XDECREF(result);
        }

        try
        {
            //the result of all commands that return something else than None is printed. This can be changed in itoDebugger by chosing compile(...,'exec') instead of 'single'
            result = PyObject_CallMethod(itomDbgInstance, "debugString", "s", command.toUtf8().data()); //command must be UTF8
        }
        catch(std::exception &exc)
        {
            result = setPyErrFromException(exc);
        }

        clearDbgCmdLoop();

        if (result == NULL) //!< syntax error
        {
            if (PyErr_Occurred() && PyErr_ExceptionMatches(PyExc_SystemExit))
            {
                std::cerr << "wish to exit (not possible yet)\n" << std::endl;
                retValue += RetVal(retError);
            }
            else
            {
                PyObject *oldTBLimit = PySys_GetObject("tracebacklimit");

                modifyTracebackDepth(3, true);
                PyErr_PrintEx(0);

                if (oldTBLimit != NULL)
                {
                    PySys_SetObject("tracebacklimit", oldTBLimit);
                }
                retValue += RetVal(retError);
            }
            PyErr_Clear();
        }

        if (m_autoReload.enabled && m_autoReload.checkStringExec)
        {
            PyObject *result = PyObject_CallMethod(m_autoReload.classAutoReload, "post_execute_hook", "");
            if (!result)
            {
                PyErr_PrintEx(0);
            }
            Py_XDECREF(result);
        }

        setGlobalDictionary();
        setLocalDictionary(NULL);

        //!< disconnect connections for live-changes in breakpoints
        shutdownBreakPointDebugConnections();
        bpModel->resetAllPyBpNumbers();

        PyGILState_Release(gstate);
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! public slot invoked by the scriptEditorWidget
/*!
    This function calls the frosted python module. This module is able to check the syntax.
    It\B4s called from ScriptEditorWidget::checkSyntax() and delivers the results by 
    calling ScriptEditorWidget::syntaxCheckResult(...).

    \param code This QString contains the code that frosted is supposed to check
    \param sender this is a pointer to the object that called this method
    \return no real return value. Results are returned by invoking ScriptEditorWidget::syntaxCheckResult(...)
*/
void PythonEngine::pythonSyntaxCheck(const QString &code, QPointer<QObject> sender)
{
    if (m_pyModSyntaxCheck)
    {
        QString firstLine;
        if (m_includeItomImportBeforeSyntaxCheck)
        {
            //add from itom import * as first line (this is afterwards removed from results)
            firstLine = m_includeItomImportString + "\n" + code; //+ m_itomMemberClasses + "\n" + code;
        }
        else
        {
            firstLine = code;
        }

        PyGILState_STATE gstate = PyGILState_Ensure();
        PyObject *result = PyObject_CallMethod(m_pyModSyntaxCheck, "check", "s", firstLine.toUtf8().data());

        if (result && PyList_Check(result) && PyList_Size(result) >= 2)
        {
            QString unexpectedErrors;
            QString flakes;

            bool ok;
            unexpectedErrors = PythonQtConversion::PyObjGetString(PyList_GetItem(result, 0), false, ok);
            if (!ok)
            {
                unexpectedErrors = "<<error>>";
            }

            flakes = PythonQtConversion::PyObjGetString(PyList_GetItem(result, 1), false, ok);
            if (!ok)
            {
                flakes = "<<error>>";
            }
            else
            {   
                if (m_includeItomImportBeforeSyntaxCheck)
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
#ifdef _DEBUG
        else if (!result)
        {
            std::cerr << "Error when calling the syntax check module of python\n" << std::endl;
            PyErr_PrintEx(0);
        }
#endif

        Py_XDECREF(result);

        PyGILState_Release(gstate);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::pythonAddBreakpoint(const QString &filename, const int lineno, const bool enabled, const bool temporary, const QString &condition, const int ignoreCount, int &pyBpNumber)
{
    RetVal retval;
    //when calling this method, the Python GIL must already be locked
    PyObject *result = NULL;

    pyBpNumber = -1;

    if (itomDbgInstance == NULL)
    {
        retval += RetVal(retError, 0, "Debugger not available");
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
            //this is an exception case that should not occure under normal circumstances
            std::cerr << tr("Error while transmitting breakpoints to debugger.").toLatin1().data() << "\n" << std::endl;
            printPythonErrorWithoutTraceback(); //traceback is sense-less, since the traceback is in itoDebugger.py only!
            retval += RetVal(retError, 0, tr("Exception raised while adding breakpoint in debugger.").toLatin1().data());
        }
        else
        {
            if (PyLong_Check(result))
            {
                long retNumber = PyLong_AsLong(result);
                if (retNumber < 0)
                {
                    pyBpNumber = -1;
                    retval += RetVal::format(retError, 0, tr("Adding breakpoint to file '%s', line %i failed in Python debugger (invalid breakpoint id).").toLatin1().data(), 
                        filename.toLatin1().data(), lineno + 1);
                }
                else
                {
                    //!> retNumber is new pyBpNumber, must now be added to BreakPointModel
                    pyBpNumber = static_cast<int>(retNumber);
                }
            }
            else
            {
                bool ok;
                QByteArray error = PythonQtConversion::PyObjGetString(result, true, ok).toLatin1();
                if (ok)
                {
                    retval += RetVal(retError, 0, error.data());
                }
                else
                {
                    retval += RetVal::format(retError, 0, tr("Adding breakpoint to file '%s', line %i in Python debugger returned unknown error string").toLatin1().data(), 
                        filename.toLatin1().data(), lineno + 1);
                }
            }
        }

        Py_XDECREF(result);
        result = NULL;
    }
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::pythonEditBreakpoint(const int pyBpNumber, const QString &filename, const int lineno, const bool enabled, const bool temporary, const QString &condition, const int ignoreCount)
{
    RetVal retval;
    //when calling this method, the Python GIL must already be locked
    PyObject *result = NULL;

    if (itomDbgInstance == NULL)
    {
        retval += RetVal(retError, 0, tr("Debugger not available").toLatin1().data());
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
            //this is an exception case that should not occure under normal circumstances
            std::cerr << "Error while editing breakpoint in debugger." << "\n" << std::endl;
            printPythonErrorWithoutTraceback(); //traceback is sense-less, since the traceback is in itoDebugger.py only!
            retval += RetVal(retError, 0, tr("Exception raised while editing breakpoint in debugger.").toLatin1().data());
        }
        else if (PyLong_Check(result))
        {
            long val = PyLong_AsLong(result);
            if (val != 0)
            {
                retval += RetVal::format(retError, 0, tr("Editing breakpoint (file '%s', line %i) in Python debugger returned error code %i").toLatin1().data(), 
                    filename.toLatin1().data(), lineno + 1, val);
            }
        }
        else
        {
            bool ok;
            QByteArray error = PythonQtConversion::PyObjGetString(result, true, ok).toLatin1();
            if (ok)
            {
                retval += RetVal(retError, 0, error.data());
            }
            else
            {
                retval += RetVal::format(retError, 0, tr("Editing breakpoint (file '%s', line %i) in Python debugger returned unknown error string").toLatin1().data(), 
                    filename.toLatin1().data(), lineno + 1);
            }
        }

        Py_XDECREF(result);
        result = NULL;
    }
    else
    {
        retval += RetVal::format(retError, 0, tr("Breakpoint in file '%s', line %i could not be edited since it has no valid Python breakpoint number (maybe a comment or blank line in script)").toLatin1().data(), 
            filename.toLatin1().data(), lineno + 1);
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::pythonDeleteBreakpoint(const int pyBpNumber)
{
    ito::RetVal retval;
    //when calling this method, the Python GIL must already be locked
    PyObject *result = NULL;
    if (itomDbgInstance == NULL)
    {
        retval += RetVal(retError, 0, tr("Debugger not available").toLatin1().data());
    }
    else if (pyBpNumber >= 0)
    {
        result = PyObject_CallMethod(itomDbgInstance, "clearBreakPoint", "i", pyBpNumber); //returns 0 (int) or str with error message
        if (result == NULL)
        {
            //this is an exception case that should not occure under normal circumstances
            std::cerr << "Error while clearing breakpoint in debugger." << "\n" << std::endl;
            printPythonErrorWithoutTraceback(); //traceback is sense-less, since the traceback is in itoDebugger.py only!
            retval += RetVal(retError, 0, tr("Exception raised while clearing breakpoint in debugger.").toLatin1().data());
        }
        else if (PyLong_Check(result))
        {
            long val = PyLong_AsLong(result);
            if (val != 0)
            {
                retval += RetVal::format(retError, 0, tr("Deleting breakpoint in Python debugger returned error code %i").toLatin1().data(), val);
            }
        }
        else
        {
            bool ok;
            QByteArray error = PythonQtConversion::PyObjGetString(result, true, ok).toLatin1();
            if (ok)
            {
                retval += RetVal(retError, 0, error.data());
            }
            else
            {
                retval += RetVal(retError, 0, tr("Deleting breakpoint in Python debugger returned unknown error string").toLatin1().data());
            }
        }

        Py_XDECREF(result);
        result = NULL;
    }
    return retval;
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

        errType = PythonQtConversion::PyObjGetString(pyErrType);
        errText = PythonQtConversion::PyObjGetString(pyErrValue);

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
        {
        pythonStateTransition(pyTransBeginRun);
        runString(ba.data());

        PyGILState_STATE gstate = PyGILState_Ensure();
        emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
        PyGILState_Release(gstate);

        pythonStateTransition(pyTransEndRun);
        }
        break;
    case pyStateRunning:
    case pyStateDebugging:
        // no command execution allowed if running or debugging without being in waiting mode
        qDebug() << "it is not allowed to run a python string in mode pyStateRunning or pyStateDebugging";
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
        {
        pythonStateTransition(pyTransBeginRun);
        list = filename.split(";");
        foreach (const QString &filenameTemp, list)
        {
            if (filenameTemp != "")
            {
                runPyFile(filenameTemp);
            }
        }

        PyGILState_STATE gstate = PyGILState_Ensure();
        emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
        PyGILState_Release(gstate);
        
        pythonStateTransition(pyTransEndRun);
        }
        break;
    case pyStateRunning:
    case pyStateDebugging:
    case pyStateDebuggingWaiting:
    case pyStateDebuggingWaitingButBusy:
        // no command execution allowed if running or debugging without being in waiting mode
        qDebug() << "it is not allowed to run a python file in mode pyStateRunning, pyStateDebugging, pyStateDebuggingWaiting or pyStateDebuggingWaitingButBusy";
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonDebugFile(QString filename)
{
    switch (pythonState)
    {
    case pyStateIdle:
        {
        pythonStateTransition(pyTransBeginDebug);
        debugFile(filename);

        PyGILState_STATE gstate = PyGILState_Ensure();
        emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
        PyGILState_Release(gstate);

        pythonStateTransition(pyTransEndDebug);
        }
        break;
    case pyStateRunning:
    case pyStateDebugging:
    case pyStateDebuggingWaiting:
    case pyStateDebuggingWaitingButBusy:
        // no command execution allowed if running or debugging without being in waiting mode
        qDebug() << "it is not allowed to debug a python file in mode pyStateRunning, pyStateDebugging, pyStateDebuggingWaiting or pyStateDebuggingWaitingButBusy";
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
        {
        pythonStateTransition(pyTransBeginDebug);
        debugString(cmd.toLatin1().data());

        PyGILState_STATE gstate = PyGILState_Ensure();
        emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
        PyGILState_Release(gstate);

        pythonStateTransition(pyTransEndDebug);
        }
        break;
    case pyStateRunning:
    case pyStateDebugging:
    case pyStateDebuggingWaiting:
    case pyStateDebuggingWaitingButBusy:
        // no command execution allowed if running or debugging without being in waiting mode
        qDebug() << "it is not allowed to debug a python string in mode pyStateRunning, pyStateDebugging, pyStateDebuggingWaiting or pyStateDebuggingWaitingButBusy";
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
            PyGILState_STATE gstate = PyGILState_Ensure();
            emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
            PyGILState_Release(gstate);
            pythonStateTransition(pyTransEndDebug);
        }
        else
        {
            pythonStateTransition(pyTransBeginRun);
            runString(cmd);
            PyGILState_STATE gstate = PyGILState_Ensure();
            emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
            PyGILState_Release(gstate);
            pythonStateTransition(pyTransEndRun);
        }
        break;
    case pyStateRunning:
    case pyStateDebugging:
        // no command execution allowed if running or debugging without being in waiting mode
        std::cerr << "it is not allowed to run a python string in mode pyStateRunning or pyStateDebugging\n" << std::endl;
        break;
    case pyStateDebuggingWaiting:
        {
        pythonStateTransition(pyTransDebugExecCmdBegin);
        runString(cmd);
        PyGILState_STATE gstate = PyGILState_Ensure();
        emitPythonDictionary(true, true, getGlobalDictionary(), getLocalDictionary());
        PyGILState_Release(gstate);
        pythonStateTransition(pyTransDebugExecCmdEnd);
        }
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonDebugFunction(PyObject *callable, PyObject *argTuple, bool gilExternal /*= false*/)
{
    switch (pythonState)
    {
    case pyStateIdle:
        pythonStateTransition(pyTransBeginDebug);
        debugFunction(callable, argTuple, gilExternal);

        if (gilExternal)
        {
            emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
        }
        else
        {
            PyGILState_STATE gstate = PyGILState_Ensure();
            emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
            PyGILState_Release(gstate);
        }

        pythonStateTransition(pyTransEndDebug);
        break;
    case pyStateRunning:
    case pyStateDebugging:
        // no command execution allowed if running or debugging without being in waiting mode
        std::cerr << "it is not allowed to debug a function or python string in mode pyStateRunning or pyStateDebugging\n" << std::endl;
        break;
    case pyStateDebuggingWaiting:
    case pyStateDebuggingWaitingButBusy:
        pythonStateTransition(pyTransDebugExecCmdBegin);
        std::cout << "Function will be executed instead of debugged since another debug session is currently running.\n" << std::endl;
        pythonRunFunction(callable, argTuple, gilExternal);
        pythonStateTransition(pyTransDebugExecCmdEnd);
        // no command execution allowed if running or debugging without being in waiting mode
        //qDebug() << "it is now allowed to debug a python function or method in mode pyStateRunning, pyStateDebugging, pyStateDebuggingWaiting or pyStateDebuggingWaitingButBusy";
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//do not execute this method from another thread, only execute it within python-thread since this method is not thread safe
void PythonEngine::pythonRunFunction(PyObject *callable, PyObject *argTuple, bool gilExternal /*= false*/)
{
    m_interruptCounter = 0;
    switch (pythonState)
    {
        case pyStateIdle:
            pythonStateTransition(pyTransBeginRun);
            runFunction(callable, argTuple, gilExternal);

            if (gilExternal)
            {
                emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
            }
            else
            {
                PyGILState_STATE gstate = PyGILState_Ensure();
                emitPythonDictionary(true, true, getGlobalDictionary(), NULL);
                PyGILState_Release(gstate);
            }

            pythonStateTransition(pyTransEndRun);
        break;

        case pyStateRunning:
        case pyStateDebugging:
        case pyStateDebuggingWaitingButBusy: //functions (from signal-calls) can be executed whenever another python method is executed (only possible if another method executing python code is calling processEvents. processEvents stops until this "runFunction" has been terminated
            runFunction(callable, argTuple, gilExternal);

            if (gilExternal)
            {
                emitPythonDictionary(true, true, getGlobalDictionary(), getLocalDictionary());
            }
            else
            {
                PyGILState_STATE gstate = PyGILState_Ensure();
                emitPythonDictionary(true, true, getGlobalDictionary(), getLocalDictionary());
                PyGILState_Release(gstate);
            }
        break;

        case pyStateDebuggingWaiting:
            pythonStateTransition(pyTransDebugExecCmdBegin);
            runFunction(callable, argTuple, gilExternal);

            if (gilExternal)
            {
                emitPythonDictionary(true, true, getGlobalDictionary(), getLocalDictionary());
            }
            else
            {
                PyGILState_STATE gstate = PyGILState_Ensure();
                emitPythonDictionary(true, true, getGlobalDictionary(), getLocalDictionary());
                PyGILState_Release(gstate);
            }

            pythonStateTransition(pyTransDebugExecCmdEnd);
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonRunStringOrFunction(QString cmdOrFctHash)
{
    size_t hashValue;
    m_interruptCounter = 0;
    if (cmdOrFctHash.startsWith(PythonEngine::fctHashPrefix))
    {
        bool success;

        QString cmdOrFctHashCropped = cmdOrFctHash.mid(PythonEngine::fctHashPrefix.length());
        hashValue = cmdOrFctHashCropped.toUInt(&success);
        
        if (!success)
        {
            std::cerr << "The command '" << cmdOrFctHashCropped.toLatin1().data() << "' seems to be a hashed function or method, but no handle value can be extracted (size_t required)\n" << std::endl;
            return;
        }

        PyGILState_STATE gstate = PyGILState_Ensure();

        QHash<size_t, FuncWeakRef>::iterator it = m_pyFuncWeakRefHashes.find(hashValue);
        if (it != m_pyFuncWeakRefHashes.end())
        {
            PyObject *callable = (PyObject*)(it->getProxyObject()); //borrowed reference
            PyObject *argTuple = it->getArguments(); //borrowed reference
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
                std::cerr << "The method associated with the key '" << cmdOrFctHashCropped.toLatin1().data() << "' does not exist any more\n" << std::endl;
            }
            Py_XDECREF(argTuple);    
        }
        else
        {
            std::cerr << "No action associated with key '" << cmdOrFctHashCropped.toLatin1().data() << "' could be found in internal hash table\n" << std::endl;
        }

        PyGILState_Release(gstate);
    }
    else
    {
        pythonRunString(cmdOrFctHash);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonDebugStringOrFunction(QString cmdOrFctHash)
{
    size_t hashValue;
    m_interruptCounter = 0;
    if (cmdOrFctHash.startsWith(PythonEngine::fctHashPrefix))
    {
        bool success;

        QString cmdOrFctHashCropped = cmdOrFctHash.mid(PythonEngine::fctHashPrefix.length());
        hashValue = cmdOrFctHashCropped.toUInt(&success);
        
        if (!success)
        {
            std::cerr << "The command '" << cmdOrFctHashCropped.toLatin1().data() << "' seems to be a hashed function or method, but no handle value can be extracted (size_t required)\n" << std::endl;
            return;
        }

        PyGILState_STATE gstate = PyGILState_Ensure();

        QHash<size_t, FuncWeakRef>::iterator it = m_pyFuncWeakRefHashes.find(hashValue);
        if (it != m_pyFuncWeakRefHashes.end())
        {
            PyObject *callable = (PyObject*)(it->getProxyObject()); //borrowed reference
            PyObject *argTuple = it->getArguments(); //borrowed reference
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
                Py_XDECREF(callable);
            }
            else
            {
                std::cerr << "The method associated with the key '" << cmdOrFctHashCropped.toLatin1().data() << "' does not exist any more\n" << std::endl;
            }
            Py_XDECREF(argTuple);    
        }
        else
        {
            std::cerr << "No action associated with key '" << cmdOrFctHashCropped.toLatin1().data() << "' could be found in internal hash table\n" << std::endl;
        }

        PyGILState_Release(gstate);
        
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
    PyGILState_STATE gstate = PyGILState_Ensure();
    pythonAddBreakpoint(bp.filename, bp.lineno, bp.enabled, bp.temporary, bp.condition, bp.ignoreCount, pyBpNumber);
    PyGILState_Release(gstate);
    bpModel->setPyBpNumber(bp, pyBpNumber);
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::breakPointDeleted(QString /*filename*/, int /*lineNo*/, int pyBpNumber)
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    ito::RetVal ret = pythonDeleteBreakpoint(pyBpNumber);
    if (ret.containsError())
    {
        std::cerr << (ret.hasErrorMessage() ? ret.errorMessage() : "unknown error while deleting breakpoint") << "\n" << std::endl;
    }
    PyGILState_Release(gstate);
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::breakPointChanged(BreakPointItem /*oldBp*/, ito::BreakPointItem newBp)
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    ito::RetVal ret = pythonEditBreakpoint(newBp.pythonDbgBpNumber, newBp.filename, newBp.lineno, newBp.enabled, newBp.temporary, newBp.condition, newBp.ignoreCount);
    if (ret.containsError())
    {
        std::cerr << (ret.hasErrorMessage() ? ret.errorMessage() : "unknown error while editing breakpoint") << "\n" << std::endl;
    }
    PyGILState_Release(gstate);
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
        PyGILState_STATE gstate = PyGILState_Ensure();
        emitPythonDictionary(true, true, getGlobalDictionary(), getLocalDictionary());
        PyGILState_Release(gstate);
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
/* \brief gets PyObject from local or global workspace that is described by a itom-specific path name

The path name is a delimiter (/) separated string list where each item has the following form XY:name.
The meaning for XY:name corresponds to PyWorkspaceItem::m_key.

This method returns a new reference to the found PyObject* or NULL. This function can only be
called if the Python GIL is already locked.
*/
PyObject* PythonEngine::getPyObjectByFullName(bool globalNotLocal, const QStringList &fullNameSplittedByDelimiter, QString *validVariableName /*= NULL*/)
{
#if defined _DEBUG && (PY_VERSION_HEX >= 0x03040000)
    if (!PyGILState_Check())
    {
        std::cerr << "Python GIL must be locked when calling getPyObjectByFullName\n" << std::endl;
        return NULL;
    }
#endif

    PyObject *obj = NULL;
    PyObject *current_obj = NULL;
    QStringList items = fullNameSplittedByDelimiter;
    int i=0;
    float f=0.0;
    PyObject *tempObj = NULL;
    PyObject *number = NULL;

    char itemKeyType, itemType;
    QByteArray itemName;
    QByteArray itemKey;
    bool ok;
    bool objIsNewRef = false;

    if (items.count() > 0 && items[0] == "")
    {
        items.removeFirst();
    }

    if (items.count() == 1 && items[0].indexOf(":") == -1)
    {
        //this is a compatibility thing. This function can also be used to
        //get a variable from the global or local dictionary. If only a
        //variable name is passed, we prepend PY_DICT PY_STRING : to the string
        //such that an item from the gobal or local dictionary is chosen.
        const char prepend[] = { PY_DICT, PY_STRING, ':', '\0' };
        items[0].prepend(prepend);
    }

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
        current_obj = obj;

        itemName = items[0].toLatin1();

        if (itemName.size() < 4) //every item has the form "as:name" where a,s... are values of the enumeration PyWorkspaceContainer:WorkspaceItemType
        {
            return NULL;
        }
        else
        {
            itemKey = itemName.mid(3);
            itemType = itemName.at(0);
            itemKeyType = itemName.at(1); //keyword is a number of a string
        }

        if (PyDict_Check(obj))
        {
            if (itemKeyType == 's') //string
            {
                tempObj = PyDict_GetItemString(obj, itemKey); //borrowed
                if (validVariableName)
                {
                    *validVariableName = itemKey;
                }
            }
            else if (itemKeyType == 'n') //number
            {
                i = itemKey.toInt(&ok);
                if (ok)
                {
                    number = PyLong_FromLong(i);
                    tempObj = PyDict_GetItem(obj, number); //borrowed

                    if (validVariableName)
                    {
                        *validVariableName = QString("item%1").arg(i);
                    }

                    Py_XDECREF(number);
                }
                if (!ok || tempObj == NULL)
                {
                    f = items[0].toFloat(&ok); //here, often, a rounding problem occurres... (this could not be fixed until now)
                    if (ok)
                    {
                        number = PyFloat_FromDouble(f);
                        tempObj = PyDict_GetItem(obj, number); //borrowed

                        if (validVariableName)
                        {
                            *validVariableName = QString("item%1").arg(f).replace(".", "dot").replace(",", "dot");
                        }

                        Py_XDECREF(number);
                    }
                }
            }

            obj = tempObj;

            if (objIsNewRef)
            {
                Py_DECREF(current_obj); 
                objIsNewRef = false; //in the overall if-case, no new obj is a new reference, all borrowed
            }
        }
        else if (PyList_Check(obj))
        {
            i = itemKey.toInt(&ok);
            if (!ok || i < 0 || i >= PyList_Size(obj)) return NULL; //error
            obj = PyList_GET_ITEM(obj,i); //borrowed

            if (validVariableName)
            {
                *validVariableName = QString("item%1").arg(i);
            }

            if (objIsNewRef)
            {
                Py_DECREF(current_obj); 
                objIsNewRef = false; //no new obj is a new reference, all borrowed
            }
        }
        else if (PyTuple_Check(obj))
        {
            i = itemKey.toInt(&ok);
            if (!ok || i < 0 || i >= PyTuple_Size(obj)) return NULL; //error
            obj = PyTuple_GET_ITEM(obj,i); //borrowed

            if (validVariableName)
            {
                *validVariableName = QString("item%1").arg(i);
            }

            if (objIsNewRef)
            {
                Py_DECREF(current_obj); 
                objIsNewRef = false; //no new obj is a new reference, all borrowed
            }
        }
        else if (PyObject_HasAttr(obj, dictUnicode))
        {
            PyObject *temp = PyObject_GetAttr(obj, dictUnicode); //new reference
            if (temp)
            {
                if (itemKeyType == 's') //string
                {
                    tempObj = PyDict_GetItemString(temp, itemKey); //borrowed
                    if (!tempObj)
                    {
                        obj = PyObject_GetAttrString(obj, itemKey); //new reference (only for this case, objIsNewRef is true (if nothing failed))
                        if (validVariableName)
                        {
                            *validVariableName = itemKey;
                        }

                        if (objIsNewRef)
                        {
                            Py_DECREF(current_obj); 
                        }

                        objIsNewRef = (obj != NULL);
                    }
                    else
                    {
                        obj = tempObj;
                        if (objIsNewRef)
                        {
                            Py_DECREF(current_obj); 
                            objIsNewRef = false;  //no new obj is a new reference, all borrowed
                        }
                    }
                }
                else if (itemKeyType == 'n') //number
                {
                    i = itemKey.toInt(&ok);
                    if (ok)
                    {
                        number = PyLong_FromLong(i);
                        tempObj = PyDict_GetItem(temp, number); //borrowed

                        if (validVariableName)
                        {
                            *validVariableName = QString("item%1").arg(i);
                        }

                        Py_XDECREF(number);
                    }
                    if (!ok || tempObj == NULL)
                    {
                        f = items[0].toFloat(&ok); //here, often, a rounding problem occurres... (this could not be fixed until now)
                        if (ok)
                        {
                            number = PyFloat_FromDouble(f);
                            tempObj = PyDict_GetItem(temp, number); //borrowed

                            if (validVariableName)
                            {
                                *validVariableName = QString("item%1").arg(f).replace(".", "dot").replace(",", "dot");
                            }

                            Py_XDECREF(number);
                        }
                    }

                    obj = tempObj;
                    if (objIsNewRef)
                    {
                        Py_DECREF(current_obj); 
                        objIsNewRef = false;
                    }
                }
                
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

    if (objIsNewRef == false)
    {
        Py_XINCREF(obj);
    }

    return obj; //always new reference
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject *PythonEngine::getPyObjectByFullName(bool globalNotLocal, const QString &fullName, QString *validVariableName /*= NULL*/)
{
    return getPyObjectByFullName(globalNotLocal, fullName.split(ito::PyWorkspaceContainer::delimiter), validVariableName);
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::workspaceGetChildNode(PyWorkspaceContainer *container, QString fullNameParentItem)
{
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *obj = getPyObjectByFullName(container->isGlobalWorkspace(), fullNameParentItem);
    
    if (obj)
    {
        container->loadDictionary(obj, fullNameParentItem);
        Py_DECREF(obj);
    }

    PyGILState_Release(gstate);
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::workspaceGetValueInformation(PyWorkspaceContainer *container, const QString &fullItemName, QSharedPointer<QString> extendedValue, ItomSharedSemaphore *semaphore)
{
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *obj = getPyObjectByFullName(container->isGlobalWorkspace(), fullItemName);

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

        Py_DECREF(obj);
    }

    PyGILState_Release(gstate);

    if (semaphore)
    {
        semaphore->release();
        semaphore->deleteSemaphore();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::emitPythonDictionary(bool emitGlobal, bool emitLocal, PyObject* globalDict, PyObject* localDict)
{
#if defined _DEBUG && PY_VERSION_HEX >= 0x03040000
    if (!PyGILState_Check())
    {
        std::cerr << "Python GIL must be locked when calling emitPythonDictionary\n" << std::endl;
        return;
    }
#endif

    //if localDict is equal to globalDict, the localDict is the current global dict (currently debugging at top level) -> it is sufficient to only show the global dict and delete the local dict
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
        if (localDict != NULL && localDict != globalDict)
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
        PyErr_PrintEx(0);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
int PythonEngine::queuedInterrupt(void * state) 
{ 
    // ok this is REALLY ugly, BUT if we want to break python constructs like:
    // while 1:
    //      try:
    //          a = 1
    //      except:
    //          pass
    //
    // we have to raise an except while exception handling. Therefore
    // we accumulate some keyboards interrupts and force their
    // excecution afterwards with setInterrupt ...
    // Anyway deeper nested try - except constructs we cannot terminate this way
//    while ((*(ito::tPythonState *)state) == pyStateRunning)
    {
        PyErr_SetNone(PyExc_KeyboardInterrupt);
        PyErr_SetNone(PyExc_KeyboardInterrupt);
        PyErr_SetInterrupt();
    }
    PythonEngine::getInstanceInternal()->m_interruptCounter.deref();
    PyErr_Clear();

    return -1; 
} 

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ bool PythonEngine::isInterruptQueued()
{
    ito::PythonEngine *pyEng = PythonEngine::getInstanceInternal();
    if (pyEng)
    {
#if QT_VERSION > 0x050000
        return (pyEng->m_interruptCounter.load() > 0);
#else
        return ((int)(pyEng->m_interruptCounter) > 0);
#endif
    }
    return false;
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonEngine::pythonInterruptExecution()
{
//    PyGILState_STATE gstate;
//    gstate = PyGILState_Ensure();

    // only queue the interrupt event if not yet done.
    // ==operator(int) of QAtomicInt does not exist for all versions of Qt5. testAndSetRelaxed returns true, if the value was 0 (and assigns one to it)
    if (m_interruptCounter.testAndSetRelaxed(0, 1)) 
    {
        if (isPythonDebugging() && isPythonDebuggingAndWaiting())
        {
            dbgCmdMutex.lock();
            debugCommandQueue.insert(0, ito::pyDbgQuit);
            dbgCmdMutex.unlock();
        }
        else
        {
            Py_AddPendingCall(&PythonEngine::queuedInterrupt, &pythonState);
        }
    }

    // Release the thread. No Python API allowed beyond this point.
//    PyGILState_Release(gstate);

    qDebug("PyErr_SetInterrupt() in pythonThread");
};

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
            PyErr_PrintEx(0);
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
                pyEngine->emitPythonDictionary(true,true,globalDict,NULL);
            }
        }

        PyThreadState *_save;

        while(!pyEngine->DbgCommandsAvailable()) //->isValidDbgCmd())
        {
            Py_UNBLOCK_THREADS //from here, python can do something else... (e.g. executing another code snippet)

            //tests showed that the CPU consumption becomes very high, if
            //this while loop iterates without a tiny sleep.
            //The subsequent processEvents however is necessary to t(5get
            //the next debug command.
            Sleeper::msleep(50);

            QCoreApplication::processEvents();

            Py_BLOCK_THREADS

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
                PyErr_PrintEx(0);
            }
            break;
        case ito::pyDbgContinue:
            if (!PyObject_CallMethod(self, "set_continue", ""))
            {
                PyErr_PrintEx(0);
            }
            break;
        case ito::pyDbgStepOver:
            if (!PyObject_CallMethod(self, "set_next", "O", frame))
            {
                PyErr_PrintEx(0);
            }
            break;
        case ito::pyDbgStepOut:
            if (!PyObject_CallMethod(self,"set_return", "O", frame))
            {
                PyErr_PrintEx(0);
            }
            break;
        case ito::pyDbgQuit:
            if (!PyObject_CallMethod(self,"do_quit", "O", frame)) //!< do_quit instead of set_quit, since one member-variable is set in itoDebugger.py
            {
                PyErr_PrintEx(0);
            }
            PythonEngine::getInstanceInternal()->m_interruptCounter.deref();
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
bool PythonEngine::renameVariable(bool globalNotLocal, const QString &oldFullItemName, QString newKey, ItomSharedSemaphore *semaphore)
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
            std::cerr << "variable can not be renamed, since dictionary is not available\n" << std::endl;
        }
        else
        {
            PyGILState_STATE gstate = PyGILState_Ensure();

            if (!PyUnicode_IsIdentifier(PyUnicode_DecodeLatin1(newKey.toLatin1().data(), newKey.length(), NULL)))
            {
                PyErr_Clear();
                retVal = false;
                std::cerr << "variable name " << newKey.toLatin1().data() << " is invalid.\n" << std::endl;
            }
            else
            {
                QStringList fullNameSplit = oldFullItemName.split(PyWorkspaceContainer::delimiter);
                if (fullNameSplit.size() > 0 && fullNameSplit[0] == "")
                {
                    fullNameSplit.removeFirst();
                }

                PyObject *oldItem = getPyObjectByFullName(globalNotLocal, fullNameSplit); //new reference

                if (oldItem)
                {
                    if (fullNameSplit.size() > 0)
                    {
                        QStringList old = fullNameSplit.last().split(":");
                        PyObject *oldName = old[0][1].toLatin1() == PY_STRING ? PythonQtConversion::QStringToPyObject(old[1]) : PyLong_FromLong(old[1].toInt()); //new reference
                        char parentContainerType = old[0][0].toLatin1();
                        fullNameSplit.removeLast(); 
                        PyObject *parentContainer = NULL;
                        if (fullNameSplit.size() > 0)
                        {
                            parentContainer = getPyObjectByFullName(globalNotLocal, fullNameSplit); //new reference
                        }
                        else
                        {
                            parentContainer = dict;
                            Py_INCREF(parentContainer);
                        }

                        switch (parentContainerType)
                        {
                        case PY_DICT:
                            value = PyDict_GetItemString(parentContainer, newKey.toLatin1().data()); //borrowed reference
                            if (value != NULL)
                            {
                                retVal = false;
                                std::cerr << "variable " << newKey.toLatin1().data() << " already exists in dictionary\n" << std::endl;
                            }
                            else
                            {
                                PyDict_SetItemString(parentContainer, newKey.toLatin1().data(), oldItem); //first set new, then delete in order not to loose the reference in-between (ref of value is automatically incremented)
                                PyDict_DelItem(parentContainer, oldName);

                                if (PyErr_Occurred())
                                {
                                    retVal = false;
                                    PyErr_PrintEx(0);
                                }
                            }
                            break;
                        case PY_MAPPING:
                            value = PyMapping_GetItemString(parentContainer, newKey.toLatin1().data()); //new reference
                            if (value != NULL)
                            {
                                retVal = false;
                                std::cerr << "variable " << newKey.toLatin1().data() << " already exists in dictionary\n" << std::endl;
                            }
                            else
                            {
                                PyMapping_SetItemString(parentContainer, newKey.toLatin1().data(), oldItem); //first set new, then delete in order not to loose the reference in-between (ref of value is automatically incremented)
                                PyMapping_DelItem(parentContainer, oldName);

                                if (PyErr_Occurred())
                                {
                                    retVal = false;
                                    PyErr_PrintEx(0);
                                }

                                Py_DECREF(value);
                            }
                            break;
                        case PY_ATTR:
                            value = PyObject_GetAttrString(parentContainer, newKey.toLatin1().data()); //new reference
                            if (value != NULL)
                            {
                                retVal = false;
                                std::cerr << "variable " << newKey.toLatin1().data() << " already exists in dictionary\n" << std::endl;
                            }
                            else
                            {
                                PyObject_SetAttrString(parentContainer, newKey.toLatin1().data(), oldItem); //first set new, then delete in order not to loose the reference in-between (ref of value is automatically incremented)
                                PyObject_DelAttr(parentContainer, oldName);

                                if (PyErr_Occurred())
                                {
                                    retVal = false;
                                    PyErr_PrintEx(0);
                                }

                                Py_DECREF(value);
                            }
                            break;
                        case PY_LIST_TUPLE:
                            retVal = false;
                            std::cerr << "variable " << newKey.toLatin1().data() << " is part of a list or tuple and cannot be renamed\n" << std::endl;
                        }

                        Py_DECREF(parentContainer);
                        Py_DECREF(oldName);
                    }

                    Py_DECREF(oldItem);
                }
                else
                {
                    retVal = false;
                    std::cerr << "variable that should be renamed could not be found.\n" << std::endl;
                }
            }

            PyGILState_Release(gstate);
        }

        if (semaphore != NULL) //release semaphore now, since the following emit command will be a blocking connection, too.
        {
            semaphore->release();
        }

        PyGILState_STATE gstate = PyGILState_Ensure();
        if (globalNotLocal)
        {
            emitPythonDictionary(true, false, getGlobalDictionary(), NULL);
        }
        else
        {
            emitPythonDictionary(false, true, NULL, getLocalDictionary());
        }
        PyGILState_Release(gstate);

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
/*
    delete one or multiple variables from python global or local workspace

    \param globalNotLocal is true, if deletion from global workspace, else: local workspace
    \param fullItemNames is a list of full item names to all python variables that should be deleted from workspace. This list must not contain child values if the parent is part of the list, too.
*/
bool PythonEngine::deleteVariable(bool globalNotLocal, const QStringList &fullItemNames, ItomSharedSemaphore *semaphore)
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
            PyGILState_STATE gstate = PyGILState_Ensure();

            QStringList fullNameSplit;
            PyObject *parentContainer = NULL;
            PyObject *name = NULL;

            foreach(const QString &fullItemName, fullItemNames)
            {
                fullNameSplit = fullItemName.split(PyWorkspaceContainer::delimiter);
                if (fullNameSplit.size() > 0 && fullNameSplit[0] == "")
                {
                    fullNameSplit.removeFirst();
                }

                PyObject *item = getPyObjectByFullName(globalNotLocal, fullNameSplit); //new reference

                if (item)
                {
                    if (fullNameSplit.size() > 0)
                    {
                        QStringList old = fullNameSplit.last().split(":");
                        name = old[0][1].toLatin1() == PY_STRING ? PythonQtConversion::QStringToPyObject(old[1]) : PyLong_FromLong(old[1].toInt()); //new reference
                        char parentContainerType = old[0][0].toLatin1();
                        fullNameSplit.removeLast();
                        if (fullNameSplit.size() > 0)
                        {
                            parentContainer = getPyObjectByFullName(globalNotLocal, fullNameSplit); //new reference
                        }
                        else
                        {
                            parentContainer = dict;
                            Py_INCREF(parentContainer);
                        }

                        switch (parentContainerType)
                        {
                        case PY_DICT:
                            PyDict_DelItem(parentContainer, name);
                            break;
                        case PY_MAPPING:
                            PyMapping_DelItem(parentContainer, name);
                            break;
                        case PY_ATTR:
                            PyObject_DelAttr(parentContainer, name); 
                            break;
                        case PY_LIST_TUPLE:
                            if (PySequence_DelItem(parentContainer, old[1].toInt()) < 0)
                            {
                                retVal = false;
                                std::cerr << "Item could not be deleted from list or tuple. It is never allowed to delete from a tuple.\n" << std::endl;
                            }
                            break;
                        }

                        if (PyErr_Occurred())
                        {
                            retVal = false;
                            PyErr_PrintEx(0);
                        }

                        Py_DECREF(parentContainer);
                        Py_DECREF(name);
                    }

                    Py_DECREF(item);
                }
            }

            PyGILState_Release(gstate);

        }

        if (semaphore != NULL) semaphore->release();

        PyGILState_STATE gstate = PyGILState_Ensure();
        if (globalNotLocal)
        {
            emitPythonDictionary(true, false, getGlobalDictionary(), NULL);
        }
        else
        {
            emitPythonDictionary(false, true, NULL, getLocalDictionary());
        }
        PyGILState_Release(gstate);


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
        retVal += RetVal(retError, 0, tr("it is not allowed to save a variable in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy").toLatin1().data());
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
            retVal += RetVal(retError, 0, tr("variables can not be saved since dictionary is not available").toLatin1().data());
        }
        else
        {
            PyGILState_STATE gstate = PyGILState_Ensure();

            //build dictionary, which should be pickled
            PyObject* pArgs = PyTuple_New(3);
            PyTuple_SetItem(pArgs,0, PyUnicode_DecodeLatin1(filename.toLatin1().data(), filename.length(), NULL));
            

            PyObject* keyList = PyList_New(0);
            PyObject* valueList = PyList_New(0);
            PyObject* tempElem = NULL;
            QString validVariableName;

            for (int i = 0 ; i < varNames.size() ; i++)
            {
                tempElem = getPyObjectByFullName(globalNotLocal, varNames[i], &validVariableName); //new reference

                if (tempElem == NULL)
                {
                    std::cerr << "variable '" << varNames.at(i).toLatin1().data() << "' can not be found in dictionary and will not be exported.\n" << std::endl;
                }
                else
                {
                    PyList_Append(keyList, PyUnicode_DecodeLatin1(validVariableName.toLatin1().data(), validVariableName.length(), NULL));
                    PyList_Append(valueList, tempElem);
                    Py_DECREF(tempElem);
                }
            }

            PyTuple_SetItem(pArgs,1,valueList);
            PyTuple_SetItem(pArgs,2,keyList);
            PyObject* pyRet = ito::PythonItom::PySaveMatlabMat(NULL, pArgs);

            retVal += checkForPyExceptions();

            Py_XDECREF(pArgs);
            Py_XDECREF(pyRet);

            PyGILState_Release(gstate);
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
/** save a single DataObject, PointCloud or PolygonMesh to a Matlab *.mat file using the python module 'scipy'.
*
*  Invoke this method by another thread (e.g. any GUI) to save a single object to an 'mat' file.
*
*  \param filename is the filename of the mat file
*  \param value is the given DataObject, PointCloud or PolygonMesh in terms of ito::Param
*  \param valueName is the name of the variable in the mat file
*  \param semaphore is the control semaphore for an asychronous call.
*/
ito::RetVal PythonEngine::saveMatlabSingleParam(QString filename, QSharedPointer<ito::Param> value, const QString &valueName, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);

    tPythonState oldState = pythonState;
    RetVal retVal;
    PyObject* dict = NULL;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += ito::RetVal(retError, 0, tr("it is not allowed to pickle a variable in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy").toLatin1().data());
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

        PyObject *item = NULL;

        if (value.isNull())
        {
            retVal += ito::RetVal(retError, 0, tr("Given value is empty. No save to matlab possible.").toLatin1().data());
        }
        else
        {
            switch (value->getType())
            {
            case (ito::ParamBase::DObjPtr & paramTypeMask) :
            {
                const ito::DataObject *obj = value->getVal<const ito::DataObject*>();
                if (obj)
                {
                    PyGILState_STATE gstate = PyGILState_Ensure();
                    item = PythonQtConversion::DataObjectToPyObject(*obj);
                    PyGILState_Release(gstate);
                }
                else
                {
                    retVal += ito::RetVal(retError, 0, tr("could not save dataObject since it is not available.").toLatin1().data());
                }
            }
            break;

#if ITOM_POINTCLOUDLIBRARY > 0 
            case (ito::ParamBase::PointCloudPtr & paramTypeMask) :
            {
                const ito::PCLPointCloud *cloud = value->getVal<const ito::PCLPointCloud*>();
                if (cloud)
                {
                    PyGILState_STATE gstate = PyGILState_Ensure();
                    item = PythonQtConversion::PCLPointCloudToPyObject(*cloud);
                    PyGILState_Release(gstate);
                }
                else
                {
                    retVal += ito::RetVal(retError, 0, tr("could not save dataObject since it is not available.").toLatin1().data());
                }
            }
            break;

            case (ito::ParamBase::PolygonMeshPtr & paramTypeMask) :
            {
                const ito::PCLPolygonMesh *mesh = value->getVal<const ito::PCLPolygonMesh*>();
                if (mesh)
                {
                    PyGILState_STATE gstate = PyGILState_Ensure();
                    item = PythonQtConversion::PCLPolygonMeshToPyObject(*mesh);
                    PyGILState_Release(gstate);
                }
                else
                {
                    retVal += ito::RetVal(retError, 0, tr("could not save dataObject since it is not available.").toLatin1().data());
                }
            }
            break;
#endif
            default:
                retVal += ito::RetVal(retError, 0, tr("unsupported data type to save to matlab.").toLatin1().data());
            }

            if (item == NULL)
            {
                retVal += ito::RetVal(retError, 0, tr("error converting object to Python object. Save to matlab not possible.").toLatin1().data());
            }
        }

        if (!retVal.containsError())
        {
            PyGILState_STATE gstate = PyGILState_Ensure();

            PyObject* pArgs = PyTuple_New(3);
            PyTuple_SetItem(pArgs, 0, PyUnicode_DecodeLatin1(filename.toLatin1().data(), filename.length(), NULL)); //steals ref.
            PyTuple_SetItem(pArgs, 1, item); //steals ref.
            PyTuple_SetItem(pArgs, 2, PyUnicode_DecodeLatin1(valueName.toLatin1().data(), valueName.length(), NULL)); //steals ref.
            PyObject *pyRet = ito::PythonItom::PySaveMatlabMat(NULL, pArgs);
            retVal += checkForPyExceptions();
            Py_XDECREF(pyRet);
            Py_XDECREF(pArgs);

            PyGILState_Release(gstate);
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
/*
\param packedVarName -> if this string is != "", the dictionary loaded from the mat file will be kept as dictionary and saved in the workspace under this variable name
*/
ito::RetVal PythonEngine::loadMatlabVariables(bool globalNotLocal, QString filename, QString packedVarName, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);
    tPythonState oldState = pythonState;
    RetVal retVal;
    PyObject* destinationDict = NULL;
    bool released = false;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += RetVal(retError, 0, tr("it is not allowed to load matlab variables in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy").toLatin1().data());
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
            retVal += RetVal(retError, 0, tr("variables can not be load since dictionary is not available").toLatin1().data());
        }
        else
        {
            PyGILState_STATE gstate = PyGILState_Ensure();

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
                if (packedVarName != "")
                {
                    PyObject *key = PythonQtConversion::QStringToPyObject(packedVarName); //new ref
                    if (key)
                    {
                        PyDict_SetItem(destinationDict, key, dict);
                    }
                    Py_XDECREF(key);
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
            }

            Py_XDECREF(dict);

            PyGILState_Release(gstate);

            if (semaphore) 
            {
                semaphore->returnValue = retVal;
                semaphore->release();
                released = true;
            }

            gstate = PyGILState_Ensure();
            if (globalNotLocal)
            {
                emitPythonDictionary(true, false, getGlobalDictionary(), NULL);
            }
            else
            {
                emitPythonDictionary(false, true, NULL, getLocalDictionary());
            }
            PyGILState_Release(gstate);
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
ito::RetVal PythonEngine::checkVarnamesInWorkspace(bool globalNotLocal, const QStringList &names, QSharedPointer<IntList> existing, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(semaphore);
    tPythonState oldState = pythonState;
    ito::RetVal retVal;
    PyObject* destinationDict = NULL;
    PyObject* value = NULL;
    bool released = false;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += ito::RetVal(ito::retError, 0, tr("It is not allowed to check names of variables in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy").toLatin1().data());
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
            retVal += ito::RetVal(ito::retError, 0, tr("values cannot be saved since workspace dictionary not available.").toLatin1().data());
        }
        else
        {
            PyGILState_STATE gstate = PyGILState_Ensure();

            PyObject *existingItem = NULL;
            PyObject *varname = NULL;

            existing->clear();
            existing->reserve(names.size());

            for (int i = 0; (i < names.size()) && (!retVal.containsError()); i++)
            {
                varname = getAndCheckIdentifier(names[i], retVal); //new ref
                existingItem = varname ? PyDict_GetItem(destinationDict, varname) : NULL; //borrowed ref

                if (existingItem)
                {
                    if (PyFunction_Check(existingItem) || PyCFunction_Check(existingItem)
                        || PyMethod_Check(existingItem) || PyType_Check(existingItem) ||
                        PyModule_Check(existingItem))
                    {
                        existing->push_back(2); //existing, non overwritable
                    }
                    else
                    {
                        existing->push_back(1); //existing, but overwritable
                    }
                }
                else
                {
                    existing->push_back(0); //non existing
                }

                Py_DECREF(varname);
            }

            PyGILState_Release(gstate);

            if (semaphore != NULL)
            {
                semaphore->returnValue = retVal;
                semaphore->release();
                released = true;
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

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::getVarnamesListInWorkspace(bool globalNotLocal, const QString &find, QSharedPointer<QStringList> varnameList, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(semaphore);
    tPythonState oldState = pythonState;
    ito::RetVal retVal;
    PyObject* destinationDict = NULL;
    PyObject* value = NULL;
    bool released = false;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += ito::RetVal(ito::retError, 0, tr("It is not allowed to check names of variables in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy").toLatin1().data());
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
            retVal += ito::RetVal(ito::retError, 0, tr("values cannot be saved since workspace dictionary not available.").toLatin1().data());
        }
        else
        {
            PyGILState_STATE gstate = PyGILState_Ensure();
            varnameList->clear();
            PyObject *key, *value;
            Py_ssize_t pos = 0;
            QRegExp rx(find);
            rx.setPatternSyntax(QRegExp::Wildcard);
            bool ok;

            while (PyDict_Next(destinationDict, &pos, &key, &value))
            {
                QString qstringKey = PythonQtConversion::PyObjGetString(key, true, ok);
                if (ok && rx.exactMatch(qstringKey))
                {
                    varnameList->append(qstringKey);
                }
            }

            PyGILState_Release(gstate);

            if (semaphore != NULL)
            {
                semaphore->returnValue = retVal;
                semaphore->release();
                released = true;
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

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*
This method tries to acquire the Python GIL before putting the values to the workspace. However, the current state of the 
state machine is not considered. This is a first test for this behaviour and should work quite well in this case, since
the operation will not take a lot of time. \TODO: think about similar behaviours and the role of the state machine in the context
of the GIL.
*/
ito::RetVal PythonEngine::putParamsToWorkspace(bool globalNotLocal, const QStringList &names, const QVector<SharedParamBasePointer > &values, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);
    //tPythonState oldState = pythonState;
    ito::RetVal retVal;
    PyObject* destinationDict = NULL;
    PyObject* value = NULL;
    bool released = false;

    if (names.size() != values.size())
    {
        retVal += ito::RetVal(ito::retError, 0, tr("The number of names and values must be equal").toLatin1().data());
    }
    /*else if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += ito::RetVal(ito::retError, 0, tr("It is not allowed to put variables in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy").toLatin1().data());
    }*/
    else
    {
        //if (pythonState == pyStateIdle)
        //{
        //    pythonStateTransition(pyTransBeginRun);
        //}
        //else if (pythonState == pyStateDebuggingWaiting)
        //{
        //    pythonStateTransition(pyTransDebugExecCmdBegin);
        //}

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
            retVal += ito::RetVal(ito::retError, 0, tr("values cannot be saved since workspace dictionary not available.").toLatin1().data());
        }
        else
        {
            PyGILState_STATE gstate = PyGILState_Ensure();

            PyObject *existingItem = NULL;
            PyObject *varname = NULL;

            for (int i = 0; (i < names.size()) && (!retVal.containsError()); i++)
            {
                varname = getAndCheckIdentifier(names[i], retVal); //new ref
                existingItem = varname ? PyDict_GetItem(destinationDict, varname) : NULL; //borrowed ref

                if (existingItem)
                {
                    if (PyFunction_Check(existingItem) || PyCFunction_Check(existingItem))
                    {
                        retVal += ito::RetVal::format(ito::retError, 0, tr("Function '%s' in this workspace can not be overwritten.").toLatin1().data(), names[i].toLatin1().data());
                        Py_XDECREF(varname);
                        break;
                    }
                    else if (PyMethod_Check(existingItem))
                    {
                        retVal += ito::RetVal::format(ito::retError, 0, tr("Method '%s' in this workspace can not be overwritten.").toLatin1().data(), names[i].toLatin1().data());
                        Py_XDECREF(varname);
                        break;
                    }
                    else if (PyType_Check(existingItem))
                    {
                        retVal += ito::RetVal::format(ito::retError, 0, tr("Type or class '%s' in this workspace can not be overwritten.").toLatin1().data(), names[i].toLatin1().data());
                        Py_XDECREF(varname);
                        break;
                    }
                    else if (PyModule_Check(existingItem))
                    {
                        retVal += ito::RetVal::format(ito::retError, 0, tr("Module '%s' in this workspace can not be overwritten.").toLatin1().data(), names[i].toLatin1().data());
                        Py_XDECREF(varname);
                        break;
                    }
                }

                if (varname)
                {
                    value = PythonParamConversion::ParamBaseToPyObject(*(values[i]));
                    if (value == NULL)
                    {
                        retVal += ito::RetVal::format(ito::retError, 0, tr("error while transforming value '%s' to PyObject*.").toLatin1().data(), names[i].toLatin1().data());
                    }
                    else
                    {
                        PyDict_SetItem(destinationDict, varname, value); //existing is automatically decremented
                        Py_XDECREF(value);
                    }
                }
                
                Py_XDECREF(varname);
            }

            //PyGILState_Release(gstate);

            if (semaphore != NULL) 
            {
                semaphore->returnValue = retVal;
                semaphore->release();
                released = true;
            }

            //gstate = PyGILState_Ensure();
            if (globalNotLocal)
            {
                emitPythonDictionary(true, false, getGlobalDictionary(), NULL);
            }
            else
            {
                emitPythonDictionary(false, true, NULL, getLocalDictionary());
            }
            PyGILState_Release(gstate);
        }

        /*if (oldState == pyStateIdle)
        {
            pythonStateTransition(pyTransEndRun);
        }
        else if (oldState == pyStateDebuggingWaiting)
        {
            pythonStateTransition(pyTransDebugExecCmdEnd);
        }*/
    }

    if (semaphore != NULL && !released) 
    {
        semaphore->returnValue = retVal;
        semaphore->release();
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PythonEngine::getParamsFromWorkspace(bool globalNotLocal, const QStringList &names, QVector<int> paramBaseTypes, QSharedPointer<SharedParamBasePointerVector > values, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);
    tPythonState oldState = pythonState;
    ito::RetVal retVal;
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
        retVal += ito::RetVal(ito::retError, 0, tr("it is not allowed to load variables in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy").toLatin1().data());
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

        if ((globalNotLocal && getGlobalDictionary() == NULL) || (!globalNotLocal && getLocalDictionary() == NULL))
        {
            retVal += ito::RetVal(ito::retError, 0, tr("values cannot be obtained since workspace dictionary not available.").toLatin1().data());
        }
        else
        {
            PyGILState_STATE gstate = PyGILState_Ensure();
            QString validVariableName;

            for (int i = 0; i < names.size(); i++)
            {
                value = getPyObjectByFullName(globalNotLocal, names[i], &validVariableName); //new reference
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
                    param = PythonParamConversion::PyObjectToParamBase(value, validVariableName.toLatin1().data(), retVal, paramBaseTypes[i], false);

                    Py_DECREF(value);

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

            PyGILState_Release(gstate);

            if (semaphore != NULL) 
            {
                semaphore->returnValue = retVal;
                semaphore->release();
                released = true;
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

    return retVal;
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
            PyGILState_STATE gstate = PyGILState_Ensure();

            PyObject *pyVarname = getAndCheckIdentifier(varname, retVal); //new reference

            if (pyVarname)
            {
                if (PyDict_GetItem(dict, pyVarname) != NULL)
                {
                    QString ErrStr = tr("variable name '%1' already exists in dictionary").arg(varname);
                    retVal += RetVal(retError, 0, ErrStr.toLatin1().data());
                }
                else
                {
                    if (instance->getBasePlugin()->getType() & ito::typeDataIO)
                    {
                        PythonPlugins::PyDataIOPlugin *dataIOPlugin = (PythonPlugins::PyDataIOPlugin*)PythonPlugins::PyDataIOPluginType.tp_new(&PythonPlugins::PyDataIOPluginType,NULL,NULL); //new ref
                        if (dataIOPlugin == NULL)
                        {
                            retVal += RetVal(retError, 0, tr("No instance of python class dataIO could be created").toLatin1().data());
                        }
                        else
                        {
                            instance->getBasePlugin()->incRef(instance);
                            dataIOPlugin->dataIOObj = (ito::AddInDataIO*)instance;
                            value = (PyObject*)dataIOPlugin;
                        }
                    }
                    else if (instance->getBasePlugin()->getType() & ito::typeActuator)
                    {
                        PythonPlugins::PyActuatorPlugin *actuatorPlugin = (PythonPlugins::PyActuatorPlugin*)PythonPlugins::PyActuatorPluginType.tp_new(&PythonPlugins::PyActuatorPluginType,NULL,NULL); //new ref
                        if (actuatorPlugin == NULL)
                        {
                            retVal += RetVal(retError, 0, tr("No instance of python class actuator could be created").toLatin1().data());
                        }
                        else
                        {
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
                        PyDict_SetItem(dict, pyVarname, value); //increments reference of value
                        Py_XDECREF(value);
                        if (PyErr_Occurred())
                        {
                            retVal += RetVal(retError, 0, tr("Dictionary is not available").toLatin1().data());
                            PyErr_PrintEx(0);
                        }
                    }
                }
            }

            Py_XDECREF(pyVarname);

            PyGILState_Release(gstate);
        }

        if (semaphore != NULL) //release semaphore now, since the following emit command will be a blocking connection, too.
        {
            semaphore->returnValue = retVal;
            semaphore->release();
        }

        PyGILState_STATE gstate = PyGILState_Ensure();
        if (globalNotLocal)
        {
            emitPythonDictionary(true, false, getGlobalDictionary(), NULL);
        }
        else
        {
            emitPythonDictionary(false, true, NULL, getLocalDictionary());
        }
        PyGILState_Release(gstate);

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
//! get the unicode object from identifier and checks if it is a valid python identifier (variable name). This returns a new reference of the unicode object or NULL with a corresponding error message (python error flag is cleared)
PyObject* PythonEngine::getAndCheckIdentifier(const QString &identifier, ito::RetVal &retval) const
{
    QByteArray ba = identifier.toLatin1();
    PyObject *obj = PyUnicode_DecodeLatin1(ba.data(), ba.size(), NULL);
    if (obj)
    {
        if (!PyUnicode_IsIdentifier(obj))
        {
            Py_DECREF(obj);
            obj = NULL;
            retval += ito::RetVal::format(ito::retError, 0, "string '%s' is no valid python identifier", ba.data());
        }
    }
    else
    {
        PyErr_Clear();
        retval += ito::RetVal::format(ito::retError, 0, "string '%s' cannot be interpreted as unicode", ba.data());
    }

    return obj;
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
            PyGILState_STATE gstate = PyGILState_Ensure();

            PyObject *result = PyObject_CallMethod(itomFunctions, "getModules", "");

            if (!result)
            {
                retValue += RetVal(retError, 0, tr("error while loading the modules").toLatin1().data());
                PyErr_PrintEx(0);
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

            PyGILState_Release(gstate);
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
            PyGILState_STATE gstate = PyGILState_Ensure();

            PyObject *stringList = PythonQtConversion::QStringListToPyList(*modNames);
            modNames->clear();

            PyObject *result = PyObject_CallMethod(itomFunctions, "reloadModules", "O", stringList);

            if (!result)
            {
                retValue += RetVal(retError, 0, tr("error while reloading the modules").toLatin1().data());
                PyErr_PrintEx(0);
            }
            else
            {
                
            }
            Py_XDECREF(result);
            Py_XDECREF(stringList);

            PyGILState_Release(gstate);
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
        retVal += ito::RetVal(retError, 0, tr("it is not allowed to pickle a variable in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy").toLatin1().data());
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

        PyGILState_STATE gstate = PyGILState_Ensure();

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
            retVal += ito::RetVal(retError, 0, tr("variables can not be pickled since dictionary is not available").toLatin1().data());
        }
        else
        {
            //build dictionary, which should be pickled
            PyObject* exportDict = PyDict_New();
            PyObject* tempElem = NULL;
            QString validVariableName;

            for (int i = 0 ; i < varNames.size() ; i++)
            {
                tempElem = getPyObjectByFullName(globalNotLocal, varNames[i], &validVariableName); //new reference

                if (tempElem == NULL)
                {
                    std::cerr << "variable '" << validVariableName.toLatin1().data() << "' can not be found in dictionary and will not be exported.\n" << std::endl;
                }
                else
                {
                    PyDict_SetItemString(exportDict, validVariableName.toLatin1().data(), tempElem); //increments tempElem by itsself
                    Py_DECREF(tempElem);
                }
            }

            retVal += pickleDictionary(exportDict, filename);

            PyDict_Clear(exportDict);
            Py_DECREF(exportDict);
            exportDict = NULL;
        }

        PyGILState_Release(gstate);


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
/** save a single DataObject, PointCloud or PolygonMesh to an *.idc file using the python module 'pickle'.
*
*  Invoke this method by another thread (e.g. any GUI) to save a single object to an 'idc' file.
*  
*  \param filename is the filename of the idc file
*  \param value is the given DataObject, PointCloud or PolygonMesh in terms of ito::Param
*  \param valueName is the name of the variable in the idc file
*  \param semaphore is the control semaphore for an asychronous call.
*/
ito::RetVal PythonEngine::pickleSingleParam(QString filename, QSharedPointer<ito::Param> value, const QString &valueName, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);

    tPythonState oldState = pythonState;
    RetVal retVal;
    PyObject* dict = NULL;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += ito::RetVal(retError, 0, tr("it is not allowed to pickle a variable in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy").toLatin1().data());
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

        PyObject *item = NULL;

        if (value.isNull())
        {
            retVal += ito::RetVal(retError, 0, tr("could not pickle since value is empty.").toLatin1().data());
        }
        else
        {
            switch (value->getType())
            {
            case (ito::ParamBase::DObjPtr & paramTypeMask) :
            {
                const ito::DataObject *obj = value->getVal<const ito::DataObject*>();
                if (obj)
                {
                    PyGILState_STATE gstate = PyGILState_Ensure();
                    item = PythonQtConversion::DataObjectToPyObject(*obj);
                    PyGILState_Release(gstate);
                }
                else
                {
                    retVal += ito::RetVal(retError, 0, tr("could not pickle dataObject since it is not available.").toLatin1().data());
                }
            }
            break;
#if ITOM_POINTCLOUDLIBRARY > 0 
            case (ito::ParamBase::PointCloudPtr & paramTypeMask) :
            {
                const ito::PCLPointCloud *cloud = value->getVal<const ito::PCLPointCloud*>();
                if (cloud)
                {
                    PyGILState_STATE gstate = PyGILState_Ensure();
                    item = PythonQtConversion::PCLPointCloudToPyObject(*cloud);
                    PyGILState_Release(gstate);
                }
                else
                {
                    retVal += ito::RetVal(retError, 0, tr("could not pickle dataObject since it is not available.").toLatin1().data());
                }
            }
            break;

            case (ito::ParamBase::PolygonMeshPtr & paramTypeMask) :
            {
                const ito::PCLPolygonMesh *mesh = value->getVal<const ito::PCLPolygonMesh*>();
                if (mesh)
                {
                    PyGILState_STATE gstate = PyGILState_Ensure();
                    item = PythonQtConversion::PCLPolygonMeshToPyObject(*mesh);
                    PyGILState_Release(gstate);
                }
                else
                {
                    retVal += ito::RetVal(retError, 0, tr("could not pickle dataObject since it is not available.").toLatin1().data());
                }
            }
            break;
#endif
            default:
                retVal += ito::RetVal(retError, 0, tr("unsupported data type to pickle.").toLatin1().data());
            }

            if (item == NULL)
            {
                retVal += ito::RetVal(retError, 0, tr("error converting object to Python object. No pickle possible.").toLatin1().data());
            }
        }

        if (!retVal.containsError())
        {
            PyGILState_STATE gstate = PyGILState_Ensure();

            //build dictionary, which should be pickled
            PyObject* exportDict = PyDict_New();
            PyDict_SetItemString(exportDict, valueName.toLatin1().data(), item); //creates new reference for item

            retVal += pickleDictionary(exportDict, filename);

            PyDict_Clear(exportDict);
            Py_DECREF(exportDict);
            exportDict = NULL;

            PyGILState_Release(gstate);
        }

        Py_XDECREF(item);

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
ito::RetVal PythonEngine::pickleDictionary(PyObject *dict, const QString &filename)
{
#if defined _DEBUG && PY_VERSION_HEX >= 0x03040000
	if (!PyGILState_Check())
	{
		std::cerr << "Python GIL must be locked when calling pickleDictionary\n" << std::endl;
		return ito::retError;
	}
#endif

    RetVal retval;

    if (mainModule == NULL)
    {
        return RetVal(retError, 0, tr("mainModule is empty or cannot be accessed").toLatin1().data());
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
    
    if (pyFileName != NULL)
    {
        fileHandle = PyObject_CallFunctionObjArgs(openMethod, pyFileName, pyMode, NULL);
        Py_DECREF(pyFileName);
    }
    
    if (pyMode) Py_DECREF(pyMode);


    if (fileHandle == NULL)
    {
        retval += checkForPyExceptions();
    }
    else
    {
        PyObject *result = NULL;
        PyObject *version = PyLong_FromLong(3); //Use pickle protocol version 3 as default. This is readable by all itom version that have been published (default for Python 3).
        
        try
        {
            result = PyObject_CallMethodObjArgs(pickleModule, PyUnicode_FromString("dump"), dict, fileHandle, version, NULL);
        }
        catch(std::bad_alloc &/*ba*/)
        {
            retval += RetVal(retError, 0, tr("No more memory available during pickling.").toLatin1().data());
        }
        catch(std::exception &exc)
        {
            if (exc.what())
            {
                retval += ito::RetVal::format(ito::retError, 0, tr("The exception '%s' has been thrown during pickling.").toLatin1().data(), exc.what());
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, tr("Pickle error. An unspecified exception has been thrown.").toLatin1().data());
            }
        }
        catch (...)
        {
            retval += ito::RetVal(ito::retError, 0, tr("Pickle error. An unspecified exception has been thrown.").toLatin1().data());
        }

        Py_DECREF(version);

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
/*
\param packedVarName -> if this string is != "", the dictionary loaded from the idc file will be kept as dictionary and saved in the workspace under this variable name
*/
ito::RetVal PythonEngine::unpickleVariables(bool globalNotLocal, QString filename, QString packedVarName, ItomSharedSemaphore *semaphore)
{
    ItomSharedSemaphoreLocker locker(semaphore);
    tPythonState oldState = pythonState;
    RetVal retVal;
    bool released = false;
    PyObject* destinationDict = NULL;

    if (pythonState == pyStateRunning || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaitingButBusy)
    {
        retVal += RetVal(retError, 0, tr("it is not allowed to unpickle a data collection in modes pyStateRunning, pyStateDebugging or pyStateDebuggingWaitingButBusy").toLatin1().data());
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
            retVal += RetVal(retError, 0, tr("variables can not be unpickled since dictionary is not available").toLatin1().data());
        }
        else
        {
			PyGILState_STATE gstate = PyGILState_Ensure();
            if (packedVarName != "")
            {
                PyObject *dict = PyDict_New();
                retVal += unpickleDictionary(dict, filename, true);
                if (!retVal.containsError())
                {
                    PyObject *key = PythonQtConversion::QStringToPyObject(packedVarName);
                    if (key)
                    {
                        PyDict_SetItem(destinationDict, key, dict);
                    }
                    Py_XDECREF(key);
                }
                Py_XDECREF(dict);
            }
            else
            {
                retVal += unpickleDictionary(destinationDict, filename, true);
            }
			PyGILState_Release(gstate);

            if (semaphore && !released)
            {
                semaphore->returnValue = retVal;
                semaphore->release();
                released = true;
            }

            gstate = PyGILState_Ensure();
            if (globalNotLocal)
            {
                emitPythonDictionary(true, false, getGlobalDictionary(), NULL);
            }
            else
            {
                emitPythonDictionary(false, true, NULL, getLocalDictionary());
            }
            PyGILState_Release(gstate);
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
ito::RetVal PythonEngine::unpickleDictionary(PyObject *destinationDict, const QString &filename, bool overwrite)
{
#if defined _DEBUG && PY_VERSION_HEX >= 0x03040000
	if (!PyGILState_Check())
	{
		std::cerr << "Python GIL must be locked when calling unpickleDictionary\n" << std::endl;
		return ito::retError;
	}
#endif

    RetVal retval;

    if (mainModule == NULL)
    {
        return RetVal(retError, 0, tr("mainModule is empty or cannot be accessed").toLatin1().data());
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
    
    if (pyFileName != NULL)
    {
        fileHandle = PyObject_CallFunctionObjArgs(openMethod, pyFileName, pyMode, NULL);
        Py_DECREF(pyFileName);
    }
    
    if (pyMode) Py_DECREF(pyMode);

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
            retval += RetVal(retError, 0, tr("No more memory available during unpickling.").toLatin1().data());
        }
        catch(std::exception &exc)
        {
            if (exc.what())
            {
                retval += ito::RetVal::format(ito::retError, 0, tr("The exception '%s' has been thrown during unpickling.").toLatin1().data(), exc.what());
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, tr("Unpickling error. An unspecified exception has been thrown.").toLatin1().data());
            }
        }
        catch (...)
        {
            retval += ito::RetVal(ito::retError, 0, tr("Unpickling error. An unspecified exception has been thrown.").toLatin1().data());
        }


        if (unpickledItem == NULL)
        {
            retval += checkForPyExceptions();
        }
        else if (!PyDict_Check(unpickledItem))
        {
            retval += RetVal(retError, 0, tr("unpickling error. This file contains no dictionary as base element.").toLatin1().data());
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
#if QT_VERSION >= 0x050000
void PythonEngine::connectNotify(const QMetaMethod &signal)
{
     if (signal == QMetaMethod::fromSignal(&PythonEngine::pythonAutoReloadChanged))
#else
void PythonEngine::connectNotify(const char* signal)
{
     if (QLatin1String(signal) == SIGNAL(pythonAutoReloadChanged(bool,bool,bool,bool)))
#endif
     {
        emit pythonAutoReloadChanged(m_autoReload.enabled, m_autoReload.checkFileExec, m_autoReload.checkStringExec, m_autoReload.checkFctExec);
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

