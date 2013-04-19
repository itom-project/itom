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

#ifndef PYTHONENGINE_H
#define PYTHONENGINE_H

/*if you add any include to this file you will DIE an immediate, horrible, painful death*/

#include <string>
//#ifndef Q_MOC_RUN
//    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API
//    #define NO_IMPORT_ARRAY
//#endif

//#define NPY_NO_DEPRECATED_API 0x00000007 //see comment in pythonNpDataObject.cpp

//python
// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#ifdef _DEBUG
    #undef _DEBUG
    #if (defined linux) | (defined CMAKE)
        #include "Python.h"
        #include "node.h"
        #include "numpy/arrayobject.h"
    #else
        #include "Python.h"
        #include "node.h"
        #include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
    #endif
    #define _DEBUG
#else
#ifdef linux
    #include "Python.h"
    #include "node.h"
    #include "numpy/arrayobject.h"
#else
    #include "Python.h"
    #include "node.h"
    #include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
#endif
#endif

/* includes */

#include "pythonNpDataObject.h"
#include "pythonItom.h"

/*
#ifdef linux
    #include "frameobject.h"
    #include "traceback.h"
#else
    #include "include/frameobject.h" //!< for traceback
    #include "include/traceback.h"
#endif
*/

#include "../models/breakPointModel.h"
#include "../../common/sharedStructuresQt.h"
#include "../../common/addInInterface.h"

#include "pythonWorkspace.h"

#include <qstringlist.h>
#include <qqueue.h>
#include <qset.h>


/* definition and macros */

/* global variables (avoid) */

/* content */

class QDesktopWidget;
class QTimer;

class PythonEngine : public QObject
{
    Q_OBJECT

public:
    PythonEngine();                                 //constructor
    ~PythonEngine();                                //destructor

    Q_INVOKABLE void pythonSetup(ito::RetVal *retValue);               //setup
    Q_INVOKABLE ito::RetVal scanAndRunAutostartFolder(QString currentDirAfterScan = QString() );
    Q_INVOKABLE ito::RetVal pythonShutdown(ItomSharedSemaphore *aimWait = NULL);            //shutdown

    Q_INVOKABLE ito::RetVal stringEncodingChanged();

    inline BreakPointModel *getBreakPointModel() const { return bpModel; }
    inline bool isPythonBusy() const { return pythonState != pyStateIdle; }
    inline bool isPythonDebugging() const { return (pythonState == pyStateDebuggingWaitingButBusy || pythonState == pyStateDebugging || pythonState == pyStateDebuggingWaiting); }
    inline bool isPythonDebuggingAndWaiting() const { return pythonState == pyStateDebuggingWaiting; }
    inline bool execInternalCodeByDebugger() const { return m_executeInternalPythonCodeInDebugMode; }
    inline void setExecInternalCodeByDebugger(bool value) { m_executeInternalPythonCodeInDebugMode = value; }

    void printPythonError();
    void printPythonErrorWithoutTraceback();

    void pythonDebugFunction(PyObject *callable, PyObject *argTuple);
    void pythonRunFunction(PyObject *callable, PyObject *argTuple);

	inline PyObject *getGlobalDictionary()  { return globalDictionary;  }  /*!< returns reference to main dictionary (main workspace) */

    static const PythonEngine *getInstance();

    QList<int> parseAndSplitCommandInMainComponents(const char *str, QByteArray &encoding) const; //can be directly called from different thread

protected:
    RetVal syntaxCheck(char* pythonFileName);       // syntaxCheck for file with filename pythonFileName
    RetVal runPyFile(char* pythonFileName);         // run file pythonFileName
    RetVal debugFile(char* pythonFileName);         // debug file pythonFileName
    RetVal runString(const char *command);          // run string command
    RetVal debugString(const char *command);        // debug string command
    RetVal debugFunction(PyObject *callable, PyObject *argTuple);
    RetVal runFunction(PyObject *callable, PyObject *argTuple);

    RetVal modifyTracebackDepth(int NrOfLevelsToPopAtFront = -1, bool showTraceback = true);

    static char* asString(PyObject *value, char *defaultString = NULL);

private:
    static PythonEngine *getInstanceInternal();

    inline PyObject *getLocalDictionary() { return localDictionary; } /*!< returns reference to local dictionary (workspace of method, which is handled right now). Is NULL if no method is executed right now. */

    PyObject *getPyObjectByFullName(bool globalNotLocal, QStringList &fullName);

    void setGlobalDictionary(PyObject* mainDict = NULL);
    void setLocalDictionary(PyObject* localDict);

    void emitPythonDictionary(bool emitGlobal, bool emitLocal, PyObject* globalDict, PyObject* localDict);

    RetVal pickleDictionary(PyObject *dict, QString filename);
    RetVal unpickleDictionary(PyObject *destinationDict, QString filename, bool overwrite);
    RetVal saveDictAsMatlab(PyObject *dict, QString filename);
    RetVal loadMatlabToDict(PyObject *destinationDict, QString filename);

    //methods for maintaining python functionality
    RetVal addMethodToModule(PyMethodDef* def);
    RetVal delMethodFromModule(const char* ml_name);
    RetVal pythonAddBuiltinMethods();

    //methods for debugging
    void enqueueDbgCmd(tPythonDbgCmd dbgCmd);
    tPythonDbgCmd dequeueDbgCmd();
    bool DbgCommandsAvailable();
    void clearDbgCmdLoop();

    RetVal pythonStateTransition(tPythonTransitions transition);

    //methods for breakpoint
    RetVal pythonAddBreakpoint(const QString &filename, const int lineno, const bool enabled, const bool temporary, const QString &condition, const int ignoreCount, int &pyBpNumber);
    RetVal pythonEditBreakpoint(const int pyBpNumber, const QString &filename, const int lineno, const bool enabled, const bool temporary, const QString &condition, const int ignoreCount);
    RetVal pythonDeleteBreakpoint(const int pyBpNumber);

    //member variables
    bool started;

    //PyGILState_STATE threadState;

    QMutex dbgCmdMutex;
    QMutex pythonStateChangeMutex;
    QMutex dictChangeMutex;
    QDesktopWidget *m_pDesktopWidget;
    QQueue<tPythonDbgCmd> debugCommandQueue;
    tPythonDbgCmd debugCommand;
    
    tPythonState pythonState;
    
    BreakPointModel *bpModel;

    PyObject* mainModule;          //!< main module of python (builtin) [borrowed]
    PyObject* mainDictionary;      //!< main dictionary of python [borrowed]
    PyObject* localDictionary;     //!< local dictionary of python [borrowed], usually NULL unless if debugger is in "interaction-mode", then globalDictionary is equal to the local dictionary of the current frame
	PyObject* globalDictionary;    //!< global dictionary of python [borrowed], equals to mainDictionary unless if debugger is in "interaction-mode", then globalDictionary is equal to the global dictionary of the current frame
    PyObject *itomDbgModule;       //!< debugger module
    PyObject *itomDbgInstance;     //!< debugger instance
    PyObject *itomModule;          //!< itom module [new ref]
    PyObject *itomFunctions;       //!< ito functions [additional python methods] [new ref]
    PyObject *gcModule;
    //PyObject *itomReturnException; //!< if this exception is thrown, the execution of the main application is stopped

    PyObject *dictUnicode;

    QSet<PyWorkspaceContainer*> m_mainWorkspaceContainer;
    QSet<PyWorkspaceContainer*> m_localWorkspaceContainer;
    QHash<QString, QPair<PyObject*,PyObject*> > m_pyFuncWeakRefHashes; //!< hash table containing weak reference to callable python methods or functions and as second, optional PyObject* an tuple, passed as argument to that function. These functions are for example executed by menu-clicks in the main window.
    int m_pyFuncWeakRefHashesAutoInc;
    bool m_executeInternalPythonCodeInDebugMode; //!< if true, button events, user interface connections to python methods... will be executed by debugger
    PyMethodDef* PythonAdditionalModuleITOM;

    //!< debugger functionality
    static PyMethodDef PyMethodItomDbg[];
    static PyModuleDef PyModuleItomDbg;
    static PyObject* PyInitItomDbg(void);
    static PyObject* PyDbgCommandLoop(PyObject *pSelf, PyObject *pArgs);

    //helper methods
    //static PyObject* checkForTimeoutHelper(ItomSharedSemaphore* semaphore, int timeout, PyObject *retValueOk);

    //other static members
    static QMutex instatiated;
    static QMutex instancePtrProtection;

    static PythonEngine* instance;

    // friend class
    friend class PythonDataObject;
    friend class ito::PythonItom;

signals:
    void pythonDebugPositionChanged(QString filename, int lineNo);
    void pythonStateChanged(tPythonTransitions pyTransition);
    void pythonModifyLocalDict(PyObject* localDict, ItomSharedSemaphore* semaphore);
    void pythonModifyGlobalDict(PyObject* globalDict, ItomSharedSemaphore* semaphore);
    void pythonAddToolbarButton(QString toolbarName, QString buttonName, QString buttonIconFilename, QString pythonCode);
    void pythonRemoveToolbarButton(QString toolbarName, QString buttonName);
    void pythonAddMenuElement(int typeID, QString key, QString name, QString code, QString icon);
    void pythonRemoveMenuElement(QString key);
    void pythonCurrentDirChanged();
	void updateCallStack(QStringList filenames, IntList lines, QStringList methods);
	void deleteCallStack();

    void pythonSetCursor(const Qt::CursorShape cursor);
    void pythonResetCursor();

public slots:
    void pythonRunString(QString cmd);
    void pythonDebugString(QString cmd);
    void pythonExecStringFromCommandLine(QString cmd);
    void pythonRunFile(QString filename);
    void pythonDebugFile(QString filename);
    void pythonRunStringOrFunction(QString cmdOrFctHash);
    void pythonDebugStringOrFunction(QString cmdOrFctHash);
    void pythonInterruptExecution() const;
    void pythonDebugCommand(tPythonDbgCmd cmd);

    void pythonGenericSlot(PyObject* callable, PyObject *argumentTuple);

    //!< these slots are only connected if python in debug-mode; while waiting these slots will be treated due to progressEvents-call in PythonEngine::PyDbgCommandLoop
    void breakPointAdded(BreakPointItem bp, int row);
    void breakPointDeleted(QString filename, int lineNo, int pyBpNumber);
    void breakPointChanged(BreakPointItem oldBp, BreakPointItem newBp);
    RetVal setupBreakPointDebugConnections();
    RetVal shutdownBreakPointDebugConnections();

    bool renameVariable(bool globalNotLocal, QString oldKey, QString newKey, ItomSharedSemaphore *semaphore = NULL);
    bool deleteVariable(bool globalNotLocal, QStringList keys, ItomSharedSemaphore *semaphore = NULL);
    bool pickleVariables(bool globalNotLocal, QString filename, QStringList varNames, ItomSharedSemaphore *semaphore = NULL);
    bool unpickleVariables(bool globalNotLocal, QString filename, ItomSharedSemaphore *semaphore = NULL);
    bool saveMatlabVariables(bool globalNotLocal, QString filename, QStringList varNames, ItomSharedSemaphore *semaphore = NULL);
    bool loadMatlabVariables(bool globalNotLocal, QString filename, ItomSharedSemaphore *semaphore = NULL);
    RetVal registerAddInInstance(QString varname, ito::AddInBase *instance, ItomSharedSemaphore *semaphore = NULL);
    RetVal getSysModules(QSharedPointer<QStringList> modNames, QSharedPointer<QStringList> modFilenames, QSharedPointer<IntList> modTypes, ItomSharedSemaphore *semaphore = NULL);
    RetVal reloadSysModules(QSharedPointer<QStringList> modNames, ItomSharedSemaphore *semaphore = NULL);

    void registerWorkspaceContainer(PyWorkspaceContainer *container, bool registerNotUnregister, bool globalNotLocal);
    void workspaceGetChildNode(ito::PyWorkspaceContainer *container, QString fullNameParentItem);
    void workspaceGetValueInformation(PyWorkspaceContainer *container, QString fullItemName, QSharedPointer<QString> extendedValue, ItomSharedSemaphore *semaphore = NULL);

    void putParamsToWorkspace(bool globalNotLocal, QStringList names, QVector<SharedParamBasePointer > values, ItomSharedSemaphore *semaphore = NULL);
    void getParamsFromWorkspace(bool globalNotLocal, QStringList names, QVector<int> paramBaseTypes, QSharedPointer<SharedParamBasePointerVector > values, ItomSharedSemaphore *semaphore = NULL);

private slots:

};


#endif
