/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut fuer Technische Optik (ITO),
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

#ifndef PYTHONENGINE_H
#define PYTHONENGINE_H

/*if you add any include to this file you will DIE an immediate, horrible, painful death*/

#include <string>
#include <exception>
//#ifndef Q_MOC_RUN
//    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API
//    #define NO_IMPORT_ARRAY
//#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //see comment in pythonNpDataObject.cpp

#ifndef Q_MOC_RUN
    //python
    // see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
    #if (defined _DEBUG) && (defined WIN32)
        #undef _DEBUG
        #if (defined linux) | (defined CMAKE)
            #include "Python.h"
            #include "node.h"
            #include "numpy/arrayobject.h"
        #elif (defined __APPLE__) | (defined CMAKE)
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
    #elif (defined __APPLE__)
        #include "Python.h"
        #include "node.h"
        #include "numpy/arrayobject.h"
    #else
        #include "Python.h"
        #include "node.h"
        #include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
    #endif
    #endif
#endif // Q_MOC_RUN

/* includes */

#include "pythonNpDataObject.h"
#include "pythonItom.h"
#include "pythonProxy.h"

#include "../models/breakPointModel.h"
#include "../../common/sharedStructuresQt.h"
#include "../../common/addInInterface.h"

#include "pythonWorkspace.h"

#include <qstringlist.h>
#include <qqueue.h>
#include <qset.h>
#include <qpointer.h>
#include <qatomic.h>


/* definition and macros */

/* global variables (avoid) */

/* content */

class QDesktopWidget;
class QTimer;

namespace ito
{

class FuncWeakRef
{
public:
    FuncWeakRef();
    FuncWeakRef(PythonProxy::PyProxy *proxyObject, PyObject *argTuple = NULL);
    FuncWeakRef(const FuncWeakRef &rhs);
    ~FuncWeakRef();
    FuncWeakRef& operator =(FuncWeakRef rhs);

    PythonProxy::PyProxy* getProxyObject() const { return m_proxyObject; } //borrowed reference
    PyObject* getArguments() const { return m_argument; } //borrowed reference
    bool isValid() const { return (m_proxyObject != NULL); }

    void setHandle(const size_t &handle);
    size_t getHandle() const { return m_handle; }
private:
    PythonProxy::PyProxy *m_proxyObject;
    PyObject *m_argument;
    size_t m_handle;
};


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

    inline ito::BreakPointModel *getBreakPointModel() const { return bpModel; }
    inline bool isPythonBusy() const                { return pythonState != ito::pyStateIdle; }
    inline bool isPythonDebugging() const           { return (pythonState == ito::pyStateDebuggingWaitingButBusy || pythonState == ito::pyStateDebugging || pythonState == ito::pyStateDebuggingWaiting); }
    inline bool isPythonDebuggingAndWaiting() const { return pythonState == ito::pyStateDebuggingWaiting; }
    inline bool execInternalCodeByDebugger() const  { return m_executeInternalPythonCodeInDebugMode; }
    inline void setExecInternalCodeByDebugger(bool value) { m_executeInternalPythonCodeInDebugMode = value; }
    ito::RetVal checkForPyExceptions();
    void printPythonErrorWithoutTraceback();
    void pythonDebugFunction(PyObject *callable, PyObject *argTuple, bool gilExternal = false);
    void pythonRunFunction(PyObject *callable, PyObject *argTuple, bool gilExternal = false);    
    inline PyObject *getGlobalDictionary()  const { return globalDictionary;  }  /*!< returns reference to main dictionary (main workspace) */
    inline bool pySyntaxCheckAvailable() const { return (m_pyModSyntaxCheck != NULL); }
    QList<int> parseAndSplitCommandInMainComponents(const char *str, QByteArray &encoding) const; //can be directly called from different thread
    QString getPythonExecutable() const { return m_pythonExecutable; }
    Qt::HANDLE getPythonThreadId() const { return m_pythonThreadId; }

    static bool isInterruptQueued();
    static const PythonEngine *getInstance();
protected:
    //RetVal syntaxCheck(char* pythonFileName);       // syntaxCheck for file with filename pythonFileName
    ito::RetVal runPyFile(const QString &pythonFileName);         // run file pythonFileName
    ito::RetVal debugFile(const QString &pythonFileName);         // debug file pythonFileName
    ito::RetVal runString(const QString &command);          // run string command
    ito::RetVal debugString(const QString &command);        // debug string command
    ito::RetVal debugFunction(PyObject *callable, PyObject *argTuple, bool gilExternal = false);
    ito::RetVal runFunction(PyObject *callable, PyObject *argTuple, bool gilExternal = false);

    ito::RetVal modifyTracebackDepth(int NrOfLevelsToPopAtFront = -1, bool showTraceback = true);

    PyObject* setPyErrFromException(const std::exception &exc);

#if QT_VERSION >= 0x050000
    void connectNotify(const QMetaMethod &signal);
#else
    void connectNotify(const char* signal);
#endif

private:
    static PythonEngine *getInstanceInternal();

    inline PyObject *getLocalDictionary() { return localDictionary; } /*!< returns reference to local dictionary (workspace of method, which is handled right now). Is NULL if no method is executed right now. */

    PyObject *getPyObjectByFullName(bool globalNotLocal, const QStringList &fullName); //Python GIL must be locked when calling this function!

    void setGlobalDictionary(PyObject* mainDict = NULL);
    void setLocalDictionary(PyObject* localDict);

    void emitPythonDictionary(bool emitGlobal, bool emitLocal, PyObject* globalDict, PyObject* localDict);

    ito::RetVal pickleDictionary(PyObject *dict, const QString &filename);
    ito::RetVal unpickleDictionary(PyObject *destinationDict, const QString &filename, bool overwrite);

    //methods for maintaining python functionality
    ito::RetVal addMethodToModule(PyMethodDef* def);
    ito::RetVal delMethodFromModule(const char* ml_name);
    ito::RetVal pythonAddBuiltinMethods();

    //methods for debugging
    void enqueueDbgCmd(ito::tPythonDbgCmd dbgCmd);
    ito::tPythonDbgCmd dequeueDbgCmd();
    bool DbgCommandsAvailable();
    void clearDbgCmdLoop();

    ito::RetVal pythonStateTransition(tPythonTransitions transition);

    //methods for breakpoint
    ito::RetVal pythonAddBreakpoint(const QString &filename, const int lineno, const bool enabled, const bool temporary, const QString &condition, const int ignoreCount, int &pyBpNumber);
    ito::RetVal pythonEditBreakpoint(const int pyBpNumber, const QString &filename, const int lineno, const bool enabled, const bool temporary, const QString &condition, const int ignoreCount);
    ito::RetVal pythonDeleteBreakpoint(const int pyBpNumber);

    ito::RetVal autoReloaderCheck();

    static int queuedInterrupt(void *state); 

    PyObject* getAndCheckIdentifier(const QString &identifier, ito::RetVal &retval) const;
    

    //member variables
    bool m_started;
    //QString m_itomMemberClasses;

    //PyGILState_STATE threadState;

    QMutex dbgCmdMutex;
    QMutex pythonStateChangeMutex;
    QMutex dictChangeMutex;
    QDesktopWidget *m_pDesktopWidget;
    QQueue<ito::tPythonDbgCmd> debugCommandQueue;
    ito::tPythonDbgCmd debugCommand;
    
    ito::tPythonState pythonState;
    
    ito::BreakPointModel *bpModel;

    PyObject* mainModule;          //!< main module of python (builtin) [borrowed]
    PyObject* mainDictionary;      //!< main dictionary of python [borrowed]
    PyObject* localDictionary;     //!< local dictionary of python [borrowed], usually NULL unless if debugger is in "interaction-mode", then globalDictionary is equal to the local dictionary of the current frame
    PyObject* globalDictionary;    //!< global dictionary of python [borrowed], equals to mainDictionary unless if debugger is in "interaction-mode", then globalDictionary is equal to the global dictionary of the current frame
    PyObject *itomDbgModule;       //!< debugger module
    PyObject *itomDbgInstance;     //!< debugger instance
    PyObject *itomModule;          //!< itom module [new ref]
    PyObject *itomFunctions;       //!< ito functions [additional python methods] [new ref]
    PyObject *m_pyModGC;
    PyObject *m_pyModSyntaxCheck;
    //PyObject *itomReturnException; //!< if this exception is thrown, the execution of the main application is stopped

    Qt::HANDLE m_pythonThreadId;

    PyObject *dictUnicode;

    QSet<ito::PyWorkspaceContainer*> m_mainWorkspaceContainer;
    QSet<ito::PyWorkspaceContainer*> m_localWorkspaceContainer;
    QHash<size_t, FuncWeakRef> m_pyFuncWeakRefHashes; //!< hash table containing weak reference to callable python methods or functions and as second, optional PyObject* an tuple, passed as argument to that function. These functions are for example executed by menu-clicks in the main window.
    size_t m_pyFuncWeakRefAutoInc;

    QString m_pythonExecutable; //!< absolute path to the python executable

    bool m_executeInternalPythonCodeInDebugMode; //!< if true, button events, user interface connections to python methods... will be executed by debugger
    PyMethodDef* PythonAdditionalModuleITOM;

    // decides if itom is automatically included in every source file before it is handed to the syntax checker
    bool m_includeItom;

    struct AutoReload
    {
        PyObject *modAutoReload;
        PyObject *classAutoReload;
        bool enabled;
        bool checkFileExec;
        bool checkStringExec;
        bool checkFctExec;
    };

    AutoReload m_autoReload;

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
    static QString fctHashPrefix;

    static PythonEngine* instance;

    QAtomicInt m_interruptCounter; //protects that a python interrupt can only be placed if there is no interrupt event queued yet.

    // friend class
    friend class ito::PythonItom;

signals:
    void pythonDebugPositionChanged(QString filename, int lineNo);
    void pythonStateChanged(tPythonTransitions pyTransition);
    void pythonModifyLocalDict(PyObject* localDict, ItomSharedSemaphore* semaphore);
    void pythonModifyGlobalDict(PyObject* globalDict, ItomSharedSemaphore* semaphore);
    void pythonCurrentDirChanged();
    void updateCallStack(QStringList filenames, IntList lines, QStringList methods);
    void deleteCallStack();

    void pythonSetCursor(const Qt::CursorShape cursor);
    void pythonResetCursor();
    void pythonAutoReloadChanged(bool enabled, bool checkFile, bool checkCmd, bool checkFct);
    void clearCommandLine();

public slots:
    void pythonRunString(QString cmd);
    void pythonDebugString(QString cmd);
    void pythonExecStringFromCommandLine(QString cmd);
    void pythonRunFile(QString filename);
    void pythonDebugFile(QString filename);
    void pythonRunStringOrFunction(QString cmdOrFctHash);
    void pythonDebugStringOrFunction(QString cmdOrFctHash);
    void pythonInterruptExecution();
    void pythonDebugCommand(tPythonDbgCmd cmd);

    void setAutoReloader(bool enabled, bool checkFile, bool checkCmd, bool checkFct);

    // Settings are neccesary for automatic itom inclusion and syntax check
    void readSettings();
    void propertiesChanged();

    void pythonSyntaxCheck(const QString &code, QPointer<QObject> sender);

    void pythonGenericSlot(PyObject* callable, PyObject *argumentTuple);

    //!< these slots are only connected if python in debug-mode; while waiting these slots will be treated due to progressEvents-call in PythonEngine::PyDbgCommandLoop
    void breakPointAdded(BreakPointItem bp, int row);
    void breakPointDeleted(QString filename, int lineNo, int pyBpNumber);
    void breakPointChanged(BreakPointItem oldBp, BreakPointItem newBp);
    ito::RetVal setupBreakPointDebugConnections();
    ito::RetVal shutdownBreakPointDebugConnections();

    bool renameVariable(bool globalNotLocal, QString oldKey, QString newKey, ItomSharedSemaphore *semaphore = NULL);
    bool deleteVariable(bool globalNotLocal, QStringList keys, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal pickleVariables(bool globalNotLocal, QString filename, QStringList varNames, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal unpickleVariables(bool globalNotLocal, QString filename, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal saveMatlabVariables(bool globalNotLocal, QString filename, QStringList varNames, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal loadMatlabVariables(bool globalNotLocal, QString filename, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal registerAddInInstance(QString varname, ito::AddInBase *instance, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal getSysModules(QSharedPointer<QStringList> modNames, QSharedPointer<QStringList> modFilenames, QSharedPointer<IntList> modTypes, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal reloadSysModules(QSharedPointer<QStringList> modNames, ItomSharedSemaphore *semaphore = NULL);

    void registerWorkspaceContainer(PyWorkspaceContainer *container, bool registerNotUnregister, bool globalNotLocal);
    void workspaceGetChildNode(PyWorkspaceContainer *container, QString fullNameParentItem);
    void workspaceGetValueInformation(PyWorkspaceContainer *container, const QString &fullItemName, QSharedPointer<QString> extendedValue, ItomSharedSemaphore *semaphore = NULL);

    ito::RetVal putParamsToWorkspace(bool globalNotLocal, const QStringList &names, const QVector<SharedParamBasePointer > &values, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal getParamsFromWorkspace(bool globalNotLocal, const QStringList &names, QVector<int> paramBaseTypes, QSharedPointer<SharedParamBasePointerVector > values, ItomSharedSemaphore *semaphore = NULL);

private slots:

};

} //end namespace ito


#endif
