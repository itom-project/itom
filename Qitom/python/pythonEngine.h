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

#ifndef PYTHONENGINE_H
#define PYTHONENGINE_H

/*if you add any include to this file you will DIE an immediate, horrible, painful death*/

#include <string>
#include <exception>
//#ifndef Q_MOC_RUN
//    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API
//    #define NO_IMPORT_ARRAY
//#endif

#include "pythonJedi.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //see comment in pythonNpDataObject.cpp

#ifndef Q_MOC_RUN
    //python
    // see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
    #if (defined _DEBUG) && (defined WIN32)
        #undef _DEBUG
        #include "pythonWrapper.h"
        #include "node.h"
        #include "numpy/arrayobject.h"
        #define _DEBUG
    #else
        #include "pythonWrapper.h"
        #include "node.h"
        #include "numpy/arrayobject.h"
    #endif

#endif // Q_MOC_RUN

/* includes */

#include "pythonItom.h"
#include "pythonProxy.h"
#include "pythonStream.h"
#include "pythonCommon.h"
#include "pythonJediRunner.h"

#include "../models/breakPointModel.h"
#include "../../common/sharedStructuresQt.h"
#include "../../common/addInInterface.h"
#include "../../common/functionCancellationAndObserver.h"
#include "../codeEditor/codeCheckerItem.h"
#include "../global.h"

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

    Q_INVOKABLE void pythonSetup(ito::RetVal *retValue, QSharedPointer<QVariantMap> infoMessages);               //setup
    Q_INVOKABLE ito::RetVal scanAndRunAutostartFolder(QString currentDirAfterScan = QString() );
    Q_INVOKABLE ito::RetVal pythonShutdown(ItomSharedSemaphore *aimWait = NULL);            //shutdown
    Q_INVOKABLE ito::RetVal stringEncodingChanged();

    inline ito::BreakPointModel *getBreakPointModel() const { return bpModel; }
    inline bool isPythonBusy() const                { return m_pythonState != ito::pyStateIdle; }
    inline bool isPythonDebugging() const           { return (m_pythonState == ito::pyStateDebuggingWaitingButBusy || m_pythonState == ito::pyStateDebugging || m_pythonState == ito::pyStateDebuggingWaiting); }
    inline bool isPythonDebuggingAndWaiting() const { return m_pythonState == ito::pyStateDebuggingWaiting; }
    inline bool execInternalCodeByDebugger() const  { return m_executeInternalPythonCodeInDebugMode; }
    inline void setExecInternalCodeByDebugger(bool value) { m_executeInternalPythonCodeInDebugMode = value; }
    void printPythonErrorWithoutTraceback();
    void pythonDebugFunction(PyObject *callable, PyObject *argTuple, bool gilExternal = false);
    void pythonRunFunction(PyObject *callable, PyObject *argTuple, bool gilExternal = false);    
    inline PyObject *getGlobalDictionary()  const { return m_pGlobalDictionary;  }  /*!< returns reference to main dictionary (main workspace) */
    inline bool pySyntaxCheckAvailable() const { return (m_pyModCodeChecker != NULL); }
    bool tryToLoadJediIfNotYetDone(); //returns true, if Jedi is already loaded or could be loaded; else false

    //!< parses a (multiline) python string and returns the line numbers of the start of every major block.
    /* From Python 3.7 on, the AST is parsed to check for the major blocks. Then the encoding is always empty.
    Before, PyParser_SimpleParseString is used with the old Python compiler and an optional encoidng can be detected.

    This methods acquires the GIL, therefore it must directly called from a non-python thread.
    */
    QList<int> parseAndSplitCommandInMainComponents(const QString &str, QByteArray &encoding) const;

    QString getPythonExecutable() const { return m_pythonExecutable; }
    Qt::HANDLE getPythonThreadId() const { return m_pythonThreadId; }

    //!< thread-safe method (can be called from any thread) to enqueue a jedi completion request
    void enqueueJediCompletionRequest(const ito::JediCompletionRequest &request);

    //!< thread-safe method (can be called from any thread) to enqueue a jedi calltip request
    void enqueueJediCalltipRequest(const ito::JediCalltipRequest &request);

    //!< thread-safe method (can be called from any thread) to enqueue a jedi calltip request
    void enqueueGoToAssignmentRequest(const ito::JediAssignmentRequest &request);

    //!< thread-safe method (can be called from any thread) to enqueue a jedi get-help request
    void enqueueJediGetHelpRequest(const ito::JediGetHelpRequest &request);

    static bool isInterruptQueued();
    static const PythonEngine *getInstance();

    //!< add a new function cancellation / observer. Each valid observer on the list will be requested to be cancelled if a script executed is interrupted
    void addFunctionCancellationAndObserver(QWeakPointer<ito::FunctionCancellationAndObserver> observer);

    //!< will remove the given observer from the list of function cancellations and observers. Even if observer is NULL, the list of current observers will be cleanup from deleted instances
    void removeFunctionCancellationAndObserver(ito::FunctionCancellationAndObserver* observer = NULL); 

protected:
    ito::RetVal runPyFile(const QString &pythonFileName);         // run file pythonFileName
    ito::RetVal debugFile(const QString &pythonFileName);         // debug file pythonFileName
    ito::RetVal runString(const QString &command);          // run string command
    ito::RetVal debugString(const QString &command);        // debug string command
    ito::RetVal debugFunction(PyObject *callable, PyObject *argTuple, bool gilExternal = false);
    ito::RetVal runFunction(PyObject *callable, PyObject *argTuple, bool gilExternal = false);

    ito::RetVal modifyTracebackDepth(int NrOfLevelsToPopAtFront = -1, bool showTraceback = true);

    PyObject* setPyErrFromException(const std::exception &exc);

    void connectNotify(const QMetaMethod &signal);

    enum DebuggerErrorCode
    {
        DbgErrorNo = 0,
        DbgErrorInvalidBp = 1, // the breakpoint candidate could not be set and will be deleted
        DbgErrorOther = 2 // any other error
    };

private:
    enum DictUpdateFlag
    {
        DictUpdate,
        DictReset,
        DictNoAction
    };

    static PythonEngine *getInstanceInternal();

    /*!< returns reference to local dictionary (workspace of method, 
    which is handled right now). Is NULL if no method is executed right now. */
    inline PyObject *getLocalDictionary() { return m_pLocalDictionary; } 

    PyObject *getPyObjectByFullName(bool globalNotLocal, const QStringList &fullNameSplittedByDelimiter, QString *validVariableName = NULL); //Python GIL must be locked when calling this function!
    PyObject *getPyObjectByFullName(bool globalNotLocal, const QString &fullName, QString *validVariableName = NULL); //Python GIL must be locked when calling this function!

    void setGlobalDictionary(PyObject* mainDict = NULL);
    void setLocalDictionary(PyObject* localDict);

    void emitPythonDictionary(DictUpdateFlag globalDict, DictUpdateFlag localDict, bool lockGIL);

    ito::RetVal pickleDictionary(PyObject *dict, const QString &filename);
    ito::RetVal unpickleDictionary(PyObject *destinationDict, const QString &filename, bool overwrite);

    //!< runs the given Python string command
    void pythonRunString(QString cmd);

    //!< debugs the given Python string command
    void pythonDebugString(QString cmd);

    //methods for debugging
    void enqueueDbgCmd(ito::tPythonDbgCmd dbgCmd);
    ito::tPythonDbgCmd dequeueDbgCmd();
    bool DbgCommandsAvailable();
    void clearDbgCmdLoop();

    ito::RetVal pythonStateTransition(tPythonTransitions transition, bool immediate = true);

    //methods for breakpoint
    ito::RetVal pythonAddBreakpoint(const BreakPointItem &breakpoint, int &pyBpNumber);
    ito::RetVal pythonEditBreakpoint(const int pyBpNumber, const BreakPointItem &newBreakpoint);
    ito::RetVal pythonDeleteBreakpoint(const int pyBpNumber);
    void submitAllBreakpointsToDebugger();

    ito::RetVal autoReloaderCheck();

    static int queuedInterrupt(void *arg); 

    PyObject* getAndCheckIdentifier(const QString &identifier, ito::RetVal &retval) const;

	QVariantMap checkCodeCheckerRequirements();

	struct CodeCheckerOptions
	{
        PythonCommon::CodeCheckerMode mode;
		bool includeItomModuleBeforeCheck;
        PythonCommon::CodeCheckerMessageType minVisibleMessageTypeLevel; //!< minimum message class that should be shown in editor margin
        QByteArray furtherPropertiesJson; //!< these parameters are parsed from a QVariantMap to json and will be passed to itomSyntaxCheck.py
	};
    
    //member variables
    bool m_started;
	CodeCheckerOptions m_codeCheckerOptions;

    QMutex dbgCmdMutex;
    QMutex pythonStateChangeMutex;
    QDesktopWidget *m_pDesktopWidget;
    QQueue<ito::tPythonDbgCmd> debugCommandQueue;
    ito::tPythonDbgCmd debugCommand;
    
    ito::tPythonState m_pythonState;
    
    ito::BreakPointModel *bpModel;

    PyObject* mainModule;          //!< main module of python (builtin) [borrowed]
    PyObject* m_pMainDictionary;   //!< main dictionary of python [borrowed]
    PyObject* m_pLocalDictionary;  //!< local dictionary of python [borrowed], usually NULL unless if debugger is in "interaction-mode", then m_pGlobalDictionary is equal to the local dictionary of the current frame
    PyObject* m_pGlobalDictionary; //!< global dictionary of python [borrowed], equals to m_pMainDictionary unless if debugger is in "interaction-mode", then m_pGlobalDictionary is equal to the global dictionary of the current frame
    PyObject *itomDbgModule;       //!< debugger module
    PyObject *itomDbgInstance;     //!< debugger instance
    PyObject *itomModule;          //!< itom module [new ref]
    PyObject *itomFunctions;       //!< ito functions [additional python methods] [new ref]
    PyObject *m_pyModGC;
    PyObject *m_pyModCodeChecker;
	bool m_pyModCodeCheckerHasPyFlakes; //!< true if m_pyModCodeChecker could be loaded and pretends to have the syntax check feature (package: pyflakes)
	bool m_pyModCodeCheckerHasFlake8; //!< true if m_pyModCodeChecker could be loaded and pretends to have the syntax and style check feature (package: flake8)
    
    QSharedPointer<PythonJediRunner> m_jediRunner;

    Qt::HANDLE m_pythonThreadId;

    PyObject *dictUnicode;
    PyObject *slotsUnicode;

    QSet<ito::PyWorkspaceContainer*> m_mainWorkspaceContainer;
    QSet<ito::PyWorkspaceContainer*> m_localWorkspaceContainer;
    QHash<size_t, FuncWeakRef> m_pyFuncWeakRefHashes; //!< hash table containing weak reference to callable python methods or functions and as second, optional PyObject* an tuple, passed as argument to that function. These functions are for example executed by menu-clicks in the main window.
    size_t m_pyFuncWeakRefAutoInc;

    QString m_pythonExecutable; //!< absolute path to the python executable

    bool m_executeInternalPythonCodeInDebugMode; //!< if true, button events, user interface connections to python methods... will be executed by debugger
    PyMethodDef* PythonAdditionalModuleITOM;

    //!< decides if itom is automatically included in every source file before it is handed to the syntax checker
    bool m_includeItomImportBeforeCodeAnalysis;

    //!< string that is prepended to each script before syntax check (if m_includeItomImportBeforeCodeAnalysis is true)
    QString m_includeItomImportString; 

    wchar_t *m_pUserDefinedPythonHome;

    QList<QWeakPointer<ito::FunctionCancellationAndObserver> > m_activeFunctionCancellations;

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
    static QMutex instantiated;
    static QMutex instancePtrProtection;
    static QString fctHashPrefix;

    static PythonEngine* instance;

    QAtomicInt m_interruptCounter; //protects that a python interrupt can only be placed if there is no interrupt event queued yet.

    // friend class
    friend class ito::PythonItom;
	friend class ito::PyStream;

signals:
    void pythonDebugPositionChanged(QString filename, int lineNo);
    void pythonStateChanged(tPythonTransitions pyTransition, bool immediate);
    void pythonModifyLocalDict(PyObject* localDict, ItomSharedSemaphore* semaphore);
    void pythonModifyGlobalDict(PyObject* globalDict, ItomSharedSemaphore* semaphore);
    void pythonCurrentDirChanged();
    void updateCallStack(QStringList filenames, IntList lines, QStringList methods);
    void deleteCallStack();

    void pythonSetCursor(const Qt::CursorShape cursor);
    void pythonResetCursor();
    void pythonAutoReloadChanged(bool enabled, bool checkFile, bool checkCmd, bool checkFct);
    void clearCommandLine();
    void startInputCommandLine(QSharedPointer<QByteArray> buffer, ItomSharedSemaphore *semaphore);

public slots:
    void pythonExecStringFromCommandLine(QString cmd);
    void pythonRunFile(QString filename);
    void pythonDebugFile(QString filename);
    void pythonRunStringOrFunction(QString cmdOrFctHash);
    void pythonDebugStringOrFunction(QString cmdOrFctHash);
    void pythonInterruptExecutionThreadSafe(bool *interruptActuatorsAndTimers = NULL); //if interruptActuatorsAndTimers is NULL, the itom settings will be read, else this boolean variable decides
    void pythonDebugCommand(tPythonDbgCmd cmd);

    void setAutoReloader(bool enabled, bool checkFile, bool checkCmd, bool checkFct);

    // Settings are neccesary for automatic itom inclusion and syntax check
    void readSettings();
    void propertiesChanged();

    void pythonCodeCheck(const QString &code, const QString &filename, bool fileSaved, QPointer<QObject> sender, QByteArray callbackFctName);

    //!< these slots are only connected if python in debug-mode; while waiting these slots will be treated due to progressEvents-call in PythonEngine::PyDbgCommandLoop
    void breakPointAdded(BreakPointItem bp, int row);
    void breakPointDeleted(QString filename, int lineNo, int pyBpNumber);
    void breakPointChanged(BreakPointItem oldBp, BreakPointItem newBp);
    ito::RetVal setupBreakPointDebugConnections();
    ito::RetVal shutdownBreakPointDebugConnections();

    bool renameVariable(bool globalNotLocal, const QString &oldFullItemName, QString newKey, ItomSharedSemaphore *semaphore = NULL);
    bool deleteVariable(bool globalNotLocal, const QStringList &fullItemNames, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal pickleVariables(bool globalNotLocal, QString filename, QStringList varNames, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal pickleSingleParam(QString filename, QSharedPointer<ito::Param> value, const QString &valueName, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal unpickleVariables(bool globalNotLocal, QString filename, QString packedVarName, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal saveMatlabVariables(bool globalNotLocal, QString filename, QStringList varNames, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal saveMatlabSingleParam(QString filename, QSharedPointer<ito::Param> value, const QString &valueName, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal loadMatlabVariables(bool globalNotLocal, QString filename, QString packedVarName, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal registerAddInInstance(QString varname, ito::AddInBase *instance, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal getSysModules(QSharedPointer<QStringList> modNames, QSharedPointer<QStringList> modFilenames, QSharedPointer<IntList> modTypes, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal reloadSysModules(QSharedPointer<QStringList> modNames, ItomSharedSemaphore *semaphore = NULL);

    void registerWorkspaceContainer(PyWorkspaceContainer *container, bool registerNotUnregister, bool globalNotLocal);
    void workspaceGetChildNode(PyWorkspaceContainer *container, QString fullNameParentItem);
    void workspaceGetValueInformation(PyWorkspaceContainer *container, const QString &fullItemName, QSharedPointer<QString> extendedValue, ItomSharedSemaphore *semaphore = NULL);

    ito::RetVal checkVarnamesInWorkspace(bool globalNotLocal, const QStringList &names, QSharedPointer<IntList> existing, ItomSharedSemaphore *semaphore = NULL); /*!< check if variable already exist in workspace, existing is 0 (non existing), 1 (existing, but can be overwritten), 2 (existing, not overwritable, e.g. function, method...)*/
    ito::RetVal putParamsToWorkspace(bool globalNotLocal, const QStringList &names, const QVector<SharedParamBasePointer > &values, ItomSharedSemaphore *semaphore = NULL);
    ito::RetVal getVarnamesListInWorkspace(bool globalNotLocal, const QString &find, QSharedPointer<QStringList> varnameList, ItomSharedSemaphore *semaphore /*= NULL*/);
    ito::RetVal getParamsFromWorkspace(bool globalNotLocal, const QStringList &names, QVector<int> paramBaseTypes, QSharedPointer<SharedParamBasePointerVector > values, ItomSharedSemaphore *semaphore = NULL);

    ito::RetVal pythonGetClearAllValues();
    ito::RetVal pythonClearAll();

};

} //end namespace ito


#endif
