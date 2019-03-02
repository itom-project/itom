/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#include "../python/pythonEngineInc.h"

#include "mainWindow.h"

#include "../AppManagement.h"
#include "../global.h"

#include "../../AddInManager/addInManager.h"
#include "../organizer/processOrganizer.h"
#include "../organizer/uiOrganizer.h"
#include "../organizer/userOrganizer.h"

#include "../ui/dialogProperties.h"
#include "../ui/dialogAbout.h"
#include "../ui/dialogReloadModule.h"
#include "../ui/dialogLoadedPlugins.h"
#include "../ui/widgetInfoBox.h"
#include "../ui/dialogPipManager.h"
#include "../ui/dialogTimerManager.h"

#include "../helper/versionHelper.h"

#include <qapplication.h>
#include <qstatusbar.h>
#include <qmessagebox.h>
#include <qfiledialog.h>
#include <qdesktopwidget.h>
#include <qmessagebox.h>
#include <qdir.h>
#include "../organizer/scriptEditorOrganizer.h"

#ifdef ITOM_USEHELPVIEWER
#include "../helpViewer/helpViewer.h"
#endif

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
//! constructor
/*!
    establishes widgets being part of the main window including necessary actions
*/
MainWindow::MainWindow() :
	m_console(NULL),
	m_contentLayout(NULL),
	m_breakPointDock(NULL),
	m_lastCommandDock(NULL),
	//    m_pythonMessageDock(NULL),
	m_helpDock(NULL),
	m_globalWorkspaceDock(NULL),
	m_localWorkspaceDock(NULL),
	m_callStackDock(NULL),
	m_fileSystemDock(NULL),
	m_pAIManagerWidget(NULL),
	m_aboutToolBar(NULL),
	m_appToolBar(NULL),
	m_toolToolBar(NULL),
	m_pythonToolBar(NULL),
	m_userDefinedSignalMapper(NULL),
	m_appFileNew(NULL),
	m_appFileOpen(NULL),
	m_aboutQt(NULL),
	m_aboutQitom(NULL),
    m_pMenuFigure(NULL),
	m_pMenuHelp(NULL),
	m_pMenuFile(NULL),
	m_pMenuPython(NULL),
	m_pMenuReloadModule(NULL),
	m_pMenuView(NULL),
	m_pHelpSystem(NULL),
	m_pStatusLblCurrentDir(NULL),
	m_pStatusLblPythonBusy(NULL),
	m_pythonBusy(false),
	m_pythonDebugMode(false),
	m_pythonInWaitingMode(false),
	m_isFullscreen(false),
	m_userDefinedActionCounter(0),
	m_lastFilesMapper(NULL),
	m_openScriptsMapper(NULL),
    m_openFigureMapper(NULL)
{
    //qDebug() << "mainWindow. Thread: " << QThread::currentThreadId ();
#ifdef __APPLE__
    // Setting high res icon for OS X
    QApplication::setWindowIcon(QIcon(":/application/icons/itomicon/itomIcon1024"));
#else
    QApplication::setWindowIcon(QIcon(":/application/icons/itomicon/itomIcon32"));
#endif

    qDebug("build main window");
    const PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    // general windows settings
    if (sizeof(void*) > 4) //was before a check using QT_POINTER_SIZE
    {
        setWindowTitle(tr("itom (x64)"));
    }
    else
    {
        setWindowTitle(tr("itom"));
    }

    setUnifiedTitleAndToolBarOnMac(true);

    setCorner(Qt::TopLeftCorner, Qt::LeftDockWidgetArea);
    setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
    setCorner(Qt::TopRightCorner, Qt::RightDockWidgetArea);
    setCorner(Qt::BottomRightCorner, Qt::RightDockWidgetArea);

    //content
    m_contentLayout = new QVBoxLayout;

    // user
    ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();
    if (uOrg && (uOrg->hasFeature(featConsoleRead) || uOrg->hasFeature(featConsoleReadWrite)))
    {
        qDebug(".. before loading console widget");
        //console (central widget):
        m_console = new ConsoleWidget(this);
        m_console->setObjectName("console"); //if a drop event onto a scripteditor comes from this object name, the drop event is always executed as copy event such that no text is deleted in the console.
        //setCentralWidget(m_console);
        qDebug(".. console widget loaded");
        m_contentLayout->addWidget(m_console);
    }

    m_contentLayout->setContentsMargins(0, 0, 0, 0);
    m_contentLayout->setSpacing(1);

    QWidget *centralWidget = new QWidget(this);
    centralWidget->setLayout(m_contentLayout);
    setCentralWidget(centralWidget); 

    if (uOrg && uOrg->hasFeature(featFileSystem))
    {
        // FileDir-Dock
        m_fileSystemDock = new FileSystemDockWidget(tr("File System"), "itomFileSystemDockWidget", this, true, true, AbstractDockWidget::floatingStandard);
        m_fileSystemDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
        connect(m_fileSystemDock, SIGNAL(currentDirChanged()), this, SLOT(currentDirectoryChanged()));
        addDockWidget(Qt::LeftDockWidgetArea, m_fileSystemDock);
    }

    if (uOrg && uOrg->hasFeature(featDeveloper))
    {
        // breakPointDock
        m_breakPointDock = new BreakPointDockWidget(tr("Breakpoints"), "itomBreakPointDockWidget", this, true, true, AbstractDockWidget::floatingStandard);
        m_breakPointDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
        addDockWidget(Qt::LeftDockWidgetArea, m_breakPointDock);

        // lastCommandDock
        m_lastCommandDock = new LastCommandDockWidget(tr("Command History"), "itomLastCommandDockWidget", this, true, true, AbstractDockWidget::floatingStandard);
        m_lastCommandDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
        addDockWidget(Qt::LeftDockWidgetArea, m_lastCommandDock);
        
        // pythonMessageDock
        //m_pythonMessageDock = new PythonMessageDockWidget(tr("Python Messages"), "itomPythonMessageDockWidget", this, true, true, AbstractDockWidget::floatingStandard);
        //m_pythonMessageDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
        //addDockWidget(Qt::BottomDockWidgetArea, m_pythonMessageDock);

        // helpDock
        m_helpDock = new HelpDockWidget(tr("Help Viewer"), "itomHelpDockWidget", this, true, true, AbstractDockWidget::floatingWindow);
        m_helpDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
        addDockWidget(Qt::LeftDockWidgetArea, m_helpDock);

        // CallStack-Dock
        m_callStackDock = new CallStackDockWidget(tr("Call Stack"), "itomCallStackDockWidget", this, true, true, AbstractDockWidget::floatingStandard);
        m_callStackDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
        addDockWidget(Qt::LeftDockWidgetArea, m_callStackDock);

        if (m_fileSystemDock)
        {
            // tabify file-directory and breakpoints
            tabifyDockWidget(m_callStackDock, m_fileSystemDock);
            tabifyDockWidget(m_fileSystemDock, m_breakPointDock);
            m_fileSystemDock->raise();
        }
        else
        {
            tabifyDockWidget(m_callStackDock, m_breakPointDock);
        }

        // global workspace widget (Python)
        m_globalWorkspaceDock = new WorkspaceDockWidget(tr("Global Variables"), "itomGlobalWorkspaceDockWidget", true, this, true, true, AbstractDockWidget::floatingStandard);
        m_globalWorkspaceDock->setAllowedAreas(Qt::AllDockWidgetAreas);
        addDockWidget(Qt::RightDockWidgetArea, m_globalWorkspaceDock);
        connect(m_globalWorkspaceDock, SIGNAL(setStatusInformation(QString,int)), this, SLOT(setStatusText(QString, int)));

        // local workspace widget (Python)
        m_localWorkspaceDock = new WorkspaceDockWidget(tr("Local Variables"), "itomLocalWorkspaceDockWidget", false, this, true, true, AbstractDockWidget::floatingStandard);
        m_localWorkspaceDock->setAllowedAreas(Qt::AllDockWidgetAreas);
        addDockWidget(Qt::RightDockWidgetArea, m_localWorkspaceDock);
        connect(m_localWorkspaceDock, SIGNAL(setStatusInformation(QString, int)), this, SLOT(setStatusText(QString, int)));

        // tabify global and local workspace
        tabifyDockWidget(m_globalWorkspaceDock, m_localWorkspaceDock);
        //splitDockWidget(m_globalWorkspaceDock, m_localWorkspaceDock, Qt::Horizontal);
        m_globalWorkspaceDock->raise();
    }

    if (uOrg && uOrg->hasFeature(featPlugins))
    {
        // AddIn-Manager
        m_pAIManagerWidget = new AIManagerWidget(tr("Plugins"), "itomPluginsDockWidget", this, true, true, AbstractDockWidget::floatingStandard, AbstractDockWidget::movingEnabled);
        qDebug(".. plugin manager widget loaded");

        addDockWidget(Qt::RightDockWidgetArea, m_pAIManagerWidget);

        if (m_helpDock)
        {
            tabifyDockWidget(m_pAIManagerWidget, m_helpDock);
            m_pAIManagerWidget->raise();
        }
    }

    if (m_pAIManagerWidget != NULL && m_helpDock != NULL)
    {
        connect(m_pAIManagerWidget, SIGNAL(showPluginInfo(QString, int)), m_helpDock, SLOT(mnuShowInfo(QString, int)));
        connect(m_pAIManagerWidget, SIGNAL(showDockWidget()), this, SLOT(mnuShowScriptReference()));
    }

    // connections
    if (pyEngine != NULL)
    {
        connect(pyEngine, SIGNAL(pythonStateChanged(tPythonTransitions)), this, SLOT(pythonStateChanged(tPythonTransitions)));

        connect(pyEngine, SIGNAL(pythonCurrentDirChanged()), this, SLOT(currentDirectoryChanged()));
        connect(this, SIGNAL(pythonDebugCommand(tPythonDbgCmd)), pyEngine, SLOT(pythonDebugCommand(tPythonDbgCmd)));

        connect(pyEngine, SIGNAL(pythonSetCursor(Qt::CursorShape)), this, SLOT(setCursor(Qt::CursorShape)));
        connect(pyEngine, SIGNAL(pythonResetCursor()), this, SLOT(resetCursor()));

        if (m_console)
        {
            connect(pyEngine, SIGNAL(clearCommandLine()), m_console, SLOT(clearCommandLine()));
            connect(pyEngine, SIGNAL(startInputCommandLine(QSharedPointer<QByteArray>, ItomSharedSemaphore*)), m_console, SLOT(startInputCommandLine(QSharedPointer<QByteArray>, ItomSharedSemaphore*)));
        }
    }
    else
    {
        showInfoMessageLine(tr("Python could not be started. itom cannot be used in the desired way. \nStart itom again with the argument 'log' and look-up the error message in the file itomlog.txt."));
        if (m_console)
            m_console->setReadOnly(true);
    }

    // signal mapper for user defined actions
    m_userDefinedSignalMapper = new QSignalMapper(this);
    connect(m_userDefinedSignalMapper, SIGNAL(mapped(const QString &)), this, SLOT(userDefinedActionTriggered(const QString &)));

    connect(m_lastCommandDock, SIGNAL(runPythonCommand(QString)), m_console, SLOT(pythonRunSelection(QString)));
    connect(m_console, SIGNAL(sendToLastCommand(QString)), m_lastCommandDock, SLOT(addLastCommand(QString)));
//    connect(m_console, SIGNAL(sendToPythonMessage(QString)), m_pythonMessageDock, SLOT(addPythonMessage(QString)));

    // Signalmapper for dynamic lastFile Menu
    m_lastFilesMapper = new QSignalMapper(this);
    connect(m_lastFilesMapper, SIGNAL(mapped(const QString &)), this, SLOT(lastFileOpen(const QString &)));

    m_openScriptsMapper = new QSignalMapper(this);
    connect(m_openScriptsMapper, SIGNAL(mapped(const QString &)), this, SLOT(openScript(const QString &)));

    m_openFigureMapper = new QSignalMapper(this);
    connect(m_openFigureMapper, SIGNAL(mapped(int)), this, SLOT(raiseFigureByHandle(int)));

    //
    createActions();
    createMenus();
    createToolBars();
    createStatusBar();
    updatePythonActions();

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);

    if (pyEngine)
    {
        connect(pyEngine, SIGNAL(pythonAutoReloadChanged(bool,bool,bool,bool)), this, SLOT(pythonAutoReloadChanged(bool,bool,bool,bool)));
        connect(this, SIGNAL(pythonSetAutoReloadSettings(bool,bool,bool,bool)), pyEngine, SLOT(setAutoReloader(bool,bool,bool,bool)));

        settings.beginGroup("Python");

        bool pyReloadEnabled = settings.value("pyReloadEnabled", false).toBool();
        bool pyReloadCheckFile = settings.value("pyReloadCheckFile", true).toBool();
        bool pyReloadCheckCmd = settings.value("pyReloadCheckCmd", true).toBool();
        bool pyReloadCheckFct = settings.value("pyReloadCheckFct", false).toBool();

        emit pythonSetAutoReloadSettings(pyReloadEnabled, pyReloadCheckFile, pyReloadCheckCmd, pyReloadCheckFct);

        settings.endGroup();
    }

    
    settings.beginGroup("MainWindow");

    bool maximized = settings.value("maximized", false).toBool();
    QDesktopWidget desktop;
    QRect mainScreen = desktop.screenGeometry(desktop.primaryScreen());
    mainScreen.adjust(mainScreen.width()/6, mainScreen.height()/6, -mainScreen.width()/6, -mainScreen.height()/6);
    QRect geometry = settings.value("geometry", mainScreen).toRect();

    if (geometry != mainScreen) //check if valid
    {
        //check whether top/left and bottom/right lie in any available desktop
        QRect r1, r2;
        r1 = desktop.availableGeometry(geometry.topLeft());
        r2 = desktop.availableGeometry(geometry.bottomRight());
        if (r1.isValid() == false || r2.isValid() == false  || r1.contains(geometry.topLeft()) == false || r2.contains(geometry.bottomRight()) == false)
        {
            //reset to default
            geometry = mainScreen;
            maximized = false;
        }
    }

    restoreState(settings.value("state", "").toByteArray());

    settings.endGroup();

    //if restore state set some dock widgets inherited from abstractDockWidget to a top level state, it must be converted to a windows style using the following method:
    if (m_fileSystemDock)
    {
        m_fileSystemDock->restoreState("itomFileSystemDockWidget");
    }

    if (m_helpDock) 
    {
        m_helpDock->synchronizeTopLevelState();
        m_helpDock->restoreState("itomHelpDockWidget");
    }

    setGeometry(geometry);
    m_geometryNormalState = geometry; //geometry in normal state

    if (maximized)
    {
        showMaximized();
    }

    qDebug(".. main window build done");
}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!
    disconnects connections between main window and python engine
*/
MainWindow::~MainWindow()
{
    if (m_fileSystemDock)
    {
        m_fileSystemDock->saveState("itomFileSystemDockWidget");
    }
    if (m_helpDock)
    {
        m_helpDock->saveState("itomHelpDockWidget");
    }
    if (m_globalWorkspaceDock)
    {
        m_globalWorkspaceDock->saveState("itomGlobalWorkspaceDockWidget");
    }
    if (m_localWorkspaceDock)
    {
        m_localWorkspaceDock->saveState("itomLocalWorkspaceDockWidget");
    }
    if (m_pAIManagerWidget)
    {
        m_pAIManagerWidget->saveState("itomPluginsDockWidget");
    }

    QSettings *settings = new QSettings(AppManagement::getSettingsFile(), QSettings::IniFormat);

    if (m_actions["py_autoReloadEnabled"])
    {
        settings->beginGroup("Python");
        settings->setValue("pyReloadEnabled", m_actions["py_autoReloadEnabled"]->isChecked());
        settings->setValue("pyReloadCheckFile", m_actions["py_autoReloadFile"]->isChecked());
        settings->setValue("pyReloadCheckCmd", m_actions["py_autoReloadCmd"]->isChecked());
        settings->setValue("pyReloadCheckFct", m_actions["py_autoReloadFunc"]->isChecked());
        settings->endGroup();
    }

    settings->beginGroup("MainWindow");
    settings->setValue("maximized", isMaximized());
    settings->setValue("geometry", m_geometryNormalState);
    //settings->setValue("geometry", saveGeometry());
    
    QByteArray state = saveState();
    settings->setValue("state", state);
    settings->endGroup();

    delete settings;

    //QByteArray ba = storeDockWidgetStatus();

    const PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (pyEngine != NULL)
    {
        disconnect(pyEngine, SIGNAL(pythonStateChanged(tPythonTransitions)), this, SLOT(pythonStateChanged(tPythonTransitions)));
        disconnect(this, SIGNAL(pythonDebugCommand(tPythonDbgCmd)), pyEngine, SLOT(pythonDebugCommand(tPythonDbgCmd)));
    }

    if (m_globalWorkspaceDock)
    {
        disconnect(m_globalWorkspaceDock, SIGNAL(setStatusInformation(QString,int)), this, SLOT(setStatusText(QString, int)));
    }
    if (m_localWorkspaceDock)
    {
        disconnect(m_localWorkspaceDock, SIGNAL(setStatusInformation(QString,int)), this, SLOT(setStatusText(QString, int)));
    }

    if (m_lastCommandDock && m_console)
    {
        disconnect(m_lastCommandDock, SIGNAL(runPythonCommand(QString)), m_console, SLOT(pythonRunSelection(QString)));
        disconnect(m_console, SIGNAL(sendToLastCommand(QString)), m_lastCommandDock, SLOT(addLastCommand(QString)));
    }

/*    if (m_pythonMessageDock && m_console)
    {
        disconnect(m_console, SIGNAL(sendToPythonMessage(QString)), m_pythonMessageDock, SLOT(addPythonMessage(QString)));
    }*/

    DELETE_AND_SET_NULL(m_pAIManagerWidget);
    DELETE_AND_SET_NULL(m_fileSystemDock);
    DELETE_AND_SET_NULL(m_helpDock);
    DELETE_AND_SET_NULL(m_globalWorkspaceDock);
    DELETE_AND_SET_NULL(m_localWorkspaceDock);
    DELETE_AND_SET_NULL(m_localWorkspaceDock);

#ifdef ITOM_USEHELPVIEWER
		HelpViewer *hv = m_helpViewer.data();
		DELETE_AND_SET_NULL(hv);
#endif

    //delete remaining user-defined toolbars and actions
    QMap<QString, QToolBar*>::iterator it = m_userDefinedToolBars.begin();
    while (it != m_userDefinedToolBars.end())
    {
        removeToolBar(*it);
        delete *it;
        ++it;
    }
    m_userDefinedToolBars.clear();

    //delete remaining user-defined menu elements
    QMap<QString, QMenu* >::iterator it2 = m_userDefinedRootMenus.begin();
    while (it2 != m_userDefinedRootMenus.end())
    {
        (*it2)->deleteLater();
        ++it2;
    }
    m_userDefinedRootMenus.clear();

    DELETE_AND_SET_NULL(m_userDefinedSignalMapper);

    if (m_pHelpSystem)
    {
        //delete m_pHelpSystem;
        m_pHelpSystem = NULL;
    }

    QMapIterator<QString, QPointer<WidgetInfoBox> > i(m_infoBoxWidgets);
    while (i.hasNext()) 
    {
        i.next();
        if (i.value().isNull() == false)
        {
            i.value().data()->deleteLater();
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by ScriptEditorOrganizer, if any ScriptDockWidget should be added to main window's dock widgets
/*!
    This method is also called to dock any figure to the main window.

    \param dockWidget ScriptDockWidget to add to any docking area
    \param area docking area, where dockWidget should be shown
*/
void MainWindow::addAbstractDock(AbstractDockWidget* dockWidget, Qt::DockWidgetArea area /*= Qt::TopDockWidgetArea*/)
{
    if (dockWidget)
    {
        bool hadParent = dockWidget->parent();
        dockWidget->setParent(this);

        if (!hadParent)
        {
            if (dockWidget->docked())
            {
                dockWidget->dockWidget();
            }
            else
            {
                dockWidget->undockWidget();
            }
        }

        if (area == Qt::NoDockWidgetArea)
        {
            addDockWidget(Qt::TopDockWidgetArea , dockWidget);
            dockWidget->setFloating(true);
        }
        else
        {
            addDockWidget(area, dockWidget);
            dockWidget->setFloating(false);
            //qDebug() << "restoreDockWidget:" << restoreDockWidget(dockWidget); //does not work until now, since the state of docked script windows is not saved. they are deleted before destructing the main window.
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by ScriptEditorOrganizer, if any ScriptDockWidget should be removed from docking area
/*!
    notice, that even a ScriptDockWidget is actually undocked, it belongs to the docking area NoDockWidgetArea

    \param dockWidget ScriptDockWidget to remove from docking area
*/
void MainWindow::removeAbstractDock(AbstractDockWidget* dockWidget)
{
    if (dockWidget)
    {
        dockWidget->setParent(NULL);
        removeDockWidget(dockWidget);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::connectPythonMessageBox(QListWidget* pythonMessageBox)
{
    connect(m_console, SIGNAL(sendToPythonMessage(QString)), pythonMessageBox, SLOT(addNewMessage(QString)));
}

//----------------------------------------------------------------------------------------------------------------------------------
//! close event invoked if main window should be closed (and therefore the whole application too)
/*!
    if this event is invoked the signal mainWindowCloseRequest is emitted, which invokes the slot mainWindowCloseRequest
    in class MainApplication in order to proceed the entire closing process. Therefore the event is ignored.

    \param event event of type QCloseEvent, describing the close request
    \sa MainApplication
*/
void MainWindow::closeEvent(QCloseEvent *event)
{
    emit(mainWindowCloseRequest());
    event->ignore(); //!< if mainWindowCloseRequest is handled and accepted by mainApplication, MainWindow will be destroyed
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::resizeEvent(QResizeEvent * event)
{
    if (!isMaximized() && !isMinimized())
    {
        m_geometryNormalState.setSize(event->size());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::moveEvent(QMoveEvent * event)
{
    if (!isMaximized() && !isMinimized())
    {
        m_geometryNormalState.moveTo(event->pos());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! creates actions for menu and toolbar
void MainWindow::createActions()
{
    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    QAction *a = NULL;

    //app actions
    ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();
    if (uOrg->hasFeature(featDeveloper))
    {
        m_appFileNew = new QAction(QIcon(":/files/icons/new.png"), tr("New Script..."), this);
        connect(m_appFileNew, SIGNAL(triggered()), this, SLOT(mnuNewScript()));
        m_appFileNew->setShortcut(QKeySequence::New);
    }

    m_appFileOpen = new QAction(QIcon(":/files/icons/open.png"), tr("Open File..."), this);
    connect(m_appFileOpen, SIGNAL(triggered()), this, SLOT(mnuOpenFile()));
    m_appFileOpen->setShortcut(QKeySequence::Open);

    m_actions["exit"] = new QAction(tr("Exit"), this);
    connect(m_actions["exit"], SIGNAL(triggered()), this, SLOT(mnuExitApplication()));

    if (uOrg->hasFeature(featProperties))
    {
        m_actions["properties"] = new QAction(QIcon(":/application/icons/adBlockAction.png"), tr("Properties..."), this);
        connect(m_actions["properties"] , SIGNAL(triggered()), this, SLOT(mnuShowProperties()));
    }

    if (uOrg->hasFeature(featUserManag))
    {
        m_actions["usermanagement"] = new QAction(QIcon(":/misc/icons/User.png"), tr("User Management..."), this);
        connect(m_actions["usermanagement"] , SIGNAL(triggered()), this, SLOT(mnuShowUserManagement()));
    }

    m_aboutQt = new QAction(QIcon(":/application/icons/helpAboutQt.png"), tr("About Qt..."), this);
    connect(m_aboutQt, SIGNAL(triggered()), qApp, SLOT(aboutQt()));
    //m_aboutQt->setShortcut(QKeySequence("F3"));

    m_aboutQitom = new QAction(QIcon(":/application/icons/itomicon/itomLogo3_64.png"), tr("About itom..."), this);
    connect(m_aboutQitom, SIGNAL(triggered()), this, SLOT(mnuAboutQitom()));

    m_actions["show_loaded_plugins"] = new QAction(QIcon(":/plugins/icons/plugin.png"), tr("Loaded Plugins..."), this);
    connect(m_actions["show_loaded_plugins"], SIGNAL(triggered()), this, SLOT(mnuShowLoadedPlugins()));

    if (uOrg->hasFeature(featDeveloper))
    {
        a = m_actions["open_assistant"] = new QAction(QIcon(":/application/icons/help.png"), tr("Help..."), this);
        a->setShortcut(QKeySequence::HelpContents);
        connect(a , SIGNAL(triggered()), this, SLOT(mnuShowAssistant()));

        a = m_actions["script_reference"] = new QAction(QIcon(":/application/icons/scriptReference.png"), tr("Script Reference"), this);
        connect(a , SIGNAL(triggered()), this, SLOT(mnuShowScriptReference()));

        a = m_actions["open_designer"] = new QAction(QIcon(":/application/icons/designer4.png"), tr("UI Designer"), this);
        connect(a, SIGNAL(triggered()), this, SLOT(mnuShowDesigner()));

        a = m_actions["python_global_runmode"] = new QAction(QIcon(":/application/icons/pythonDebug.png"), tr("Run Python Code In Debug Mode"), this);
        a->setToolTip(tr("Set whether internal python code should be executed in debug mode"));
        a->setCheckable(true);
        if (pyEngine)
        {
            a->setChecked(pyEngine->execInternalCodeByDebugger());
        }
        connect(m_actions["python_global_runmode"], SIGNAL(triggered(bool)), this, SLOT(mnuToggleExecPyCodeByDebugger(bool)));

        a = m_actions["close_all_plots"] = new QAction(QIcon(":/application/icons/closePlots.png"), tr("Close All Floatable Figures"), this);
        connect(m_actions["close_all_plots"], SIGNAL(triggered(bool)), this, SLOT(mnuCloseAllPlots()));

        a = m_actions["show_all_plots"] = new QAction(QIcon(":/application/icons/showAllPlots.png"), tr("Show All Floatable Figures"), this);
        connect(m_actions["show_all_plots"], SIGNAL(triggered(bool)), this, SLOT(mnuShowAllPlots()));
        
        a = m_actions["minimize_all_plots"] = new QAction(QIcon(":/application/icons/hideAllPlots"), tr("Minimize All Floatable Figures"), this);
        connect(m_actions["minimize_all_plots"], SIGNAL(triggered(bool)), this, SLOT(mnuMinimizeAllPlots()));

        a = m_actions["python_stopAction"] = new QAction(QIcon(":/script/icons/stopScript.png"), tr("Stop"), this);
        a->setShortcut(tr("Shift+F5"));
        a->setShortcutContext(Qt::WidgetWithChildrenShortcut);
        connect(a, SIGNAL(triggered()), this, SLOT(mnuScriptStop()));

        a = m_actions["python_continueAction"] = new QAction(QIcon(":/script/icons/continue.png"), tr("Continue"), this);
        a->setShortcut(tr("F6"));
        a->setShortcutContext(Qt::WidgetWithChildrenShortcut);
        connect(a, SIGNAL(triggered()), this, SLOT(mnuScriptContinue()));

        m_actions["python_stepAction"] = new QAction(QIcon(":/script/icons/step.png"), tr("Step"), this);
        m_actions["python_stepAction"]->setShortcut(tr("F11"));
        m_actions["python_stepAction"]->setShortcutContext(Qt::WidgetWithChildrenShortcut);
        connect(m_actions["python_stepAction"], SIGNAL(triggered()), this, SLOT(mnuScriptStep()));

        a = m_actions["python_stepOverAction"] = new QAction(QIcon(":/script/icons/stepOver.png"), tr("Step Over"), this);
        a->setShortcut(tr("F10"));
        a->setShortcutContext(Qt::WidgetWithChildrenShortcut);
        connect(a, SIGNAL(triggered()), this, SLOT(mnuScriptStepOver()));

        a = m_actions["python_stepOutAction"] = new QAction(QIcon(":/script/icons/stepOut.png"), tr("Step Out"), this);
        a->setShortcut(tr("Shift+F11"));
        a->setShortcutContext(Qt::WidgetWithChildrenShortcut);
        connect(a, SIGNAL(triggered()), this, SLOT(mnuScriptStepOut()));

        a = m_actions["python_reloadModules"] = new QAction(QIcon(":/application/icons/reload.png"), tr("Reload Modules..."), this);
        connect(a, SIGNAL(triggered()), this, SLOT(mnuPyReloadModules()));

        a = m_actions["py_autoReloadEnabled"] = new QAction(tr("Autoreload Modules"), this);
        a->setCheckable(true);
        connect(a, SIGNAL(triggered(bool)), this, SLOT(mnuPyAutoReloadTriggered(bool)));

        a = m_actions["py_autoReloadFile"] = new QAction(tr("Autoreload Before Script Execution"), this);
        a->setCheckable(true);
        connect(a, SIGNAL(triggered(bool)), this, SLOT(mnuPyAutoReloadTriggered(bool)));

        a = m_actions["py_autoReloadCmd"] = new QAction(tr("Autoreload Before Single Command"), this);
        a->setCheckable(true);
        connect(a, SIGNAL(triggered(bool)), this, SLOT(mnuPyAutoReloadTriggered(bool)));

        a = m_actions["py_autoReloadFunc"] = new QAction(tr("Autoreload Before Events And Function Calls"), this);
        a->setCheckable(true);
        connect(a, SIGNAL(triggered(bool)), this, SLOT(mnuPyAutoReloadTriggered(bool)));

        a = m_actions["py_packageManager"] = new QAction(tr("Package Manager..."), this);
        connect(a, SIGNAL(triggered()), this, SLOT(mnuPyPipManager()));

		a = m_actions["python_timerManager"] = new QAction(tr("Timer Manager..."), this);
		connect(a, SIGNAL(triggered()), this, SLOT(mnuPyTimerManager()));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! creates toolbar
void MainWindow::createToolBars()
{
    m_appToolBar = addToolBar(tr("Application"));
    m_appToolBar->setObjectName("toolbarApplication");
    ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();
    if (uOrg->hasFeature(featDeveloper))
    {
        m_appToolBar->addAction(m_appFileNew);
    }
    m_appToolBar->addAction(m_appFileOpen);
    m_appToolBar->setFloatable(false);

    m_toolToolBar = addToolBar(tr("Tools"));
    m_toolToolBar->setObjectName("toolbarTools");
    if (uOrg->hasFeature(featDeveloper))
    {
        m_toolToolBar->addAction(m_actions["open_designer"]);
    }
    m_toolToolBar->setFloatable(false);

    m_aboutToolBar = addToolBar(tr("About"));
    m_aboutToolBar->setObjectName("toolbarAbout");
    m_aboutToolBar->setFloatable(false);
    m_aboutToolBar->addAction(m_actions["open_assistant"]);

    if (uOrg->hasFeature(featDeveloper))
    {
        m_pythonToolBar = addToolBar(tr("Python"));
        m_pythonToolBar->setObjectName("toolbarPython");
        m_pythonToolBar->addAction(m_actions["python_global_runmode"]);
        m_pythonToolBar->setFloatable(false);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::createMenus()
{
    m_pMenuFile = menuBar()->addMenu(tr("File"));
    m_pMenuFile->addAction(m_appFileNew);
    m_pMenuFile->addAction(m_appFileOpen);

    // dynamically created Menu with the last files
    m_plastFilesMenu = m_pMenuFile->addMenu(QIcon(":/files/icons/filePython.png"), tr("Recently Used Files"));
    connect(this->m_plastFilesMenu, SIGNAL(aboutToShow()), this, SLOT(menuLastFilesAboutToShow()));
    // Add these menus dynamically

    ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();

    if (uOrg->hasFeature(featProperties))
    {
        m_pMenuFile->addAction(m_actions["properties"]);
    }

    if (uOrg->hasFeature(featUserManag))
    {
        m_pMenuFile->addAction(m_actions["usermanagement"]);
    }

    m_pMenuFile->addAction(m_actions["show_loaded_plugins"]);
    m_pMenuFile->addSeparator();
    m_pMenuFile->addAction(m_actions["exit"]);

    m_pMenuView = menuBar()->addMenu(tr("View"));
    connect(m_pMenuView, SIGNAL(aboutToShow()), this, SLOT(mnuViewAboutToShow()));

    m_pMenuFigure = menuBar()->addMenu(tr("Figure"));
    m_pMenuFigure->addAction(m_actions["close_all_plots"]);
    m_pMenuFigure->addAction(m_actions["show_all_plots"]);
    m_pMenuFigure->addAction(m_actions["minimize_all_plots"]);
    m_pShowOpenFigure = m_pMenuFigure->addMenu(QIcon(":/application/icons/showPlot.png"), tr("Current Figures"));
    connect(m_pShowOpenFigure, SIGNAL(aboutToShow()), this, SLOT(mnuFigureAboutToShow()));

    if (uOrg->hasFeature(featDeveloper))
    {
        m_pMenuPython = menuBar()->addMenu(tr("Script"));
        m_pMenuPython->addAction(m_actions["python_stopAction"]);
        m_pMenuPython->addAction(m_actions["python_continueAction"]);
        m_pMenuPython->addAction(m_actions["python_stepAction"]);
        m_pMenuPython->addAction(m_actions["python_stepOverAction"]);
        m_pMenuPython->addAction(m_actions["python_stepOutAction"]);
        m_pMenuPython->addSeparator();
        m_pMenuPython->addAction(m_actions["python_global_runmode"]);
        

        m_pMenuReloadModule = m_pMenuPython->addMenu(QIcon(":/application/icons/reload.png"), tr("Reload Modules"));
        m_pMenuReloadModule->addAction(m_actions["py_autoReloadEnabled"]);
        m_pMenuReloadModule->addSeparator();
        m_pMenuReloadModule->addAction(m_actions["py_autoReloadFile"]);
        m_pMenuReloadModule->addAction(m_actions["py_autoReloadCmd"]);
        m_pMenuReloadModule->addAction(m_actions["py_autoReloadFunc"]);
        m_pMenuReloadModule->addSeparator();
        m_pMenuReloadModule->addAction(m_actions["python_reloadModules"]);

        m_pMenuPython->addAction(m_actions["python_timerManager"]);
        m_pMenuPython->addAction(m_actions["py_packageManager"]);
    }

    m_pMenuHelp = menuBar()->addMenu(tr("Help"));
    if (uOrg->hasFeature(featDeveloper))
    {
        m_pMenuHelp->addAction(m_actions["open_assistant"]);
        m_pMenuHelp->addAction(m_actions["script_reference"]);
    }
    m_pMenuHelp->addAction(m_aboutQt);
    m_pMenuHelp->addAction(m_aboutQitom);
//    m_pMenuHelp->addAction(m_actions["show_loaded_plugins"]);
    
    //linux: in some linux distributions, the menu bar did not appear if it is displayed
    //on top of the desktop. Therefore, native menu bars (as provided by the OS) are disabled here.
    //see: qt-project.org/forums/viewthread/7445
#ifndef __APPLE__
    menuBar()->setNativeMenuBar(false);
#else // __APPLE__
    // OS X: without the native menu bar option, the menu bar is displayed within the window which might be irritating.
    menuBar()->setNativeMenuBar(true);
#endif // __APPLE__
}

//----------------------------------------------------------------------------------------------------------------------------------
/*Slot aboutToOpen*/
void MainWindow::menuLastFilesAboutToShow()
{
    // Delete old actions
    for (int i = 0; i < m_plastFilesMenu->actions().length(); ++i)
    {
        m_plastFilesMenu->actions().at(i)->deleteLater();
    }
    m_plastFilesMenu->clear();
    
    // Get StringList of last Files
    QStringList fileList;
    QObject *seoO = AppManagement::getScriptEditorOrganizer();
    if (seoO)
    {
        ScriptEditorOrganizer *sEO = qobject_cast<ScriptEditorOrganizer*>(seoO);
        if (sEO)
        {
            if (sEO->getRecentlyUsedFiles().isEmpty())
            {
                QAction *a = m_plastFilesMenu->addAction(tr("No Entries"));
                a->setEnabled(false);
            }
            else
            {
                QAction *a;

                // Create new menus
                foreach (const QString &path, sEO->getRecentlyUsedFiles()) 
                {
                    QString displayedPath = path;
                    IOHelper::elideFilepathMiddle(displayedPath, 200);
                    a = new QAction(QIcon(":/icons/filePython.png"), displayedPath, this);
                    m_plastFilesMenu->addAction(a);
                    connect(a, SIGNAL(triggered()), m_lastFilesMapper, SLOT(map()));
                    m_lastFilesMapper->setMapping(a, path);
                }
            }
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuFigureAboutToShow()
{
    if (m_pMenuView)
    {
        // Delete old actions
        for (int i = 0; i < m_pShowOpenFigure->actions().length(); ++i)
        {
            m_pShowOpenFigure->actions().at(i)->deleteLater();
        }
        m_pShowOpenFigure->clear();
    }
    ito::RetVal retval = ito::retOk;
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if (uiOrga == NULL)
    {
        retval += ito::RetVal(ito::retError, 0, QString("Instance of UiOrganizer not available").toLatin1().data());

    }
    if (!retval.containsError())
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QSharedPointer<QList<unsigned int> > widgetNames(new QList<unsigned int>);
        QSharedPointer<QString> title(new QString);
        QMetaObject::invokeMethod(uiOrga, "getAllAvailableHandles", Q_ARG(QSharedPointer<QList<unsigned int> >, widgetNames), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        unsigned int val;
        QAction *a;
        if (widgetNames->isEmpty())
        {
            a = new QAction(QString("No Figures Available"), this);
            m_pShowOpenFigure->addAction(a);
        }
        else
        {
            qSort(*widgetNames);
            
            foreach(val, *widgetNames)
            {
                ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
                QMetaObject::invokeMethod(uiOrga, "getPlotWindowTitlebyHandle", Q_ARG(unsigned int, val), Q_ARG(QSharedPointer<QString>, title), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
                
                a = new QAction(*title, this);
                
                m_pShowOpenFigure->addAction(a);
                connect(a, SIGNAL(triggered()), m_openFigureMapper, SLOT(map()));
                m_openFigureMapper->setMapping(a, val);
            }
        }

    }
    return;
}
//----------------------------------------------------------------------------------------------------------------------------------
/*Slot aboutToOpen*/
void MainWindow::mnuViewAboutToShow()
{
    if (m_pMenuView)
    {
        m_pMenuView->clear();
        
        QMenu *dockWidgets = createPopupMenu();
        if (dockWidgets)
        {
            dockWidgets->menuAction()->setIcon(QIcon(":/application/icons/preferences-general.png"));
            dockWidgets->menuAction()->setText(tr("Toolboxes"));
            m_pMenuView->addMenu(dockWidgets);
            m_pMenuView->addSeparator();
        }

        ito::ScriptEditorOrganizer *sew = qobject_cast<ito::ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());
        QAction *a;

        if (sew != NULL)
        {
            QStringList filenames = sew->openedScripts();
            QString filenameElided;

            if (filenames.size() > 0)
            {
                foreach(const QString &filename, filenames)
                {
                    filenameElided = filename;
                    IOHelper::elideFilepathMiddle(filenameElided, 200);
                    a = new QAction(QIcon(":/files/icons/filePython.png"), filenameElided, this);
                    m_pMenuView->addAction(a);
                    connect(a, SIGNAL(triggered()), m_openScriptsMapper, SLOT(map()));
                    m_openScriptsMapper->setMapping(a, filename);
                }
            }
            else
            {
                a = m_plastFilesMenu->addAction(tr("No Opened Scripts"));
                a->setEnabled(false);
                m_pMenuView->addAction(a);
            }
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------------------
// Slot that is invoked by the lastfile Buttons over the signalmapper
void MainWindow::lastFileOpen(const QString &path)
{
    QString fileName;
    fileName = path;

    if (!fileName.isEmpty())
    {
        QDir::setCurrent(QFileInfo(fileName).path());
        IOHelper::openGeneralFile(fileName, false, true, this);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::openScript(const QString &filename)
{
    ito::ScriptEditorOrganizer *sew = qobject_cast<ito::ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());

    if (sew)
    {
        sew->openScript(filename);
    }
}
//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::raiseFigureByHandle(int handle)
{
    ito::RetVal retval = ito::retOk;
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if (uiOrga == NULL)
    {
        retval += ito::RetVal(ito::retError, 0, QString("Instance of UiOrganizer not available").toLatin1().data());

    }
    if (!retval.containsError())
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QMetaObject::invokeMethod(uiOrga, "figureShow",Q_ARG(unsigned int, handle), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    }
}
//----------------------------------------------------------------------------------------------------------------------------------
//! initializes status bar
void MainWindow::createStatusBar()
{
	m_pStatusLblCurrentDir = new QLabel("cd: ", this);
    statusBar()->addPermanentWidget(m_pStatusLblCurrentDir);

	m_pStatusLblPythonBusy = new QLabel(tr("Python is being executed"), this);
	m_pStatusLblPythonBusy->setVisible(false);
	statusBar()->addWidget(m_pStatusLblPythonBusy);

    currentDirectoryChanged(); //actualize the label of m_statusLblCurrentDir

    statusBar()->showMessage(tr("Ready"));
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot connected to signal pythonStateChanged in PythonEngine which is invoked by every change of the python state
/*!
    Actually, this slot is only evaluated in the main window in order to show python's busy state in the statusBar.

    \param pyTransition Python transition to the next state
    \sa PythonEngine
*/
void MainWindow::pythonStateChanged(tPythonTransitions pyTransition)
{
    QToolBar* tempToolBar;
    QAction* tempAction;

    switch(pyTransition)
    {
    case pyTransBeginRun:
        m_pythonInWaitingMode=false;
        m_pythonDebugMode = false;
        m_pythonBusy = true;
		statusBar()->clearMessage();
		m_pStatusLblPythonBusy->setVisible(true);

        //disable every userDefined-Action
        foreach(tempToolBar, m_userDefinedToolBars)
        {
            foreach(tempAction, tempToolBar->actions())
            {
                tempAction->setEnabled(false);
            }
        }
        break;
    case pyTransBeginDebug:
        m_pythonInWaitingMode=false;
        m_pythonDebugMode = true;
        m_pythonBusy = true;
		statusBar()->clearMessage();
		m_pStatusLblPythonBusy->setVisible(true);

        //disable every userDefined-Action
        foreach(tempToolBar, m_userDefinedToolBars)
        {
            foreach(tempAction, tempToolBar->actions())
            {
                tempAction->setEnabled(false);
            }
        }
        break;
    case pyTransDebugExecCmdBegin:
        m_pythonInWaitingMode=false;
        m_pythonDebugMode = true;
        m_pythonBusy = true;
        break;
    case pyTransDebugContinue:
        m_pythonBusy = true;
        m_pythonDebugMode = true;
        m_pythonInWaitingMode=false;
        break;
    case pyTransEndRun:
    case pyTransEndDebug:
        m_pythonDebugMode = false;
        m_pythonBusy = false;
        m_pythonInWaitingMode=false;

		m_pStatusLblPythonBusy->setVisible(false);

        //enable every userDefined-Action
        foreach(tempToolBar, m_userDefinedToolBars)
        {
            foreach(tempAction, tempToolBar->actions())
            {
                tempAction->setEnabled(true);
            }
        }
        break;
    case pyTransDebugWaiting:
    case pyTransDebugExecCmdEnd:
        m_pythonInWaitingMode=true;
        m_pythonDebugMode = true;
        m_pythonBusy = true;
        break;
    }

    updatePythonActions();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! updates actions which deal with python commands
void MainWindow::updatePythonActions()
{
    ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();
    if (uOrg->hasFeature(featDeveloper))
    {
        m_actions["python_stopAction"]->setEnabled(pythonBusy());
        m_actions["python_continueAction"]->setEnabled(pythonBusy() && pythonDebugMode() && pythonInWaitingMode());
        m_actions["python_stepAction"]->setEnabled(pythonBusy() && pythonDebugMode() && pythonInWaitingMode());
        m_actions["python_stepOverAction"]->setEnabled(pythonBusy() && pythonDebugMode() && pythonInWaitingMode());
        m_actions["python_stepOutAction"]->setEnabled(pythonBusy() && pythonDebugMode() && pythonInWaitingMode());

        bool enableUserDefMenu = (pythonBusy() && pythonInWaitingMode()) || !pythonBusy();
        foreach(QMenu *mnu, m_userDefinedRootMenus)
        {
            mnu->setEnabled(enableUserDefMenu);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by action to open a new python script
/*!
    invokes method \a newScript in ScriptEditorOrganizer
*/
void MainWindow::mnuNewScript()
{
    QObject *sew = AppManagement::getScriptEditorOrganizer();

    if (sew != NULL)
    {
        QMetaObject::invokeMethod(sew, "newScript");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by action to open any known file format
/*!
    Py-macro files will be opened by ScriptEditorOrganizer

    \sa ScriptEditorOrganizer
*/
void MainWindow::mnuOpenFile()
{
    QString fileName;
    RetVal retValue(retOk);

    QString filter = IOHelper::getFileFilters(IOHelper::IOFilters(IOHelper::IOInput | IOHelper::IOPlugin | IOHelper::IOAllFiles | IOHelper::IOMimeAll));
    static QString selectedFilter; //since this variable is static, it will remember the last set filter.
    fileName = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::currentPath(), filter, &selectedFilter); //tr("python (*.py);;itom data collection (*.idc);;images (*.rpm *.bmp *.png);;matlab (*.mat);;itom files(*.py *.idc *.rpm *.bmp *.png *.mat);;all files (*.*)"));

    QFileInfo info(fileName);

    if (fileName.isEmpty()) return;

    QDir::setCurrent(QFileInfo(fileName).path());
    IOHelper::openGeneralFile(fileName, false, true, this);
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuShowAssistant()
{
	showAssistant();
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::showAssistant(const QString &collectionFile /*= ""*/)
{
#ifdef __APPLE__
	QString appName = "Assistant";
#else
	QString appName = "assistant";
#endif

	ito::RetVal retval;
	QString collectionFile_;

	if (collectionFile == "") //create internal help, if not yet done
	{
		if (this->m_pHelpSystem == NULL)
		{
			m_pHelpSystem = HelpSystem::getInstance();
			QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
			retval += m_pHelpSystem->rebuildHelpIfNotUpToDate();
			collectionFile_ = m_pHelpSystem->getHelpCollectionAbsFileName();
			QApplication::restoreOverrideCursor();
		}
		else
		{
			collectionFile_ = m_pHelpSystem->getHelpCollectionAbsFileName();
			QFileInfo collectionFileInfo(collectionFile_);
			if (!collectionFileInfo.exists())
			{
				QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
				retval += m_pHelpSystem->rebuildHelpIfNotUpToDate();
				collectionFile_ = m_pHelpSystem->getHelpCollectionAbsFileName();
				QApplication::restoreOverrideCursor();
			}
		}
	}
	else
	{
		QFileInfo fileInfo(collectionFile);
		if (fileInfo.suffix() == "qhc" && fileInfo.exists())
		{
			collectionFile_ = collectionFile;
		}
		else
		{
			retval += ito::RetVal::format(ito::retError, 0, "The file '%s' is not a valid help collection file or does not exist.", collectionFile.toLatin1().data());
		}
	}

	if (!retval.containsError()) //warning is ok
	{
#ifdef ITOM_USEHELPVIEWER
		if (m_helpViewer.isNull())
		{
			m_helpViewer = QPointer<HelpViewer>(new HelpViewer(NULL));
			m_helpViewer->setAttribute(Qt::WA_DeleteOnClose, true);
		}
		m_helpViewer->setCollectionFile(collectionFile_);
		m_helpViewer->show();

#else
		ProcessOrganizer *po = qobject_cast<ProcessOrganizer*>(AppManagement::getProcessOrganizer());
		if (po)
		{
			bool existingProcess = false;
			QProcess *process = po->getProcess(appName, true, existingProcess, true);

			if (existingProcess && process->state() == QProcess::Running)
			{
				//assistant is already loaded. try to activate it by sending the activateIdentifier command without arguments (try-and-error to find this way to activate it)
				QByteArray ba;
				ba.append("activateIdentifier \n");
				process->write(ba);
			}
			else
			{
				QStringList args;

				args << QLatin1String("-collectionFile");
				args << QLatin1String(collectionFile_.toLatin1().data());
				args << QLatin1String("-enableRemoteControl");

				QString app = ProcessOrganizer::getAbsQtToolPath(appName);

				process->start(app, args);

				connect(process, SIGNAL(error(QProcess::ProcessError)), this, SLOT(helpAssistantError(QProcess::ProcessError)));
			}
		}
		else
		{
			retval += ito::RetVal(ito::retError, 0, "Process Organizer could not be loaded");
		}
#endif
	}

	if (retval != ito::retOk)
	{
		QString title;
		QString text;
		if (retval.hasErrorMessage()) text = QString("\n%1").arg(QLatin1String(retval.errorMessage()));
		if (retval.containsError())
		{
			text.prepend(tr("Error when preparing help or showing assistant."));
			QMessageBox::critical(this, tr("Error while showing assistant."), text);
		}
		else if (retval.containsWarning())
		{
			text.prepend(tr("Warning when preparing help or showing assistant."));
			QMessageBox::warning(this, tr("Warning while showing assistant."), text);
		}
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuCloseAllPlots()
{
    QObject *uiOrga = AppManagement::getUiOrganizer();
    if (uiOrga == NULL)
    {
        QMessageBox::critical(this, "UiOrganizer", "The UiOrganizer is not available");
        return;
    }

    QMetaObject::invokeMethod(uiOrga, "closeAllFloatableFigures", Q_ARG(ItomSharedSemaphore*, NULL)); 
}
//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuShowAllPlots()
{
    QObject *uiOrga = AppManagement::getUiOrganizer();
    if (uiOrga == NULL)
    {
        QMessageBox::critical(this, "UiOrganizer", "The UiOrganizer is not available");
        return;
    }
    
    QMetaObject::invokeMethod(uiOrga, "figureShow", Q_ARG(unsigned int, 0), Q_ARG(ItomSharedSemaphore*, NULL));
}
//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuMinimizeAllPlots()
{
    QObject *uiOrga = AppManagement::getUiOrganizer();
    if (uiOrga == NULL)
    {
        QMessageBox::critical(this, "UiOrganizer", "The UiOrganizer is not available");
        return;
    }

    QMetaObject::invokeMethod(uiOrga, "figureMinimizeAll", Q_ARG(ItomSharedSemaphore*, NULL));
}
//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuShowScriptReference()
{
    if (m_helpDock)
    {
        m_helpDock->raiseAndActivate();
        /*m_helpDock->setVisible(true);
        m_helpDock->raise();*/
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuAboutQitom()
{
    QMap<QString, QString> versionList = getItomVersionMap();

    DialogAboutQItom *dlgAbout = new DialogAboutQItom(versionList);
    dlgAbout->exec();
    DELETE_AND_SET_NULL(dlgAbout);
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::helpAssistantError (QProcess::ProcessError /*error*/)
{
    QMessageBox msgBox(this);
    msgBox.setText(tr("The help assistant could not be started."));
    msgBox.exec();
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuShowProperties()
{
    DialogProperties *dlg = new DialogProperties();
    dlg->exec();
    if (dlg->result() == QDialog::Accepted)
    {

    }

    DELETE_AND_SET_NULL(dlg);
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuShowUserManagement()
{
    DialogUserManagement *dlg = new DialogUserManagement();
    dlg->exec();
    if (dlg->result() == QDialog::Accepted)
    {

    }

    DELETE_AND_SET_NULL(dlg);
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::setStatusText(QString message, int timeout)
{
    if (message == "")
    {
        this->statusBar()->clearMessage();
    }
    else
    {
        statusBar()->showMessage(message, std::max<int>(timeout, 0));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*
An existing button with the same name (text) will not be deleted but a new button is added. This is because the possibly related
python methods or functions cannot be deleted from this method. However, each button has its unique buttonHandle that can be used
to explicitely delete the button.
*/
ito::RetVal MainWindow::addToolbarButton(const QString &toolbarName, const QString &buttonName, const QString &buttonIconFilename, const QString &pythonCode, QSharedPointer<size_t> buttonHandle, ItomSharedSemaphore *waitCond /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval;
    QMap<QString, QToolBar*>::const_iterator it = m_userDefinedToolBars.constFind(toolbarName);
    QToolBar *toolbar = NULL;
    QAction *action = NULL;
    
    if (it == m_userDefinedToolBars.constEnd())
    {
        m_userDefinedToolBars[toolbarName] = toolbar = new QToolBar(toolbarName, this);
        addToolBar(toolbar);
    }
    else
    {
        toolbar = *it;

        ////check if this action already exists, if so delete it first
        //foreach(action, (*it)->actions())
        //{
        //    if (action->text() == buttonName)
        //    {
        //        (*it)->removeAction(action);
        //        DELETE_AND_SET_NULL(action);
        //        break;
        //    }
        //}
    }

    QIcon icon = IOHelper::searchIcon(buttonIconFilename, IOHelper::SFAll);
    action = new QAction(icon, buttonName, toolbar);
    action->setProperty("itom__buttonHandle", ++m_userDefinedActionCounter);
    action->setToolTip(buttonName);

    connect(action, SIGNAL(triggered()), m_userDefinedSignalMapper, SLOT(map()));
    m_userDefinedSignalMapper->setMapping(action, pythonCode);
    toolbar->addAction(action);

    *buttonHandle = m_userDefinedActionCounter;

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MainWindow::removeToolbarButton(const QString &toolbarName, const QString &buttonName, QSharedPointer<size_t> buttonHandle, bool showMessage /*= true*/, ItomSharedSemaphore *waitCond /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval;
    QMap<QString, QToolBar*>::iterator it = m_userDefinedToolBars.find(toolbarName);
    QAction* tempAction;
    bool found = false;
    *buttonHandle = (size_t)NULL;

    if (it != m_userDefinedToolBars.end())
    {
        foreach(tempAction, (*it)->actions())
        {
            if (tempAction->text() == buttonName)
            {
                (*it)->removeAction(tempAction);
                *buttonHandle = (size_t)(tempAction->property("itom__buttonHandle").toUInt()); //0 if invalid
                DELETE_AND_SET_NULL(tempAction);
                found = true;
                break;
            }
        }
        
        if ((*it)->actions().size() == 0) //remove this toolbar
        {
            QString tmpName = it.key();
            removeToolBar(*it);
            m_userDefinedToolBars.remove(tmpName);
        }

        if (!found)
        {
            retval += ito::RetVal::format(ito::retError, 0, "The button '%s' of toolbar '%s' could not be found.", buttonName.toLatin1().data(), toolbarName.toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal::format(ito::retError, 0, "The toolbar '%s' could not be found.", toolbarName.toLatin1().data());
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    if (showMessage && retval.containsWarningOrError())
    {
        QMessageBox msgBox;
        msgBox.setText(QLatin1String(retval.errorMessage()));
        msgBox.exec();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MainWindow::removeToolbarButton(const size_t buttonHandle, bool showMessage /*= true*/, ItomSharedSemaphore *waitCond /*= NULL*/)
{
    //buttonHandle is the pointer-address to the QAction of the button
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval;
    QAction* tempAction;

    bool found = false;

    for (QMap<QString, QToolBar*>::iterator it = m_userDefinedToolBars.begin(); !found && it != m_userDefinedToolBars.end(); ++it)
    {
        foreach (tempAction, (*it)->actions())
        {
            if ((size_t)(tempAction->property("itom__buttonHandle").toUInt()) == buttonHandle)
            {
                (*it)->removeAction(tempAction);
                DELETE_AND_SET_NULL(tempAction);
                found = true;
                break;
            }
        }

        if (found && (*it)->actions().size() == 0) //remove this toolbar
        {
            QString key = it.key();
            removeToolBar(*it);
            m_userDefinedToolBars.remove(key);
            break;
        }
    }

    if (!found)
    {
        retval += ito::RetVal::format(ito::retError, 0, "The button (%i) could not be found.", buttonHandle);
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    if (showMessage && retval.containsWarningOrError())
    {
        QMessageBox msgBox;
        msgBox.setText(QLatin1String(retval.errorMessage()));
        msgBox.exec();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MainWindow::addMenuElement(int typeID, const QString &key, const QString &name, const QString &code, const QString &buttonIconFilename, QSharedPointer<size_t> menuHandle, bool showMessage /*= true*/, ItomSharedSemaphore *waitCond /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    RetVal retValue(retOk);

    //key is a slash-splitted value: e.g. rootKey/parentKey/nextParentKey/.../myKey
    QStringList keys = key.split("/");
    QString current_key;
    QAction *act;
    QMenu *parent_menu = NULL;
    QMap<QString, QMenu*>::iterator root_it;
    bool found = false;

    //check icon
    QIcon icon = IOHelper::searchIcon(buttonIconFilename, IOHelper::SFAll);

    //some sanity checks
    if (keys.size() == 1 && typeID != 2)
    {
        retValue += ito::RetVal(ito::retError, 0, tr("One single menu element must be of type MENU [2]").toLatin1().data());
    }
    else if (keys.size() == 0)
    {
        retValue += ito::RetVal(ito::retError, 0, tr("Key must not be empty.").toLatin1().data());
    }
    else if (typeID < 0 || typeID > 2)
    {
        retValue += ito::RetVal(ito::retError, 0, tr("Invalid menu item type.").toLatin1().data());
    }
    else
    {
        //check first level entry (is more special than all other ones since it is register in the m_userDefinedRootMenus map
        current_key = keys.takeFirst();
        root_it = m_userDefinedRootMenus.find(current_key);
        
        if (root_it != m_userDefinedRootMenus.end())
        {
            parent_menu = root_it.value();
            *menuHandle = (size_t)parent_menu->menuAction()->property("itom__menuHandle").toUInt();
        }
        else //exist new root menu item
        {
            parent_menu = menuBar()->addMenu(keys.size() == 0 ? name : current_key); //only the last item gets 'name' as visible name, all the others get their key component
            parent_menu->menuAction()->setData(current_key);
            parent_menu->menuAction()->setIconText(current_key);
            parent_menu->menuAction()->setProperty("itom__menuHandle", ++m_userDefinedActionCounter);
            *menuHandle = m_userDefinedActionCounter;
            m_userDefinedRootMenus[current_key] = parent_menu;
        }

        //now parent_menu is fixed and we can now recursively create the menu item tree until the last item
        while (parent_menu && keys.size() > 0 && !retValue.containsError())
        {
            current_key = keys.takeFirst();

            if (keys.size() > 0) //must be a tree item (menu item, since not the last one)
            {
                //check if parent_menu contains a child-action that is a submenu with the same current_key
                //if so, use this, else create a new sub-menu
                found = false;
                foreach(QAction* a, parent_menu->actions())
                {
                    if (a->menu() && a->data().toString() == current_key)
                    {
                        //existing one, use it
                        found = true;
                        parent_menu = a->menu();
                        *menuHandle = (size_t)a->property("itom__menuHandle").toUInt();
                        break;
                    }
                }

                if (!found)
                {
                    parent_menu = parent_menu->addMenu(icon,current_key);
                    parent_menu->menuAction()->setProperty("itom__menuHandle", ++m_userDefinedActionCounter);
                    parent_menu->menuAction()->setIconText(current_key);
                    parent_menu->menuAction()->setData(current_key);
                    *menuHandle = m_userDefinedActionCounter;
                }
            }
            else
            {
                //append a menu, separator or button to parent_menu and returns its menuHandle.
                if (typeID == 0) //BUTTON
                {
                    act = parent_menu->addAction(icon,name);
                    act->setProperty("itom__menuHandle", ++m_userDefinedActionCounter);
                    act->setIconText(name);
                    act->setData(current_key);
                    connect(act, SIGNAL(triggered()), m_userDefinedSignalMapper, SLOT(map()));
                    m_userDefinedSignalMapper->setMapping(act, code);
                    *menuHandle = m_userDefinedActionCounter;
                }
                else if (typeID == 2 /*MENU*/)
                {
                    parent_menu = parent_menu->addMenu(icon,name);
                    parent_menu->menuAction()->setProperty("itom__menuHandle", ++m_userDefinedActionCounter);
                    parent_menu->menuAction()->setIconText(name);
                    parent_menu->menuAction()->setData(current_key);
                    *menuHandle = m_userDefinedActionCounter;
                }
                else // if (typeID == 1) //SEPARATOR
                {
                    act = parent_menu->addSeparator();
                    act->setProperty("itom__menuHandle", ++m_userDefinedActionCounter);
                    act->setIconText(name);
                    act->setData(current_key);
                    *menuHandle = m_userDefinedActionCounter;
                } 
            }
        }
    }

    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    if (showMessage && retValue.containsError())
    {
        QMessageBox::critical(this, tr("Add menu element"), QLatin1String(retValue.errorMessage()));
    }
    else if (showMessage && retValue.containsWarning())
    {
        QMessageBox::warning(this, tr("Add menu element"), QLatin1String(retValue.errorMessage()));
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MainWindow::removeMenuElement(const QString &key, QSharedPointer<QVector<size_t> > removedMenuHandles, bool showMessage /*= true*/, ItomSharedSemaphore *waitCond /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval;
    QStringList keys = key.split("/");
    QString tempKey;
    QMenu *parentMenu = NULL;
    QAction *actToDelete = NULL;
    QMap<QString, QMenu*>::iterator it;

    if (keys.size() == 1)
    {
        tempKey = keys[0];
        keys.pop_front();
        it = m_userDefinedRootMenus.find(tempKey);
        if (it != m_userDefinedRootMenus.end())
        {
            removedMenuHandles->append((size_t)(it.value()->menuAction()->property("itom__menuHandle").toUInt()));
            getMenuHandlesRecursively(it.value(), removedMenuHandles);
            (*it)->deleteLater();
            it = m_userDefinedRootMenus.erase(it);
        }
        else
        {
            retval += ito::RetVal::format(ito::retError, 0, "A user-defined menu with the key sequence '%s' could not be found", key.toLatin1().data());
        }
    }
    else if (keys.size() > 1)
    {
        tempKey = keys[0];
        keys.pop_front();
        it = m_userDefinedRootMenus.find(tempKey);
        if (it != m_userDefinedRootMenus.end())
        {
            parentMenu = *it;
        }

        while (keys.size() > 0 && parentMenu)
        {
            tempKey = keys[0];
            keys.pop_front();
            actToDelete = NULL;

            foreach(QAction *a, parentMenu->actions())
            {
                if (a->data().toString() == tempKey)
                {
                    actToDelete = a;
                    parentMenu = actToDelete->menu();
                    break;
                }
            }
        }

        if (keys.size() == 0 && actToDelete)
        {
            removedMenuHandles->append((size_t)(actToDelete->property("itom__menuHandle").toUInt()));
            getMenuHandlesRecursively(actToDelete->menu(), removedMenuHandles);

            if (actToDelete->menu()) //this action belongs to a QMenu -> delete the QMenu
            {
                actToDelete->menu()->deleteLater();
            }
            else //this action is a real action -> directly delete it
            {
                actToDelete->deleteLater();
            }
        }
        else
        {
            retval += ito::RetVal::format(ito::retError, 0, "A user-defined menu with the key sequence '%s' could not be found", key.toLatin1().data());
        }
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    if (showMessage && retval.containsWarningOrError())
    {
        QMessageBox::warning(this, tr("Remove menu element"), QLatin1String(retval.errorMessage()));
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MainWindow::removeMenuElement(const size_t menuHandle, QSharedPointer<QVector<size_t> > removedMenuHandles, bool showMessage /*= true*/, ItomSharedSemaphore *waitCond /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval;
    QString tempKey;
    // QMenu *parentMenu = NULL; // unused
    QAction *actToDelete = NULL;
    bool found = false;
    QMap<QString, QMenu*>::iterator it = m_userDefinedRootMenus.begin();

    while (it != m_userDefinedRootMenus.end() && !found)
    {
        if ((size_t)it.value()->menuAction()->property("itom__menuHandle").toUInt() == menuHandle)
        {
            found = true;
            removedMenuHandles->append(menuHandle);
            getMenuHandlesRecursively(it.value(), removedMenuHandles);
            it.value()->deleteLater();
            it = m_userDefinedRootMenus.erase(it);
        }

        if (!found)
        {
            actToDelete = searchActionRecursively(menuHandle, it.value());
            if (actToDelete)
            {
                found = true;
                removedMenuHandles->append((size_t)(actToDelete->property("itom__menuHandle").toUInt()));
                getMenuHandlesRecursively(actToDelete->menu(), removedMenuHandles);
                
                if (actToDelete->menu()) //this action belongs to a QMenu -> delete the QMenu
                {
                    actToDelete->menu()->deleteLater();
                }
                else //this action is a real action -> directly delete it
                {
                    actToDelete->deleteLater();
                }
            }
        }

        ++it;
    }

    if (!found)
    {
        retval += ito::RetVal::format(ito::retError, 0, "A user-defined menu with the handle '%i' could not be found", menuHandle);
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    if (showMessage && retval.containsWarningOrError())
    {
        QMessageBox::warning(this, tr("Remove menu element"), QLatin1String(retval.errorMessage()));
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::getMenuHandlesRecursively(const QMenu *parent, QSharedPointer<QVector<size_t> > menuHandles)
{
    if (parent)
    {
        foreach (const QAction *a, parent->actions())
        {
            if (a)
            {
                menuHandles->append((size_t)a->property("itom__menuHandle").toUInt());
                getMenuHandlesRecursively(a->menu(), menuHandles);
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QAction* MainWindow::searchActionRecursively(const size_t menuHandle, const QMenu *parent)
{
    if (!parent)
    {
        return NULL;
    }

    QAction *a2;

    foreach (QAction *a, parent->actions())
    {
        if ((size_t)a->property("itom__menuHandle").toUInt() == menuHandle)
        {
            return a;
        }

        a2 = searchActionRecursively(menuHandle, a->menu());
        if (a2)
        {
            return a2;
        }
    }

    return NULL;
}

QString dumpChildMenus(const QString &baseKey, const QAction *parent)
{
	unsigned int id;
	bool ok;
	QStringList items;
	QString key;

	if (parent->menu())
	{
		foreach(const QAction *a, parent->menu()->actions())
		{
			id = a->property("itom__menuHandle").toUInt(&ok);
			key = baseKey + "/" + a->data().toString();

			if (ok)
			{
				items.append(QString("{'id': %1, 'key':'%2', 'name': '%3', 'children': %4}").arg(id).arg(key).arg(a->text()).arg(dumpChildMenus(key, a)));
			}
			else
			{
				items.append(QString("{'id': %1, 'key':'%2', 'name': '%3', 'children': %4}").arg(-1).arg(key).arg(a->text()).arg(dumpChildMenus(key, a)));
			}
		}
	}
	
	return QString("[%1]").arg(items.join(","));
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MainWindow::dumpToolbarsAndButtons(QSharedPointer<QString> pythonCodeString, ItomSharedSemaphore *waitCond /*= NULL*/)
{
	ito::RetVal retval;

	QStringList toolbar_list;
	QStringList menu_list;

	QStringList actions;
	unsigned int id;
	bool ok;
	QAction *act;


	//toolbars
	foreach(const QString &key, m_userDefinedToolBars.keys())
	{
		actions.clear();

		foreach(const QAction *a, m_userDefinedToolBars[key]->actions())
		{
			id = a->property("itom__buttonHandle").toUInt(&ok);
			if (ok)
			{
				actions.append(QString("{'id': %1, 'name': '%2'}").arg(id).arg(a->text()));
			}
			else
			{
				actions.append(QString("{'id': -1, 'name': '%1'}").arg(a->text()));
			}
		}

		toolbar_list.append(QString("'%1':[%2]").arg(key).arg(actions.join(",")));
	}

	//menus
	QMap<QString, QMenu *>::ConstIterator it = m_userDefinedRootMenus.constBegin();
	while (it != m_userDefinedRootMenus.constEnd())
	{
		act = it.value()->menuAction();

		id = act->property("itom__menuHandle").toUInt(&ok);
		if (ok)
		{
			menu_list.append(QString("{'id': %1, 'key':'%2', 'name': '%3', 'children': %4}").arg(id).arg(it.key()).arg(act->text()).arg(dumpChildMenus(it.key(), act)));
		}
		else
		{
			menu_list.append(QString("{'id': %1, 'key':'%2', 'name': '%3', 'children': %4}").arg(-1).arg(it.key()).arg(act->text()).arg(dumpChildMenus(it.key(), act)));
		}

		it++;
	}

	*pythonCodeString = QString("{'toolbars':{%1}, 'menus':[%2]}").arg(toolbar_list.join(",")).arg(menu_list.join(","));

	if (waitCond)
	{
		waitCond->returnValue = retval;
		waitCond->release();
	}

	return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::pythonRunSelection(QString selectionText)
{
    m_console->pythonRunSelection(selectionText);
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuPyPipManager()
{
    DialogPipManager *dpm = new DialogPipManager(this);
    dpm->exec();
    DELETE_AND_SET_NULL(dpm);
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::currentDirectoryChanged()
{
    QString cd = QDir::cleanPath(QDir::currentPath());
    if (m_pStatusLblCurrentDir)
    {
        m_pStatusLblCurrentDir->setText(tr("Current Directory: %1").arg(cd));
    }

    if (m_fileSystemDock)
    {
        m_fileSystemDock->changeBaseDirectory(cd);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::setCursor(const Qt::CursorShape cursor)
{
    QApplication::setOverrideCursor(QCursor(cursor));
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::resetCursor()
{
    QApplication::restoreOverrideCursor();
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::userDefinedActionTriggered(const QString &pythonCode)
{
    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (pyEngine == NULL)
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Python is not available. This action cannot be executed."));
        msgBox.exec();
    }
    else if (pythonCode == "")
    {
        QMessageBox msgBox;
        msgBox.setText(tr("There is no python code associated with this action."));
        msgBox.exec();
    }
    else
    {
        QByteArray ba(pythonCode.toLatin1());
        ba.replace("\\n",QByteArray(1,'\n'));
        ba.replace("\n",QByteArray(1,'\n'));

        if (pyEngine->execInternalCodeByDebugger())
        {
            QMetaObject::invokeMethod(pyEngine, "pythonDebugStringOrFunction", Q_ARG(QString, pythonCode));
        }
        else
        {
            QMetaObject::invokeMethod(pyEngine, "pythonRunStringOrFunction", Q_ARG(QString, pythonCode));
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuShowDesigner()
{
#ifdef __APPLE__
    QString appName = "Designer";
#else
    QString appName = "designer";
#endif
    
    ProcessOrganizer *po = qobject_cast<ProcessOrganizer*>(AppManagement::getProcessOrganizer());
    if (po)
    {
        bool existingProcess = false;
        QProcess *process = po->getProcess(appName, true, existingProcess, false);

        if (existingProcess && process->state() == QProcess::Running)
        {
            //designer is already loaded. try to activate it by sending the activateIdentifier command without arguments (try-and-error to find this way to activate it)
            QByteArray ba("activateIdentifier \n");
            process->write(ba);
        }
        else
        {
            QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
            QString appPath = QDir::cleanPath(QCoreApplication::applicationDirPath());
            env.insert("QT_PLUGIN_PATH", appPath);
            
#ifndef WIN32
            QString pathEnv = env.value("PATH");
            pathEnv.prepend(appPath + ":");
            env.insert("PATH", pathEnv);
#else
            QString pathEnv = env.value("path");
            pathEnv.prepend(appPath + ";");
            env.insert("path", pathEnv);
#endif
            
#ifdef __APPLE__
            env.insert("PATH", env.value("PATH") + ":" + env.value("HOME") + "/Applications");
            env.insert("PATH", env.value("PATH") + ":/Applications");
#endif // __APPLE__
            
            process->setProcessEnvironment(env);
            
            connect(process, SIGNAL(error(QProcess::ProcessError)), this, SLOT(designerError(QProcess::ProcessError)));

            po->clearStandardOutputBuffer(appName);

            QStringList arguments;
            arguments << "-server"/* << filename*/;
            QString app = ProcessOrganizer::getAbsQtToolPath(appName);
            //qDebug() << app << arguments;
            process->start(app, arguments); //the arguments stringlist must be given here, else the process cannot be started in a setup environment!
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::designerError (QProcess::ProcessError error)
{
    QMessageBox msgBox(this);
    msgBox.setText(tr("The UI designer (Qt designer) could not be started (%1).").arg(error));
    msgBox.exec();
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuToggleExecPyCodeByDebugger(bool checked)
{
    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (pyEngine != NULL)
    {
        pyEngine->setExecInternalCodeByDebugger(checked);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuScriptStop()
{
    if (pythonDebugMode() && pythonInWaitingMode())
    {
        PythonEngine *pyeng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
        if (pyeng)
        {
            pyeng->pythonInterruptExecution();
        }
//        emit(pythonDebugCommand(ito::pyDbgQuit));
        raise();
    }
    else
    {
        PythonEngine *pyeng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
        if (pyeng)
        {
            pyeng->pythonInterruptExecution();
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to continue debugging process if actually waiting at breakpoint
void MainWindow::mnuScriptContinue()
{
    emit(pythonDebugCommand(ito::pyDbgContinue));
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute a python debugging step
void MainWindow::mnuScriptStep()
{
    emit(pythonDebugCommand(ito::pyDbgStep));
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute a python debugging step over
void MainWindow::mnuScriptStepOver()
{
    emit(pythonDebugCommand(ito::pyDbgStepOver));
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute a python debugging step out
void MainWindow::mnuScriptStepOut()
{
    emit(pythonDebugCommand(ito::pyDbgStepOut));
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuPyReloadModules()
{
    DialogReloadModule *dlgReloadModules = new DialogReloadModule(this);
    dlgReloadModules->exec();
    DELETE_AND_SET_NULL(dlgReloadModules);
}
//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuPyTimerManager()
{
	DialogTimerManager *dlgTimerManager = new DialogTimerManager(this);
	dlgTimerManager->exec();
	DELETE_AND_SET_NULL(dlgTimerManager);
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuPyAutoReloadTriggered(bool checked)
{
    if (m_actions["py_autoReloadEnabled"] && m_actions["py_autoReloadFile"] && m_actions["py_autoReloadCmd"] && m_actions["py_autoReloadFunc"])
    {
        bool enabled = m_actions["py_autoReloadEnabled"]->isChecked();
        bool checkFile = m_actions["py_autoReloadFile"]->isChecked();
        bool checkCmd = m_actions["py_autoReloadCmd"]->isChecked();
        bool checkFct = m_actions["py_autoReloadFunc"]->isChecked();
        emit pythonSetAutoReloadSettings(enabled, checkFile, checkCmd, checkFct);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuShowLoadedPlugins()
{
    DialogLoadedPlugins *dlgLoadedPlugins = new DialogLoadedPlugins(this);
    dlgLoadedPlugins->exec();
    DELETE_AND_SET_NULL(dlgLoadedPlugins);
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuExitApplication()
{
    //does not call the closeEvent-method!
    emit(mainWindowCloseRequest());
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::showInfoMessageLine(QString text, QString winKey /*= ""*/)
{
    WidgetInfoBox *w = NULL;

    if (winKey != "" && m_infoBoxWidgets.contains(winKey))
    {
        w = m_infoBoxWidgets[winKey].data();
        if (w == NULL)
        {
            m_infoBoxWidgets.remove(winKey);
        }
    }

    if (w == NULL)
    {
        w = new WidgetInfoBox(text, this);
        m_contentLayout->insertWidget(0, w);
        if (winKey != "")
        {
            m_infoBoxWidgets[winKey] = QPointer<WidgetInfoBox>(w);
        }
    }
    else if (text != "")
    {
        w->setInfoText(text);
    }
    else
    {
        w->deleteLater();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::pythonAutoReloadChanged(bool enabled, bool checkFile, bool checkCmd, bool checkFct)
{
    if (m_actions["py_autoReloadEnabled"]) m_actions["py_autoReloadEnabled"]->setChecked(enabled);

    if (m_actions["py_autoReloadFile"]) m_actions["py_autoReloadFile"]->setChecked(checkFile);
    if (m_actions["py_autoReloadCmd"]) m_actions["py_autoReloadCmd"]->setChecked(checkCmd);
    if (m_actions["py_autoReloadFunc"]) m_actions["py_autoReloadFunc"]->setChecked(checkFct);

    if (m_actions["py_autoReloadFile"]) m_actions["py_autoReloadFile"]->setEnabled(enabled);
    if (m_actions["py_autoReloadCmd"]) m_actions["py_autoReloadCmd"]->setEnabled(enabled);
    if (m_actions["py_autoReloadFunc"]) m_actions["py_autoReloadFunc"]->setEnabled(enabled);

}

} //end namespace ito

