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

#include "../python/pythonEngineInc.h"

#include "mainWindow.h"

#include "../AppManagement.h"
#include "../global.h"

#include "../organizer/addInManager.h"
#include "../organizer/processOrganizer.h"
#include "../organizer/uiOrganizer.h"
#include "../organizer/userOrganizer.h"

#include "../ui/dialogProperties.h"
#include "../ui/dialogAbout.h"
#include "../ui/dialogReloadModule.h"
#include "../ui/dialogLoadedPlugins.h"
#include "../ui/widgetInfoBox.h"

#include "../helper/versionHelper.h"

#include <qapplication.h>
#include <qstatusbar.h>
#include <qmessagebox.h>
#include <qfiledialog.h>
#include <qdesktopwidget.h>
#include <qdir.h>

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
    m_pMenuHelp(NULL),
    m_pMenuFile(NULL),
    m_pMenuPython(NULL),
	m_pMenuView(NULL),
    m_pHelpSystem(NULL),
    m_statusLblCurrentDir(NULL),
    m_pythonBusy(false),
    m_pythonDebugMode(false),
    m_pythonInWaitingMode(false),
    m_isFullscreen(false)
{
    //qDebug() << "mainWindow. Thread: " << QThread::currentThreadId ();
    QApplication::setWindowIcon(QIcon(":/application/icons/itomicon/curAppIcon.png"));

    qDebug("build main window");
    const PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    // general windows settings
#if QT_POINTER_SIZE == 8
    setWindowTitle(tr("itom (x64)"));
#else
    setWindowTitle(tr("itom"));
#endif

    setUnifiedTitleAndToolBarOnMac(true);

    setCorner(Qt::TopLeftCorner, Qt::LeftDockWidgetArea);
    setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
    setCorner(Qt::TopRightCorner, Qt::RightDockWidgetArea);
    setCorner(Qt::BottomRightCorner, Qt::RightDockWidgetArea);


    //content
    m_contentLayout = new QVBoxLayout;

    ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();
    if (uOrg->hasFeature(featConsole))
    {
        qDebug(".. before loading console widget");
        //console (central widget):
        m_console = new ConsoleWidget(this);
        //setCentralWidget(m_console);
        qDebug(".. console widget loaded");
        m_contentLayout->addWidget(m_console);
    }

    m_contentLayout->setContentsMargins(0,0,0,0);
    m_contentLayout->setSpacing(1);

    QWidget *centralWidget = new QWidget(this);
    centralWidget->setLayout(m_contentLayout);
    setCentralWidget(centralWidget); 

    if (uOrg->hasFeature(featFileSystem))
    {
        // FileDir-Dock
        m_fileSystemDock = new FileSystemDockWidget(tr("File System"), this, true, true, AbstractDockWidget::floatingStandard);
	    m_fileSystemDock->setObjectName("itomFileSystemDockWidget");
	    m_fileSystemDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
        connect(m_fileSystemDock, SIGNAL(currentDirChanged()), this, SLOT(currentDirectoryChanged()));
        addDockWidget(Qt::LeftDockWidgetArea, m_fileSystemDock);
    }

    if (uOrg->hasFeature(featDeveloper))
    {
        // breakPointDock
        m_breakPointDock = new BreakPointDockWidget(tr("Breakpoints"), this, true, true, AbstractDockWidget::floatingStandard);
	    m_breakPointDock->setObjectName("itomBreakPointDockWidget");
        m_breakPointDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
        addDockWidget(Qt::LeftDockWidgetArea, m_breakPointDock);

	    // CallStack-Dock
	    m_callStackDock = new CallStackDockWidget(tr("Call Stack"), this, true, true, AbstractDockWidget::floatingStandard);
	    m_callStackDock->setObjectName("itomCallStackDockWidget");
	    m_callStackDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea | Qt::TopDockWidgetArea);
	    addDockWidget(Qt::LeftDockWidgetArea, m_callStackDock);

        // tabify file-directory and breakpoints
	    tabifyDockWidget(m_callStackDock, m_fileSystemDock);
        tabifyDockWidget(m_fileSystemDock, m_breakPointDock);
        m_fileSystemDock->raise();

        // global workspace widget (Python)
        m_globalWorkspaceDock = new WorkspaceDockWidget(tr("Global Variables"), true, this, true, true, AbstractDockWidget::floatingStandard);
	    m_globalWorkspaceDock->setObjectName("itomGlobalWorkspaceDockWidget");
        m_globalWorkspaceDock->setAllowedAreas(Qt::AllDockWidgetAreas);
        addDockWidget(Qt::RightDockWidgetArea, m_globalWorkspaceDock);
        connect(m_globalWorkspaceDock, SIGNAL(setStatusInformation(QString,int)), this, SLOT(setStatusText(QString,int)));

        // local workspace widget (Python)
        m_localWorkspaceDock = new WorkspaceDockWidget(tr("Local Variables"), false, this, true, true, AbstractDockWidget::floatingStandard);
	    m_localWorkspaceDock->setObjectName("itomLocalWorkspaceDockWidget");
        m_localWorkspaceDock->setAllowedAreas(Qt::AllDockWidgetAreas);
        addDockWidget(Qt::RightDockWidgetArea, m_localWorkspaceDock);
        connect(m_localWorkspaceDock, SIGNAL(setStatusInformation(QString,int)), this, SLOT(setStatusText(QString,int)));

        // tabify global and local workspace
        tabifyDockWidget(m_globalWorkspaceDock, m_localWorkspaceDock);
        //splitDockWidget(m_globalWorkspaceDock, m_localWorkspaceDock, Qt::Horizontal);
        m_globalWorkspaceDock->raise();
    }

    if (uOrg->hasFeature(featPlugins))
    {
        // AddIn-Manager
    //    m_pAIManagerWidget = new AIManagerWidget();
        m_pAIManagerWidget = new AIManagerWidget(tr("Plugins"), this, true, true, AbstractDockWidget::floatingStandard, AbstractDockWidget::movingEnabled);
	    m_pAIManagerWidget->setObjectName("itomPluginsDockWidget");
        qDebug(".. plugin manager widget loaded");

        addDockWidget(Qt::RightDockWidgetArea, m_pAIManagerWidget);
    }

    // connections
    if (pyEngine != NULL)
    {
        //connect(pyEngine, SIGNAL(pythonModifyGlobalDict(PyObject*,ItomSharedSemaphore*)), m_globalWorkspaceDock, SLOT(loadPythonDictionary(PyObject *,ItomSharedSemaphore*)));
        //connect(pyEngine, SIGNAL(pythonModifyLocalDict(PyObject*,ItomSharedSemaphore*)), m_localWorkspaceDock, SLOT(loadPythonDictionary(PyObject *,ItomSharedSemaphore*)));
        connect(pyEngine, SIGNAL(pythonStateChanged(tPythonTransitions)), this, SLOT(pythonStateChanged(tPythonTransitions)));
        connect(pyEngine, SIGNAL(pythonAddToolbarButton(QString, QString, QString, QString)), this, SLOT(pythonAddToolbarButton(QString, QString, QString, QString)));
        connect(pyEngine, SIGNAL(pythonRemoveToolbarButton(QString, QString)), this, SLOT(pythonRemoveToolbarButton(QString, QString)));

        connect(pyEngine, SIGNAL(pythonAddMenuElement(int,QString,QString,QString,QString)), this, SLOT(pythonAddMenuElement(int,QString,QString,QString,QString)));
        connect(pyEngine, SIGNAL(pythonRemoveMenuElement(QString)), this, SLOT(pythonRemoveMenuElement(QString)));

        connect(pyEngine, SIGNAL(pythonCurrentDirChanged()), this, SLOT(currentDirectoryChanged()));
        connect(this, SIGNAL(pythonDebugCommand(tPythonDbgCmd)), pyEngine, SLOT(pythonDebugCommand(tPythonDbgCmd)));

        connect(pyEngine, SIGNAL(pythonSetCursor(Qt::CursorShape)), this, SLOT(setCursor(Qt::CursorShape)));
        connect(pyEngine, SIGNAL(pythonResetCursor()), this, SLOT(resetCursor()));
    }
	else
	{
		showInfoMessageLine(tr("Python could not be started. itom cannot be used in the desired way."));
		m_console->setReadOnly(true);
	}

    // signal mapper for user defined actions
    m_userDefinedSignalMapper = new QSignalMapper(this);
    connect(m_userDefinedSignalMapper, SIGNAL(mapped(const QString &)), this, SLOT(userDefinedActionTriggered(const QString &)));

    //
    createActions();
    createMenus();
    createToolBars();
    createStatusBar();
    updatePythonActions();


    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("MainWindow");

    //restoreGeometry(settings.value("geometry").toByteArray());

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

    setGeometry(geometry);
    m_geometryNormalState = geometry; //geometry in normal state

    if (maximized)
    {
        showMaximized();
    }

    qDebug(".. main window build done");

    //showInfoMessageLine("Hey folks...!", "TestInfo");
}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!
    disconnects connections between main window and python engine
*/
MainWindow::~MainWindow()
{
    QSettings *settings = new QSettings(AppManagement::getSettingsFile(), QSettings::IniFormat);
/*
    settings->beginGroup("MainWindow");
    settings->setValue("maximized", isMaximized());
    settings->setValue("geometry", m_geometryNormalState);
    //settings->setValue("geometry", saveGeometry());
    
    QByteArray state = saveState();
    settings->setValue("state", state);
    settings->endGroup();
*/
    delete settings;

	
	//QByteArray ba = storeDockWidgetStatus();

    const PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (pyEngine != NULL)
    {
        //disconnect(pyEngine, SIGNAL(pythonModifyGlobalDict(PyObject*,ItomSharedSemaphore*)), m_globalWorkspaceDock, SLOT(loadPythonDictionary(PyObject *,ItomSharedSemaphore*)));
        //disconnect(pyEngine, SIGNAL(pythonModifyLocalDict(PyObject*,ItomSharedSemaphore*)), m_localWorkspaceDock, SLOT(loadPythonDictionary(PyObject *,ItomSharedSemaphore*)));
        disconnect(pyEngine, SIGNAL(pythonStateChanged(tPythonTransitions)), this, SLOT(pythonStateChanged(tPythonTransitions)));
        disconnect(this, SIGNAL(pythonDebugCommand(tPythonDbgCmd)), pyEngine, SLOT(pythonDebugCommand(tPythonDbgCmd)));
    }

    if (m_globalWorkspaceDock) disconnect(m_globalWorkspaceDock, SIGNAL(setStatusInformation(QString,int)), this, SLOT(setStatusText(QString,int)));
    if (m_localWorkspaceDock)  disconnect(m_localWorkspaceDock, SIGNAL(setStatusInformation(QString,int)), this, SLOT(setStatusText(QString,int)));

//    delete m_pAIManagerView;
//    delete m_pAIManagerDock;
    delete m_pAIManagerWidget;

    //delete remaining user-defined toolbars and actions
    QMap<QString, QToolBar*>::iterator it = m_userDefinedToolBars.begin();
    while(it != m_userDefinedToolBars.end())
    {
        removeToolBar(*it);
        delete *it;
        it++;
    }
    m_userDefinedToolBars.clear();

    //delete remaining user-defined menu elements
    QMap<QString, QMenu* >::iterator it2 = m_userDefinedRootMenus.begin();
    while(it2 != m_userDefinedRootMenus.end())
    {
        (*it2)->deleteLater();
        it2++;
    }
    m_userDefinedRootMenus.clear();

    DELETE_AND_SET_NULL(m_userDefinedSignalMapper);
    m_pHelpSystem = NULL;

    QMapIterator<QString, QWeakPointer<WidgetInfoBox> > i(m_infoBoxWidgets);
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
    \param dockWidget ScriptDockWidget to add to any docking area
    \param area docking area, where dockWidget should be shown
*/
void MainWindow::addAbstractDock(AbstractDockWidget* dockWidget, Qt::DockWidgetArea area)
{
    if (dockWidget)
    {
        dockWidget->setParent(this);

        if (area == Qt::NoDockWidgetArea)
        {
            addDockWidget(Qt::TopDockWidgetArea , dockWidget);
            dockWidget->setFloating(true);
        }
        else
        {
            addDockWidget(area, dockWidget);
            dockWidget->setFloating(false);
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
//! close event invoked if main window should be closed (and therefore the whole application too)
/*!
    if this event is invoked the signal mainWindowCloseRequest is emitted, which invokes the slot mainWindowCloseRequest
    in class MainApplication in order to proceed the entire closing process. Therefore the event is ignored.

    \param event event of type QCloseEvent, describing the close request
    \sa MainApplication
*/
void MainWindow::closeEvent(QCloseEvent *event)
{
    //QSettings settings;
     //settings.setValue("geometry", saveGeometry());
     //settings.setValue("windowState", saveState());

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

    if (uOrg->hasFeature(featUserManag))
    {
        m_actions["properties"] = new QAction(QIcon(":/application/icons/adBlockAction.png"), tr("Properties..."), this);
        connect(m_actions["properties"] , SIGNAL(triggered()), this, SLOT(mnuShowProperties()));

        m_actions["usermanagement"] = new QAction(QIcon(":/misc/icons/User.png"), tr("User Management..."), this);
        connect(m_actions["usermanagement"] , SIGNAL(triggered()), this, SLOT(mnuShowUserManagement()));
    }

    m_aboutQt = new QAction(QIcon(":/application/icons/helpAboutQt.png"), tr("About Qt..."), this);
    connect(m_aboutQt, SIGNAL(triggered()), qApp, SLOT(aboutQt()));
    //m_aboutQt->setShortcut(QKeySequence("F3"));

    m_aboutQitom = new QAction(QIcon(":/application/icons/itomicon/q_itoM32.png"), tr("About itom..."), this);
    connect(m_aboutQitom, SIGNAL(triggered()), this, SLOT(mnuAboutQitom()));

    m_actions["show_loaded_plugins"] = new QAction(QIcon(":/plugins/icons/plugin.png"), tr("Loaded plugins..."), this);
    connect(m_actions["show_loaded_plugins"], SIGNAL(triggered()), this, SLOT(mnuShowLoadedPlugins()));

    if (uOrg->hasFeature(featDeveloper))
    {
        m_actions["open_assistant"] = new QAction(QIcon(":/application/icons/help.png"), tr("Help..."), this);
        connect(m_actions["open_assistant"] , SIGNAL(triggered()), this, SLOT(mnuShowAssistant()));
        m_actions["open_assistant"]->setShortcut(QKeySequence::HelpContents);

        m_actions["open_designer"] = new QAction(QIcon(":/application/icons/designer4.png"), tr("UI Designer"), this);
        connect(m_actions["open_designer"], SIGNAL(triggered()), this, SLOT(mnuShowDesigner()));

        m_actions["python_global_runmode"] = new QAction(QIcon(":/application/icons/pythonDebug.png"), tr("Run python code in debug mode"), this);
        m_actions["python_global_runmode"]->setToolTip(tr("set whether internal python code should be executed in debug mode"));
        m_actions["python_global_runmode"]->setCheckable(true);
        if (pyEngine)
        {
            m_actions["python_global_runmode"]->setChecked(pyEngine->execInternalCodeByDebugger());
        }
        connect(m_actions["python_global_runmode"], SIGNAL(triggered(bool)), this, SLOT(mnuToogleExecPyCodeByDebugger(bool)));

        m_actions["python_stopAction"] = new QAction(QIcon(":/script/icons/stopScript.png"), tr("stop"), this);
        m_actions["python_stopAction"]->setShortcut(tr("Shift+F10"));
        m_actions["python_stopAction"]->setShortcutContext(Qt::WidgetWithChildrenShortcut);
        connect(m_actions["python_stopAction"], SIGNAL(triggered()), this, SLOT(mnuScriptStop()));

        m_actions["python_continueAction"] = new QAction(QIcon(":/script/icons/continue.png"), tr("continue"), this);
        m_actions["python_continueAction"]->setShortcut(tr("F6"));
        m_actions["python_continueAction"]->setShortcutContext(Qt::WidgetWithChildrenShortcut);
        connect(m_actions["python_continueAction"], SIGNAL(triggered()), this, SLOT(mnuScriptContinue()));

        m_actions["python_stepAction"] = new QAction(QIcon(":/script/icons/step.png"), tr("step"), this);
        m_actions["python_stepAction"]->setShortcut(tr("F11"));
        m_actions["python_stepAction"]->setShortcutContext(Qt::WidgetWithChildrenShortcut);
        connect(m_actions["python_stepAction"], SIGNAL(triggered()), this, SLOT(mnuScriptStep()));

        m_actions["python_stepOverAction"] = new QAction(QIcon(":/script/icons/stepOver.png"), tr("step over"), this);
        m_actions["python_stepOverAction"]->setShortcut(tr("F10"));
        m_actions["python_stepOverAction"]->setShortcutContext(Qt::WidgetWithChildrenShortcut);
        connect(m_actions["python_stepOverAction"], SIGNAL(triggered()), this, SLOT(mnuScriptStepOver()));

        m_actions["python_stepOutAction"] = new QAction(QIcon(":/script/icons/stepOut.png"), tr("step out"), this);
        m_actions["python_stepOutAction"]->setShortcut(tr("Shift+F11"));
        m_actions["python_stepOutAction"]->setShortcutContext(Qt::WidgetWithChildrenShortcut);
        connect(m_actions["python_stepOutAction"], SIGNAL(triggered()), this, SLOT(mnuScriptStepOut()));


        m_actions["python_reloadModules"] = new QAction(QIcon(":/application/icons/reload.png"), tr("Reload modules..."), this);
        connect(m_actions["python_reloadModules"], SIGNAL(triggered()), this, SLOT(mnuPyReloadModules()));
    }
}

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

    m_toolToolBar = addToolBar(tr("Tools"));
    m_toolToolBar->setObjectName("toolbarTools");
    if (uOrg->hasFeature(featDeveloper))
    {
        m_toolToolBar->addAction(m_actions["open_designer"]);
    }

    m_aboutToolBar = addToolBar(tr("About"));
    m_aboutToolBar->setObjectName("toolbarAbout");

    m_aboutToolBar->addAction(m_actions["open_assistant"]);

    if (uOrg->hasFeature(featDeveloper))
    {
        m_pythonToolBar = addToolBar(tr("Python"));
        m_pythonToolBar->setObjectName("toolbarPython");
        m_pythonToolBar->addAction(m_actions["python_global_runmode"]);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::createMenus()
{
    m_pMenuFile = menuBar()->addMenu(tr("File"));
    m_pMenuFile->addAction(m_appFileNew);
    m_pMenuFile->addAction(m_appFileOpen);

    ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();
    if (uOrg->hasFeature(featUserManag))
    {
        m_pMenuFile->addAction(m_actions["properties"]);
        m_pMenuFile->addAction(m_actions["usermanagement"]);
    }
    m_pMenuFile->addAction(m_actions["show_loaded_plugins"]);
    m_pMenuFile->addSeparator();
    m_pMenuFile->addAction(m_actions["exit"]);

	m_pMenuView = menuBar()->addMenu(tr("View"));
	QMenu *dockWidgets = createPopupMenu();
    if (dockWidgets)
    {
        dockWidgets->menuAction()->setIcon(QIcon(":/application/icons/preferences-general.png"));
	    dockWidgets->menuAction()->setText(tr("Toolboxes"));
	    m_pMenuView->addMenu(dockWidgets);
    }

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
        m_pMenuPython->addSeparator();
        m_pMenuPython->addAction(m_actions["python_reloadModules"]);
    }

    m_pMenuHelp = menuBar()->addMenu(tr("Help"));
    m_pMenuHelp->addAction(m_actions["open_assistant"]);
    m_pMenuHelp->addAction(m_aboutQt);
    m_pMenuHelp->addAction(m_aboutQitom);
//    m_pMenuHelp->addAction(m_actions["show_loaded_plugins"]);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! initializes status bar
void MainWindow::createStatusBar()
{
    m_statusLblCurrentDir = new QLabel("cd: ");
    statusBar()->addPermanentWidget(m_statusLblCurrentDir);

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
        statusBar()->showMessage(tr("python is being executed"));

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
        statusBar()->showMessage(tr("python is being executed"));

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

        if (statusBar()->currentMessage() == tr("python is being executed"))
        {
            statusBar()->clearMessage();
        }

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
        QMetaObject::invokeMethod(sew,"newScript");
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
    fileName = QFileDialog::getOpenFileName(this, tr("open file"), QDir::currentPath(), filter, &selectedFilter); //tr("python (*.py);;itom data collection (*.idc);;images (*.rpm *.bmp *.png);;matlab (*.mat);;itom files(*.py *.idc *.rpm *.bmp *.png *.mat);;all files (*.*)"));

    QFileInfo info(fileName);

    if (fileName.isEmpty()) return;

    QDir::setCurrent(QFileInfo(fileName).path());
    IOHelper::openGeneralFile(fileName, false, true, this);
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuShowAssistant()
{
    if (this->m_pHelpSystem == NULL)
    {
        m_pHelpSystem = HelpSystem::getInstance();
        m_pHelpSystem->rebuildHelpIfNotUpToDate();
    }

    ProcessOrganizer *po = qobject_cast<ProcessOrganizer*>(AppManagement::getProcessOrganizer());
    if (po)
    {
        bool existingProcess = false;
        QProcess *process = po->getProcess("assistant", true, existingProcess, true);

        if (existingProcess && process->state() == QProcess::Running)
        {
            //assistant is already loaded. try to activate it by sending the activateIdentifier command without arguments (try-and-error to find this way to activate it)
            QByteArray ba;
            ba.append("activateIdentifier \n");
            process->write(ba);
        }
        else
        {
			QString collectionFile = m_pHelpSystem->getHelpCollectionAbsFileName();            
			QStringList args;
			QFileInfo collectionFileInfo(collectionFile);

			if (collectionFileInfo.exists() == false)
			{
				m_pHelpSystem->rebuildHelpIfNotUpToDate();
				collectionFile = m_pHelpSystem->getHelpCollectionAbsFileName();
			}
			
            args << QLatin1String("-collectionFile");
            args << QLatin1String(collectionFile.toAscii().data());
            args << QLatin1String("-enableRemoteControl");

            QString app = ProcessOrganizer::getAbsQtToolPath( "assistant" );

            process->start(app, args);

            connect(process, SIGNAL(error(QProcess::ProcessError)), this, SLOT(helpAssistantError(QProcess::ProcessError)));
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuAboutQitom()
{
    QMap<QString, QString> versionList = getItomVersionMap();

    DialogAboutQItom *dlgAbout = new DialogAboutQItom(versionList);
    dlgAbout->exec();
    if (dlgAbout->result() == QDialog::Accepted)
    {

    }

    delete dlgAbout;
    dlgAbout = NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::helpAssistantError (QProcess::ProcessError /*error*/)
{
    QMessageBox msgBox(this);
    msgBox.setText("The help assistant could not be started.");
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

    delete dlg;
    dlg = NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuShowUserManagement()
{
    DialogUserManagement *dlg = new DialogUserManagement();
    dlg->exec();
    if (dlg->result() == QDialog::Accepted)
    {

    }

    delete dlg;
    dlg = NULL;
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
void MainWindow::pythonAddToolbarButton(QString toolbarName, QString buttonName, QString buttonIconFilename, QString pythonCode)
{
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

        //check if this action already exists, if so delete it first
        foreach(action, (*it)->actions())
        {
            if (action->data() == buttonName)
            {
                (*it)->removeAction(action);
                DELETE_AND_SET_NULL(action);
                break;
            }
        }
    }

    //check icon
    QDir basePath;
    bool iconFound = false;
    int i = 0;
    QIcon icon(buttonIconFilename);

    if (buttonIconFilename != "" && icon.isNull())
    {
        while(!iconFound && i < 1000)
        {
            switch(i)
            {
            case 0:
                basePath = QDir::current();
                break;
            case 1:
                basePath = QCoreApplication::applicationDirPath();
                break;
            case 2:
                basePath = QCoreApplication::applicationDirPath();
                basePath.cd("Qitom");
                break;
            default:
                i = 1000;
                break;
            }

            i++;

            if (basePath.exists())
            {
                if (basePath.exists(buttonIconFilename))
                {
                    icon = QIcon(basePath.absoluteFilePath(buttonIconFilename));
                    iconFound = true;
                }
            }
        }
    }

    if (icon.isNull() || icon.availableSizes().size() == 0)
    {
        icon = QIcon("");
    }

    action = new QAction(icon, buttonName, toolbar);
    action->setData(buttonName);
    action->setToolTip(buttonName);
    connect(action, SIGNAL(triggered()), m_userDefinedSignalMapper, SLOT(map()));
    m_userDefinedSignalMapper->setMapping(action, pythonCode);
    toolbar->addAction(action);
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::pythonRemoveToolbarButton(QString toolbarName, QString buttonName)
{
    QMap<QString, QToolBar*>::iterator it = m_userDefinedToolBars.find(toolbarName);
    QAction* tempAction;

    if (it != m_userDefinedToolBars.end())
    {
        foreach(tempAction, (*it)->actions())
        {
            if (tempAction->data() == buttonName)
            {
                (*it)->removeAction(tempAction);
                DELETE_AND_SET_NULL(tempAction);
                break;
            }
        }

        if ((*it)->actions().size() == 0) //remove this toolbar
        {
            removeToolBar(*it);
            m_userDefinedToolBars.remove(it.key());
        }
    }
    else
    {
        QMessageBox msgBox;
        msgBox.setText("The toolbar '" + toolbarName + "' could not be found");
        msgBox.exec();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::pythonAddMenuElement(int typeID, QString key, QString name, QString code, QString buttonIconFilename)
{
    //key is a slash-splitted value: e.g. rootKey/parentKey/nextParentKey/.../myKey
    QStringList keys = key.split("/");
    QString tempKey = "";
    QAction *act;
    QMenu *parentMnu = NULL;
    //QMenu *mnu;
    RetVal retValue(retOk);
    QList<QAction*> actionList;

    //check icon
    QIcon icon(buttonIconFilename);
    QDir basePath;
    bool iconFound = false;
    int i = 0;

    if (buttonIconFilename != "" && icon.isNull())
    {
        while(!iconFound && i < 1000)
        {
            switch(i)
            {
            case 0:
                basePath = QDir::current();
                break;
            case 1:
                basePath = QCoreApplication::applicationDirPath();
                break;
            case 2:
                basePath = QCoreApplication::applicationDirPath();
                basePath.cd("Qitom");
                break;
            default:
                i = 1000;
                break;
            }

            i++;

            if (basePath.exists())
            {
                if (basePath.exists(buttonIconFilename))
                {
                    icon = QIcon(basePath.absoluteFilePath(buttonIconFilename));
                    iconFound = true;
                }
            }
        }
    }

    if (icon.isNull() || icon.availableSizes().size() == 0)
    {
        icon = QIcon("");
    }

    //check root element
    if (keys.size() >= 1)
    {
        if (keys.size() == 1) //single element, must be type MENU
        {
            if (typeID != 2)
            {
                retValue += ito::RetVal(ito::retError,0,tr("one single menu element must be of type MENU [2]").toAscii().data());
            }
        }

        tempKey = keys[0];
        keys.pop_front();
        if (!retValue.containsError())
        {
            if (m_userDefinedRootMenus.contains(tempKey))
            {
                parentMnu = m_userDefinedRootMenus[tempKey];
            }
            else
            {
                if (keys.size() == 0)
                {
                    parentMnu = menuBar()->addMenu(name);
                }
                else
                {
                    parentMnu = menuBar()->addMenu(tempKey);
                }
                m_userDefinedRootMenus[tempKey] = parentMnu;
            }
        }
    }
    else
    {
        retValue += ito::RetVal(ito::retError,0,tr("no menu element is indicated").toAscii().data());
    }

    //check further elements
    while(keys.size() > 0 && !retValue.containsError())
    {
        actionList = parentMnu->actions();
        tempKey = keys[0];
        keys.pop_front();
        act = NULL;

        //search for tempKey in actionList
        foreach(QAction* a, actionList)
        {
            if (a->data().toString() == tempKey)
            {
                act = a;
                break;
            }
        }

        if (act) //element already exists
        {
            if (keys.size() > 0) //not the last element
            {
                if (act->menu() == NULL) //item is no menu, but has to be a menu
                {
                    retValue += RetVal::format(retError,0,tr("The menu item '%s' does already exist but is no menu type").toAscii().data(), act->iconText().toAscii().data());
                }
                else
                {
                    parentMnu = act->menu();
                }
            }
            else
            {
                retValue += RetVal(retError,0,tr("menu item already exists.").toAscii().data());
            }
        }
        else //element has to be created
        {
            if (keys.size() > 0) //not the last element, create menu
            {      
                parentMnu = parentMnu->addMenu(tempKey);
                parentMnu->menuAction()->setIconText(tempKey);
                parentMnu->menuAction()->setData(QVariant(tempKey));
            }
            else
            {
                switch(typeID)
                {
                case 0: //BUTTON
                    act = parentMnu->addAction(icon,name);
                    act->setIconText(name);
                    act->setData(QVariant(tempKey));
                    connect(act, SIGNAL(triggered()), m_userDefinedSignalMapper, SLOT(map()));
                    m_userDefinedSignalMapper->setMapping(act, code);
                    break;
                case 1: //SEPARATOR
                    act = parentMnu->addSeparator();
                    act->setIconText(name);
                    act->setData(QVariant(tempKey));
                    break;
                case 2: //MENU
                    parentMnu = parentMnu->addMenu(icon,name);
                    parentMnu->menuAction()->setIconText(name);
                    parentMnu->menuAction()->setData(QVariant(tempKey));

                    /*act = parentMnu->addAction(icon,name);
                    act->setIconText(name);
                    act->setData(QVariant(tempKey));
                    act->setMenu(new QMenu(name));
                    parentMnu = act->menu();*/
                    break;
                default:
                    retValue += RetVal(retError,0,tr("Invalid typeID.").toAscii().data());
                    break;
                }
            }
        }
    }

    if (retValue.containsError())
    {
        QMessageBox::critical(this, tr("Add menu element"), retValue.errorMessage());
    }
    else if (retValue.containsWarning())
    {
        QMessageBox::warning(this, tr("Add menu element"), retValue.errorMessage());
    }
    
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::pythonRemoveMenuElement(QString key)
{
    QStringList keys = key.split("/");
    QString tempKey;
    QMenu *parentMenu = NULL;
    QAction *actToDelete = NULL;
    QList<QAction*> actionList;
    QMap<QString, QMenu*>::iterator it;

    if (keys.size() == 1)
    {
        tempKey = keys[0];
        keys.pop_front();
        it = m_userDefinedRootMenus.find(tempKey);
        if (it != m_userDefinedRootMenus.end())
        {
            (*it)->deleteLater();
            it = m_userDefinedRootMenus.erase(it);
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

        while(keys.size() > 0 && parentMenu)
        {
            tempKey = keys[0];
            keys.pop_front();
            actToDelete = NULL;

            actionList = parentMenu->actions();
            foreach(QAction *a, actionList)
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
            actToDelete->deleteLater();
        }
        else
        {
            QMessageBox::warning(this, tr("Remove menu element"), QString(tr("A user-defined menu with the key sequence '%1' could not be found")).arg(key));
        }
    }

}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::pythonRunSelection(QString selectionText)
{
    m_console->pythonRunSelection(selectionText);
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::currentDirectoryChanged()
{
    QString cd = QDir::cleanPath(QDir::currentPath());
    if (m_statusLblCurrentDir)
    {
        m_statusLblCurrentDir->setText(tr("Current Directory: %1").arg(cd));
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
        msgBox.setText("Python is not available. This action cannot be executed.");
        msgBox.exec();
    }
    else if (pythonCode == "")
    {
        QMessageBox msgBox;
        msgBox.setText("there is no python code associated to this action.");
        msgBox.exec();
    }
    else
    {
        QByteArray ba(pythonCode.toAscii());
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
    ProcessOrganizer *po = qobject_cast<ProcessOrganizer*>(AppManagement::getProcessOrganizer());
    if (po)
    {
        bool existingProcess = false;
        QProcess *process = po->getProcess("designer", true, existingProcess, false);

        if (existingProcess && process->state() == QProcess::Running)
        {
            //assistant is already loaded. try to activate it by sending the activateIdentifier command without arguments (try-and-error to find this way to activate it)
            QByteArray ba;
            ba.append("activateIdentifier \n");
            process->write(ba);
        }
        else
        {
            QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
            QString appPath = QDir::cleanPath(QCoreApplication::applicationDirPath());
            env.insert("QT_PLUGIN_PATH", appPath);
            process->setProcessEnvironment(env);

            connect(process, SIGNAL(error(QProcess::ProcessError)), this, SLOT(designerError(QProcess::ProcessError)));

            QString app = ProcessOrganizer::getAbsQtToolPath( "designer" );
            process->start(app, QStringList());
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::designerError (QProcess::ProcessError /*error*/)
{
    QMessageBox msgBox(this);
    msgBox.setText("The UI designer (Qt designer) could not be started.");
    msgBox.exec();
}

//----------------------------------------------------------------------------------------------------------------------------------
void MainWindow::mnuToogleExecPyCodeByDebugger(bool checked)
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
        emit(pythonDebugCommand(ito::pyDbgQuit));
        raise();
    }
    else if (PythonEngine::getInstance())
    {
        PythonEngine::getInstance()->pythonInterruptExecution();
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
            m_infoBoxWidgets[winKey] = QWeakPointer<WidgetInfoBox>(w);
        }
    }
    else
    {
        w->setInfoText(text);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
} //end namespace ito

