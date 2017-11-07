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

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "workspaceDockWidget.h"
#include "callStackDockWidget.h"
#include "consoleWidget.h"
#include "AIManagerWidget.h"
#include "fileSystemDockWidget.h"
#include "breakPointDockWidget.h"
#include "helpDockWidget.h"
#include "lastCommandDockWidget.h"
//#include "pythonMessageDockWidget.h"
#include "userManagement.h"

#include <qtableview.h>
#include <qprocess.h>

#include <qsignalmapper.h>

#include "../organizer/helpSystem.h"

#include <qsharedpointer.h>

class QSignalMapper; //forward declaration

namespace ito {

class WidgetInfoBox; //forward declaration

#ifdef ITOM_USEHELPVIEWER
class HelpViewer; //forward declaration
#endif

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow();
    ~MainWindow();


    
protected:
    void closeEvent(QCloseEvent *event);
    void resizeEvent(QResizeEvent * event);
    void moveEvent (QMoveEvent * event);

    inline bool pythonBusy() const { return m_pythonBusy; }                    /*!<  returns if python is busy (true) */
    inline bool pythonDebugMode() const { return m_pythonDebugMode; }          /*!<  returns if python is in debug mode (true) */
    inline bool pythonInWaitingMode() const { return m_pythonInWaitingMode; }  /*!<  returns if python is in waiting mode (true) \sa m_pythonInWaitingMode */

private:
    void createActions();
    void createMenus();
    void createToolBars();
    void createStatusBar();
    void updateMenus();
    void updatePythonActions();

    void getMenuHandlesRecursively(const QMenu *parent, QSharedPointer<QVector<size_t> > menuHandles);
    QAction* searchActionRecursively(const size_t menuHandle, const QMenu *parent);

    ConsoleWidget *m_console;

    QVBoxLayout *m_contentLayout;
    
    BreakPointDockWidget  *m_breakPointDock;
    LastCommandDockWidget *m_lastCommandDock;
//    PythonMessageDockWidget *m_pythonMessageDock;
    HelpDockWidget        *m_helpDock;
    WorkspaceDockWidget   *m_globalWorkspaceDock;
    WorkspaceDockWidget   *m_localWorkspaceDock;
    CallStackDockWidget   *m_callStackDock;
    FileSystemDockWidget  *m_fileSystemDock;

    AIManagerWidget* m_pAIManagerWidget;

    QSignalMapper *m_lastFilesMapper;       /*!<  Maps signal from the "last opened files" buttons */
    QSignalMapper *m_openScriptsMapper;       /*!<  Maps signal from the opened scripts" actions */

    QToolBar* m_aboutToolBar;
    QToolBar* m_appToolBar;
    QToolBar* m_toolToolBar;
    QToolBar* m_pythonToolBar;

    QMap<QString, QToolBar*> m_userDefinedToolBars;
    QMap<QString, QMenu* > m_userDefinedRootMenus;
    QSignalMapper *m_userDefinedSignalMapper;
    unsigned int m_userDefinedActionCounter;

    QAction *m_appFileNew;
    QAction *m_appFileOpen;
    QAction *m_aboutQt;
    QAction *m_aboutQitom;

    QMap<QString, QAction*> m_actions;

    QMenu *m_pMenuHelp;
    QMenu *m_pMenuFile;
    QMenu *m_plastFilesMenu;
    QMenu *m_pMenuPython;
    QMenu *m_pMenuReloadModule;
    QMenu *m_pMenuView;

    HelpSystem *m_pHelpSystem;

    QLabel *m_pStatusLblCurrentDir; //label for showing current directory
	QLabel *m_pStatusLblPythonBusy; //label for showing the busy status of python (hidden, if python is currently not working)

#ifdef ITOM_USEHELPVIEWER
    QPointer<HelpViewer> m_helpViewer;
#endif

    QRect m_geometryNormalState;

    bool m_pythonBusy;                  /*!<  if true, python is busy right now */
    bool m_pythonDebugMode;             /*!<  if true, python is in debug mode right now */
    bool m_pythonInWaitingMode;         /*!<  if true, python is in debug mode but waiting for next user command (e.g. the debugger waits at a breakpoint) */
    bool m_isFullscreen;

    QMap<QString, QPointer<WidgetInfoBox> > m_infoBoxWidgets;

signals:
    void mainWindowCloseRequest();  /*!<  signal emitted if user would like to close the main window and therefore the entire application */
    void pythonDebugCommand(tPythonDbgCmd cmd); /*!<  will be received by PythonThread, directly */
    void pythonSetAutoReloadSettings(bool enabled, bool checkFile, bool checkCmd, bool checkFct);

public slots:
    void addAbstractDock(AbstractDockWidget* dockWidget, Qt::DockWidgetArea area = Qt::TopDockWidgetArea);
    void removeAbstractDock(AbstractDockWidget* dockWidget);
    void connectPythonMessageBox(QListWidget* pythonMessageBox);

    void pythonStateChanged(tPythonTransitions pyTransition);

    void setStatusText(QString message, int timeout);

    ito::RetVal addToolbarButton(const QString &toolbarName, const QString &buttonName, const QString &buttonIconFilename, const QString &pythonCode, QSharedPointer<size_t> buttonHandle, ItomSharedSemaphore *waitCond = NULL);
    ito::RetVal removeToolbarButton(const QString &toolbarName, const QString &buttonName, QSharedPointer<size_t> buttonHandle, bool showMessage = true, ItomSharedSemaphore *waitCond = NULL);
    ito::RetVal removeToolbarButton(const size_t buttonHandle, bool showMessage = true, ItomSharedSemaphore *waitCond = NULL);

    ito::RetVal addMenuElement(int typeID, const QString &key, const QString &name, const QString &code, const QString &buttonIconFilename, QSharedPointer<size_t> menuHandle, bool showMessage = true, ItomSharedSemaphore *waitCond = NULL);
    ito::RetVal removeMenuElement(const QString &key, QSharedPointer<QVector<size_t> > removedMenuHandles, bool showMessage = true, ItomSharedSemaphore *waitCond = NULL);
    ito::RetVal removeMenuElement(const size_t menuHandle, QSharedPointer<QVector<size_t> > removedMenuHandles, bool showMessage = true, ItomSharedSemaphore *waitCond = NULL);
    void pythonRunSelection(QString selectionText);

    void setCursor(const Qt::CursorShape cursor);
    void resetCursor();

    void currentDirectoryChanged();

    void showInfoMessageLine( QString text, QString winKey = "" );

	void showAssistant(const QString &collectionFile = "");

private slots:
    void mnuAboutQitom();
    void mnuExitApplication();

    void mnuNewScript();
    void mnuOpenFile();
    void mnuShowAssistant();
    void mnuShowScriptReference();
    void mnuShowDesigner();
    void mnuShowProperties();
    void mnuShowUserManagement();
    void mnuToggleExecPyCodeByDebugger(bool checked);

    void mnuScriptStop();
    void mnuScriptContinue();
    void mnuScriptStep();
    void mnuScriptStepOver();
    void mnuScriptStepOut();
    void mnuPyReloadModules();
    void mnuShowLoadedPlugins();
    void mnuPyPipManager();
	void mnuPyTimerManager();

    void mnuPyAutoReloadTriggered(bool checked);

    void helpAssistantError ( QProcess::ProcessError error );
    void designerError ( QProcess::ProcessError error );

    void userDefinedActionTriggered(const QString &pythonCode);

    void pythonAutoReloadChanged(bool enabled, bool checkFile, bool checkCmd, bool checkFct);

    void menuLastFilesAboutToShow();
    void lastFileOpen(const QString &path);
    void openScript(const QString &filename);
    
    void mnuViewAboutToShow();

};

} //end namespace ito

#endif
