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
*********************************************************************** *///#ifndef SCRIPTTABWIDGET_H
//#define SCRIPTTABWIDGET_H
//
//#include "scriptEditorWidget.h"
//#include "../common/sharedStructures.h"
//
//#include "QTabWidgetItom.h"
//
//#include <QWidget>
//#include <QVBoxLayout>
//#include <qstring.h>
//#include <qaction.h>
//
//class ScriptTabWidget : public QWidget
//{
//    Q_OBJECT
//
//public:
//    ScriptTabWidget(QWidget* parent = NULL);
//    ~ScriptTabWidget() {};
//
//    RetVal closeAndDeleteScriptEditorTab( ScriptEditorWidget* seWidget);
//
//    RetVal addEditor(ScriptEditorWidget* seWidget);
//    RetVal appendEditor( ScriptEditorWidget* editorWidget); //!< appends widget, without creating it (for drag&drop, (un)-docking...)
//    RetVal removeEditor( ScriptEditorWidget* editorWidget); //!< removes widget, without deleting it (for drag&drop, (un)-docking...)
//
//    inline void setDocked(bool docked) { this->docked = docked; };
//
//    QStringList getModifiedFileNames(bool ignoreUnsavedFiles = false);
//    
//    
//
//protected:
//
//
//private:
//
//    RetVal init();
//
//    void contextMenuEvent (QContextMenuEvent * event);
//
//    QTabWidgetItom* tab;
//    //RetVal addNewEditor(QString filename); //!< open editor and open filename
//    //RetVal addNewEditor(); //!< empty editor
//
//    bool docked;
//
//    std::map<QString,QAction*> menuActions;
//    QMenu *contextMenu;
//
//    int actTabIndex;
//
//    
//
//signals:
//    void scriptTabCloseRequest( ScriptEditorWidget* widget );
//    void scriptTabDockTab (ScriptEditorWidget* widget);
//    void scriptTabUndockTab (ScriptEditorWidget* widget);
//
//    void scriptNewRequest(ScriptTabWidget* tabWidget = NULL);
//    void scriptOpenRequest(ScriptTabWidget* tabWidget = NULL);
//    void scriptSaveAllScripts();
//
//public slots:
//    void tabCloseRequested (int index);
//    void tabCurrentChanged (int index);
//    //void scriptEditorRemove ( ScriptEditorWidget* widget );
//
//    void scriptModificationChanged (bool change);
////    void changeTabStyle(bool status);
////    void setTabAttribute(int attribute);
////    void setPosition(bool b);
////    void getCurrentPosition(int pos);
//
//private slots:
//    void menuTabMoveLeft();
//    void menuTabMoveRight();
//    void menuTabMoveFirst();
//    void menuTabMoveLast();
//    void menuCloseTab();
//    void menuCloseOthers();
//    void menuCloseAll();
//    void menuNewFile();
//    void menuOpenFile();
//    void menuSaveFile();
//    void menuSaveFileAs();
//    void menuSaveAllFiles();
//
//    void menuRunScript();
//    void menuDebugScript();
//    void menuStopScript();
//
//    void menuUndockTab();
//    void menuDockTab();
//
//    RetVal preShowContextMenu();
//
//
//
//};
//
//#endif