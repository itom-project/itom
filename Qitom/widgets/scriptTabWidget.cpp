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
*********************************************************************** *///
//#include "scriptTabWidget.h"
//
//ScriptTabWidget::ScriptTabWidget( QWidget *parent ) : QWidget(parent), actTabIndex(-1)
//{
//    tab = new QTabWidgetItom();
//
//    //!< tab-settings
//    tab->setElideMode(Qt::ElideMiddle);
//    tab->setTabShape(QTabWidget::Rounded);
//    tab->setTabsClosable(true);
//    tab->setMovable(true);
//    tab->setTabPosition(QTabWidget::South);
//
//
//    QVBoxLayout *layout = new QVBoxLayout();
//    layout->addWidget(tab);
//    layout->setContentsMargins(2,2,2,2);
//
//    setLayout(layout);
//    setMinimumWidth(500);
//
//    init();
//
//    //addNewEditor();
//    //addNewEditor();
//}
//
//RetVal ScriptTabWidget::init()
//{
//    menuActions.clear();
//
//    contextMenu = new QMenu();
//    menuActions["moveLeft"] = contextMenu->addAction(QIcon("icons/1leftarrow.png"),tr("move left"),this,SLOT(menuTabMoveLeft()));
//    menuActions["moveRight"] = contextMenu->addAction(QIcon("icons/1rightarrow.png"),tr("move right"),this,SLOT(menuTabMoveRight()));
//    menuActions["moveFirst"] = contextMenu->addAction(QIcon("icons/2leftarrow.png"),tr("move first"),this,SLOT(menuTabMoveFirst()));
//    menuActions["moveLast"] = contextMenu->addAction(QIcon("icons/2rightarrow.png"),tr("move last"),this,SLOT(menuTabMoveLast()));
//    contextMenu->addSeparator();
//    menuActions["close"] = contextMenu->addAction(QIcon("icons/close.png"),tr("close"),this,SLOT(menuCloseTab()));
//    menuActions["closeOthers"] = contextMenu->addAction(QIcon("icons/close.png"),tr("close all but this"),this,SLOT(menuCloseOthers()));
//    menuActions["closeAll"] = contextMenu->addAction(QIcon("icons/close.png"),tr("close all"),this,SLOT(menuCloseAll()));
//    contextMenu->addSeparator();
//    menuActions["newFile"] = contextMenu->addAction(QIcon("icons/new.png"),tr("new file"),this,SLOT(menuNewFile()));
//    menuActions["openFile"] = contextMenu->addAction(QIcon("icons/open.png"),tr("open file"),this,SLOT(menuOpenFile()));
//    menuActions["saveFile"] = contextMenu->addAction(QIcon("icons/fileSave.png"),tr("save file"),this,SLOT(menuSaveFile()));
//    menuActions["saveFileAs"] = contextMenu->addAction(QIcon("icons/fileSaveAs.png"),tr("save file as..."),this,SLOT(menuSaveFileAs()));
//    menuActions["saveAllFiles"] = contextMenu->addAction(QIcon("icons/fileSaveAll.png"),tr("save all files"),this,SLOT(menuSaveAllFiles()));
//    contextMenu->addSeparator();
//    menuActions["dockTab"] = contextMenu->addAction(QIcon("icons/attribute.png"), tr("dock tab"),this, SLOT(menuDockTab()));
//    menuActions["undockTab"] = contextMenu->addAction(QIcon("icons/attributes.png"), tr("undock tab"), this, SLOT(menuUndockTab()));
//    
//    connect(contextMenu, SIGNAL(aboutToShow()), this, SLOT(preShowContextMenu()));
//
//    connect( tab, SIGNAL(currentChanged(int)), this, SLOT(tabCurrentChanged(int)));
//    connect( tab, SIGNAL(tabCloseRequested(int)), this, SLOT(tabCloseRequested(int)));
//
//    return RetVal(retOk);
//}
//
//
//
////RetVal ScriptTabWidget::addNewEditor(QString filename)
////{
////    QFile file;
////
////    if(file.exists(filename))
////    {
////        ScriptEditorWidget* scriptEditor = new ScriptEditorWidget(filename, this);
////        tab->addTab(scriptEditor, scriptEditor->getFilename());
////        connect(scriptEditor, SIGNAL(modificationChanged(bool)), this, SLOT(scriptModificationChanged(bool)));
////
////        scriptModificationChanged(false);
////
////        return RetVal(retOk);
////    }
////    else
////    {
////        return RetVal(retError);
////    }
////
////}
////
////RetVal ScriptTabWidget::addNewEditor()
////{
////    ScriptEditorWidget* scriptEditor = new ScriptEditorWidget(this);
////    //editors.push_back(scriptEditor);
////    tab->addTab(scriptEditor, scriptEditor->getFilename());
////    connect(scriptEditor, SIGNAL(modificationChanged(bool)), this, SLOT(scriptModificationChanged(bool)));
////
////    scriptModificationChanged(false);
////
////    return RetVal(retOk);
////
////}
//
//void ScriptTabWidget::contextMenuEvent(QContextMenuEvent * event)
//{
//    QRect tabRectangle;
//    event->accept();
//    
//    actTabIndex = -1;
//    
//    for(int i=0;i<tab->count();i++)
//    {
//        tabRectangle = tab->getTabBar()->tabRect(i);
//
//        if(tabRectangle.contains(event->pos()-tab->pos()-tab->getTabBar()->pos()))
//        {
//            actTabIndex = i;
//        }
//
//    }
//
//    contextMenu->exec(event->globalPos());
//
//}
//
//RetVal ScriptTabWidget::preShowContextMenu()
//{
//    menuActions["moveLeft"]->setEnabled(actTabIndex > 0);
//    menuActions["moveRight"]->setEnabled(actTabIndex > -1 && actTabIndex < tab->count()-1);
//    menuActions["moveFirst"]->setEnabled(actTabIndex > 0);
//    menuActions["moveLast"]->setEnabled(actTabIndex > -1 && actTabIndex < tab->count()-1);
//
//    menuActions["close"]->setEnabled(actTabIndex > -1);
//    menuActions["closeAll"]->setEnabled(actTabIndex > -1);
//    menuActions["closeOthers"]->setEnabled(actTabIndex > -1 && tab->count()>1);
//
//    ScriptEditorWidget *sew = NULL;
//
//    if(actTabIndex > -1)
//    {
//        sew = static_cast<ScriptEditorWidget *>(tab->widget(actTabIndex));
//    }
//
//    menuActions["newFile"]->setEnabled(true);
//    menuActions["saveFile"]->setEnabled(sew != NULL && sew->isModified());
//    menuActions["saveFileAs"]->setEnabled(sew != NULL);
//    menuActions["saveAllFiles"]->setEnabled(false);
//
//    for(int i=0;i<tab->count();i++)
//    {
//        if(static_cast<ScriptEditorWidget *>(tab->widget(i))->isModified())
//        {
//            menuActions["saveAllFiles"]->setEnabled(true);
//        }
//    }
//
//    menuActions["dockTab"]->setVisible(!docked);
//    menuActions["undockTab"]->setVisible(docked);
//
//    return RetVal(retOk);
//}
//
//void ScriptTabWidget::menuTabMoveLeft() //!< ok
//{
//    if(actTabIndex > 0) 
//    {
//        tab->getTabBar()->moveTab(actTabIndex,actTabIndex-1);
//    }
//}
//
//void ScriptTabWidget::menuTabMoveRight() //!< ok
//{
//    if(actTabIndex < tab->count()-1) 
//    {
//        tab->getTabBar()->moveTab(actTabIndex,actTabIndex+1);
//    }
//}
//
//void ScriptTabWidget::menuTabMoveFirst() //!< ok
//{
//    if(actTabIndex >= 0) 
//    {
//        tab->getTabBar()->moveTab(actTabIndex,0);
//    }
//}
//
//void ScriptTabWidget::menuTabMoveLast() //!< ok
//{
//    if(actTabIndex <= tab->count()-1) 
//    {
//        tab->getTabBar()->moveTab(actTabIndex,tab->count()-1);
//    }
//}
//void ScriptTabWidget::menuCloseTab() //!< ok
//{
//    if(actTabIndex >= 0) 
//    {
//        tabCloseRequested (actTabIndex);
//    }
//}
//
//void ScriptTabWidget::menuCloseOthers() //!< ok
//{
//    for(int i = tab->count()-1; i >= 0 ; i--)
//    {
//        if(i != actTabIndex)
//        {
//            tabCloseRequested (i);
//        }
//    }
//}
//
//void ScriptTabWidget::menuCloseAll() //!< ok
//{
//    for(int i = tab->count()-1; i >= 0 ; i--)
//    {
//        tabCloseRequested (i);
//    }
//}
//
//void ScriptTabWidget::menuNewFile() //!< ok
//{
//    /*addNewEditor();
//    tab->setCurrentIndex(tab->count()-1);*/
//    emit(scriptNewRequest(this));
//}
//
//void ScriptTabWidget::menuOpenFile() //!< ok
//{
//    /*addNewEditor();
//    tab->setCurrentIndex(tab->count()-1);*/
//    emit(scriptOpenRequest(this));
//}
//
//void ScriptTabWidget::menuSaveFile() //!< ok
//{
//    ScriptEditorWidget *sew = NULL;
//
//    if(actTabIndex > -1)
//    {
//        sew = static_cast<ScriptEditorWidget *>(tab->widget(actTabIndex));
//        sew->saveFile(false, "");
//    }
//}
//
//void ScriptTabWidget::menuSaveFileAs() //!< ok
//{
//    ScriptEditorWidget *sew = NULL;
//
//    if(actTabIndex > -1)
//    {
//        sew = static_cast<ScriptEditorWidget *>(tab->widget(actTabIndex));
//        sew->saveFile(true, "");
//    }
//}
//
//void ScriptTabWidget::menuSaveAllFiles() //!< ok
//{
//    /*ScriptEditorWidget *sew = NULL;
//
//    for(int i = 0; i < tab->count() ; i++)
//    {
//        sew = static_cast<ScriptEditorWidget *>(tab->widget(i));
//        sew->menuSave();
//    }
//*/
//    emit(scriptSaveAllScripts());
//}
//
//void ScriptTabWidget::menuRunScript()
//{
//}
//
//void ScriptTabWidget::menuDebugScript()
//{
//}
//
//void ScriptTabWidget::menuStopScript()
//{
//}
//
//void ScriptTabWidget::menuDockTab() //!< ok
//{
//    ScriptEditorWidget *sew = NULL;
//
//    if(actTabIndex > -1)
//    {
//        sew = static_cast<ScriptEditorWidget *>(tab->widget(actTabIndex));
//        emit(scriptTabDockTab(sew) );
//    }
//}
//
//void ScriptTabWidget::menuUndockTab() //!< ok
//{
//    ScriptEditorWidget *sew = NULL;
//
//    if(actTabIndex > -1)
//    {
//        sew = static_cast<ScriptEditorWidget *>(tab->widget(actTabIndex));
//        emit(scriptTabUndockTab(sew) );
//    }
//}
//
//void ScriptTabWidget::tabCloseRequested (int index)
//{
//    ScriptEditorWidget *sew = static_cast<ScriptEditorWidget *>(tab->widget(index));
//
//    emit(scriptTabCloseRequest(sew));
//}
//
//void ScriptTabWidget::tabCurrentChanged (int index)
//{
//    QWidget *widget = tab->widget(index);
//    ScriptEditorWidget *scriptEditorWidget = static_cast<ScriptEditorWidget *>(widget);
//}
//
//void ScriptTabWidget::scriptModificationChanged (bool change)
//{
//    QString title;
//    QFileInfo filename;
//    ScriptEditorWidget* sew;
//    for(int i=0; i<tab->count();i++)
//    {
//        sew = static_cast<ScriptEditorWidget*>(tab->widget(i));
//
//        filename.setFile(sew->getFilename());
//
//        title = filename.fileName();
//        if(title=="") title=tr("unknown");
//
//        if(sew->isModified())
//        {
//            title.append("*");
//        }
//
//        tab->setTabText(i,title );
//        tab->setTabToolTip(i, filename.absoluteFilePath());
//    }
//}
//
//RetVal ScriptTabWidget::closeAndDeleteScriptEditorTab( ScriptEditorWidget* seWidget)
//{
//    for( int i = 0 ; i < tab->count() ; i++)
//    {
//        if(tab->widget(i) == seWidget)
//        {
//            disconnect(seWidget, SIGNAL(modificationChanged(bool)), this, SLOT(scriptModificationChanged(bool)));
//            tab->removeTab(i);
//            delete seWidget;
//            return RetVal(retOk);
//        }
//    }
//    return RetVal(retError);
//}
//
//RetVal ScriptTabWidget::appendEditor( ScriptEditorWidget* editorWidget)
//{
//    QString name = editorWidget->getFilename();
//    if(name == "") name = tr("unknown");
//    tab->addTab(editorWidget, name);
//    tab->setCurrentIndex(tab->count()-1);
//
//    connect(editorWidget, SIGNAL(modificationChanged(bool)), this, SLOT(scriptModificationChanged(bool)));
//
//    scriptModificationChanged(editorWidget->isModified());
//
//    return RetVal(retOk);
//}
//
//
//RetVal ScriptTabWidget::removeEditor(ScriptEditorWidget* editorWidget)
//{
//    for( int i = 0 ; i < tab->count() ; i++)
//    {
//        if(static_cast<ScriptEditorWidget*>(tab->widget(i)) == editorWidget)
//        {
//            disconnect(editorWidget, SIGNAL(modificationChanged(bool)), this, SLOT(scriptModificationChanged(bool)));
//            tab->removeTab(i);
//
//            return RetVal(retOk);
//        }
//    }
//    return RetVal(retError);
//}