/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

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

#include "python/pythonEngineInc.h"
#include "global.h"

#include "scriptEditorOrganizer.h"
#include "../widgets/scriptEditorWidget.h"
#include "../AppManagement.h"

#include <qmessagebox.h>
#include <qmetaobject.h>
#include <qsettings.h>
#include <qfileinfo.h>
#include <qmainwindow.h>
#include <qdir.h>

namespace ito
{
    QDataStream &operator<<(QDataStream &out, const ito::ScriptEditorStorage &obj)
    {
        out << obj.filename << obj.firstVisibleLine << obj.bookmarkLines;
        return out;
    }

    QDataStream &operator>>(QDataStream &in, ito::ScriptEditorStorage &obj)
    {
        in >> obj.filename >> obj.firstVisibleLine >> obj.bookmarkLines;
        return in;
    }


/*!
    \class ScriptEditorOrganizer
    \brief organizes script editors, independent on their appearance (docked or window-style)
*/

//----------------------------------------------------------------------------------------------------------------------------------
//! constructor
/*!
    setups connections to python engine and to get a notification about focus changes.

    \param dockAvailable true if dock functionality is available
*/
ScriptEditorOrganizer::ScriptEditorOrganizer(bool dockAvailable)
{
    m_dockAvailable = dockAvailable;

    widgetFocusChanged(NULL,NULL); //sets active ScriptDockWidget to NULL

    m_scriptStackMutex.lock();
    scriptDockElements.clear();
    m_scriptStackMutex.unlock();

    const PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    connect(this, SIGNAL(pythonRunFile(QString)), pyEngine, SLOT(pythonRunFile(QString)));
    connect(this, SIGNAL(pythonDebugFile(QString)), pyEngine, SLOT(pythonDebugFile(QString)));
    connect(pyEngine, SIGNAL(pythonDebugPositionChanged(QString, int)), this, SLOT(pythonDebugPositionChanged(QString,int)));
    connect(qApp, SIGNAL(focusChanged(QWidget*,QWidget*)), this, SLOT(widgetFocusChanged(QWidget*,QWidget*)));


}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!
    disconnections remaining connections to python engine and deletes remaining ScriptDockWidgets (should no occure)
*/
ScriptEditorOrganizer::~ScriptEditorOrganizer()
{
    disconnect();

    ScriptDockWidget* sew;

    m_scriptStackMutex.lock();
    while(scriptDockElements.size()>0)
    {
        sew = scriptDockElements.last();
        delete sew;
        scriptDockElements.pop_back();
    }

    scriptDockElements.clear();
    m_scriptStackMutex.unlock();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! This function is called to save all the informations about widgets before itom is closed
/*!
*/
void ScriptEditorOrganizer::saveScriptState()
{
    QMainWindow *mainWin = qobject_cast<QMainWindow*>(AppManagement::getMainWindow());

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);

    settings.remove("ScriptEditorOrganizer"); //remove old entries and rebuild it from nothing

    settings.beginGroup("ScriptEditorOrganizer");
    settings.beginWriteArray("scriptWidgets");
    int counter = 0;
    QVariant states;

    foreach(ito::ScriptDockWidget *sdw, scriptDockElements)
    {
        settings.setArrayIndex(counter++);
        
        if (mainWin)
        {
            settings.setValue("dockWidgetArea",mainWin->dockWidgetArea(sdw));
        }
        else
        {
            settings.setValue("dockWidgetArea", Qt::TopDockWidgetArea);
        }
        
        settings.setValue("objectName", sdw->objectName());
        settings.setValue("docked",sdw->docked());
        states = QVariant::fromValue<QList<ito::ScriptEditorStorage> >(sdw->saveScriptState());
        settings.setValue("state", states);
    }

    settings.endArray();

    // Last opened files save
    settings.beginWriteArray("lastScriptWidgets");
    counter = 0;
    foreach (const QString &path, m_recentlyUsedFiles)
    {
        if (path != "")
        {
            settings.setArrayIndex(counter++);
            settings.setValue("path", path);
        }
    }
    settings.endArray();
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! This function is called to get all the saved informations about widgets after itom starts
/*!
*/
RetVal ScriptEditorOrganizer::restoreScriptState()
{
    RetVal retval;
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("ScriptEditorOrganizer");

    bool docked;
    QString objectName;
    QVariant scriptDockState;
    Qt::DockWidgetArea area;
    ScriptDockWidget *sdw;

    // reading last used files
    int counter = settings.beginReadArray("lastScriptWidgets");
    QFileInfo fi;

    for (int i = 0; i < counter; ++i)
    {
        settings.setArrayIndex(i);
        fi.setFile(settings.value("path").toString());
        if (fi.exists())
        {
            m_recentlyUsedFiles.append(QDir::toNativeSeparators(fi.absoluteFilePath()));
        }
    }
    settings.endArray();

    // open script windows
    counter = settings.beginReadArray("scriptWidgets");

    for (int i = 0; i < counter; ++i)
    {
        settings.setArrayIndex(i);
        docked = settings.value("docked", false).toBool();
        objectName = settings.value("objectName").toString();
        area = (Qt::DockWidgetArea)settings.value("dockWidgetArea", Qt::TopDockWidgetArea).toInt();

        if (!docked)
        {
            area = Qt::NoDockWidgetArea;
        }

        scriptDockState = settings.value("state");

        if (scriptDockState.isValid() && scriptDockState.canConvert<QList<ito::ScriptEditorStorage> >())
        {
            QList<ito::ScriptEditorStorage> states = scriptDockState.value<QList<ito::ScriptEditorStorage> >();

            bool valid = false;

            foreach(const ito::ScriptEditorStorage &ses, states)
            {
                QFileInfo info(ses.filename);
                if (info.exists())
                {
                    valid = true;
                    break;
                }
            }

            if (valid)
            {
                sdw = createEmptyScriptDock(docked, area, objectName);
                RetVal ret = sdw->restoreScriptState(states);

                if (ret.containsError())
                {
                    removeScriptDockWidget(sdw);
                }

                retval += ret;
            }
        }
    }

    settings.endArray();

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! This slot is called if a file is saved or stored in any widget
/*!
    This function is used to manage the "last used files" list. It stores all used 
    files in the list and keeps it up to date. If the list is longer than 10 elements, 
    the last ones are deleted.

    \param filename filename of a saved or loaded file is insert into the list at first position
*/
void ScriptEditorOrganizer::fileOpenedOrSaved(const QString &filename)
{
    m_recentlyUsedFiles.prepend(QDir::toNativeSeparators(filename));
    m_recentlyUsedFiles.removeDuplicates();
    int maxNumberLastFiles = 10;
    if (m_recentlyUsedFiles.size() > maxNumberLastFiles) 
    {
        while (m_recentlyUsedFiles.size() > maxNumberLastFiles)
        {
            m_recentlyUsedFiles.removeLast();
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! creates new ScriptDockWidget without any script editor tab.
/*!
    Since there should only be one docked widget, docked will be set to false if there exists already a docked widget.
    Setups connections between the new ScriptDockWidget and this organizer or the python engine.

    \param docked true, if widget should be docked in main window, else false (new on-top window)
    \return reference to new ScriptDockWidget
*/
ScriptDockWidget* ScriptEditorOrganizer::createEmptyScriptDock(bool docked, Qt::DockWidgetArea area /*=Qt::TopDockWidgetArea*/, const QString &objectName /*= QString()*/)
{
    ScriptDockWidget* newWidget;
    

    //QWidget *mainWin = qobject_cast<QWidget*>(AppManagement::getMainWindow());

    docked = docked && m_dockAvailable;
    if (docked && this->getFirstDockedElement() != NULL)
    {
        docked = false;
    }

    newWidget = new ScriptDockWidget(tr("Script Editor"), "", docked, m_dockAvailable, NULL /*mainWin*/); //parent will be set later by addScriptDockWidgetToMainWindow signal

    if (objectName.isNull() || m_usedObjectNames.contains(objectName))
    {
        //generate a new object name
        QString objectNameDraft;
        for (int i = 0; i < 10000; ++i)
        {
            objectNameDraft = QString("scriptEditor%1").arg(i);

            if (m_usedObjectNames.contains(objectNameDraft) == false)
            {
                newWidget->setObjectName(objectNameDraft);
                m_usedObjectNames.insert(objectNameDraft);
                break;
            }
        }
    }
    else
    {
        newWidget->setObjectName(objectName);
        m_usedObjectNames.insert(objectName);
    }
    
    m_scriptStackMutex.lock();
    scriptDockElements.push_front(newWidget);
    m_scriptStackMutex.unlock();

    connect(newWidget,SIGNAL(removeAndDeleteScriptDockWidget(ScriptDockWidget*)),this,SLOT(removeScriptDockWidget(ScriptDockWidget*)));
    connect(newWidget,SIGNAL(dockScriptTab(ScriptDockWidget*,int,bool)),this,SLOT(dockScriptTab(ScriptDockWidget*,int,bool)));
    connect(newWidget,SIGNAL(undockScriptTab(ScriptDockWidget*,int,bool,bool)),this,SLOT(undockScriptTab(ScriptDockWidget*,int,bool,bool)));

    connect(newWidget,SIGNAL(openScriptRequest(QString,ScriptDockWidget*)), this, SLOT(openScriptRequested(QString,ScriptDockWidget*)));

    connect(newWidget, SIGNAL(pythonRunFileRequest(QString)), this, SLOT(pythonRunFileRequested(QString)));
    connect(newWidget, SIGNAL(pythonDebugFileRequest(QString)), this, SLOT(pythonDebugFileRequested(QString)));
    //!< setup signal/slot-connection to python thread
    qRegisterMetaType<ito::tPythonDbgCmd>("tPythonDbgCmd");
    
    const PythonEngine *pyEngine = PythonEngine::getInstance();
    if (pyEngine)
    {
        connect(newWidget, SIGNAL(pythonDebugCommand(tPythonDbgCmd)), pyEngine, SLOT(pythonDebugCommand(tPythonDbgCmd)));
        connect(newWidget, SIGNAL(pythonInterruptExecution()), pyEngine, SLOT(pythonInterruptExecution()));
    }

    if (docked)
    {
        if (area == Qt::NoDockWidgetArea) area = Qt::TopDockWidgetArea;
        emit(addScriptDockWidgetToMainWindow(newWidget, area));
    }
    else
    {
        emit(addScriptDockWidgetToMainWindow(newWidget, Qt::NoDockWidgetArea));
    }

    return newWidget;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by ScriptDockWidget close event method. The given widget should be closed and removed from the scriptDockElements-list
/*!
    Disconnects many connections between the ScriptDockWidget and the ScriptEditorOrganizer or the PythonEngine. Emits signal to equally
    remove the widget from the docking area in main window.

    \param widget ScriptDockWidget which should be closed and removed
*/
void ScriptEditorOrganizer::removeScriptDockWidget(ScriptDockWidget* widget)
{
    widget->disconnect(); //disconnect all connected to 'widget'

    emit(removeScriptDockWidgetFromMainWindow(widget));

    m_usedObjectNames.remove(widget->objectName());

    m_scriptStackMutex.lock();
    scriptDockElements.removeAll(widget);
    m_scriptStackMutex.unlock();

    widget->deleteLater(); //do not directly delete widget instead of using deleteLater. This is important e.g. for closing the windows if number of remaining tabs is zero.
}

//----------------------------------------------------------------------------------------------------------------------------------
//! saves all opened scripts, if changes exist
/*!
    \param askFirst true if user can decide whether to save the script or not
    \param ignoreNewScripts true if scripts which do not have a filename should be ignored
    \return retOk if everything done, else retError (e.g. user cancellation)
*/
RetVal ScriptEditorOrganizer::saveAllScripts(bool askFirst, bool ignoreNewScripts)
{
    RetVal retValue(retOk);
    QList<ScriptDockWidget*>::iterator it;

    if (askFirst)
    {
        QStringList unsavedFileNames;
        QMessageBox msgBox;

        m_scriptStackMutex.lock();
        for (it = scriptDockElements.begin(); it != scriptDockElements.end(); ++it)
        {
            unsavedFileNames.append((*it)->getModifiedFileNames(ignoreNewScripts));
        }
        m_scriptStackMutex.unlock();

        if (unsavedFileNames.size() > 0)
        {
            msgBox.setText(tr("The following files have been changed and should be safed:"));
            msgBox.setInformativeText(unsavedFileNames.join("\n"));
            msgBox.setStandardButtons(QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
            msgBox.setDefaultButton(QMessageBox::Save);
            int ret = msgBox.exec();

            if (ret & QMessageBox::Cancel)
            {
                return RetVal(retError);
            }
            else if (ret & QMessageBox::Discard)
            {
                return RetVal(retOk);
            }
        }
    }

    m_scriptStackMutex.lock();
    for (it = scriptDockElements.begin(); it != scriptDockElements.end(); ++it)
    {
        retValue += (*it)->saveAllScripts(false, ignoreNewScripts);
    }
    m_scriptStackMutex.unlock();

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! tries to close all opened script.
/*!
    long description

    \param saveFirst indicates whether unsaved or new scripts should be saved first.
    \return retOk if all scripts could be closed, else retError
*/
RetVal ScriptEditorOrganizer::closeAllScripts(bool saveFirst)
{
    RetVal retValue(retOk);

    QList<ScriptDockWidget*>::iterator it;

    if (saveFirst)
    {
        retValue += saveAllScripts(true, false);
    }

    if (!retValue.containsError())
    {
        m_scriptStackMutex.lock();
        QList<ScriptDockWidget*> tempList(scriptDockElements); //!< copy since scriptDockElements will change its size during closing
        m_scriptStackMutex.unlock();

        for (it = tempList.begin(); it != tempList.end(); ++it)
        {
            retValue += (*it)->closeAllScripts(false, false);
        }
    }
  
    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns first ScriptDockWidget of the widget-list which is docked. This is also the last activated docked widget.
/*!
    \return docked ScriptDockWidget or NULL, if no such widget exists.
*/
ScriptDockWidget* ScriptEditorOrganizer::getFirstDockedElement()
{
    QList<ScriptDockWidget*>::iterator it;

    QMutexLocker locker(&m_scriptStackMutex);

    for (it = scriptDockElements.begin(); it != scriptDockElements.end(); ++it)
    {
        if ((*it)->docked())
        {
            return *it;
        }
    }

    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns the ScriptDockWidget, which actually has the focus or lastly got the focus.
/*!
    \return Active ScriptDockWidget or NULL, if no ScriptDockWidget is available
*/
ScriptDockWidget* ScriptEditorOrganizer::getActiveDockWidget()
{
    QMutexLocker locker(&m_scriptStackMutex);

    if (scriptDockElements.isEmpty())
    {
        return NULL;
    }
    else
    {
        return scriptDockElements.first();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! 
/*!
    \return ScriptDockWidget
*/
ScriptDockWidget* ScriptEditorOrganizer::getFirstUndockedElement()
{
    QMutexLocker locker(&m_scriptStackMutex);
    foreach(ito::ScriptDockWidget *sdw, scriptDockElements)
    {
        if (sdw->docked() == false)
        {
            return sdw;
        }
    }
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot is connected to signal "focusChanged" of QApplication and indicates every change in the active widget.
/*!
    This slot is evaluated in order to check, whether a ScriptDockWidget has been activated (has got the focus). If so,
    this ScriptDockWidget will be moved on top of the scriptDockElements-list, since the first element should always be the
    active one. Write action to scriptDockElements is protected by scriptStackMutex.

    \param now widget which just got the focus
*/
void ScriptEditorOrganizer::widgetFocusChanged(QWidget* /*old*/, QWidget* now)
{
    if (now != NULL)
    {
        //active widget can also be a sub-widget of ScriptDockWidget, therefore look iteratively for ScriptDockWidget
        QWidget* temp = now;
        ScriptDockWidget* sdwNew = NULL;
        int index;

        while(temp != NULL)
        {
            sdwNew = qobject_cast<ScriptDockWidget*>(temp);

            if (sdwNew != NULL)
            {
                temp = NULL;
                //m_scriptStackMutex.lock(); //commented due to crashes in debug mode

                index = scriptDockElements.indexOf(sdwNew);

                if (index >= 0)
                {
                    scriptDockElements.removeAt(index);
                    scriptDockElements.push_front(sdwNew);
                }

                //m_scriptStackMutex.unlock();
            }
            else
            {
                temp = temp->parentWidget();
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if tab, defined by its index, if the given widget should be docked
/*!
    The script editor which should be docked is given by its ScriptDockWidget container, named widget, and its tab index.
    If there exists at least one docked ScriptDockWidget, widget will be docked there, otherwise a new docked ScriptDockWidget is opened first.
    If the source ScriptDockWidget does not contain any other tabs and if closeDockIfEmpty is set, the source widget will be closed.

    \param widget ScriptDockWidget container, which contains the tab
    \param index tab-index of the tab
    \param closeDockIfEmpty see method description
*/
void ScriptEditorOrganizer::dockScriptTab(ScriptDockWidget* widget, int index, bool closeDockIfEmpty)
{
    if (!widget->docked() && widget->isTabIndexValid(index))
    {
        ScriptDockWidget *dockedWidget = this->getFirstDockedElement();
        if (dockedWidget == NULL)
        {
            dockedWidget = createEmptyScriptDock(true);
        }
        
        ScriptEditorWidget* editor = widget->removeEditor(index);
        dockedWidget->appendEditor(editor);

        if (widget->getTabCount() == 0 && closeDockIfEmpty)
        {
            widget->close();
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if tab, defined by its index, in the given widget should be undocked
/*!
    The script editor which should be undocked is given by its ScriptDockWidget container, named widget, and its tab index.
    If the active ScriptDockWidget is already undocked, widget will be docked there, otherwise a new undocked ScriptDockWidget is opened first.
    If the source ScriptDockWidget does not contain any other tabs and if closeDockIfEmpty is set, the source widget will be closed.

    \param widget ScriptDockWidget container, which contains the tab
    \param index tab-index of the tab
    \param closeDockIfEmpty see method description
*/
void ScriptEditorOrganizer::undockScriptTab(ScriptDockWidget* widget, int index, bool undockToNewScriptWindow /*= false*/, bool closeDockIfEmpty /*= false*/)
{
    if (widget->isTabIndexValid(index))
    {
        ScriptDockWidget *undockedWidget = NULL;
        undockedWidget = getFirstUndockedElement(); //the really first element is per default the active one. //getActiveDockWidget();
        if (undockedWidget == NULL || undockToNewScriptWindow) // activeWidget is docked, so open a new one
        {
            undockedWidget = createEmptyScriptDock(false);
        }

        ScriptEditorWidget* editor = widget->removeEditor(index);
        undockedWidget->appendEditor(editor);

        if (widget->getTabCount() == 0 && closeDockIfEmpty)
        {
            widget->close();
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if empty new script window should be created and displayed
/*!
    long description

    \param docked indicates whether script window should be docked in MainWindow or not
    \param waitCond ItomSharedSemaphore which will be waked up if process is finished. Use NULL if nothing should happen
*/
void ScriptEditorOrganizer::openNewScriptWindow(bool docked, ItomSharedSemaphore *semaphore)
{
    createEmptyScriptDock(docked);

    if (semaphore != NULL)
    {
        semaphore->release();
        semaphore->deleteSemaphore();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot, invoked if new script should be opened
/*!
    \param waitCond ItomSharedSemaphore which will be waked up if process is finished. Use NULL if nothing should happen
    \return retOk if success, else retError
*/
RetVal ScriptEditorOrganizer::newScript(ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retError);
    ScriptDockWidget* activeWidget = getActiveDockWidget();
    if (activeWidget != NULL)
    {
        retValue += activeWidget->newScript();
        activeWidget->raiseAndActivate();
    }
    else
    {
        activeWidget = createEmptyScriptDock(false);
        if (activeWidget != NULL) 
        {
            retValue += activeWidget->newScript();
            activeWidget->raiseAndActivate();
        }
    }

    if (semaphore != NULL)
    {
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot, invoked if python macro file should be opened as new tab in active script editor window
/*!
    \param filename Filename of the python macro
    \param semaphore ItomSharedSemaphore which will be woken up if opening process is finished. Use NULL if nothing should happen
    \param visibleLineNr is the line number that should be visible and where the cursor should be positioned (default: -1, no cursor positioning)
    \return retOk if success, else retError
*/
RetVal ScriptEditorOrganizer::openScript(const QString &filename, ItomSharedSemaphore *semaphore, int visibleLineNr)
{
    RetVal retValue(retOk);

    bool exist = false;

    QList<ScriptDockWidget*>::iterator it;

    m_scriptStackMutex.lock();

    for (it = scriptDockElements.begin(); it != scriptDockElements.end() && !exist; ++it)
    {
        if ((*it)->activateTabByFilename(filename))
        {
            exist = true;
            fileOpenedOrSaved(filename);
            (*it)->raiseAndActivate();
            if (visibleLineNr >= 0)
            {
                (*it)->activeTabEnsureLineVisible(visibleLineNr);
            }
        }
    }
    m_scriptStackMutex.unlock();

    if (!exist)
    {
        ScriptDockWidget* activeWidget = getActiveDockWidget();
        if (activeWidget == NULL)
        {
            activeWidget = createEmptyScriptDock(false);
        }
    
        if (activeWidget != NULL) 
        {
            if (filename.isNull())
            {
                retValue += activeWidget->openScript();
            }
            else
            {
                retValue += activeWidget->openScript(filename, false);
            }

            if (visibleLineNr >= 0)
            {
                activeWidget->activeTabEnsureLineVisible(visibleLineNr);
            }

            activeWidget->raiseAndActivate();
        }
    }

    if (semaphore != NULL)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if a file open command has been executed in any script window.
/*!
    Checks if filename already has been opened in another script window. If yes only activates this tab, else
    openes the script in the given widget (if NULL opens a new script window)

    \param filename Filename of the python macro which should be opened
    \param widget ScriptDockWidget where this macro should appear as new tab. If NULL, new script window will be created
    \sa ScriptDockWidget
*/
ScriptDockWidget* ScriptEditorOrganizer::openScriptRequested(const QString &filename, ScriptDockWidget* widget)
{
    bool exist = false;

    QList<ScriptDockWidget*>::iterator it;
    ScriptDockWidget* tempWidget = NULL;

    m_scriptStackMutex.lock();

    for (it = scriptDockElements.begin(); it != scriptDockElements.end() && !exist; ++it)
    {
        if ((*it)->activateTabByFilename(filename))
        {
            exist = true;
            tempWidget = *it;
            fileOpenedOrSaved(filename);
        }
    }

    m_scriptStackMutex.unlock();

    if (!exist)
    {
        if (widget == NULL)
        {
            widget = createEmptyScriptDock(false);
        }
         widget->openScript(filename, true);
         tempWidget = widget;
    }

    return tempWidget;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if someone wants to run a python file with filename.
/*!
    Before signaling the execution command, checks that every opened script already having a filename is saved.

    \param filename Filename of the python script where the execution should start
*/
void ScriptEditorOrganizer::pythonRunFileRequested(QString filename)
{
    RetVal retValue(retOk);

    retValue += this->saveAllScripts(true, true);

    if (!retValue.containsError())
    {
        emit(pythonRunFile(filename));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if someone wants to debug a python file with filename.
/*!
    Before signaling the debug command, checks that every opened script already having a filename is saved.

    \param filename Filename of the python script where the debugging should start
*/
void ScriptEditorOrganizer::pythonDebugFileRequested(QString filename)
{
    RetVal retValue(retOk);

    retValue += this->saveAllScripts(true, true);

    if (!retValue.containsError())
    {
        emit(pythonDebugFile(filename));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if line in python debugging process has been changed
/*!
    Checks, if filename is already opened in one script editor. If yes, activates this script editor.
    If no, opens this the specified script in a new tab.

    \param filename Filename of actual executed python macro
    \param lineNo line number in file (here not used)
*/
void ScriptEditorOrganizer::pythonDebugPositionChanged(QString filename, int lineNo)
{
    QList<ScriptDockWidget*>::iterator it;
    bool found = false;

    m_scriptStackMutex.lock();

    for (it = scriptDockElements.begin(); it != scriptDockElements.end(); ++it)
    {
        found = found || (*it)->activateTabByFilename(filename);
    }

    m_scriptStackMutex.unlock();

    if (!found)
    {
        ScriptDockWidget* activeWidget = getActiveDockWidget();
        ScriptDockWidget* widget = openScriptRequested(filename, activeWidget);
        widget->activateTabByFilename(filename, lineNo);
    }
}

} //end namespace ito