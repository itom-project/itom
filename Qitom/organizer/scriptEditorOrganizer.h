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

#ifndef SCRIPTEDITORORGANIZER_H
#define SCRIPTEDITORORGANIZER_H

#include "../widgets/scriptDockWidget.h"
#include "../common/sharedStructuresQt.h"

namespace ito
{
QDataStream &operator<<(QDataStream &out, const ito::ScriptEditorStorage &obj); 

QDataStream &operator>>(QDataStream &in, ito::ScriptEditorStorage &obj); 

class ScriptEditorOrganizer : public QObject
{
    Q_OBJECT
public:
    ScriptEditorOrganizer( bool dockAvailable);
    ~ScriptEditorOrganizer();

    RetVal saveAllScripts(bool askFirst = true, bool ignoreNewScripts = false);
    RetVal closeAllScripts(bool saveFirst);

    void saveScriptState();
    RetVal restoreScriptState();

    const QStringList &getRecentlyUsedFiles() const { return m_recentlyUsedFiles; }

protected:
    ScriptDockWidget* createEmptyScriptDock(bool docked, Qt::DockWidgetArea area = Qt::TopDockWidgetArea, const QString &objectName = QString());

    

private:
    ScriptDockWidget* getFirstDockedElement();
    ScriptDockWidget* getFirstUndockedElement();
    ScriptDockWidget* getActiveDockWidget();

    QList<ScriptDockWidget*> scriptDockElements;    //! list with references to all ScriptDockWidgets (docked or windows-style)
    QSet<QString> m_usedObjectNames;               //! currently used objectNames for script windows
    bool m_dockAvailable;                             //! true if docking mode is available, else: false

    QMutex m_scriptStackMutex;                        //! mutex locking any changes to scriptDockElements

    QStringList m_recentlyUsedFiles;

signals:
    void addScriptDockWidgetToMainWindow(AbstractDockWidget *dockWidget, Qt::DockWidgetArea area); //! signal emitted if dockWidget should be added to docking area in main window
    void removeScriptDockWidgetFromMainWindow(AbstractDockWidget *dockWidget);                     //! signal emitted if dockWidget should be removed from main window

    void pythonRunFile(QString filename);           //! signal emitted if macro (filename) should be executed in python
    void pythonDebugFile(QString filename);         //! signal emitted if macro (filename) should be debugged in python

public slots:
    void removeScriptDockWidget(ScriptDockWidget* widget);

    void dockScriptTab(ScriptDockWidget* widget, int index, bool closeDockIfEmpty = false);
    void undockScriptTab(ScriptDockWidget* widget, int index, bool undockToNewScriptWindow = false, bool closeDockIfEmpty = false);

    void openNewScriptWindow(bool docked, ItomSharedSemaphore* semaphore = NULL);
    RetVal newScript(ItomSharedSemaphore* semaphore = NULL);
    RetVal openScript(const QString &filename, ItomSharedSemaphore* semaphore = NULL, int visibleLineNr = -1);

    ScriptDockWidget* openScriptRequested(const QString &filename, ScriptDockWidget* widget);

    void pythonRunFileRequested(QString filename);
    void pythonDebugFileRequested(QString filename);

    void pythonDebugPositionChanged(QString filename, int lineNo);

    void fileOpenedOrSaved(const QString &filename);

private slots:
    void widgetFocusChanged(QWidget* old, QWidget* now);

};

} //end namespace ito

#endif