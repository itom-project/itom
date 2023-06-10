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

#ifndef SCRIPTEDITORORGANIZER_H
#define SCRIPTEDITORORGANIZER_H

#include "../widgets/scriptDockWidget.h"
#include "../widgets/outlineSelectorWidget.h"
#include "../common/sharedStructuresQt.h"
#include "../models/bookmarkModel.h"

#include <qsignalmapper.h>
#include <qlist.h>

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

    RetVal saveAllScripts(bool askFirst = true, bool ignoreNewScripts = false, int *saveScriptState = NULL);
    RetVal closeAllScripts(bool saveFirst);

    void saveScriptState();
    RetVal restoreScriptState();

    const QStringList &getRecentlyUsedFiles() const { return m_recentlyUsedFiles; }

    inline const ScriptEditorActions& getScriptEditorActions() const { return m_commonScriptEditorActions; }

    QStringList openedScripts() const;

    inline BookmarkModel* getBookmarkModel() const
    {
        return m_pBookmarkModel;
    }

    ScriptDockWidget* activateOpenedScriptByFilename(const QString &filename, int currentDebugLine = -1, int UID = -1);

    //!< returns the outlines of all opened scripts
    QList<OutlineSelectorWidget::EditorOutline> getAllOutlines(const ScriptDockWidget *currentScriptDockWidget, int &currentIndex) const;

protected:
    ScriptDockWidget* createEmptyScriptDock(bool docked, Qt::DockWidgetArea area = Qt::TopDockWidgetArea, const QString &objectName = QString());

    RetVal applyGoBackNavigationItem(const GoBackNavigationItem &item);
    void updateGoBackNavigationActions();

private:
    ScriptDockWidget* getFirstDockedElement() const;
    ScriptDockWidget* getFirstUndockedElement() const;
    ScriptDockWidget* getActiveDockWidget() const;

    BookmarkModel *m_pBookmarkModel;

    QList<ScriptDockWidget*> m_scriptDockElements;    //! list with references to all ScriptDockWidgets (docked or windows-style)
    QSet<QString> m_usedObjectNames;               //! currently used objectNames for script windows
    bool m_dockAvailable;                             //! true if docking mode is available, else: false
    bool m_dockedNewWidget;
    mutable QMutex m_scriptStackMutex; //! mutex locking any changes to m_scriptDockElements. This mutex can also be changed in const methods

    QStringList m_recentlyUsedFiles;

    ScriptEditorActions m_commonScriptEditorActions;
    QMenu *m_pGoBackNavigationMenu; //! menu for the backward items
    QList<GoBackNavigationItem> m_goBackNavigationHistory; //! history of go back navigation items. Newer items are at the end of the list. The list is limited to a number of maximum items.
    int m_goBackNavigationIndex;                           //! current position of script editors in goBackNavigationHistory. If equal to m_goBackNavigationHistory.size(), the current position is at the end.
    static const int MaxGoBackNavigationEntries;               //! maximum number of entries in the go back navigation history.

signals:
    void addScriptDockWidgetToMainWindow(AbstractDockWidget *dockWidget, Qt::DockWidgetArea area); //! signal emitted if dockWidget should be added to docking area in main window
    void removeScriptDockWidgetFromMainWindow(AbstractDockWidget *dockWidget);                     //! signal emitted if dockWidget should be removed from main window

    void pythonRunFile(QString filename);           //! signal emitted if macro (filename) should be executed in python
    void pythonDebugFile(QString filename);         //! signal emitted if macro (filename) should be debugged in python

public slots:
    void removeScriptDockWidget(ScriptDockWidget* widget);

    void dockScriptTab(ScriptDockWidget* widget, int index, bool closeDockIfEmpty = false);
    void undockScriptTab(ScriptDockWidget* widget, int index, bool undockToNewScriptWindow = false, bool closeDockIfEmpty = false);

    RetVal openNewScriptWindow(bool docked, ItomSharedSemaphore* semaphore = NULL);
    RetVal newScript(ItomSharedSemaphore* semaphore = NULL);
    RetVal openScript(const QString &filename, ItomSharedSemaphore* semaphore = NULL, int visibleLineNr = -1, bool errorMessageClick = false, bool showSelectedCallstackLine = false);

    ScriptDockWidget* openScriptRequested(const QString &filename, ScriptDockWidget* widget);

    void pythonRunFileRequested(QString filename);
    void pythonDebugFileRequested(QString filename);

    void pythonDebugPositionChanged(QString filename, int lineNo);

    void fileOpenedOrSaved(const QString &filename);

private slots:
    void widgetFocusChanged(QWidget* old, QWidget* now);

    void onAddGoBackNavigationItem(const GoBackNavigationItem &item);
    void onGotoBookmark(const BookmarkItem &item);

    //Action slots
    void mnuNavigateForward();
    void mnuNavigateBackward();
    void mnuNavigateBackwardItem(int index);
};

} //end namespace ito

#endif
