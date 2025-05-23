/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2024, Institut für Technische Optik (ITO),
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

#pragma once

#include "../helper/IOHelper.h"
#include "abstractDockWidget.h"

#include "itomQWidgets.h"
#include "../models/itomFileSystemModel.h"

#include <qwidget.h>
#include <qaction.h>
#include <qtoolbar.h>

#include <qtextbrowser.h>
#include <qtreeview.h>
#include <qlabel.h>
#include <qhash.h>
#include <qmutex.h>
#include <qmenu.h>
#include <qcombobox.h>
#include <qprocess.h>
#include <qevent.h>
#include <qurl.h>
#include <qfileiconprovider.h>
#include <qstack.h>
#include <qsignalmapper.h>


namespace ito
{
    class FileSystemDockWidget : public AbstractDockWidget
    {
        Q_OBJECT

            Q_PROPERTY(QColor linkColor READ linkColor WRITE setLinkColor DESIGNABLE true);

        public:
            FileSystemDockWidget(const QString &title, const QString &objName, QWidget *parent = NULL, bool docked = true, bool isDockAvailable = true, tFloatingStyle floatingStyle = floatingNone, tMovingStyle movingStyle = movingEnabled, const QString &baseDirectory = QDir::currentPath());
            ~FileSystemDockWidget();

            QColor linkColor() { return m_linkColor; }
            void setLinkColor(const QColor &color);

        protected:
            void createActions();
            void createMenus();
            void createToolBars();
            void createStatusBar(){}
            void updateActions();
            void updatePythonActions(){ updateActions(); }

            QString getHtmlTag(const QString &tag);

            bool eventFilter(QObject *obj, QEvent *event);

        private:
            void fillFilterList();
            void showInGraphicalShell(const QString &filePath);
            void treeViewHideOrShowColumns(const bool& hide);

            QMenu* m_pShowDirListMenu;
            QMenu* m_pFileSystemSettingMenu;
            QMenu* m_pContextMenu;
            QTextBrowser* m_pPathEdit;
            QToolBar* m_pMainToolbar;
            QTreeViewItom* m_pTreeView;
            QLabel* m_pLblFilter;
            QComboBox* m_pCmbFilter;
            ItomFileSystemModel* m_pFileSystemModel;
            QString m_baseDirectory;
            QHash<QString,QStringList> m_defaultFilterPatterns;
            QMutex m_baseDirChangeMutex;
            QList<QUrl> m_clipboardCutData; //this mime-data has recently be selected by a cut action and is no available in QClipboard
            bool m_showColumnDetails;
            QList<int> m_detailColumnsWidth;
            QColor m_linkColor;

            QStack<QString> m_historyStack;
            QStack<QString> m_forwardStack;
            int m_maxHistorySize = 20;

#if QT_VERSION >= QT_VERSION_CHECK(6,0,0)
            // since (at least) Qt 6.7, QFileSystemModel
            // creates a default QAbstractFileIconProvider,
            // that shows very basic folder icons. However, we
            // would like to see the real folder icons of the
            // operating system. Therefore, we have to pass
            // our own instance of QFileIconProvider to the model.
            QFileIconProvider m_fileIconProvider;
#endif

            ShortcutAction* m_pActMoveCDUp;
            ShortcutAction* m_pActSelectCD;
            ShortcutAction* m_pActCopyDir;
            ShortcutAction* m_pActPasteDir;
            ShortcutAction* m_pActOpenFile;
            ShortcutAction* m_pActExecuteFile;
            ShortcutAction* m_pActLocateOnDisk;
            ShortcutAction* m_pActRenameItem;
            ShortcutAction* m_pActDeleteItems;
            ShortcutAction* m_pActCutItems;
            ShortcutAction* m_pActCopyItems;
            ShortcutAction* m_pActPasteItems;
            ShortcutAction* m_pActNewDir;
            ShortcutAction* m_pActNewPyFile;
            ShortcutAction* m_pViewList;
            ShortcutAction* m_pViewDetails;

            QAction *m_lastMovedShowDirAction;

        signals:
            void currentDirChanged();

        private slots:
            void mnuMoveCDUp();
            void mnuSelectCD();
            void mnuCopyDir();
            void mnuPasteDir();
            void mnuLocateOnDisk();
            void mnuExecuteFile();
            void mnuOpenFile();
            void mnuRenameItem();
            void mnuDeleteItems();
            void mnuCutItems();
            void mnuCopyItems();
            void mnuPasteItems();
            void mnuNewDir();
            void mnuNewPyFile();
            void showList();
            void showDetails();
            void mnuToggleView();
            void newDirSelected(const QString& text);
            void cmbFilterEditTextChanged(const QString &text);
            void openFile(const QModelIndex& index);
            void treeViewContextMenuRequested(const QPoint &pos);
            void removeActionFromDirList(const int &pos);
            void itemDoubleClicked(const QModelIndex &index);
            void pathAnchorClicked(const QUrl &link);
            void onMouseReleased(QMouseEvent* e);

            void navigateBackward();
            void navigateForward();

        public slots:
            RetVal changeBaseDirectory(QString dir, bool clearHistory=false, bool addToHistory=true);
            void processError(QProcess::ProcessError error);
    };

} //end namespace ito
