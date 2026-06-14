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

#ifndef BOOKMARKDOCKWIDGET_H
#define BOOKMARKDOCKWIDGET_H

#include "abstractDockWidget.h"

#include <qwidget.h>
#include <qaction.h>
#include <qtoolbar.h>
#include <qitemselectionmodel.h>
#include "../models/bookmarkModel.h"

#include "itomQWidgets.h"


namespace ito
{
    /*!
        \class BookmarkDockWidget
        \brief Provides the bookmark toolbox which is mainly a view of the BookmarkModel.
    */
    class BookmarkDockWidget : public AbstractDockWidget
    {
        Q_OBJECT

        public:

            //! Constructor for the toolbox
            /*!
                \param title is the title of the toolbox
                \param objName is an internal object name for this toolbox, used to store its geometry and state at shutdown
                \param parent is an optional parent widget
                \param docked indicate if the toolbox should be docked per default
                \param isDockAvailable indicates if this toolbox can be docked in any case
                \param floatingStyle indicates the window floating style behaviour of this toolbox
                \param movingStyle indicates if this toolbox might be moved from one dockable area of itom's main window to another one or not.
            */
            BookmarkDockWidget(const QString &title, const QString &objName, QWidget *parent = NULL,
                bool docked = true, bool isDockAvailable = true,
                tFloatingStyle floatingStyle = floatingNone,
                tMovingStyle movingStyle = movingEnabled);

            //! Destructor for the toolbox
            ~BookmarkDockWidget();

            //! Set the BookmarkModel for this toolbox
            /*!
                Usually, the BookmarkModel is not available during construction of this toolbox, since the main window
                of itom is loaded earlier than the ScriptEditorOrganizer, which is the owner of the BookmarkModel.

                Therefore the model is set via this method at a later time. However the model can only be set once.
                Further calls of this method will do nothing.

                \param model is a reference to the inialized BookmarkModel
            */
            void setBookmarkModel(BookmarkModel *model);

        protected:

            //! this method creates all actions of this toolbox and is overloaded from AbstractDockWidget.
            void createActions();

            //! this method creates all menus of this toolbox and is overloaded from AbstractDockWidget.
            void createMenus();

            //! this method creates all menus of this toolbox and is overloaded from AbstractDockWidget.
            void createToolBars();

            //! this method is overloaded from AbstractDockWidget to initialize any status bars. Here it does nothing.
            void createStatusBar(){}

            //! this method is overloaded from AbstractDockWidget and is called if any actions should be updated.
            void updateActions();

            //! this method is overloaded from AbstractDockWidget and is called whenever any Python relevant actions are changed. It does nothing.
            void updatePythonActions(){}

        private:
            QTreeViewItom   *m_bookmarkView;        /*!< QTreeViewItom derived from QTreeView with some special selection behaviour (see QItomWidgets)*/
            QToolBar    *m_pMainToolbar;            /*!< Toolbar with QActions */
            QMenu *m_pContextMenu;                  /*!< Context menu with the same actions as the toolbar */
            BookmarkModel *m_pModel;                /*!< reference to the BookmarkModel. This widget is not the owner of the model. */
            QAction *m_pSpacerAction;               /*!< since the model is usually provided after having constructed this toolbox, some actions will be added to the toolbar in front of this action. */

        private Q_SLOTS:

            //! This slot is executed when a bookmark has been double clicked.
            /*
                \param index is the QModelIndex of the clicked entry.
            */
            void doubleClicked(const QModelIndex &index);

            //! This slot is executed when a context has been requested on the tree view of this toolbox
            /*
                \param pos is the position of the mouse click
            */
            void treeViewContextMenuRequested(const QPoint &pos);
    };

} //end namespace ito

#endif
