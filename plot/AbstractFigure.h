/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef ABSTRACTFIGURE_H
#define ABSTRACTFIGURE_H

#include "../common/commonGlobal.h"
#include "AbstractNode.h"

#include "../common/apiFunctionsGraphInc.h"
#include "../common/apiFunctionsInc.h"

#include <qmainwindow.h>
#include <qlabel.h>
#include <qtoolbar.h>
#include <qevent.h>
#include <qdockwidget.h>

class QPropertyEditorWidget; //forward declaration

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito {

class AbstractFigure;

//void ITOMCOMMONQT_EXPORT initialize(AbstractFigure *fig);

class ITOMCOMMONQT_EXPORT AbstractFigure : public QMainWindow, public AbstractNode
{
    Q_OBJECT
    Q_ENUMS(WindowMode)
    Q_PROPERTY(bool toolbarVisible READ getToolbarVisible WRITE setToolbarVisible DESIGNABLE true USER true)
    Q_PROPERTY(bool contextMenuEnabled READ getContextMenuEnabled WRITE setContextMenuEnabled DESIGNABLE true)

    Q_CLASSINFO("prop://toolbarVisible", "Toggles the visibility of the toolbar of the plot.")
    Q_CLASSINFO("prop://contextMenuEnabled", "Defines whether the context menu of the plot should be enabled or not.")

    public:
        enum WindowMode { ModeInItomFigure, ModeStandaloneInUi, ModeStandaloneWindow };

        struct ToolBarItem {
            ToolBarItem() : toolbar(NULL), visible(1), section(0), key("") {}
            QToolBar *toolbar;
            bool visible;
            int section;
            Qt::ToolBarArea area;
            QString key;
        };

        AbstractFigure(const QString &itomSettingsFile, WindowMode windowMode = ModeStandaloneInUi, QWidget *parent = 0);
        virtual ~AbstractFigure();

        virtual bool event(QEvent *e);
        void setApiFunctionGraphBasePtr(void **apiFunctionGraphBasePtr);
        void setApiFunctionBasePtr(void **apiFunctionBasePtr);
        void ** getApiFunctionGraphBasePtr(void) { return m_apiFunctionsGraphBasePtr; }
        void ** getApiFunctionBasePtr(void) { return m_apiFunctionsBasePtr; }

        virtual RetVal addChannel(AbstractNode *child, ito::Param* parentParam, ito::Param* childParam, Channel::ChanDirection direction, bool deleteOnParentDisconnect, bool deleteOnChildDisconnect);
        virtual RetVal addChannel(Channel *newChannel);
        virtual RetVal removeChannelFromList(unsigned int uniqueID);
        virtual RetVal removeChannel(Channel *delChannel);

        virtual RetVal update(void) = 0; /*!> Calls apply () and updates all children*/

        //properties
        virtual void setToolbarVisible(bool visible);
        virtual bool getToolbarVisible() const;
        virtual void setContextMenuEnabled(bool show) = 0; 
        virtual bool getContextMenuEnabled() const = 0;

        virtual QDockWidget *getPropertyDockWidget() const { return m_propertyDock; }

        QList<QMenu*> getMenus() const;
        QList<AbstractFigure::ToolBarItem> getToolbars() const;

    protected:

        virtual RetVal init() { return retOk; } //this method is called from after construction and after that the api pointers have been transmitted


        void addToolBar(QToolBar *toolbar, const QString &key, Qt::ToolBarArea area = Qt::TopToolBarArea, int section = 1);
        void addToolBarBreak(const QString &key, Qt::ToolBarArea area = Qt::TopToolBarArea);

        void showToolBar(const QString &key);
        void hideToolBar(const QString &key);

        void addMenu(QMenu *menu);

        void updatePropertyDock();
        void setPropertyObservedObject(QObject* obj);

        RetVal initialize();

        QMenu *m_contextMenu;

        WindowMode m_windowMode;
        QString m_itomSettingsFile;
        QWidget *m_mainParent; //the parent of this figure is only set to m_mainParent, if the stay-on-top behaviour is set to the right value

        void **m_apiFunctionsGraphBasePtr;
        void **m_apiFunctionsBasePtr;
        
        bool m_toolbarsVisible;

    private:
        QList<QMenu*> m_menus;
        QList<ToolBarItem> m_toolbars;

        QDockWidget *m_propertyDock;
        QPropertyEditorWidget *m_propertyEditorWidget;
        QObject *m_propertyObservedObject;

    signals:
        
    private slots:

        inline void mnuShowToolbar(bool /*checked*/) { setToolbarVisible(true); }
        inline void mnuShowProperties(bool checked) { if (m_propertyDock) { m_propertyDock->setVisible(checked); } }

    public slots:
        void refreshPlot() { update(); }
};

} // namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif // ABSTRACTFIGURE_H
