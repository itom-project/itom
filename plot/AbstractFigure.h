/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut f�r Technische
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

#if QT_VERSION < 0x050000
#include <qmainwindow.h>
#include <qlabel.h>
#include <qtoolbar.h>
#include <qevent.h>
#include <qdockwidget.h>
#else
#include <QtWidgets/qmainwindow.h>
//#include <QtWidgets/qlabel.h>
#include <QtWidgets/qtoolbar.h>
#include <QtWidgets/qdockwidget.h>
//#include <QtGui/qevent.h>
#endif

class QPropertyEditorWidget; //forward declaration
class MarkerLegendWidget; //forward declaration

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

//place this macro in the header file of the designer plugin widget class right before the first section (e.g. public:)
#define DESIGNER_PLUGIN_ITOM_API \
            protected: \
                void importItomApi(void** apiPtr) \
                {ito::ITOM_API_FUNCS = apiPtr;} \
                void importItomApiGraph(void** apiPtr) \
                { ito::ITOM_API_FUNCS_GRAPH = apiPtr;} \
            public: \
                //.

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

    Q_CLASSINFO("slot://refreshPlot", "Triggeres an update of the current plot window.")

    public:
        enum WindowMode { ModeInItomFigure, ModeStandaloneInUi, ModeStandaloneWindow };
        enum CompilerFeatures 
        { 
            tOpenCV        = 0x01,
            tPointCloudLib = 0x02
        };

        int getCompilerFeatures(void) const 
        {
            int retval = tOpenCV;
            #if defined USEPCL || ITOM_POINTCLOUDLIBRARY
            retval |= tPointCloudLib;
            #endif
            return retval;
        }

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
        virtual QDockWidget *getMarkerLegendDockWidget() const { return m_markerLegendDock; }

        QList<QMenu*> getMenus() const;
        QList<AbstractFigure::ToolBarItem> getToolbars() const;

    protected:

        virtual RetVal init() { return retOk; } //this method is called from after construction and after that the api pointers have been transmitted

        virtual void importItomApi(void** apiPtr) = 0; //this methods are implemented in the plugin itsself. Therefore place ITOM_API right after Q_INTERFACE in the header file and replace Q_EXPORT_PLUGIN2 by Q_EXPORT_PLUGIN2_ITOM in the source file.
        virtual void importItomApiGraph(void** apiPtr) = 0;

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

        QDockWidget *m_markerLegendDock;
        MarkerLegendWidget *m_markerLegendWidget;

    signals:
        
    private slots:

        inline void mnuShowToolbar(bool /*checked*/) { setToolbarVisible(true); }
        inline void mnuShowProperties(bool checked) { if (m_propertyDock) { m_propertyDock->setVisible(checked); } }
        inline void mnuShowMarkerLegend(bool checked) 
        { 
            if (m_markerLegendDock) 
            { 
                m_markerLegendDock->setVisible(checked); 
            } 
        }

    public slots:

        int getPlotID() 
        { 
            if(!ito::ITOM_API_FUNCS_GRAPH) return 0;
            //return getUniqueID();
            ito::uint32 thisID = 0;
            ito::RetVal retval = apiGetFigureIDbyHandle(this, thisID);

            if(retval.containsError())
            {
                return 0;
            }
            return thisID; 
        }
        void refreshPlot() { update(); }
};

} // namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif // ABSTRACTFIGURE_H
