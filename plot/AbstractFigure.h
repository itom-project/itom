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

#include "AbstractNode.h"

#include "../common/apiFunctionsGraphInc.h"
#include "../common/apiFunctionsInc.h"

#include <qmainwindow.h>
#include <qlabel.h>
#include <qtoolbar.h>
#include <qevent.h>

namespace ito {

class AbstractFigure;
void initialize(AbstractFigure *fig);

class AbstractFigure : public QMainWindow, public AbstractNode
{
    Q_OBJECT
    Q_ENUMS(WindowMode)
    Q_PROPERTY(bool toolbarVisible READ toolbarVisible WRITE setToolbarVisible DESIGNABLE true)
    Q_PROPERTY(bool showContextMenu READ showContextMenu WRITE setShowContextMenu DESIGNABLE true)

    public:
        enum WindowMode { ModeInItomFigure, ModeStandaloneInUi, ModeStandaloneWindow };

        struct ToolBarItem {
            QToolBar *toolbar;
            bool visible;
            int section;
            Qt::ToolBarArea area;
            QString key;
        };

        AbstractFigure(const QString &itomSettingsFile, WindowMode windowMode = ModeStandaloneInUi, QWidget *parent = 0);
        ~AbstractFigure();

        virtual bool event(QEvent *e);
        void setApiFunctionGraphBasePtr(void **apiFunctionGraphBasePtr);
        void setApiFunctionBasePtr(void **apiFunctionBasePtr);
        void ** getApiFunctionGraphBasePtr(void) { return m_apiFunctionsGraphBasePtr; }
        void ** getApiFunctionBasePtr(void) { return m_apiFunctionsBasePtr; }

        virtual RetVal addChannel(AbstractNode *child, ito::Param* parentParam, ito::Param* childParam, Channel::ChanDirection direction, bool deleteOnParentDisconnect, bool deleteOnChildDisconnect);
        virtual RetVal addChannel(Channel *newChannel);
        virtual RetVal removeChannelFromList(unsigned int uniqueID);
        virtual RetVal removeChannel(Channel *delChannel);

        //properties
        virtual void setToolbarVisible(bool visible);
        virtual bool toolbarVisible() const;
        virtual void setShowContextMenu(bool show) = 0; 
        virtual bool showContextMenu() const = 0;

        QList<QMenu*> getMenus() const;
        QList<AbstractFigure::ToolBarItem> getToolbars() const;

    protected:

		void addToolBar(QToolBar *toolbar, const QString &key, Qt::ToolBarArea area = Qt::TopToolBarArea, int section = 1);
        void addToolBarBreak(const QString &key, Qt::ToolBarArea area = Qt::TopToolBarArea);

		void showToolBar(const QString &key);
		void hideToolBar(const QString &key);

        void addMenu(QMenu *menu);

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

    signals:

    private slots:

        inline void mnuShowToolbar(bool /*checked*/) { setToolbarVisible(true); }

    public slots:
};

}; // namespace ito

#endif // ABSTRACTFIGURE_H
