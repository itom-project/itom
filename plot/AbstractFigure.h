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
    Q_ENUMS(TopLevelMode)
    Q_ENUMS(WindowMode)
    Q_PROPERTY(bool toolbarVisible READ toolbarVisible WRITE setToolbarVisible DESIGNABLE true)
    Q_PROPERTY(bool showContextMenu READ showContextMenu WRITE setShowContextMenu DESIGNABLE true)
    Q_PROPERTY(TopLevelMode topLevelMode READ getTopLevelMode WRITE setTopLevelMode DESIGNABLE false) //designable is not important, since topLevelMode only is relevant for window-based-plots.

    public:
        AbstractFigure(const QString &itomSettingsFile, QWidget *parent = 0);
        ~AbstractFigure();

        enum WindowMode { ModeWindow, ModeEmbedded };
        enum TopLevelMode { TopLevelNothing, TopLevelParentOnly, TopLevelOverall };

        void setWindowMode(const WindowMode mode);

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

        void setTopLevelMode(TopLevelMode mode);
        TopLevelMode getTopLevelMode();

		//toolbar methods
		void addToolBar(QToolBar *toolbar, const QString &key, Qt::ToolBarArea area = Qt::TopToolBarArea);
		void insertToolBar(const QString &key_before, QToolBar *toolbar, const QString &key);
		void insertToolBarBreak(const QString &key_before);
		void removeToolBar(const QString &key);
		void removeToolBarBreak(const QString &key_before);
		void showToolBar(const QString &key);
		void hideToolBar(const QString &key);

    protected:

        RetVal initialize();
        void addMenu(QMenu *menu);

		QMap<QString, QPair<QToolBar*, bool> > m_toolbars;

        QMenu    *m_contextMenu;

        QAction *m_actTopLevelParent;
        QAction *m_actTopLevelOverall;

        QMenu *m_menuWindow;

        WindowMode m_windowMode;
        TopLevelMode m_topLevelMode;
        QString m_itomSettingsFile;
        QWidget *m_mainParent; //the parent of this figure is only set to m_mainParent, if the stay-on-top behaviour is set to the right value

        void **m_apiFunctionsGraphBasePtr;
        void **m_apiFunctionsBasePtr;
        
        //friend void ito::initialize(ito::AbstractFigure *fig);

		bool m_toolbarsVisible;

        

    private:

    signals:

    private slots:

        inline void mnuShowToolbar(bool /*checked*/) { setToolbarVisible(true); }

        void mnuTopLevelOverall(bool checked);
        void mnuTopLevelParent(bool checked);

    public slots:
};

}; // namespace ito

Q_DECLARE_METATYPE(ito::AbstractFigure::TopLevelMode)

#endif // ABSTRACTFIGURE_H
