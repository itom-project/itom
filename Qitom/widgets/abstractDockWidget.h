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

#ifndef ABSTRACTDOCKWIDGET_H
#define ABSTRACTDOCKWIDGET_H

#include "../global.h"
#include "../common/sharedStructures.h"

#include <qmainwindow.h>
#include <qdockwidget.h>
#include <qmenubar.h>
#include <qevent.h>
#include <qmap.h>
#include <qtoolbar.h>
#include <qdebug.h>
#include <qstring.h>
#include <qaction.h>
#include <qshortcut.h>
#include <qrect.h>
#include <qwidget.h>


namespace ito
{
    class AbstractDockWidget : public QDockWidget
    {
        Q_OBJECT
        
        //these properties are taken from QWidget in order to redirect them either to the QDockWidget or to the main window (depending on dock status)
        Q_PROPERTY(bool visible READ isVisible WRITE setVisible DESIGNABLE false)
        Q_PROPERTY(bool enabled READ isEnabled WRITE setEnabled)

        Q_PROPERTY(QRect geometry READ geometry WRITE setGeometry)
        Q_PROPERTY(QRect frameGeometry READ frameGeometry)
        Q_PROPERTY(QRect normalGeometry READ normalGeometry)
        Q_PROPERTY(int x READ x)
        Q_PROPERTY(int y READ y)
        Q_PROPERTY(QPoint pos READ pos WRITE move DESIGNABLE false STORED false)
        Q_PROPERTY(QSize frameSize READ frameSize)
        Q_PROPERTY(QSize size READ size WRITE resize DESIGNABLE false STORED false)
        Q_PROPERTY(int width READ width)
        Q_PROPERTY(int height READ height)
        Q_PROPERTY(QRect rect READ rect)
        Q_PROPERTY(QRect childrenRect READ childrenRect)
        Q_PROPERTY(QRegion childrenRegion READ childrenRegion)

        public:

            enum tFloatingStyle { floatingNone, floatingStandard, floatingWindow }; /*!< if floatingNone, this widget can not be undocked, if floatingStandard or floatingWindow it can be undocked. If floatingWindow appearance in undocked mode changes to real window style */
            enum tMovingStyle { movingDisabled, movingEnabled };    /*!<  if movingDisabled dockWidget must not be moved from one docking area to another one, else: movingEnabled */
            enum tTopLevelStyle { topLevelOverall, topLevelParentOnly, topLevelNothing };

            struct Toolbar
            {
                Toolbar() : section(0), key(""), tb(NULL) {}
                Qt::ToolBarArea area;
                int section;
                QString key;
                QToolBar *tb;
            };

            AbstractDockWidget(bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle, const QString &title = QString(), const QString &objName = QString(), QWidget *parent = 0);
            virtual ~AbstractDockWidget();

            inline bool docked() const { return m_docked; }    /*!<  returns if docking widget is docked (true) or undocked (false) */

            RetVal setAdvancedWindowTitle( QString newCompleteTitle = QString(), bool appendToBasicTitle = true );

            RetVal setTopLevel( tTopLevelStyle topLevel );

            void setParent ( QWidget * parent ) { m_overallParent = parent; QDockWidget::setParent(parent); }

            QWidget *getActiveInstance() 
            { 
                if(!m_docked && m_floatingStyle == floatingWindow)
                {
                    return m_pWindow;
                }
                return this;
            }

            //these methods are mainly redirected to QDockWidget or QMainWindow (depending on dock status)
            #define QWIDGETPROPGETTER(gettername) if(m_docked && m_dockAvailable) { return QDockWidget::gettername(); } else { return m_pWindow->gettername(); }

            #define QWIDGETPROPSETTER(settername,...) \
            if (m_docked) \
            { \
                QDockWidget::settername(__VA_ARGS__);  \
            } \
            else \
            { \
                if (m_floatingStyle == floatingWindow) \
                { \
                    m_pWindow->settername(__VA_ARGS__); \
                    QDockWidget::settername(__VA_ARGS__); \
                } \
                else \
                { \
                    QDockWidget::settername(__VA_ARGS__); \
                } \
            }

            QRect frameGeometry() const;
            const QRect &geometry() const;
            QRect normalGeometry() const;

            int x() const;
            int y() const;
            QPoint pos() const;
            QSize frameSize() const;
            QSize size() const;
            int width() const;
            int height() const;
            QRect rect() const;
            QRect childrenRect() const;
            QRegion childrenRegion() const;

            void move(int x, int y);
            void move(const QPoint &);
            void resize(int w, int h);
            void resize(const QSize &);
            void setGeometry(int x, int y, int w, int h);
            void setGeometry(const QRect &);

            bool isEnabled() const;
            bool isVisible() const;

            void saveState(const QString &iniName) const;
            void restoreState(const QString &iniName) const;

        public Q_SLOTS:
            void setEnabled(bool);
            virtual void setVisible(bool visible);
   
        protected:

            class ShortcutAction : public QObject
            {
            public:
                ShortcutAction(const QString &text, AbstractDockWidget *parent) : QObject(parent), m_action(NULL), m_shortcut(NULL)
                {
                    m_action = new QAction(text, parent);
                }

                ShortcutAction(const QIcon &icon, const QString &text, AbstractDockWidget *parent) : QObject(parent), m_action(NULL), m_shortcut(NULL)
                {
                    m_action = new QAction(icon, text, parent);
                }

                ShortcutAction(const QIcon &icon, const QString &text, AbstractDockWidget *parent, const QKeySequence &key, Qt::ShortcutContext context = Qt::WindowShortcut) : QObject(parent), m_action(NULL), m_shortcut(NULL)
                {
                    QString text2 = text;
                    QString text3 = text;
                    text2.append( "\t" );
                    text2.append( key.toString(QKeySequence::NativeText) );
                    text3.append( " (" );
                    text3.append( key.toString(QKeySequence::NativeText) );
                    text3.append( ")" );
                    m_action = new QAction(icon, text2, parent);
                    m_action->setToolTip(text3);
                    m_shortcut = new QShortcut(key, parent->getCanvas());
                    m_shortcut->setContext(context);
                }

                ~ShortcutAction()
                {
                    //do not delete action and shortcut here, since it will be deleted by common parent.
                }

                void connectTrigger(const QObject *receiver, const char *method, Qt::ConnectionType type = Qt::AutoConnection)
                {
                    if(m_action)
                    {
                        QObject::connect(m_action, SIGNAL(triggered()), receiver, method, type);
                    }
                    if(m_shortcut)
                    {
                        QObject::connect(m_shortcut, SIGNAL(activated()), receiver, method, type);
                    }
                }

                void setEnabled(bool actionEnabled, bool shortcutEnabled)
                {
                    if(m_action) 
                    {
                        m_action->setEnabled(actionEnabled);
                        if(m_shortcut) m_shortcut->setEnabled(shortcutEnabled);
                    }
                }

                void setEnabled(bool enabled)
                {
                    setEnabled(enabled,enabled);
                }

                void setVisible(bool actionVisible, bool shortcutEnabled)
                {
                    if(m_action) 
                    {
                        m_action->setVisible(actionVisible);
                        if(m_shortcut) m_shortcut->setEnabled(shortcutEnabled);
                    }
                }

                void setVisible(bool visible)
                {
                    setVisible(visible,visible);
                }

                QAction* action() { return m_action; }

            private:
                QAction *m_action;
                QShortcut *m_shortcut;
            };

            //! eventFilter for m_pWindow
            /*!
                depending on m_floatingStyle and the docked property, close-events will be catched by the closeEvent-method of this docking-widget or by the closeEvent
                of m_pWindow, which is not overloaded directly. Therefore this event-filter is installed and in case of a QCloseEvent, the closeEvent-method of this
                docking widget will be invoked in order to handle the close request. Else, the event will be passed and handled by somebody else.
            */
            bool eventFilter(QObject *obj, QEvent *event)
            {
                if (event->type() == QEvent::Close)
                {
                    closeEvent((QCloseEvent*)event);
                    if(testAttribute(Qt::WA_DeleteOnClose) && event->isAccepted() ) //if window should be closed and dockwidget is assigned with WA_DeleteOnClose, delete this dockwidget first.
                    {
                       deleteLater();
                    }
                    
                    return true;
                }
                else
                {
                    return QObject::eventFilter(obj,event);
                }
            };

            void init();

            virtual void closeEvent(QCloseEvent *event);

            virtual void createActions() = 0;
            virtual void createMenus() = 0;
            virtual void createToolBars() = 0;
            virtual void createStatusBar() = 0;
            virtual void updateActions() {}
            virtual void updatePythonActions() = 0;
	    
	    Qt::WindowFlags modifyFlags(const Qt::WindowFlags &flags, const Qt::WindowFlags &setFlags, const Qt::WindowFlags &unsetFlags);

            virtual void windowStateChanged( bool /*windowNotToolbox*/ ) {}

            void setContentWidget(QWidget *widget);
            inline QWidget* getContentWidget() const { return m_pWindow->centralWidget(); }   /*!<  returns reference to central widget of docking window */
            inline QMainWindow* getCanvas() { return m_pWindow; }

            inline bool pythonBusy() const { return m_pythonBusy; }                    /*!<  returns if python is busy (true) */
            inline bool pythonDebugMode() const { return m_pythonDebugMode; }          /*!<  returns if python is in debug mode (true) */
            inline bool pythonInWaitingMode() const { return m_pythonInWaitingMode; }  /*!<  returns if python is in waiting mode (true) \sa m_pythonInWaitingMode */

            //RetVal addAndRegisterToolBar(QToolBar* tb, QString key);
            //RetVal unregisterToolBar(QString key);
            QToolBar* getToolBar(QString key) const;
            inline QMenuBar* getMenuBar() const { return (m_pWindow == NULL) ?  NULL : m_pWindow->menuBar(); }

            RetVal addToolBar(QToolBar *tb, const QString &key, Qt::ToolBarArea area = Qt::TopToolBarArea, int section = 1);
            RetVal removeToolBar(const QString &key);

            QAction *m_actStayOnTop;
            QAction *m_actStayOnTopOfApp;

        private:

            void contextMenuEvent(QContextMenuEvent *e)
            {
                e->accept();
                qDebug() << "context menu of abstractDockWidget clicked. [placeholder for (un)docking feature.] pos: " << e->pos();
            }

            QMainWindow *m_pWindow;

            bool m_docked;                      /*!<  flag indicating whether this instance is docked (true) or not (false) */
            bool m_dockAvailable;               /*!<  flag indicating whether docking functionality is available (true) */
            tFloatingStyle m_floatingStyle;     /*!<  floating style of dock widget \sa tFloatingStyle */
            tMovingStyle m_movingStyle;         /*!<  moving style of dock widget \sa tMovingStyle */
            QString m_basicTitle;               /*!<  basic title for this instance, shown if dock widget is docked */
            QString m_completeTitle;            /*!<  complete title for this instance, shown if dock widget is undocked */

            QMap<QString,QToolBar*> m_toolBars; /*!<  map of different toolbars, which are shown in undocked version with bigger icon sizes comparing to docked version */
            QList<Toolbar> m_toolbars;

            bool m_pythonBusy;                  /*!<  if true, python is busy right now */
            bool m_pythonDebugMode;             /*!<  if true, python is in debug mode right now */
            bool m_pythonInWaitingMode;         /*!<  if true, python is in debug mode but waiting for next user command (e.g. the debugger waits at a breakpoint) */

            QToolBar* m_dockToolbar;
            QAction* m_actDock;
            QAction* m_actUndock;

            QWidget* m_overallParent;           /*!< parent given by constructor, which is the parent of both the dock widget and the main window in floating mode */
            tTopLevelStyle m_recentTopLevelStyle;

            QSize m_oldMinSize;
            QSize m_oldMaxSize;
            QRect m_lastUndockedSize;
    
        //signals:
        //    void addDockToMainWindow(AbstractDockWidget *widget);       /*!<  signal emitted if widget should be docked to main window */
        //    void removeDockFromMainWindow(AbstractDockWidget *widget);  
        //    void dockCloseRequest(AbstractDockWidget *widget);

        public slots:
            virtual void dockedToMainWindow(AbstractDockWidget * /*widget*/){}
            virtual void removedFromMainWindow(AbstractDockWidget * /*widget*/){}
            virtual void pythonStateChanged(tPythonTransitions pyTransition);

            void raiseAndActivate();  /*!< activates this dock widget or window and raises it on top of all opened windows (if possible) */

            void setDockSize(int newWidth, int newHeight);

            void dockWidget();
            void undockWidget();

        private slots:
            void mnuStayOnTop(bool checked);
            void mnuStayOnTopOfApp(bool checked);

            void returnToOldMinMaxSizes();
    
    };

} //end namespace ito

#endif
