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

#include <string>

#include "../python/pythonEngineInc.h"
#include "../python/pythonStatePublisher.h"

#include "abstractDockWidget.h"
#include "../AppManagement.h"
#include "../helper/guiHelper.h"

#include <qstyle.h>
#include <QSpacerItem>
#include <qwidget.h>
#include <qlayout.h>
#include <qtimer.h>
#include <qsettings.h>
#include <QScreen>

#include <qapplication.h>

#include <iostream>
#include <sstream>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class AbstractDockWidget
    \brief abstract dock widget class which inherits QDockWidget. The content of QDockWidget consists of an instance of QMainWindow. The real widget is then placed
            as central widget of this QMainWindow. QMainWindow has the ability to show a menubar as well as a toolbar, which is used in different versions depending
            on docked state. Inherit AbstractDockWidget in order to realize content of QMainWindow.
*/

//! constructor
/*!
    basic member initializations. Afterwards method \a init is called.

    \param docked true if widget should be docked at startup, else false
    \param isDockAvailable true if docking functionality is available, else false
    \param floatingStyle FloatingStyle of AbstractDockWidget \sa tFloatingStyle
    \param movingStyle MovingStyle of AbstractDockWidget \sa tMovingStyle
    \param title initial title for docking widget. members m_basicTitle and m_completeTitle are set to title, too.
    \param parent parent-widget, default: NULL
    \sa init
*/
AbstractDockWidget::AbstractDockWidget(bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle, const QString &title, const QString &objName, QWidget *parent) :
    QDockWidget(parent),
    m_actStayOnTop(NULL),
    m_actStayOnTopOfApp(NULL),
    m_pWindow(NULL),
    m_docked(docked && isDockAvailable),
    m_dockAvailable(isDockAvailable),
    m_floatingStyle(floatingStyle),
    m_movingStyle(movingStyle),
    m_basicTitle(title),
    m_completeTitle(title),
    m_pythonBusy(false),
    m_pythonDebugMode(false),
    m_pythonInWaitingMode(false),
    m_dockToolbar(NULL),
    m_actDock(NULL),
    m_actUndock(NULL),
    m_overallParent(parent),
    m_recentTopLevelStyle(topLevelNothing)
{
    if (objName != "")
    {
        this->setObjectName(objName);
    }

    PythonEngine* pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (pyEngine)
    {
        m_pythonBusy = pyEngine->isPythonBusy();
        m_pythonDebugMode = pyEngine->isPythonDebugging();
        m_pythonInWaitingMode = pyEngine->isPythonDebuggingAndWaiting();
    }

    m_toolBars.clear();

    if (title.isNull())
    {
        QDockWidget(parent);
    }
    else
    {
        QDockWidget(title, parent);
    }

    //m_docked = docked && isDockAvailable;

    QDockWidget::setWindowTitle(title);

    m_actStayOnTop = new QAction(QIcon(""), tr("Stay On Top"), this);
    m_actStayOnTop->setToolTip(tr("Stay on top of all visible windows"));
    m_actStayOnTop->setCheckable(true);
    connect(m_actStayOnTop, SIGNAL(triggered(bool)), this, SLOT(mnuStayOnTop(bool)));
    m_actStayOnTopOfApp = new QAction(tr("Stay On Top Of Main Window"), this);
    m_actStayOnTopOfApp->setToolTip(tr("Stay on top of main window of itom"));
    m_actStayOnTopOfApp->setCheckable(true);
    connect(m_actStayOnTopOfApp, SIGNAL(triggered(bool)), this, SLOT(mnuStayOnTopOfApp(bool)));

}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
AbstractDockWidget::~AbstractDockWidget()
{
    if (m_pWindow)
    {
        DELETE_AND_SET_NULL(m_pWindow);
    }

    m_toolBars.clear();

    const PythonStatePublisher *pyStatePublisher = qobject_cast<PythonStatePublisher*>(AppManagement::getPythonStatePublisher());

    if (pyStatePublisher)
    {
        disconnect(pyStatePublisher, &PythonStatePublisher::pythonStateChanged,
            this, &AbstractDockWidget::pythonStateChanged);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! init method, called by constructor
/*!
    \sa AbstractDockWidget::AbstractDockWidget
*/
void AbstractDockWidget::init()
{
    m_pWindow = new QMainWindow(this);
    m_pWindow->installEventFilter(this);
    m_pWindow->setWindowFlags(modifyFlags(m_pWindow->windowFlags(), Qt::Widget, Qt::Window));

    //m_pWindow->menuBar()->setNativeMenuBar(true);
    //linux: in some linux distributions, the menu bar did not appear if it is displayed
    //on top of the desktop. Therefore, native menu bars (as provided by the OS) are disabled here.
    //see: qt-project.org/forums/viewthread/7445
#ifndef __APPLE__
    m_pWindow->menuBar()->setNativeMenuBar(false);
#else // __APPLE__
    // OS X: without the native menu bar option, the menu bar is displayed within the window which might be irritating.
    m_pWindow->menuBar()->setNativeMenuBar(true);
#endif // __APPLE__

    setWidget(m_pWindow);

    QDockWidget::DockWidgetFeatures features = QDockWidget::DockWidgetClosable;

    if (m_floatingStyle == floatingStandard)
    {
        features |= QDockWidget::DockWidgetFloatable;
    }
    else if (m_floatingStyle == floatingWindow)
    {
        // nothing
    }

    if (m_movingStyle == movingEnabled)
    {
        features |= QDockWidget::DockWidgetMovable;
    }

    setFeatures(features);

    if (m_floatingStyle == floatingWindow && m_dockAvailable)
    {
        // only show dock-toolbar, if this widget is able to be a full-window in undocked mode
        m_dockToolbar = new QToolBar(tr("Docking Toolbar"), this);
        m_dockToolbar->setObjectName("toolBarDocking"); // to avoid Qt warnings during saveState / restoreState

        QWidget *spacerWidget = new QWidget();
        QHBoxLayout *spacerLayout = new QHBoxLayout();
        spacerLayout->addItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Minimum));
        spacerWidget->setLayout(spacerLayout);

        m_actDock = new QAction(QIcon(":/dockWidget/icons/dockButtonGlyph.png"), tr("Dock Widget"), this);
        connect(m_actDock, SIGNAL(triggered()), this, SLOT(dockWidget()));
        m_actUndock = new QAction(QIcon(":/dockWidget/icons/undockButtonGlyph.png"), tr("Undock Widget"), this);
        connect(m_actUndock, SIGNAL(triggered()), this, SLOT(undockWidget()));

        m_dockToolbar->addWidget(spacerWidget);
        m_dockToolbar->addAction(m_actDock);
        m_dockToolbar->addAction(m_actUndock);

        m_dockToolbar->setVisible(true);
        m_dockToolbar->setMovable(false);

        m_pWindow->addToolBar(m_dockToolbar);
    }

    const PythonStatePublisher *pyStatePublisher = \
        qobject_cast<PythonStatePublisher*>(AppManagement::getPythonStatePublisher());

    if (pyStatePublisher)
    {
        connect(pyStatePublisher, &PythonStatePublisher::pythonStateChanged,
            this, &AbstractDockWidget::pythonStateChanged);
    }

    createActions();
    createMenus();
    createToolBars();
    createStatusBar();

    if (parent())
    {
        if (docked())
        {
            dockWidget();
        }
        else
        {
            undockWidget();
        }
    }

    updateActions();
}

//---------------------------------------------------------------------------------------------------------
// BEGIN of the section where many properties of QWidget as base class of this class is overwritten such that depending on the dock-status
// the right properties of the underlying windows (dockWidget or mainWindow) are get or set.
//--------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------------------
QRect AbstractDockWidget::frameGeometry() const
{
    QWIDGETPROPGETTER(frameGeometry)
}

//----------------------------------------------------------------------------------------------------------------------------------
const QRect &AbstractDockWidget::geometry() const
{
    QWIDGETPROPGETTER(geometry)
}

//----------------------------------------------------------------------------------------------------------------------------------
QRect AbstractDockWidget::normalGeometry() const
{
    QWIDGETPROPGETTER(normalGeometry)
}

//----------------------------------------------------------------------------------------------------------------------------------
int AbstractDockWidget::x() const
{
    QWIDGETPROPGETTER(x)
}

//----------------------------------------------------------------------------------------------------------------------------------
int AbstractDockWidget::y() const
{
    QWIDGETPROPGETTER(y)
}

//----------------------------------------------------------------------------------------------------------------------------------
QPoint AbstractDockWidget::pos() const
{
    QWIDGETPROPGETTER(pos)
}

//----------------------------------------------------------------------------------------------------------------------------------
QSize AbstractDockWidget::frameSize() const
{
    QWIDGETPROPGETTER(frameSize)
}

//----------------------------------------------------------------------------------------------------------------------------------
QSize AbstractDockWidget::size() const
{
    QWIDGETPROPGETTER(size)
}

//----------------------------------------------------------------------------------------------------------------------------------
int AbstractDockWidget::width() const
{
    QWIDGETPROPGETTER(width)
}

//----------------------------------------------------------------------------------------------------------------------------------
int AbstractDockWidget::height() const
{
    QWIDGETPROPGETTER(height)
}

//----------------------------------------------------------------------------------------------------------------------------------
QRect AbstractDockWidget::rect() const
{
    QWIDGETPROPGETTER(rect)
}

//----------------------------------------------------------------------------------------------------------------------------------
QRect AbstractDockWidget::childrenRect() const
{
    QWIDGETPROPGETTER(childrenRect)
}

//----------------------------------------------------------------------------------------------------------------------------------
QRegion AbstractDockWidget::childrenRegion() const
{
    QWIDGETPROPGETTER(childrenRegion)
}

//----------------------------------------------------------------------------------------------------------------------------------

void AbstractDockWidget::move(int x, int y)
{
    QWIDGETPROPSETTER(move, x, y)
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::move(const QPoint &point)
{
    QWIDGETPROPSETTER(move, point)
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::resize(int w, int h)
{
    QWIDGETPROPSETTER(resize, w, h)
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::resize(const QSize &size)
{
    QWIDGETPROPSETTER(resize, size)
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::setGeometry(int x, int y, int w, int h)
{
    QWIDGETPROPSETTER(setGeometry, x, y, w, h)
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::setGeometry(const QRect &rect)
{
    QWIDGETPROPSETTER(setGeometry, rect)
}

//----------------------------------------------------------------------------------------------------------------------------------
QString AbstractDockWidget::windowTitle()
{
    QWIDGETPROPGETTER(windowTitle)
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::setWindowTitle(const QString &title)
{
    QWIDGETPROPSETTER(setWindowTitle, title)
}

//----------------------------------------------------------------------------------------------------------------------------------
bool AbstractDockWidget::isEnabled() const
{
    QWIDGETPROPGETTER(isEnabled)
}

//----------------------------------------------------------------------------------------------------------------------------------
bool AbstractDockWidget::isVisible() const
{
    QWIDGETPROPGETTER(isVisible)
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::setEnabled(bool enabled)
{
    QWIDGETPROPSETTER(setEnabled, enabled)
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::setMinimumSize(const QSize &size)
{
    QDockWidget::setMinimumSize(size);

    if (m_pWindow)
    {
        m_pWindow->setMinimumSize(size);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::setVisible(bool visible)
{
    if (visible)
    {
        //if undocked, the dock widget should not be shown, only the main window
        QWIDGETPROPSETTERSPECIAL(setVisible, visible)
    }
    else
    {
        QWIDGETPROPSETTER(setVisible, visible);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::saveState(const QString &iniName) const
{
    if (objectName() != "")
    {
        QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
        settings.beginGroup(iniName);
        settings.setValue("state", m_pWindow->saveState());
        settings.setValue("docked", docked());

        if (m_floatingStyle == floatingWindow)
        {
            QByteArray geometry = m_pWindow->saveGeometry();

			if (!docked())
			{
				settings.setValue("geometry", geometry);
			}

            if (!docked())
            {
                settings.setValue("visible", isVisible());
            }
            else
            {
                // invalidate setting 'visible' since it is always false if docked
                // (saveState is called in destructor, where visible is already false)
                settings.setValue("visible", QVariant());
            }
        }

        settings.endGroup();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::restoreState(const QString &iniName)
{
    if (objectName() != "")
    {
        QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
        settings.beginGroup(iniName);

        m_pWindow->restoreState(settings.value("state").toByteArray());

		QByteArray geometry = settings.value("geometry").toByteArray();
		QVariant docked = settings.value("docked");
		QVariant visible_ = settings.value("visible");
		bool visible = visible_.isValid() ? visible_.toBool() : true;

        if (m_floatingStyle == floatingWindow)
        {
            if (docked.isValid())
            {
                if (docked.toBool())
                {
                    if (!m_docked)
                    {
                        dockWidget();
                    }
                }
                else
                {
                    if (m_docked)
                    {
                        undockWidget(visible);
                    }
                }
            }

            if (visible_.isValid())
            {
                if (visible)
                {
                    setVisible(true);
                    m_pWindow->restoreGeometry(geometry);
                    //see also bug-report https://bugreports.qt.io/browse/QTBUG-21371 (fixed in >= Qt 5.3.0).
                }
                else
                {
                    setVisible(false);
                    m_pendingGeometryState = geometry;
                }
            }
        }
        else
        {
            QVariant docked = settings.value("docked");

            if (docked.isValid())
            {
                setFloating(!docked.toBool());
            }
        }
        settings.endGroup();
    }
}

//---------------------------------------------------------------------------------------------------------
// END of the property overwrite section
//--------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------------------
//! sets any given QWidget as central widget of QMainWindow and inversely sets this QWidget's parent to the instance of QMainWindget
/*!
    \param widget instance of QWidget, which should be the central widget of this dock widget
*/
void AbstractDockWidget::setContentWidget(QWidget *widget)
{
    if (m_pWindow != NULL && widget != NULL)
    {
        m_pWindow->setCentralWidget(widget);
        widget->setParent(m_pWindow);
    }
    else
    {
        qDebug("call AbstractDockWidget::init() first, since window is NULL");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! changes the title of widget
/*!
    If widget is docked or m_floatingStyle is not equal to floatingWindow, this widget always gets the title m_basicTitle. Otherwise its title is set to m_completeTitle.
    This member is either set to newCompleteTitle if prependToBasicTitle is equal to false, otherwise newCompleteTitle is prepended to m_basicTitle.

    \param newCompleteTitle new complete title for this widget (e.g. containing filename and modification sign *)
    \param prependToBasicTitle true if member m_completeTitle should consist of newCompleteTitle + m_basicTitle, false if m_completeTitle is equal to newCompleteTitle
    \return retOk
*/
RetVal AbstractDockWidget::setAdvancedWindowTitle(QString newCompleteTitle, bool prependToBasicTitle)
{
    if (!newCompleteTitle.isNull())
    {
        if (prependToBasicTitle)
        {
            m_completeTitle = m_basicTitle;
            m_completeTitle.append(newCompleteTitle);
        }
        else
        {
            m_completeTitle = newCompleteTitle;
        }
    }
    if (!docked() && m_floatingStyle == floatingWindow)
    {
        QDockWidget::setWindowTitle(m_completeTitle);
        m_pWindow->setWindowTitle(m_completeTitle);
    }
    else
    {
        QDockWidget::setWindowTitle(m_completeTitle); //m_basicTitle);
        m_pWindow->setWindowTitle(m_completeTitle); //m_basicTitle);
    }
    return RetVal(retOk);
}


//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractDockWidget::addToolBar(QToolBar *tb, const QString &key, Qt::ToolBarArea area /*= Qt::TopToolBarArea*/, int section /*= 1*/)
{
    QList<Toolbar>::iterator i;
    int highestSection = 1;

    //check if key already available and quit if so
    for (i = m_toolbars.begin(); i != m_toolbars.end(); ++i)
    {
        if (i->key == key)
        {
            return RetVal(retError, 0, tr("Toolbar '%1' is already available").arg(key).toLatin1().data());
        }

        if (i->area == area && i->section > highestSection)
        {
            highestSection = i->section;
        }
    }

    if (!m_docked && m_floatingStyle == floatingWindow)
    {
        tb->setIconSize(QSize(style()->pixelMetric(QStyle::PM_ToolBarIconSize), style()->pixelMetric(QStyle::PM_ToolBarIconSize)));
    }
    else
    {
        //adjust the height of the toolbar in docked status to 16 px with a 96dpi screen and scale it to other pixels for higher resolving screens.
        int size = 16 * GuiHelper::screenDpiFactor();
        tb->setIconSize(QSize(size, size));
    }

    if (area == Qt::TopToolBarArea)
    {
        if (highestSection < section)
        {
            m_pWindow->addToolBarBreak(area);
            m_pWindow->addToolBar(tb);
        }
        else if (section == 1)
        {
            if (m_dockToolbar)
            {
                m_pWindow->insertToolBar(m_dockToolbar, tb);
            }
            else
            {
                m_pWindow->addToolBar(tb);
            }
        }
        else
        {
            m_pWindow->addToolBar(tb);
        }
    }
    else
    {
        if (highestSection < section)
        {
            m_pWindow->addToolBarBreak(area);
            m_pWindow->addToolBar(tb);
        }
        else
        {
            m_pWindow->addToolBar(tb);
        }
    }

    Toolbar t;
    t.area = area;
    t.key = key;
    t.section = section;
    t.tb = tb;
    m_toolbars.append(t);

#ifdef __APPLE__
    // Bug fix: Issue #35
    // OS X hides windows/ dialog after adding a toolbar
    m_pWindow->show();
#endif // __APPLE__

    return retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractDockWidget::removeToolBar(const QString &key)
{
    //key is unique
    QList<Toolbar>::iterator i;
    Qt::ToolBarArea area;
    int section;
    int count = 0;
    QToolBar *tb = NULL;
    int idx = 0;

    for (i = m_toolbars.begin(); i != m_toolbars.end(); ++i)
    {
        if (i->key == key)
        {
            tb = i->tb;
            area = i->area;
            section = i->section;
            break;
        }

        idx++;
    }

    if (tb)
    {
        //count toolbars in same section and area
        //do this only if section > 1, since this is in order to remove breaks, which are not available before 1 section
        if (section > 1)
        {
            for (i = m_toolbars.begin(); i != m_toolbars.end(); ++i)
            {
                if (i->section == section && i->area == area)
                {
                    count++;
                }
            }
        }

        if (count == 1)
        {
            //tb is the last in this section, remove break before this section
            m_pWindow->removeToolBarBreak(tb);
        }

        m_pWindow->removeToolBar(tb);
        m_toolbars.takeAt(idx);

        return retOk;

    }

    return RetVal(retError, 0, tr("Toolbar '%1' not found").arg(key).toLatin1().data());
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns reference to toolbar with given key-value
/*!
    \param key String with key value for desired toolbar
    \return reference to QToolBar or NULL, if key not found
*/
QToolBar* AbstractDockWidget::getToolBar(QString key) const
{
    QMap<QString, QToolBar*>::const_iterator it = m_toolBars.find(key);

    if (it != m_toolBars.end())
    {
        return *it;
    }
    else
    {
        return NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! closeEvent invoked if this AbstractDockWidget should be closed
/*!
    In this abstract class, the event is always accepted.

    \note Please overwrite this method by derived class in order to realize desired behaviour.

    \param event Event of type QCloseEvent
*/
void AbstractDockWidget::closeEvent (QCloseEvent * event)
{
    event->accept();
}


//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if python state changed. Sets the specific member variables according to the python transition.
/*!
    \param pyTransition pythonTransition (type tPythonTransitions) defining the transition to the destination python state
*/
void AbstractDockWidget::pythonStateChanged(tPythonTransitions pyTransition)
{
    switch(pyTransition)
    {
    case pyTransBeginRun:
        m_pythonInWaitingMode = false;
        m_pythonDebugMode = false;
        m_pythonBusy = true;
        break;
    case pyTransBeginDebug:
    case pyTransDebugExecCmdBegin:
        m_pythonInWaitingMode = false;
        m_pythonDebugMode = true;
        m_pythonBusy = true;
        break;
    case pyTransDebugContinue:
        m_pythonBusy = true;
        m_pythonDebugMode = true;
        m_pythonInWaitingMode = false;
        break;
    case pyTransEndRun:
    case pyTransEndDebug:
        m_pythonDebugMode = false;
        m_pythonBusy = false;
        m_pythonInWaitingMode = false;
        break;
    case pyTransDebugWaiting:
    case pyTransDebugExecCmdEnd:
        m_pythonInWaitingMode = true;
        m_pythonDebugMode = true;
        m_pythonBusy = true;
        break;
    }

    updatePythonActions();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! docks this dockWidget.
/*!
    Transforms this widget into a docking widget (e.g. from a single window) and docks it into the main window.
    The menubar is hidden and the size of all icons is decreased to 16x16 px.
*/
void AbstractDockWidget::dockWidget()
{
    if (m_docked == false && m_floatingStyle == floatingWindow)
    {
        m_lastUndockedSize = m_pWindow->geometry();
    }

    //m_pWindow->menuBar()->setNativeMenuBar(true);
    //linux: in some linux distributions, the menu bar did not appear if it is displayed
    //on top of the desktop. Therefore, native menu bars (as provided by the OS) are disabled here.
    //see: qt-project.org/forums/viewthread/7445
#ifndef __APPLE__
    m_pWindow->menuBar()->setNativeMenuBar(false);
#else // __APPLE__
    // OS X: without the native menu bar option, the menu bar is displayed within the window which might be irritating.
    m_pWindow->menuBar()->setNativeMenuBar(true);
#endif // __APPLE__

    m_docked = true;

    Qt::WindowFlags flags = m_pWindow->windowFlags();
    flags &= (~Qt::WindowStaysOnTopHint); //delete WindowStaysOnTopHint
    //the following two commented lines cause layout errors with Qt5 especially
    //flags &= (~Qt::Window); //delete Window
    //flags |= Qt::Widget; //add Qt::Widget flag
    m_pWindow->setWindowFlags(flags);
    m_pWindow->setParent(this);
    setWidget(m_pWindow);
    setParent(m_overallParent);
    setFloating(false);
    QDockWidget::setVisible(true);
    raiseAndActivate();

    m_pWindow->menuBar()->hide();

    if (m_actDock)
    {
        m_actDock->setVisible(false);
    }

    if (m_actUndock)
    {
        m_actUndock->setVisible(true);
    }

    windowStateChanged(false);

    float dpiScale = GuiHelper::screenDpiFactor();

    if (m_dockToolbar)
    {
        m_dockToolbar->setIconSize(QSize(20 * dpiScale, 15 * dpiScale));
        m_dockToolbar->setMinimumWidth(49 * dpiScale);
    }

    QList<Toolbar>::iterator it;

    for (it = m_toolbars.begin(); it != m_toolbars.end(); ++it)
    {
        it->tb->setIconSize(QSize(16 * dpiScale, 16 * dpiScale));
    }

    setAdvancedWindowTitle();

    toggleViewAction()->setVisible(true);

    emit dockStateChanged(true);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! undocks this dockWidget.
/*!
    If m_floatingStyle is equal to floatingWindow, then this widget is transformed into a single window
    with its own toolbar and menubar. All Icons are increased in size in order to have the single-window-look.
*/
void AbstractDockWidget::undockWidget(bool show_it /*= true*/)
{
    bool m_docked_old = m_docked;
    m_docked = false;
    float dpiFactor = GuiHelper::screenDpiFactor();

    if (m_floatingStyle == floatingWindow)
    {
        //m_pWindow->menuBar()->setNativeMenuBar(true);
        //linux: in some linux distributions, the menu bar did not appear if it is displayed
        //on top of the desktop. Therefore, native menu bars (as provided by the OS) are disabled here.
        //see: qt-project.org/forums/viewthread/7445
#ifndef __APPLE__
        m_pWindow->menuBar()->setNativeMenuBar(false);
#else // __APPLE__
        // OS X: without the native menu bar option, the menu bar is displayed within the window which might be irritating.
        m_pWindow->menuBar()->setNativeMenuBar(true);
#endif // __APPLE__

        m_pWindow->menuBar()->show();
        if (m_actDock)
        {
            m_actDock->setVisible(true);
        }

        if (m_actUndock)
        {
            m_actUndock->setVisible(false);
        }

        windowStateChanged(true);

        //setWindowFlags(Qt::Window);
        //setFloating(true);

        if (m_dockToolbar)
        {
            m_dockToolbar->setIconSize(QSize(style()->pixelMetric(QStyle::PM_ToolBarIconSize), style()->pixelMetric(QStyle::PM_ToolBarIconSize)));
        }

        m_pWindow->setWindowFlags(modifyFlags(m_pWindow->windowFlags(), Qt::Window, Qt::Widget));

        if (m_docked_old && !m_lastUndockedSize.isEmpty())
        {
            m_pWindow->setGeometry(m_lastUndockedSize);
#if linux //also fixes the bug in lxde such that title bar is out of window, since frameGeometry is bigger than geometry
            m_pWindow->move(m_pWindow->geometry().topLeft() - m_pWindow->pos());
#endif
        }

        if (m_docked_old && show_it)
        {
            show();
        }

        setTopLevel(m_recentTopLevelStyle, show_it);

        setFloating(true);
        QDockWidget::hide();

        if (show_it)
        {
            m_pWindow->show();
            m_pWindow->raise();
        }

#if linux
    //in LXDE the window is sometimes positioned out of the window such that
    //the title bar is not visible any more. Therefore it is center in the
    //center of the current main window.
    if (m_lastUndockedSize.isEmpty())
    {

        QScreen *ps = QGuiApplication::primaryScreen();
        QRect overallRect = ps->availableGeometry();
        QPoint centerPoint = overallRect.center();
        m_pWindow->adjustSize();
        m_pWindow->move(centerPoint - m_pWindow->rect().center());

    }
#endif

        toggleViewAction()->setVisible(false);
    }
    else
    {
        setParent(m_overallParent);
        m_pWindow->setParent(this);
        setFloating(true);

        if (m_dockToolbar)
        {
            m_dockToolbar->setIconSize(QSize(20 * dpiFactor, 15 * dpiFactor));
        }
    }

    QList<Toolbar>::iterator it;

    for (it = m_toolbars.begin(); it != m_toolbars.end(); ++it)
    {
        if (m_floatingStyle == floatingWindow)
        {
            it->tb->setIconSize(QSize(style()->pixelMetric(QStyle::PM_ToolBarIconSize), style()->pixelMetric(QStyle::PM_ToolBarIconSize)));
        }
        else
        {
            it->tb->setIconSize(QSize(16 * dpiFactor, 16 * dpiFactor));
        }
    }

    if (m_dockToolbar)
    {
        m_dockToolbar->setMinimumWidth(55 * dpiFactor);
    }

    setAdvancedWindowTitle();

    emit dockStateChanged(false);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractDockWidget::setTopLevel(tTopLevelStyle topLevel, bool showWindow /*= true*/)
{
    //only feasible if undocked and floatingStyle == floatingWindow
    if (!m_docked && m_floatingStyle == floatingWindow)
    {
        Qt::WindowFlags flags = m_pWindow->windowFlags();

        switch(topLevel)
        {
        case topLevelNothing:
            m_pWindow->setParent(NULL);
            m_pWindow->setWindowFlags(flags & ~(Qt::WindowStaysOnTopHint));
            m_actStayOnTop->setChecked(false);
            m_actStayOnTopOfApp->setChecked(false);
            break;
        case topLevelOverall:
            m_pWindow->setParent(m_overallParent);
            m_pWindow->setWindowFlags(flags | Qt::WindowStaysOnTopHint);
            m_actStayOnTop->setChecked(true);
            m_actStayOnTopOfApp->setChecked(false);
            break;
        case topLevelParentOnly:
            m_pWindow->setParent(m_overallParent);
            m_pWindow->setWindowFlags(flags & ~(Qt::WindowStaysOnTopHint));
            m_actStayOnTop->setChecked(false);
            m_actStayOnTopOfApp->setChecked(true);
            break;
        }

        if (showWindow)
        {
            m_pWindow->show();
        }
    }

    m_recentTopLevelStyle = topLevel;

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::mnuStayOnTop(bool checked)
{
    if (checked)
    {
        setTopLevel(topLevelOverall);
    }
    else
    {
        setTopLevel(topLevelNothing);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::mnuStayOnTopOfApp(bool checked)
{
    if (checked)
    {
        setTopLevel(topLevelParentOnly);
    }
    else
    {
        setTopLevel(topLevelNothing);
    }
}
//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::mini()
{
    m_pWindow->setWindowState(Qt::WindowMinimized);
}
//----------------------------------------------------------------------------------------------------------------------------------
//! activates this dock widget or window and raises it on top of all opened windows
/*!
    Depending on the docking-state of this widget and its style (docking-widget or single-window),
    this widget is activated and if undocked raised on top of the window stack.

    \return retOk
*/
void AbstractDockWidget::raiseAndActivate()
{
    if (m_docked)
    {
        activateWindow();
        QDockWidget::show();
        QDockWidget::raise();
    }
    else if (m_floatingStyle == floatingWindow)
    {
        raise();
        m_pWindow->setWindowState( (m_pWindow->windowState() & ~Qt::WindowMinimized) | Qt::WindowActive);
        m_pWindow->raise(); //for MacOS
        m_pWindow->activateWindow(); //for Windows
        m_pWindow->show();

        if (m_pendingGeometryState != "")
        {
            m_pWindow->restoreGeometry(m_pendingGeometryState);
            m_pendingGeometryState = "";
        }
    }
    else
    {
        raise();
        m_pWindow->activateWindow();
        activateWindow();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//hack from: http://qt-project.org/faq/answer/how_can_i_resize_a_qdockwidget_programatically
void AbstractDockWidget::setDockSize(int newWidth, int newHeight)
{
    if (m_docked)
    {
        m_oldMaxSize = maximumSize();
        m_oldMinSize = minimumSize();

        if (newWidth >= 0)
        {
            if (width() < newWidth)
            {
                setMinimumWidth(newWidth);
            }
            else
            {
                setMaximumWidth(newWidth);
            }
        }
        if (newHeight >= 0)
        {
            if (height() < newHeight)
            {
                setMinimumHeight(newHeight);
            }
            else
            {
                setMaximumHeight(newHeight);
            }
        }

        QTimer::singleShot(1, this, SLOT(returnToOldMinMaxSizes()));
    }
    else
    {
        if (newHeight > 0 && newWidth > 0)
        {
            m_pWindow->resize(newWidth, newHeight);
        }
        else if (newHeight > 0)
        {
            m_pWindow->resize(m_pWindow->width(), newHeight);
        }
        else if (newWidth > 0)
        {
            m_pWindow->resize(newWidth, m_pWindow->height());
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractDockWidget::returnToOldMinMaxSizes()
{
    if (m_docked)
    {
        setMinimumSize(m_oldMinSize);
        setMaximumSize(m_oldMaxSize);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
Qt::WindowFlags AbstractDockWidget::modifyFlags(const Qt::WindowFlags &flags, const Qt::WindowFlags &setFlags /*= 0*/, const Qt::WindowFlags &unsetFlags /*= 0*/)
{
  Qt::WindowFlags out = flags;
  out |= setFlags;
  out &= (~unsetFlags);
  return out;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! synchronizes the top level state of the dock widget with the floating settings of this abstract dock widget
/*!
    If the floating style is floating window and the dock widget has been set as floated dock widget (e.g. by a restoreState method call)
    the dock widget is in the top level state, however it does not correspond to the desired undocked, main window style.
    This is synchronized and corrected by this function.
*/
void AbstractDockWidget::synchronizeTopLevelState()
{
    if (m_floatingStyle == floatingWindow && m_docked == true && isFloating())
    {
        undockWidget(false);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QSize AbstractDockWidget::sizeHint() const
{
    if (m_docked)
    {
        return QDockWidget::sizeHint();
    }
    else
    {
        return m_pWindow->sizeHint();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QSize AbstractDockWidget::minimumSizeHint() const
{
    if (m_docked)
    {
        return QDockWidget::minimumSizeHint();
    }
    else
    {
        return m_pWindow->minimumSizeHint();
    }
}

} //end namespace ito
