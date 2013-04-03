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

#include <string>
//#ifndef Q_MOC_RUN
//    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must bebefore include global.h)
//    #define NO_IMPORT_ARRAY
//#endif
#include "../python/pythonEngineInc.h"

#include "abstractDockWidget.h"
#include "../AppManagement.h"

#include <qstyle.h>
#include <QSpacerItem>
#include <qwidget.h>
#include <qlayout.h>
#include <qtimer.h>

namespace ito {

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
AbstractDockWidget::AbstractDockWidget(bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle, const QString &title, QWidget *parent) :
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
    PythonEngine* pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if(pyEngine != NULL)
    {
        m_pythonBusy = pyEngine->isPythonBusy();
        m_pythonDebugMode = pyEngine->isPythonDebugging();
        m_pythonInWaitingMode = pyEngine->isPythonDebuggingAndWaiting();
    }

    m_toolBars.clear();

    if(title.isNull())
    {
        QDockWidget(parent);
    }
    else
    {
        QDockWidget(title, parent);
    }

    //m_docked = docked && isDockAvailable;

    setWindowTitle(title);

    m_actStayOnTop = new QAction(QIcon(""), tr("stay on top"), this);
    m_actStayOnTop->setCheckable(true);
    connect(m_actStayOnTop, SIGNAL(triggered(bool)), this, SLOT(mnuStayOnTop(bool)));
    m_actStayOnTopOfApp = new QAction(tr("stay on top of main window"), this);
    m_actStayOnTopOfApp->setCheckable(true);
    connect(m_actStayOnTopOfApp, SIGNAL(triggered(bool)), this, SLOT(mnuStayOnTopOfApp(bool)));
}

//! destructor
AbstractDockWidget::~AbstractDockWidget()
{
    DELETE_AND_SET_NULL(m_pWindow);
    m_toolBars.clear();

	if(PythonEngine::getInstance())
	{
		disconnect(PythonEngine::getInstance(), SIGNAL(pythonStateChanged(tPythonTransitions)), this, SLOT(pythonStateChanged(tPythonTransitions)));
	}
}

//! init method, called by constructor
/*!
    \sa AbstractDockWidget::AbstractDockWidget
*/
void AbstractDockWidget::init()
{
    m_pWindow = new QMainWindow(this);

    m_pWindow->installEventFilter(this);

    m_pWindow->setWindowFlags(Qt::Widget);
    setWidget(m_pWindow);


    QDockWidget::DockWidgetFeatures features = QDockWidget::DockWidgetClosable;

    if(m_floatingStyle == floatingStandard)
    {
        features |= QDockWidget::DockWidgetFloatable;
    }
    else if(m_floatingStyle == floatingWindow)
    {
        //nothing
    }

    if(m_movingStyle == movingEnabled)
    {
        features |= QDockWidget::DockWidgetMovable;
    }

    setFeatures(features);


    m_dockToolbar = new QToolBar("docking toolbar",this);

    QWidget *spacerWidget = new QWidget();
    QHBoxLayout *spacerLayout = new QHBoxLayout();
    spacerLayout->addItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Minimum));
    spacerWidget->setLayout(spacerLayout);

    m_actDock = new QAction(QIcon(":/dockWidget/icons/dockButtonGlyph.png"), "dock widget",this);
    connect(m_actDock, SIGNAL(triggered()), this, SLOT(dockWidget()));
    m_actUndock = new QAction(QIcon(":/dockWidget/icons/undockButtonGlyph.png"), "undock widget",this);
    connect(m_actUndock, SIGNAL(triggered()), this, SLOT(undockWidget()));

    m_dockToolbar->addWidget(spacerWidget);
    m_dockToolbar->addAction(m_actDock);
    m_dockToolbar->addAction(m_actUndock);

    m_dockToolbar->setVisible(m_floatingStyle == floatingWindow && m_dockAvailable); //only show dock-toolbar, if this widget is able to be a full-window in undocked mode
    m_dockToolbar->setMovable(false);

    m_pWindow->addToolBar(m_dockToolbar);

	if(PythonEngine::getInstance())
	{
		connect(PythonEngine::getInstance(), SIGNAL(pythonStateChanged(tPythonTransitions)), this, SLOT(pythonStateChanged(tPythonTransitions)));
	}

    createActions();
    createMenus();
    createToolBars();
    createStatusBar();

    if(docked())
    {
        dockWidget();
    }
    else
    {
        undockWidget();
    }

    updateActions();
}

void AbstractDockWidget::setVisible(bool visible)
{
    if(m_docked)
    {
        QDockWidget::setVisible(visible);
    }
    else
    {
        if(m_floatingStyle == floatingWindow)
        {
            m_pWindow->setVisible(visible);
            if(visible == false)
            {
                QDockWidget::setVisible(visible);
            }
        }
        else
        {
            QDockWidget::setVisible(visible);
        }
    }
}

//! resizes this widget to the given size
/*!
    \param width
    \param height
*/
void AbstractDockWidget::resizeDockWidget(int width, int height)
{
    resize(width, height);
    m_pWindow->resize(width, height);
}


//! sets any given QWidget as central widget of QMainWindow and inversely sets this QWidget's parent to the instance of QMainWindget
/*!
    \param widget instance of QWidget, which should be the central widget of this dock widget
*/
void AbstractDockWidget::setContentWidget(QWidget *widget)
{
    if(m_pWindow != NULL && widget != NULL)
    {
        m_pWindow->setCentralWidget(widget);
        widget->setParent(m_pWindow);
    }
    else
    {
        qDebug("call AbstractDockWidget::init() first, since window is NULL");
    }
}

//! changes the title of widget
/*!
    If widget is docked or m_floatingStyle is not equal to floatingWindow, this widget always gets the title m_basicTitle. Otherwise its title is set to m_completeTitle.
    This member is either set to newCompleteTitle if prependToBasicTitle is equal to false, otherwise newCompleteTitle is prepended to m_basicTitle.

    \param newCompleteTitle new complete title for this widget (e.g. containing filename and modification sign *)
    \param prependToBasicTitle true if member m_completeTitle should consist of newCompleteTitle + m_basicTitle, false if m_completeTitle is equal to newCompleteTitle
    \return retOk
*/
RetVal AbstractDockWidget::setAdvancedWindowTitle( QString newCompleteTitle, bool prependToBasicTitle )
{
    if(!newCompleteTitle.isNull())
    {
        if(prependToBasicTitle)
        {
            m_completeTitle = m_basicTitle;
            m_completeTitle.append(newCompleteTitle);
        }
        else
        {
            m_completeTitle = newCompleteTitle;
        }
    }
    if(!docked() && m_floatingStyle == floatingWindow )
    {
        setWindowTitle(m_completeTitle);
        m_pWindow->setWindowTitle(m_completeTitle);
    }
    else
    {
        setWindowTitle(m_completeTitle); //m_basicTitle);
        m_pWindow->setWindowTitle(m_completeTitle); //m_basicTitle);
    }
    return RetVal(retOk);
}



////! ands given toolbar and register it with given key-string in toolbar-map (m_toolBars)
///*!
//    long description
//
//    \param tb reference to toolbar of type QToolBar
//    \param key string containing key for this toolbar
//    \return retOk, if toolbar could be added, retError, if key already exists in map
//*/
//RetVal AbstractDockWidget::addAndRegisterToolBar(QToolBar* tb, QString key)
//{
//    QMap<QString,QToolBar*>::iterator it = m_toolBars.find(key);
//
//    if(it == m_toolBars.end())
//    {
//        m_pWindow->insertToolBar(m_dockToolbar, tb);
//
//        if(!m_docked && m_floatingStyle == floatingWindow)
//        {
//            tb->setIconSize(QSize(style()->pixelMetric(QStyle::PM_ToolBarIconSize),style()->pixelMetric(QStyle::PM_ToolBarIconSize)));
//        }
//        else
//        {
//            tb->setIconSize(QSize(16,16));
//        }
//
//        m_toolBars.insert(key,tb);
//        return RetVal(retOk);
//    }
//    return RetVal(retError);
//}

////! remove toolbar with given key from m_toolBars
///*!
//    \param key key-string to toolbar which should be removed
//    \return retOk, if toolbar could be removed, retError, if key has not been found in toolbar-map
//*/
//RetVal AbstractDockWidget::unregisterToolBar(QString key)
//{
//    int nr = m_toolBars.remove(key);
//    
//
//    if(nr == 0)
//    {
//        return RetVal(retError);
//    }
//    else
//    {
//        return RetVal(retOk);
//    }
//}

RetVal AbstractDockWidget::addToolBar(QToolBar *tb, const QString &key, Qt::ToolBarArea area /*= Qt::TopToolBarArea*/, int section /*= 1*/)
{
    QList<Toolbar>::iterator i;
    int highestSection = 1;

    //check if key already available and quit if so
    for(i = m_toolbars.begin() ; i != m_toolbars.end() ; ++i)
    {
        if(i->key == key)
        {
            return RetVal(retError,0,tr("toolbar '%1' is already available").arg(key).toAscii().data());
        }

        if(i->area == area && i->section > highestSection)
        {
            highestSection = i->section;
        }
    }

    if(!m_docked && m_floatingStyle == floatingWindow)
    {
        tb->setIconSize(QSize(style()->pixelMetric(QStyle::PM_ToolBarIconSize),style()->pixelMetric(QStyle::PM_ToolBarIconSize)));
    }
    else
    {
        tb->setIconSize(QSize(16,16));
    }

    if(area == Qt::TopToolBarArea)
    {
        if(highestSection < section)
        {
            m_pWindow->addToolBarBreak(area);
            m_pWindow->addToolBar(tb);
        }
        else if(section == 1)
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
        if(highestSection < section)
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

    return retOk;
}

RetVal AbstractDockWidget::removeToolBar(const QString &key)
{
    //key is unique
    QList<Toolbar>::iterator i;
    Qt::ToolBarArea area;
    int section;
    int count = 0;
    QToolBar *tb = NULL;
    int idx = 0;

    for(i = m_toolbars.begin() ; i != m_toolbars.end() ; ++i)
    {
        if(i->key == key)
        {
            tb = i->tb;
            area = i->area;
            section = i->section;
            break;
        }

        idx++;
    }

    if(tb)
    {
        //count toolbars in same section and area
        //do this only if section > 1, since this is in order to remove breaks, which are not available before 1 section
        if(section > 1)
        {
            for(i = m_toolbars.begin() ; i != m_toolbars.end() ; ++i)
            {
                if(i->section == section && i->area == area)
                {
                    count++;
                }
            }
        }

        if(count == 1)
        {
            //tb is the last in this section, remove break before this section
            m_pWindow->removeToolBarBreak(tb);
        }

        m_pWindow->removeToolBar(tb);
        m_toolbars.takeAt(idx);

        return retOk;

    }

    return RetVal(retError, 0, tr("toolbar '%1' not found").arg(key).toAscii().data());
}

//! returns reference to toolbar with given key-value
/*!
    \param key String with key value for desired toolbar
    \return reference to QToolBar or NULL, if key not found
*/
QToolBar* AbstractDockWidget::getToolBar(QString key) const
{
    QMap<QString,QToolBar*>::const_iterator it = m_toolBars.find(key);

    if(it != m_toolBars.end())
    {
        return *it;
    }
    else
    {
        return NULL;
    }
}

//! closeEvent invoked if this AbstractDockWidget should be closed
/*!
    In this abstract class, the event is always accepted.

    \note Please overwritte this method by derived class in order to realize desired behaviour.

    \param event Event of type QCloseEvent
*/
void AbstractDockWidget::closeEvent ( QCloseEvent * event )
{
    event->accept();
}

//! slot invoked if python state changed. Sets the specific member variables according to the python transition.
/*!
    \param pyTransition pythonTransition (type tPythonTransitions) defining the transition to the destination python state
*/
void AbstractDockWidget::pythonStateChanged(tPythonTransitions pyTransition)
{
    switch(pyTransition)
    {
    case pyTransBeginRun:
        m_pythonInWaitingMode=false;
        m_pythonDebugMode = false;
        m_pythonBusy = true;
        break;
    case pyTransBeginDebug:
    case pyTransDebugExecCmdBegin:
        m_pythonInWaitingMode=false;
        m_pythonDebugMode = true;
        m_pythonBusy = true;
        break;
    case pyTransDebugContinue:
        m_pythonBusy = true;
        m_pythonDebugMode = true;
        m_pythonInWaitingMode=false;
        break;
    case pyTransEndRun:
    case pyTransEndDebug:
        m_pythonDebugMode = false;
        m_pythonBusy = false;
        m_pythonInWaitingMode=false;
        break;
    case pyTransDebugWaiting:
    case pyTransDebugExecCmdEnd:
        m_pythonInWaitingMode=true;
        m_pythonDebugMode = true;
        m_pythonBusy = true;
        break;
    }

    updatePythonActions();
}

//! docks this dockWidget.
/*!
    Transforms this widget into a docking widget (e.g. from a single window) and docks it into the main window.
    The menubar is hidden and the size of all icons is decreased to 16x16 px.
*/
void AbstractDockWidget::dockWidget()
{
    if(m_docked == false && m_floatingStyle == floatingWindow)
    {
        m_lastUndockedSize = m_pWindow->geometry();
    }

    //qDebug() << "AbstractDockWidget::dockWidget start";
    m_docked = true;

    setWindowFlags(Qt::Widget);

    m_pWindow->setWindowFlags(Qt::Widget);
    m_pWindow->setWindowFlags( m_pWindow->windowFlags() & ~(Qt::WindowStaysOnTopHint) );
    m_pWindow->setParent(this);
    //qDebug() << "AbstractDockWidget::dockWidget 0" << (int)m_pWindow << " " << (int)m_overallParent;
    setWidget(m_pWindow);
    //qDebug() << "AbstractDockWidget::dockWidget 0a";
    setParent(m_overallParent);
    //qDebug() << "AbstractDockWidget::dockWidget 0b";
    setFloating(false);
    //qDebug() << "AbstractDockWidget::dockWidget 0c";
    QDockWidget::setVisible(true); //show();
    //qDebug() << "AbstractDockWidget::dockWidget 0d";
    m_pWindow->menuBar()->hide();
    m_actDock->setVisible(false);
    m_actUndock->setVisible(true);

    windowStateChanged( false );

    m_dockToolbar->setIconSize(QSize(20,15));

    QList<Toolbar>::iterator it;
    for(it = m_toolbars.begin() ; it != m_toolbars.end() ; ++it)
    {
        it->tb->setIconSize(QSize(16,16));
    }

    //QMap<QString,QToolBar*>::iterator it;
    ////qDebug() << "AbstractDockWidget::dockWidget 2";
    //for(it = m_toolBars.begin() ; it != m_toolBars.end(); ++it)
    //{
    //    (*it)->setIconSize(QSize(16,16));
    //}
    //qDebug() << "AbstractDockWidget::dockWidget 3";
    setAdvancedWindowTitle();

    toggleViewAction()->setVisible(true);
    //qDebug() << "AbstractDockWidget::dockWidget end";
}

//! undocks this dockWidget.
/*!
    If m_floatingStyle is equal to floatingWindow, then this widget is transformed into a single window
    with its own toolbar and menubar. All Icons are increased in size in order to have the single-window-look.
*/
void AbstractDockWidget::undockWidget()
{
    bool m_docked_old = m_docked;
    m_docked = false;

    if(m_floatingStyle == floatingWindow)
    {
        m_pWindow->menuBar()->show();
        m_actDock->setVisible(true);
        m_actUndock->setVisible(false);

        windowStateChanged( true );

        //setWindowFlags(Qt::Window);
        //setFloating(true);
        

        m_dockToolbar->setIconSize(QSize(style()->pixelMetric(QStyle::PM_ToolBarIconSize),style()->pixelMetric(QStyle::PM_ToolBarIconSize)));

        m_pWindow->setWindowFlags(Qt::Window);

        if(m_docked_old && !m_lastUndockedSize.isEmpty())
        {
            m_pWindow->setGeometry(m_lastUndockedSize);
        }

        if(m_docked_old)
        {
            show();
        }

        setTopLevel(m_recentTopLevelStyle);

        setWindowFlags(Qt::Widget);
        setFloating(true);
        QDockWidget::hide();

        m_pWindow->show();
        m_pWindow->raise();

        toggleViewAction()->setVisible(false);
    }
    else
    {
        setParent(m_overallParent);
        m_pWindow->setParent(this);
        setWindowFlags(Qt::Widget);
        setFloating(true);

        m_dockToolbar->setIconSize(QSize(20,15));
    }

    QList<Toolbar>::iterator it;
    for(it = m_toolbars.begin() ; it != m_toolbars.end() ; ++it)
    {
        if(m_floatingStyle == floatingWindow)
        {
            it->tb->setIconSize(QSize(style()->pixelMetric(QStyle::PM_ToolBarIconSize),style()->pixelMetric(QStyle::PM_ToolBarIconSize)));
        }
        else
        {
            it->tb->setIconSize(QSize(16,16));
        }
    }


    /*QMap<QString,QToolBar*>::iterator it;

    for(it = m_toolBars.begin() ; it != m_toolBars.end(); ++it)
    {
        if(m_floatingStyle == floatingWindow)
        {
            (*it)->setIconSize(QSize(style()->pixelMetric(QStyle::PM_ToolBarIconSize),style()->pixelMetric(QStyle::PM_ToolBarIconSize)));
        }
        else
        {
            (*it)->setIconSize(QSize(16,16));
        }
    }*/

    setAdvancedWindowTitle();
}

RetVal AbstractDockWidget::setTopLevel( tTopLevelStyle topLevel )
{
    //only feasible if undocked and floatingStyle == floatingWindow
    if(!m_docked && m_floatingStyle == floatingWindow)
    {
        Qt::WindowFlags flags = m_pWindow->windowFlags();

        switch(topLevel)
        {
        case topLevelNothing:
            m_pWindow->setParent(NULL);
            m_pWindow->setWindowFlags( flags & ~(Qt::WindowStaysOnTopHint) );
            m_actStayOnTop->setChecked(false);
            m_actStayOnTopOfApp->setChecked(false);
            break;
        case topLevelOverall:
            m_pWindow->setParent(m_overallParent);
            m_pWindow->setWindowFlags( flags | Qt::WindowStaysOnTopHint );
            m_actStayOnTop->setChecked(true);
            m_actStayOnTopOfApp->setChecked(false);
            break;
        case topLevelParentOnly:
            m_pWindow->setParent(m_overallParent);
            m_pWindow->setWindowFlags( flags & ~(Qt::WindowStaysOnTopHint) );
            m_actStayOnTop->setChecked(false);
            m_actStayOnTopOfApp->setChecked(true);
            break;
        }
        m_pWindow->show();


    }

    m_recentTopLevelStyle = topLevel;

    return ito::retOk;
}

void AbstractDockWidget::mnuStayOnTop(bool checked)
{
    if(checked)
    {
        setTopLevel(topLevelOverall);
    }
    else
    {
        setTopLevel(topLevelNothing);
    }
}

void AbstractDockWidget::mnuStayOnTopOfApp(bool checked)
{
    if(checked)
    {
        setTopLevel(topLevelParentOnly);
    }
    else
    {
        setTopLevel(topLevelNothing);
    }
}

//! activates this dock widget or window and raises it on top of all opened windows
/*!
    Depending on the docking-state of this widget and its style (docking-widget or single-window),
    this widget is activated and if undocked raised on top of the window stack.

    \return retOk
*/
void AbstractDockWidget::raiseAndActivate()
{
    if(m_docked)
    {
        activateWindow();
        QDockWidget::show();
    }
    else if(m_floatingStyle == floatingWindow)
    {
        raise();
        m_pWindow->raise();
        m_pWindow->activateWindow();
    }
    else
    {
        raise();
        m_pWindow->activateWindow();
        activateWindow();
    }
}

//hack from: http://qt-project.org/faq/answer/how_can_i_resize_a_qdockwidget_programatically
void AbstractDockWidget::setDockSize(int newWidth, int newHeight)
{
	if(m_docked)
	{
		m_oldMaxSize= maximumSize();
		m_oldMinSize= minimumSize();
 
		if (newWidth>=0)
		{
			if (width()<newWidth)
			{
				setMinimumWidth(newWidth);
			}
			else
			{
				setMaximumWidth(newWidth);
			}
		}
		if (newHeight>=0)
		{
			if (height()<newHeight)
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
		if(newHeight > 0 && newWidth > 0)
		{
			m_pWindow->resize(newWidth,newHeight);
		}
		else if(newHeight > 0)
		{
			m_pWindow->resize(m_pWindow->width(), newHeight);
		}
		else if(newWidth > 0)
		{
			m_pWindow->resize(newWidth, m_pWindow->height());
		}
	}
}

void AbstractDockWidget::returnToOldMinMaxSizes()
{
	if(m_docked)
	{
		setMinimumSize(m_oldMinSize);
		setMaximumSize(m_oldMaxSize);
	}
}

} //end namespace ito
