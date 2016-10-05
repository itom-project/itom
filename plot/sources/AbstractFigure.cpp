/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut fuer Technische
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

#include "../AbstractFigure.h"

#include <qaction.h>
#include <qtoolbar.h>
#include <qmenu.h>
#include <qmenubar.h>
#include <qevent.h>
#include <qsettings.h>
#include <qshortcut.h>

#include "../../common/typeDefs.h"
#include "../../common/addInInterface.h"
#include "../../common/apiFunctionsInc.h"
#include "QPropertyEditor/QPropertyEditorWidget.h"

namespace ito 
{

//------------------------------------------------------------------------------------------------------------------------
class AbstractFigurePrivate : QObject
{
public:
    AbstractFigurePrivate() :
        propertyDock(NULL),
        propertyEditorWidget(NULL),
        propertyObservedObject(NULL),
        toolbarsVisible(true)
    {}

    QList<QMenu*> menus;
    QList<AbstractFigure::ToolBarItem> toolbars;
    QList<AbstractFigure::ToolboxItem> toolboxes;
    QHash<QAction*, QShortcut*> shortcutActions;

    QDockWidget *propertyDock;
    QPropertyEditorWidget *propertyEditorWidget;
	
	QObject *propertyObservedObject;
    bool toolbarsVisible;
};

//----------------------------------------------------------------------------------------------------------------------------------
AbstractFigure::AbstractFigure(const QString &itomSettingsFile, WindowMode windowMode, QWidget *parent) : 
    QMainWindow(parent),
    AbstractNode(),
    d(NULL),
    m_itomSettingsFile(itomSettingsFile),
    m_apiFunctionsGraphBasePtr(NULL),
    m_apiFunctionsBasePtr(NULL),
    m_mainParent(parent),
    m_windowMode(windowMode),    
    m_lineCutType(tNoChildPlot),
    m_zSliceType(tNoChildPlot),
    m_zoomCutType(tNoChildPlot)
{
    d = new AbstractFigurePrivate();

    initialize();
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractFigure::~AbstractFigure()
{
    foreach(Channel *delChan, m_pChannels)
    {
        removeChannel(delChan);
    }
    m_pChannels.clear();

    //clear toolbars and menus. toolbars and toolboxes are only added
    //to the main window of the plot in the window modes standaloneInUi or
    //standaloneWindow. If so, they are deleted by the destructor of
    //the main window. Else they have to be deleted here.
	if (m_windowMode == ModeInItomFigure)
	{
		foreach(ToolBarItem t, d->toolbars)
		{
			if (t.toolbar)
			{
				t.toolbar->deleteLater();
			}
		}
	}
        
	if (m_windowMode == ModeInItomFigure)
	{
        foreach(ToolboxItem t, d->toolboxes)
        {
            if (t.toolbox)
            {
                t.toolbox->deleteLater();
            }
        }
    }

    foreach(QMenu *m, d->menus)
    {
        m->deleteLater();
    }

    d->menus.clear();
    d->toolbars.clear();
    d->toolboxes.clear();

    d->propertyDock = NULL;

    delete d;
    d = NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractFigure::initialize()
{
    //in all modes, plot is either embedded in itom figureWidget or in external ui-file. Therefore, it is always considered to be a widget
    switch (m_windowMode)
    {
        case AbstractFigure::ModeInItomFigure:
        case AbstractFigure::ModeStandaloneInUi:
            setWindowFlags(Qt::Widget);
            setAttribute(Qt::WA_DeleteOnClose, false);
            menuBar()->setVisible(false);
            break;
        case AbstractFigure::ModeStandaloneWindow:
            setWindowFlags(Qt::Window);
            setAttribute(Qt::WA_DeleteOnClose, true);
            menuBar()->setVisible(true);
            break;
    }

    d->propertyDock = new QDockWidget(tr("Properties"), this);
    d->propertyDock->setVisible(false);
    d->propertyDock->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetMovable);

    d->propertyEditorWidget = new QPropertyEditorWidget(d->propertyDock);
    d->propertyDock->setWidget(d->propertyEditorWidget);
	addToolbox(d->propertyDock, "properties", Qt::RightDockWidgetArea);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setPropertyObservedObject(QObject* obj)
{
    d->propertyObservedObject = obj;
    if (d->propertyEditorWidget)
    {
        d->propertyEditorWidget->setObject(obj);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::updatePropertyDock()
{
    if (d->propertyEditorWidget && d->propertyObservedObject)
    {
        d->propertyEditorWidget->updateObject(d->propertyObservedObject);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractFigure::addChannel(AbstractNode *child, ito::Param* parentParam, ito::Param* childParam, Channel::ChanDirection direction, bool deleteOnParentDisconnect, bool deleteOnChildDisconnect)
{
    ito::RetVal retVal;
    uint channelHash1 = ito::calculateChannelHash(this, parentParam, child, childParam);
    uint channelHash2 = ito::calculateChannelHash(child, childParam, this, parentParam);

    Channel *tempChannel;
    foreach(tempChannel, m_pChannels)
    {
        if ((tempChannel->getHash() == channelHash1) || (tempChannel->getHash() == channelHash2))
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("duplicate Channel, in addChannel").toLatin1().data());
        }
    }

    if (direction == Channel::parentToChild)
    {
        if (apiCompareParam(*childParam, *parentParam, retVal) == ito::tCmpFailed)
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("parameters incompatible, while adding channel").toLatin1().data());
        }
    }
    else if (direction == Channel::childToParent)
    {
        if (apiCompareParam(*parentParam, *childParam, retVal) == ito::tCmpFailed)
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("parameters incompatible, while adding channel").toLatin1().data());
        }
    }
    else
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("undefined channel direction, while adding channel").toLatin1().data());
    }

    Channel *newChannel = new Channel(this, parentParam, deleteOnParentDisconnect, child, childParam, deleteOnChildDisconnect, direction);
    m_pChannels.insert(newChannel->getUniqueID(), newChannel);
    newChannel->getChild()->addChannel(newChannel);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractFigure::addChannel(Channel *newChannel)
{
    if (newChannel->getChild() != this)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("invalid child pointer, in addChannel").toLatin1().data());
    }

    uint channelHash1 = ito::calculateChannelHash(this, newChannel->getChildParam(), newChannel->getParent(), newChannel->getParentParam());
    uint channelHash2 = ito::calculateChannelHash(newChannel->getParent(), newChannel->getParentParam(), this, newChannel->getChildParam());

    Channel *tempChannel;
    foreach(tempChannel, m_pChannels)
    {
        if ((tempChannel->getHash() == channelHash1) || (tempChannel->getHash() == channelHash2))
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("duplicate Channel, in addChannel").toLatin1().data());
        }
    }

    m_pChannels.insert(newChannel->getUniqueID(), newChannel);
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractFigure::removeChannelFromList(unsigned int uniqueID)
{
    ito::RetVal retval = ito::retOk;
    int delBehaviour = m_pChannels[uniqueID]->getDeleteBehaviour((AbstractNode*)this);

    if (!m_pChannels.contains(uniqueID))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("channel does not exist").toLatin1().data());
    }

    m_pChannels.remove(uniqueID);

    if (delBehaviour)
    {
        Channel *iterChannel;
        foreach(iterChannel, m_pChannels)
        {
            // connection is removed in the destructor of Connection so the following line is not necessary
            removeChannel(iterChannel);
        }
//        delete this;
        deleteLater();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractFigure::removeChannel(Channel *delChannel)
{
    if (!m_pChannels.contains(delChannel->getUniqueID()))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("channel does not exist").toLatin1().data());
    }

    unsigned int uniqueID = delChannel->getUniqueID();
    int delBehaviour = delChannel->getDeleteBehaviour((AbstractNode*)this);

    if (delChannel->getParent() == (AbstractNode*)this)
    {
        m_pChannels.remove(uniqueID);
        delChannel->getChild()->removeChannelFromList(uniqueID);
        delete delChannel;
    }
    else
    {
        delChannel->getParent()->removeChannel(delChannel); // maybe we do not need this function call if we check for existance here
    }

    if (delBehaviour)
    {
        Channel *iterChannel;
        foreach(iterChannel, m_pChannels)
        {
            // connection is removed in the destructor of Connection so the following line is not necessary
            removeChannel(iterChannel);
        }
        deleteLater();
    }
    
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::addMenu(QMenu *menu)
{
    //never adds to menuBar()
    d->menus.append(menu);
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<QMenu*> AbstractFigure::getMenus() const
{
    if (m_windowMode == AbstractFigure::ModeStandaloneInUi)
    {
        //in standalone mode, this plugin handles its own menus and toolbars
        return QList<QMenu*>();
    }
    else
    {
        return d->menus;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<AbstractFigure::ToolBarItem> AbstractFigure::getToolbars() const
{
    if (m_windowMode == AbstractFigure::ModeStandaloneInUi)
    {
        //in standalone mode, this plugin handles its own menus and toolbars
        return QList<AbstractFigure::ToolBarItem>();
    }
    else
    {
        return d->toolbars;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::addToolBar(QToolBar *toolbar, const QString &key, Qt::ToolBarArea area /*= Qt::TopToolBarArea*/, int section /*= 1*/)
{
    ToolBarItem item;
    item.key = key;
    item.area = area;
    item.toolbar = toolbar;
    item.visible = d->toolbarsVisible;
    item.section = section;

    int maxSection = 1;

    //get highest section for same area
    foreach (const ToolBarItem &titem, d->toolbars)
    {
        if (titem.area == area)
        {
            maxSection = std::max(maxSection, titem.section);
        }
    }
	//this signal is established in order to check if the toolbar war already destroyed
	bool test = connect(toolbar, SIGNAL(destroyed(QObject*)), this, SLOT(toolBarDestroyed(QObject*))); 

    d->toolbars.append(item);

    if (m_windowMode == AbstractFigure::ModeStandaloneInUi || m_windowMode == AbstractFigure::ModeStandaloneWindow)
    {
        if (maxSection < section)
        {
            QMainWindow::addToolBarBreak(area);
        }

        QMainWindow::addToolBar(area, toolbar);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::addToolBarBreak(const QString &key, Qt::ToolBarArea area /*= Qt::TopToolBarArea*/)
{
    ToolBarItem item;
    item.key = key;
    item.area = area;
    item.toolbar = NULL;
    item.visible = d->toolbarsVisible;
    item.section = 1;

    //get highest section for same area
    foreach(const ToolBarItem &titem, d->toolbars)
    {
        if (titem.area == area)
        {
            item.section = std::max(item.section, titem.section);
        }
    }

    d->toolbars.append(item);

    if (m_windowMode == AbstractFigure::ModeStandaloneInUi || m_windowMode == AbstractFigure::ModeStandaloneWindow)
    {
        QMainWindow::addToolBarBreak(area);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::showToolBar(const QString &key)
{
    QList<AbstractFigure::ToolBarItem>::iterator i;
    
    for (i = d->toolbars.begin(); i != d->toolbars.end(); ++i)
    {
        if (i->key == key)
        {
            i->visible = true;
            i->toolbar->setVisible(true && d->toolbarsVisible);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::hideToolBar(const QString &key)
{
    QList<AbstractFigure::ToolBarItem>::iterator i;
    
    for (i = d->toolbars.begin(); i != d->toolbars.end(); ++i)
    {
        if (i->key == key)
        {
            i->visible = false;
            i->toolbar->setVisible(false /*&& m_toolbarsVisible*/); //always false
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setApiFunctionGraphBasePtr(void **apiFunctionGraphBasePtr)
{ 
    this->importItomApiGraph(apiFunctionGraphBasePtr);
    m_apiFunctionsGraphBasePtr = apiFunctionGraphBasePtr; 
    ito::ITOM_API_FUNCS_GRAPH = apiFunctionGraphBasePtr;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setApiFunctionBasePtr(void **apiFunctionBasePtr)
{ 
    this->importItomApi(apiFunctionBasePtr);
    m_apiFunctionsBasePtr = apiFunctionBasePtr; 
    ito::ITOM_API_FUNCS = apiFunctionBasePtr;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool AbstractFigure::event(QEvent *e)
{
    //the event User+123 is emitted by UiOrganizer, if the API has been prepared and can
    //transmitted to the plugin. This assignment cannot be done directly, since 
    //the array ITOM_API_FUNCS is in another scope if called from itom. By sending an
    //event from itom to the plugin, this method is called and ITOM_API_FUNCS is in the
    //right scope. The methods above only set the pointers in the "wrong"-itom-scope (which
    //also is necessary if any methods of the plugin are directly called from itom).
    if (e->type() == (QEvent::User+123))
    {
        //importItomApi(m_apiFunctionsBasePtr);
        //importItomPlotApi(m_apiFunctionsGraphBasePtr);
        init();
    }   

    return QMainWindow::event(e);
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setToolbarVisible(bool visible)
{

    QList<AbstractFigure::ToolBarItem>::iterator i;
    
    for (i = d->toolbars.begin(); i != d->toolbars.end(); ++i)
    {
        if (i->toolbar)
        {
            i->toolbar->setVisible(visible && (*i).visible);
        }
    }

    d->toolbarsVisible = visible;
    updatePropertyDock();
}

//----------------------------------------------------------------------------------------------------------------------------------
bool AbstractFigure::getToolbarVisible() const 
{ 
    return d->toolbarsVisible;
}

//----------------------------------------------------------------------------------------------------------------------------------
QDockWidget* AbstractFigure::getPropertyDockWidget() const 
{ 
    return d->propertyDock; 
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<AbstractFigure::ToolboxItem> AbstractFigure::getToolboxes() const
{
    if (m_windowMode == AbstractFigure::ModeStandaloneInUi)
    {
        //in standalone mode, this plugin handles its own menus and toolbars
        return QList<AbstractFigure::ToolboxItem>();
    }
    else
    {
        return d->toolboxes;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::addToolbox(QDockWidget *toolbox, const QString &key, Qt::DockWidgetArea area /*= Qt::RightDockWidgetArea*/)
{
    ToolboxItem item;
    item.key = key;
    item.area = area;
    item.toolbox = toolbox;
    d->toolboxes.append(item);
	//this signal is established in order to check if the docking widget already has been deleted while destruction of mainWindows
	bool test = connect(toolbox, SIGNAL(destroyed(QObject*)), this, SLOT(toolBoxDestroyed(QObject*))); 

    switch (m_windowMode)
    {
    case AbstractFigure::ModeInItomFigure:
        /*default if figure is used for plotting data in itom, may also be part of a subfigure area.
        Then, the created DockWidget should be used by the outer window and managed/displayed by it */
        break;
    case AbstractFigure::ModeStandaloneInUi:
        /*figure is contained in an user interface. Then the dock widget is dock with floating mode (default) */
        QMainWindow::addDockWidget(Qt::RightDockWidgetArea, toolbox);
        toolbox->setFloating(true);
        break;

    case AbstractFigure::ModeStandaloneWindow:
        QMainWindow::addDockWidget(Qt::RightDockWidgetArea, toolbox);
        break;
    }
}


//----------------------------------------------------------------------------------------------------------------------------------
bool AbstractFigure::removeToolbox(const QString &key)
{
	bool state = false;
	bool found = true;
	while (found == true)
	{ 
		int index = 0;
		found = false;
		foreach (ToolboxItem item, d->toolboxes)
		{
			if (item.toolbox == NULL)
			{
				continue;
			}
			if (item.key == key)
			{ 
				if (item.toolbox->isVisible())
				{
					item.toolbox->hide();
				}
				if (m_windowMode != AbstractFigure::ModeInItomFigure)
				{
					QMainWindow::removeDockWidget(item.toolbox);
				}
				d->toolboxes.removeAt(index);
				state = true;
				found = true;
				break;
			}
			index++;
		}
	}

	return state;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::mnuShowProperties(bool checked) 
{ 
    if (d->propertyDock) 
    { 
        d->propertyDock->setVisible(checked);
    } 
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::toolBoxDestroyed(QObject *object)
{
	if (object == NULL)
	{
		return;
	}
	int index = 0;
	foreach (ToolboxItem item, d->toolboxes)
	{
		if (item.toolbox == object)
		{
			d->toolboxes.removeAt(index);
			break;
		}
		index++;
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::toolBarDestroyed(QObject *object)
{
	if (object == NULL)
	{
		return;
	}
	int index = 0;
	foreach (ToolBarItem item, d->toolbars)
	{
		if (item.toolbar == object)
		{
			d->toolbars.removeAt(index);
			break;
		}
		index++;
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractFigure::registerShortcutActions()
{
    QShortcut *shortcut;
    QAction *a;
    QWidget *p = centralWidget();
    foreach(QObject *o, children())
    {
        a = qobject_cast<QAction*>(o);
        
        if (a && d->shortcutActions.contains(a))
        {
            d->shortcutActions[a]->deleteLater(); //delete a previous shortcut
        }

        if (a && a->shortcut().isEmpty() == false)
        {
            shortcut = new QShortcut(a->shortcut(), p);
            shortcut->setContext(Qt::WidgetWithChildrenShortcut);
            connect(shortcut, SIGNAL(activated()), a, SLOT(trigger()));

            QString text2 = a->text();
            QString text3 = a->text();
            text3.replace("&", "");
            text2 += "\t" + a->shortcut().toString(QKeySequence::NativeText);
            text3 += " (" + a->shortcut().toString(QKeySequence::NativeText) + ")";
            a->setText(text2);
            a->setToolTip(text3);
            a->setShortcut(QKeySequence());
            shortcut->setEnabled(a->isEnabled());

            connect(a, SIGNAL(changed()), this, SLOT(actionChanged())); //to be notified if e.g. the enable property of the action changed
            d->shortcutActions[a] = shortcut;
        }
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::actionChanged()
{
    QObject *s = sender(); //action where any property like enabled changed...
    QAction *a = qobject_cast<QAction*>(s);

    if (a && d->shortcutActions.contains(a))
    {
        d->shortcutActions[a]->setEnabled(a->isEnabled());
    }
}

} //end namespace ito
