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

//#define ITOM_IMPORT_PLOTAPI
//#define ITOM_IMPORT_API
#include "../AbstractFigure.h"

#if QT_VERSION < 0x050000
#include <qaction.h>
#include <qtoolbar.h>
#include <qmenu.h>
#include <qmenubar.h>
#else
#include <QtWidgets/qaction.h>
#include <QtWidgets/qtoolbar.h>
#include <QtWidgets/qmenu.h>
#include <QtWidgets/qmenubar.h>
#include <QtGui/qevent.h>
#endif
#include <qsettings.h>

#include "../../common/typeDefs.h"
#include "../../common/addInInterface.h"
#include "../../common/apiFunctionsInc.h"
#include "QPropertyEditor/QPropertyEditorWidget.h"

namespace ito 
{

//----------------------------------------------------------------------------------------------------------------------------------
AbstractFigure::AbstractFigure(const QString &itomSettingsFile, WindowMode windowMode, QWidget *parent) : 
    QMainWindow(parent),
    AbstractNode(),
    m_contextMenu(NULL),
    m_itomSettingsFile(itomSettingsFile),
    m_apiFunctionsGraphBasePtr(NULL),
    m_apiFunctionsBasePtr(NULL),
    m_mainParent(parent),
    m_toolbarsVisible(true),
    m_windowMode(windowMode),
    m_propertyDock(NULL),
    m_propertyEditorWidget(NULL),
    m_propertyObservedObject(NULL)
{
    //itom_PLOTAPI = NULL;
    //importItomPlotApi(NULL);
    initialize();
    //ito::ITOM_API_FUNCS_GRAPH = NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractFigure::~AbstractFigure()
{
    Channel *delChan;
    foreach(delChan, m_pChannels)
    {
        // connection is removed in the destructor of Connection so the following line is not necessary
//            (delConn->getPartner())->disconnect(delConn->getKey());
        removeChannel(delChan);
    }
    m_pChannels.clear();

    //clear toolbars and menus
    foreach(QMenu *m, m_menus)
    {
        m->deleteLater();
    }
    m_menus.clear();

    foreach (ToolBarItem t, m_toolbars)
    {
        if (t.toolbar)
        {
            t.toolbar->deleteLater();
        }
    }
    m_toolbars.clear();

    if (m_propertyDock)
    {
        m_propertyDock->deleteLater();
    }
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

    if (m_windowMode == AbstractFigure::ModeStandaloneInUi)
    {
        foreach(const ToolBarItem &item, m_toolbars)
        {
            if (item.toolbar)
            {
                QMainWindow::addToolBar(item.area, item.toolbar);
            }
            else
            {
                QMainWindow::addToolBarBreak(item.area);
            }
        }
    }

    m_propertyDock = new QDockWidget(tr("Properties"), this);
    m_propertyDock->setVisible(false);
    m_propertyDock->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetMovable);

    m_propertyEditorWidget = new QPropertyEditorWidget(m_propertyDock);
    m_propertyDock->setWidget(m_propertyEditorWidget);

    switch (m_windowMode)
    {
        case AbstractFigure::ModeInItomFigure:
            /*default if figure is used for plotting data in itom, may also be part of a subfigure area.
            Then, the created DockWidget should be used by the outer window and managed/displayed by it */
            break;
        case AbstractFigure::ModeStandaloneInUi:
            /*figure is contained in an user interface. Then the dock widget is dock with floating mode (default) */
            addDockWidget(Qt::NoDockWidgetArea, m_propertyDock);
            break;

        case AbstractFigure::ModeStandaloneWindow:
            addDockWidget(Qt::RightDockWidgetArea, m_propertyDock);
            break;
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setPropertyObservedObject(QObject* obj)
{
    m_propertyObservedObject = obj;
    if (m_propertyEditorWidget)
    {
        m_propertyEditorWidget->setObject(obj);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::updatePropertyDock()
{
    if (m_propertyEditorWidget && m_propertyObservedObject)
    {
        m_propertyEditorWidget->updateObject(m_propertyObservedObject);
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
//            (delConn->getPartner())->disconnect(delConn->getKey()); 
            removeChannel(iterChannel);
        }
//        m_pChannels.clear();
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
//        delChannel->getChild()->removeChannel(delChannel);
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
//            (delConn->getPartner())->disconnect(delConn->getKey()); 
            removeChannel(iterChannel);
        }
//        m_pChannels.clear();
//        delete this;
        deleteLater();
    }
    
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::addMenu(QMenu *menu)
{
    //never adds to menuBar()
    m_menus.append(menu);
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
        return m_menus;
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
        return m_toolbars;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::addToolBar(QToolBar *toolbar, const QString &key, Qt::ToolBarArea area /*= Qt::TopToolBarArea*/, int section /*= 1*/)
{
    ToolBarItem item;
    item.key = key;
    item.area = area;
    item.toolbar = toolbar;
    item.visible = m_toolbarsVisible;
    item.section = section;

    int maxSection = 1;

    //get highest section for same area
    foreach (const ToolBarItem &titem, m_toolbars)
    {
        if (titem.area == area)
        {
            maxSection = std::max(maxSection, titem.section);
        }
    }

    m_toolbars.append(item);

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
    item.visible = m_toolbarsVisible;
    item.section = 1;

    //get highest section for same area
    foreach (const ToolBarItem &titem, m_toolbars)
    {
        if (titem.area == area)
        {
            item.section = std::max(item.section, titem.section);
        }
    }

    m_toolbars.append(item);

    if (m_windowMode == AbstractFigure::ModeStandaloneInUi || m_windowMode == AbstractFigure::ModeStandaloneWindow)
    {
        QMainWindow::addToolBarBreak(area);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::showToolBar(const QString &key)
{
    QList<AbstractFigure::ToolBarItem>::iterator i;
    
    for (i = m_toolbars.begin(); i != m_toolbars.end(); ++i)
    {
        if (i->key == key)
        {
            i->visible = true;
            i->toolbar->setVisible(true && m_toolbarsVisible);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::hideToolBar(const QString &key)
{
    QList<AbstractFigure::ToolBarItem>::iterator i;
    
    for (i = m_toolbars.begin(); i != m_toolbars.end(); ++i)
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
    
    for (i = m_toolbars.begin(); i != m_toolbars.end(); ++i)
    {
        if (i->toolbar)
        {
            i->toolbar->setVisible(visible && (*i).visible);
        }
    }

    m_toolbarsVisible = visible;
    updatePropertyDock();
}

//----------------------------------------------------------------------------------------------------------------------------------
bool AbstractFigure::getToolbarVisible() const 
{ 
    return m_toolbarsVisible;
}

} //end namespace ito
