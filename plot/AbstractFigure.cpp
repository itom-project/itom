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

#define ITOM_IMPORT_PLOTAPI
#define ITOM_IMPORT_API
#include "AbstractFigure.h"

#include <qaction.h>
#include <qtoolbar.h>
#include <qmenu.h>
#include <qmenubar.h>
#include <qsettings.h>

#include "../common/typeDefs.h"
#include "../common/addInInterface.h"
#include "../common/apiFunctionsInc.h"

using namespace ito;

namespace ito 
{

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
            return ito::RetVal(ito::retError, 0, QObject::tr("duplicate Channel, in addChannel").toAscii().data());
        }
    }

    if (direction == Channel::parentToChild)
    {
        if (apiCompareParam(*childParam, *parentParam, retVal) == ito::tCmpFailed)
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("parameters incompatible, while adding channel").toAscii().data());
        }
    }
    else if (direction == Channel::childToParent)
    {
        if (apiCompareParam(*parentParam, *childParam, retVal) == ito::tCmpFailed)
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("parameters incompatible, while adding channel").toAscii().data());
        }
    }
    else
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("undefined channel direction, while adding channel").toAscii().data());
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
        return ito::RetVal(ito::retError, 0, QObject::tr("invalid child pointer, in addChannel").toAscii().data());

    uint channelHash1 = ito::calculateChannelHash(this, newChannel->getChildParam(), newChannel->getParent(), newChannel->getParentParam());
    uint channelHash2 = ito::calculateChannelHash(newChannel->getParent(), newChannel->getParentParam(), this, newChannel->getChildParam());

    Channel *tempChannel;
    foreach(tempChannel, m_pChannels)
    {
        if ((tempChannel->getHash() == channelHash1) || (tempChannel->getHash() == channelHash2))
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("duplicate Channel, in addChannel").toAscii().data());
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
        return ito::RetVal(ito::retError, 0, QObject::tr("channel does not exist").toAscii().data());

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
        return ito::RetVal(ito::retError, 0, QObject::tr("channel does not exist").toAscii().data());

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

////----------------------------------------------------------------------------------------------------------------------------------
//void initialize(ito::AbstractFigure *fig)
//{
//    
//
//}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractFigure::AbstractFigure(const QString &itomSettingsFile, QWidget *parent)
    : QMainWindow(parent),
    AbstractNode(),
    m_contextMenu(NULL),
    m_itomSettingsFile(itomSettingsFile),
    m_apiFunctionsGraphBasePtr(NULL),
    m_apiFunctionsBasePtr(NULL),
    m_actTopLevelParent(NULL),
    m_actTopLevelOverall(NULL),
    m_menuWindow(NULL),
    m_mainParent(parent),
	m_toolbarsVisible(true)
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
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractFigure::initialize()
{
    //create actions
    m_actTopLevelOverall = new QAction(tr("stay on top of all windows"), this);
    m_actTopLevelOverall->setCheckable(true);
    connect(m_actTopLevelOverall, SIGNAL(triggered(bool)), this, SLOT(mnuTopLevelOverall(bool)));

    m_actTopLevelParent = new QAction(tr("stay on top of main window"), this);
    m_actTopLevelParent->setCheckable(true);
    connect(m_actTopLevelParent, SIGNAL(triggered(bool)), this, SLOT(mnuTopLevelParent(bool)));

    //create main menus
    m_menuWindow = new QMenu(tr("window"), this);
    m_menuWindow->addAction(m_actTopLevelParent);
    m_menuWindow->addAction(m_actTopLevelOverall);
    menuBar()->addMenu(m_menuWindow);

    setWindowMode( ito::AbstractFigure::ModeEmbedded ); //this is important as default. In the windows case, this mode is reset to ModeWindow
    //fig->setWindowFlags(Qt::Widget); //this is important such that this main window reacts as widget

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::addMenu(QMenu *menu)
{
    QAction *prevAct = NULL;
    if(m_menuWindow) prevAct = m_menuWindow->menuAction();
    menu->setParent(this);

    if(prevAct)
    {
        menuBar()->insertMenu(prevAct, menu);
    }
    else
    {
        menuBar()->addMenu(menu);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::addToolBar(QToolBar *toolbar, const QString &key, Qt::ToolBarArea area /*= Qt::TopToolBarArea*/)
{
	if(m_toolbars.contains(key))
	{
		m_toolbars[key].first->deleteLater();
	}
	QMainWindow::addToolBar(area,toolbar);
	m_toolbars[key] = QPair<QToolBar*,bool>( toolbar, m_toolbarsVisible );
	toolbar->setVisible( m_toolbarsVisible );
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::insertToolBar(const QString &key_before, QToolBar *toolbar, const QString &key)
{
	if(m_toolbars.contains(key_before) == false) return;

	QToolBar *before = m_toolbars[key_before].first;
	if(m_toolbars.contains(key))
	{
		m_toolbars[key].first->deleteLater();
	}
	QMainWindow::insertToolBar(before,toolbar);
	m_toolbars[key] = QPair<QToolBar*,bool>( toolbar, m_toolbarsVisible );
	toolbar->setVisible( m_toolbarsVisible );
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::insertToolBarBreak(const QString &key_before)
{
	if(m_toolbars.contains(key_before) == false) return;

	QToolBar *before = m_toolbars[key_before].first;
	QMainWindow::insertToolBarBreak(before);
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::removeToolBar(const QString &key)
{
	if(m_toolbars.contains(key))
	{
		m_toolbars[key].first->deleteLater();
		m_toolbars.remove(key);
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::removeToolBarBreak(const QString &key_before)
{
	if(m_toolbars.contains(key_before))
	{
		QMainWindow::removeToolBarBreak( m_toolbars[key_before].first );
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::showToolBar(const QString &key)
{
	if(m_toolbars.contains(key))
	{
		m_toolbars[key].second = true;
		m_toolbars[key].first->setVisible( true && m_toolbarsVisible );
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::hideToolBar(const QString &key)
{
	if(m_toolbars.contains(key))
	{
		m_toolbars[key].second = false;
		m_toolbars[key].first->setVisible( false && m_toolbarsVisible );
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setWindowMode(const WindowMode mode)
{
    switch( mode )
    {
    case ModeWindow:
        {
        setWindowFlags(Qt::Window);
        setAttribute(Qt::WA_DeleteOnClose, true);
        m_windowMode = ModeWindow;

        QSettings settings(m_itomSettingsFile, QSettings::IniFormat );
        settings.beginGroup("Figures");
        TopLevelMode tlm = (TopLevelMode)(settings.value("topLevelMode", TopLevelNothing)).toInt();
        settings.endGroup();

        setTopLevelMode(tlm);
        if(menuBar()->actions().count() > 0)
        {
            menuBar()->setVisible(true);
        }
        }
        break;
    case ModeEmbedded:
        {
        setWindowFlags(Qt::Widget);
        setAttribute(Qt::WA_DeleteOnClose, false);
        m_windowMode = ModeEmbedded;
        setTopLevelMode(TopLevelNothing);
        menuBar()->setVisible(false);
        }
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setTopLevelMode(TopLevelMode mode)
{
    Qt::WindowFlags flags = windowFlags();
    bool visible = isVisible();

    if(m_windowMode == ModeEmbedded)
    {
        mode = TopLevelNothing;
        setParent(m_mainParent);
        setWindowFlags( flags & ~(Qt::WindowStaysOnTopHint) );
        m_actTopLevelOverall->setChecked(false);
        m_actTopLevelParent->setChecked(false);
    }
    else
    {

        switch(mode)
        {
            case TopLevelNothing:
                setParent(NULL);
                setWindowFlags( flags & ~(Qt::WindowStaysOnTopHint) );
                m_actTopLevelOverall->setChecked(false);
                m_actTopLevelParent->setChecked(false);
                break;
            case TopLevelOverall:
                setParent(m_mainParent);
                setWindowFlags( flags | Qt::WindowStaysOnTopHint );
                m_actTopLevelOverall->setChecked(true);
                m_actTopLevelParent->setChecked(false);
                break;
            case TopLevelParentOnly:
                setParent(m_mainParent);
                setWindowFlags( flags & ~(Qt::WindowStaysOnTopHint) );
                m_actTopLevelOverall->setChecked(false);
                m_actTopLevelParent->setChecked(true);
                break;
        }

        QSettings settings(m_itomSettingsFile, QSettings::IniFormat );
        settings.beginGroup("Figures");
        settings.setValue("topLevelMode", mode);
        settings.endGroup();
    }

    m_topLevelMode = mode;

    if(visible) show(); //it is necessary to re-show the widget in order to activate the settings from above.
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractFigure::TopLevelMode AbstractFigure::getTopLevelMode()
{
    return m_topLevelMode;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setApiFunctionGraphBasePtr(void **apiFunctionGraphBasePtr)
{ 
    importItomPlotApi(apiFunctionGraphBasePtr);
    m_apiFunctionsGraphBasePtr = apiFunctionGraphBasePtr; 
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setApiFunctionBasePtr(void **apiFunctionBasePtr)
{ 
    importItomApi(apiFunctionBasePtr);
    m_apiFunctionsBasePtr = apiFunctionBasePtr; 
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
    if(e->type() == (QEvent::User+123))
    {
        importItomApi(m_apiFunctionsBasePtr);
        importItomPlotApi(m_apiFunctionsGraphBasePtr);
    }   
    return QMainWindow::event(e);
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::setToolbarVisible(bool visible)
{

	QMapIterator<QString, QPair<QToolBar*,bool> > i(m_toolbars);
	
	while (i.hasNext()) 
	{
		i.next();
		i.value().first->setVisible( visible && i.value().second );
	}

	m_toolbarsVisible = visible;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool AbstractFigure::toolbarVisible() const 
{ 
    return m_toolbarsVisible;
}



//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::mnuTopLevelOverall(bool checked)
{
    if(checked)
    {
        setTopLevelMode(AbstractFigure::TopLevelOverall);
    }
    else
    {    
        setTopLevelMode(AbstractFigure::TopLevelNothing);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractFigure::mnuTopLevelParent(bool checked)
{
    if(checked)
    {
        setTopLevelMode(AbstractFigure::TopLevelParentOnly);
    }
    else
    {    
        setTopLevelMode(AbstractFigure::TopLevelNothing);
    }
}

} //end namespace ito
