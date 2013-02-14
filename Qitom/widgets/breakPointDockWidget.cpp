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

#include "../python/pythonEngineInc.h"

#include "breakPointDockWidget.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qheaderview.h>


namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
BreakPointDockWidget::BreakPointDockWidget(const QString &title, QWidget *parent, bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, parent)
{
	m_breakPointView = new QTableView(this);

	AbstractDockWidget::init();

	setContentWidget(m_breakPointView);

	m_breakPointView->verticalHeader()->setDefaultSectionSize(22);
	m_breakPointView->setAlternatingRowColors(true);
	m_breakPointView->setTextElideMode( Qt::ElideLeft );

	PythonEngine *pe = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
	if(pe)
	{
		m_breakPointView->setModel( pe->getBreakPointModel() );
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
BreakPointDockWidget::~BreakPointDockWidget()
{
    
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::createActions()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::createMenus()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::createToolBars()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::updateActions()
{
}



} //end namespace ito
