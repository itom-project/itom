/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

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

#include "helpDockWidget.h"
#include "../global.h"
#include "../AppManagement.h"

#include "../organizer/scriptEditorOrganizer.h"

#include <qheaderview.h>
#include <qsettings.h>


namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
HelpDockWidget::HelpDockWidget(const QString &title, QWidget *parent, bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, parent)
{
    AbstractDockWidget::init();
}

//----------------------------------------------------------------------------------------------------------------------------------
HelpDockWidget::~HelpDockWidget()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void HelpDockWidget::createActions()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void HelpDockWidget::createMenus()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void HelpDockWidget::createToolBars()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void HelpDockWidget::updateActions()
{
}


} //end namespace ito
