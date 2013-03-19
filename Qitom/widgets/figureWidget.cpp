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

#include "../global.h"

#include "figureWidget.h"


namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------

FigureWidget::FigureWidget(const QString &title, bool docked, bool isDockAvailable, QWidget *parent, Qt::WindowFlags /*flags*/)
    : AbstractDockWidget(docked, isDockAvailable, floatingWindow, movingEnabled, title, parent)
{

    AbstractDockWidget::init();

    resizeDockWidget(700,400);

    setFocusPolicy(Qt::StrongFocus);
//    setAcceptDrops(true);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!
    cancels connections and closes every tab.
*/
FigureWidget::~FigureWidget()
{


}


void FigureWidget::createActions()
{
}

void FigureWidget::createMenus()
{
}

void FigureWidget::createToolBars()
{
}

void FigureWidget::createStatusBar()
{
}

void FigureWidget::updateActions()
{
}




} //end namespace ito
