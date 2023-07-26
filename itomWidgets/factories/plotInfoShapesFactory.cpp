/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2020, Institut fuer Technische Optik (ITO),
   Universitaet Stuttgart, Germany

   This file is part of itom.

   itom is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   itom is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */


#include <QtCore/QtPlugin>
#include "plotInfoShapesFactory.h"
#include "plotInfoShapes.h"


// --------------------------------------------------------------------------
PlotInfoShapesFactory::PlotInfoShapesFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *PlotInfoShapesFactory::createWidget(QWidget *_parent)
{
	PlotInfoShapes* widget = new PlotInfoShapes(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString PlotInfoShapesFactory::domXml() const
{
  return "<widget class=\"PlotInfoShapes\" name=\"plotInfoShapes\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon PlotInfoShapesFactory::icon() const
{
  return QIcon(":/icons/widget.png");
}

// --------------------------------------------------------------------------
QString PlotInfoShapesFactory::includeFile() const
{
    return "plotInfoShapes.h";
}

// --------------------------------------------------------------------------
bool PlotInfoShapesFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString PlotInfoShapesFactory::name() const
{
    return "PlotInfoShapes";
}

//-----------------------------------------------------------------------------
QString PlotInfoShapesFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString PlotInfoShapesFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString PlotInfoShapesFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void PlotInfoShapesFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
