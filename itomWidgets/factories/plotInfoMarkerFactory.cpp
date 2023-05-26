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
#include "plotInfoMarkerFactory.h"
#include "plotInfoMarker.h"


// --------------------------------------------------------------------------
PlotInfoMarkerFactory::PlotInfoMarkerFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *PlotInfoMarkerFactory::createWidget(QWidget *_parent)
{
	PlotInfoMarker* widget = new PlotInfoMarker(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString PlotInfoMarkerFactory::domXml() const
{
  return "<widget class=\"PlotInfoMarker\" name=\"plotInfoMarker\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon PlotInfoMarkerFactory::icon() const
{
  return QIcon(":/icons/widget.png");
}

// --------------------------------------------------------------------------
QString PlotInfoMarkerFactory::includeFile() const
{
    return "plotInfoMarker.h";
}

// --------------------------------------------------------------------------
bool PlotInfoMarkerFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString PlotInfoMarkerFactory::name() const
{
    return "PlotInfoMarker";
}

//-----------------------------------------------------------------------------
QString PlotInfoMarkerFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString PlotInfoMarkerFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString PlotInfoMarkerFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void PlotInfoMarkerFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
