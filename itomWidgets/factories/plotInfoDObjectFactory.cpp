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
#include "plotInfoDObjectFactory.h"
#include "plotInfoDObject.h"


// --------------------------------------------------------------------------
PlotInfoDObjectFactory::PlotInfoDObjectFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *PlotInfoDObjectFactory::createWidget(QWidget *_parent)
{
	PlotInfoDObject* widget = new PlotInfoDObject(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString PlotInfoDObjectFactory::domXml() const
{
  return "<widget class=\"PlotInfoDObject\" name=\"plotInfoDObject\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon PlotInfoDObjectFactory::icon() const
{
  return QIcon(":/icons/widget.png");
}

// --------------------------------------------------------------------------
QString PlotInfoDObjectFactory::includeFile() const
{
    return "plotInfoDObject.h";
}

// --------------------------------------------------------------------------
bool PlotInfoDObjectFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString PlotInfoDObjectFactory::name() const
{
    return "PlotInfoDObject";
}

//-----------------------------------------------------------------------------
QString PlotInfoDObjectFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString PlotInfoDObjectFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString PlotInfoDObjectFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void PlotInfoDObjectFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
