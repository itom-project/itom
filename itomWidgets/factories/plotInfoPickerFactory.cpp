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
#include "plotInfoPickerFactory.h"
#include "plotInfoPicker.h"


// --------------------------------------------------------------------------
PlotInfoPickerFactory::PlotInfoPickerFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *PlotInfoPickerFactory::createWidget(QWidget *_parent)
{
	PlotInfoPicker* widget = new PlotInfoPicker(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString PlotInfoPickerFactory::domXml() const
{
  return "<widget class=\"PlotInfoPicker\" name=\"plotInfoPicker\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon PlotInfoPickerFactory::icon() const
{
  return QIcon(":/icons/widget.png");
}

// --------------------------------------------------------------------------
QString PlotInfoPickerFactory::includeFile() const
{
    return "plotInfoPicker.h";
}

// --------------------------------------------------------------------------
bool PlotInfoPickerFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString PlotInfoPickerFactory::name() const
{
    return "PlotInfoPicker";
}

//-----------------------------------------------------------------------------
QString PlotInfoPickerFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString PlotInfoPickerFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString PlotInfoPickerFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void PlotInfoPickerFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
