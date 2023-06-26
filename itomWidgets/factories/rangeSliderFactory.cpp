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
#include "rangeSliderFactory.h"
#include "rangeSlider.h"


// --------------------------------------------------------------------------
RangeSliderFactory::RangeSliderFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *RangeSliderFactory::createWidget(QWidget *_parent)
{
    RangeSlider* widget = new RangeSlider(Qt::Horizontal, _parent);
    return widget;
}

// --------------------------------------------------------------------------
QString RangeSliderFactory::domXml() const
{
  return "<widget class=\"RangeSlider\" name=\"rangeSlider\">\n"
    "<property name=\"orientation\">\n"
    "  <enum>Qt::Horizontal</enum>\n"
    " </property>\n"
    "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon RangeSliderFactory::icon() const
{
  return QIcon(":/icons/hrangeslider.png");
}

// --------------------------------------------------------------------------
QString RangeSliderFactory::includeFile() const
{
    return "rangeSlider.h";
}

// --------------------------------------------------------------------------
bool RangeSliderFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString RangeSliderFactory::name() const
{
    return "RangeSlider";
}

//-----------------------------------------------------------------------------
QString RangeSliderFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString RangeSliderFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString RangeSliderFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void RangeSliderFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
