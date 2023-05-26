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
#include "doubleRangeSliderFactory.h"
#include "doubleRangeSlider.h"


// --------------------------------------------------------------------------
DoubleRangeSliderFactory::DoubleRangeSliderFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *DoubleRangeSliderFactory::createWidget(QWidget *_parent)
{
    DoubleRangeSlider* widget = new DoubleRangeSlider(Qt::Horizontal, _parent);
    return widget;
}

// --------------------------------------------------------------------------
QString DoubleRangeSliderFactory::domXml() const
{
  return "<widget class=\"DoubleRangeSlider\" name=\"doubleRangeSlider\">\n"
    "<property name=\"orientation\">\n"
    "  <enum>Qt::Horizontal</enum>\n"
    " </property>\n"
    "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon DoubleRangeSliderFactory::icon() const
{
  return QIcon(":/icons/hrangeslider.png");
}

// --------------------------------------------------------------------------
QString DoubleRangeSliderFactory::includeFile() const
{
    return "doubleRangeSlider.h";
}

// --------------------------------------------------------------------------
bool DoubleRangeSliderFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString DoubleRangeSliderFactory::name() const
{
    return "DoubleRangeSlider";
}

//-----------------------------------------------------------------------------
QString DoubleRangeSliderFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString DoubleRangeSliderFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString DoubleRangeSliderFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void DoubleRangeSliderFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
