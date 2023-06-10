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
#include "doubleSliderFactory.h"
#include "doubleSlider.h"


// --------------------------------------------------------------------------
DoubleSliderFactory::DoubleSliderFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *DoubleSliderFactory::createWidget(QWidget *_parent)
{
    DoubleSlider* widget = new DoubleSlider(Qt::Horizontal, _parent);
    return widget;
}

// --------------------------------------------------------------------------
QString DoubleSliderFactory::domXml() const
{
  return "<widget class=\"DoubleSlider\" name=\"doubleSlider\">\n"
    "<property name=\"orientation\">\n"
    "  <enum>Qt::Horizontal</enum>\n"
    " </property>\n"
    "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon DoubleSliderFactory::icon() const
{
  return QIcon(":/icons/hslider.png");
}

// --------------------------------------------------------------------------
QString DoubleSliderFactory::includeFile() const
{
    return "doubleSlider.h";
}

// --------------------------------------------------------------------------
bool DoubleSliderFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString DoubleSliderFactory::name() const
{
    return "DoubleSlider";
}

//-----------------------------------------------------------------------------
QString DoubleSliderFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString DoubleSliderFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString DoubleSliderFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void DoubleSliderFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
