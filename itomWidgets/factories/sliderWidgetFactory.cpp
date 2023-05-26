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
#include "sliderWidgetFactory.h"
#include "sliderWidget.h"


// --------------------------------------------------------------------------
SliderWidgetFactory::SliderWidgetFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *SliderWidgetFactory::createWidget(QWidget *_parent)
{
    SliderWidget* widget = new SliderWidget(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString SliderWidgetFactory::domXml() const
{
  return "<widget class=\"SliderWidget\" name=\"sliderWidget\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon SliderWidgetFactory::icon() const
{
  return QIcon(":/icons/sliderspinbox.png");
}

// --------------------------------------------------------------------------
QString SliderWidgetFactory::includeFile() const
{
    return "sliderWidget.h";
}

// --------------------------------------------------------------------------
bool SliderWidgetFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString SliderWidgetFactory::name() const
{
    return "SliderWidget";
}

//-----------------------------------------------------------------------------
QString SliderWidgetFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString SliderWidgetFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString SliderWidgetFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void SliderWidgetFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
