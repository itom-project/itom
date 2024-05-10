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
#include "rangeWidgetFactory.h"
#include "rangeWidget.h"


// --------------------------------------------------------------------------
RangeWidgetFactory::RangeWidgetFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *RangeWidgetFactory::createWidget(QWidget *_parent)
{
    RangeWidget* widget = new RangeWidget(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString RangeWidgetFactory::domXml() const
{
  return "<widget class=\"RangeWidget\" name=\"rangeWidget\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon RangeWidgetFactory::icon() const
{
  return QIcon(":/icons/rangespinbox.png");
}

// --------------------------------------------------------------------------
QString RangeWidgetFactory::includeFile() const
{
    return "rangeWidget.h";
}

// --------------------------------------------------------------------------
bool RangeWidgetFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString RangeWidgetFactory::name() const
{
    return "RangeWidget";
}

//-----------------------------------------------------------------------------
QString RangeWidgetFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString RangeWidgetFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString RangeWidgetFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void RangeWidgetFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
