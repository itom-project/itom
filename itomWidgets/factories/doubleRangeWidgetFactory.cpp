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
#include "doubleRangeWidgetFactory.h"
#include "doubleRangeWidget.h"


// --------------------------------------------------------------------------
DoubleRangeWidgetFactory::DoubleRangeWidgetFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *DoubleRangeWidgetFactory::createWidget(QWidget *_parent)
{
    DoubleRangeWidget* widget = new DoubleRangeWidget(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString DoubleRangeWidgetFactory::domXml() const
{
  return "<widget class=\"DoubleRangeWidget\" name=\"doubleRangeWidget\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon DoubleRangeWidgetFactory::icon() const
{
  return QIcon(":/icons/rangespinbox.png");
}

// --------------------------------------------------------------------------
QString DoubleRangeWidgetFactory::includeFile() const
{
    return "doubleRangeWidget.h";
}

// --------------------------------------------------------------------------
bool DoubleRangeWidgetFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString DoubleRangeWidgetFactory::name() const
{
    return "DoubleRangeWidget";
}

//-----------------------------------------------------------------------------
QString DoubleRangeWidgetFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString DoubleRangeWidgetFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString DoubleRangeWidgetFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void DoubleRangeWidgetFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
