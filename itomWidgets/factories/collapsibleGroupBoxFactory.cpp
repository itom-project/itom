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
#include "collapsibleGroupBoxFactory.h"
#include "collapsibleGroupBox.h"


// --------------------------------------------------------------------------
CollapsibleGroupBoxFactory::CollapsibleGroupBoxFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *CollapsibleGroupBoxFactory::createWidget(QWidget *_parent)
{
    CollapsibleGroupBox* widget = new CollapsibleGroupBox(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString CollapsibleGroupBoxFactory::domXml() const
{
  return "<widget class=\"CollapsibleGroupBox\" \
          name=\"collapsibleGroupBox\">\n"
          " <property name=\"geometry\">\n"
          "  <rect>\n"
          "   <x>0</x>\n"
          "   <y>0</y>\n"
          "   <width>300</width>\n"
          "   <height>100</height>\n"
          "  </rect>\n"
          " </property>\n"
          " <property name=\"title\">"
          "  <string>GroupBox</string>"
          " </property>"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon CollapsibleGroupBoxFactory::icon() const
{
  return QIcon(":/icons/groupboxcollapsible.png");
}

// --------------------------------------------------------------------------
QString CollapsibleGroupBoxFactory::includeFile() const
{
    return "collapsibleGroupBox.h";
}

// --------------------------------------------------------------------------
bool CollapsibleGroupBoxFactory::isContainer() const
{
    return true;
}

// --------------------------------------------------------------------------
QString CollapsibleGroupBoxFactory::name() const
{
    return "CollapsibleGroupBox";
}

//-----------------------------------------------------------------------------
QString CollapsibleGroupBoxFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString CollapsibleGroupBoxFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString CollapsibleGroupBoxFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void CollapsibleGroupBoxFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
