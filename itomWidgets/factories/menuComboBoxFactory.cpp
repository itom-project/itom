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
#include "menuComboBoxFactory.h"
#include "menuComboBox.h"


// --------------------------------------------------------------------------
MenuComboBoxFactory::MenuComboBoxFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *MenuComboBoxFactory::createWidget(QWidget *_parent)
{
    MenuComboBox* widget = new MenuComboBox(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString MenuComboBoxFactory::domXml() const
{
  return "<widget class=\"MenuComboBox\" \
          name=\"MenuComboBox\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon MenuComboBoxFactory::icon() const
{
  return QIcon(":/icons/combobox.png");
}

// --------------------------------------------------------------------------
QString MenuComboBoxFactory::includeFile() const
{
    return "comboBox.h";
}

// --------------------------------------------------------------------------
bool MenuComboBoxFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString MenuComboBoxFactory::name() const
{
    return "MenuComboBox";
}

//-----------------------------------------------------------------------------
QString MenuComboBoxFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString MenuComboBoxFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString MenuComboBoxFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void MenuComboBoxFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
