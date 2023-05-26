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
#include "checkableComboBoxFactory.h"
#include "checkableComboBox.h"


// --------------------------------------------------------------------------
CheckableComboBoxFactory::CheckableComboBoxFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *CheckableComboBoxFactory::createWidget(QWidget *_parent)
{
    CheckableComboBox* widget = new CheckableComboBox(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString CheckableComboBoxFactory::domXml() const
{
  return "<widget class=\"CheckableComboBox\" \
          name=\"CheckableComboBox\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon CheckableComboBoxFactory::icon() const
{
  return QIcon(":/icons/combobox.png");
}

// --------------------------------------------------------------------------
QString CheckableComboBoxFactory::includeFile() const
{
    return "checkableComboBox.h";
}

// --------------------------------------------------------------------------
bool CheckableComboBoxFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString CheckableComboBoxFactory::name() const
{
    return "CheckableComboBox";
}

//-----------------------------------------------------------------------------
QString CheckableComboBoxFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString CheckableComboBoxFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString CheckableComboBoxFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void CheckableComboBoxFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
