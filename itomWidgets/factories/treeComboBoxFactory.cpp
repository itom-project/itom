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
#include "treeComboBoxFactory.h"
#include "treeComboBox.h"


// --------------------------------------------------------------------------
TreeComboBoxFactory::TreeComboBoxFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *TreeComboBoxFactory::createWidget(QWidget *_parent)
{
    TreeComboBox* widget = new TreeComboBox(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString TreeComboBoxFactory::domXml() const
{
  return "<widget class=\"TreeComboBox\" \
          name=\"TreeComboBox\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon TreeComboBoxFactory::icon() const
{
  return QIcon(":/icons/combobox.png");
}

// --------------------------------------------------------------------------
QString TreeComboBoxFactory::includeFile() const
{
    return "treeComboBox.h";
}

// --------------------------------------------------------------------------
bool TreeComboBoxFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString TreeComboBoxFactory::name() const
{
    return "TreeComboBox";
}

//-----------------------------------------------------------------------------
QString TreeComboBoxFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString TreeComboBoxFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString TreeComboBoxFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void TreeComboBoxFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
