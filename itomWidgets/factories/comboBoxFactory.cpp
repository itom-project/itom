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
#include "comboBoxFactory.h"
#include "comboBox.h"


// --------------------------------------------------------------------------
ComboBoxFactory::ComboBoxFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *ComboBoxFactory::createWidget(QWidget *_parent)
{
    ComboBox* widget = new ComboBox(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString ComboBoxFactory::domXml() const
{
  return "<widget class=\"ComboBox\" \
          name=\"ComboBox\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon ComboBoxFactory::icon() const
{
  return QIcon(":/icons/combobox.png");
}

// --------------------------------------------------------------------------
QString ComboBoxFactory::includeFile() const
{
    return "comboBox.h";
}

// --------------------------------------------------------------------------
bool ComboBoxFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString ComboBoxFactory::name() const
{
    return "ComboBox";
}

//-----------------------------------------------------------------------------
QString ComboBoxFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString ComboBoxFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString ComboBoxFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void ComboBoxFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
