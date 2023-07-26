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
#include "doubleSpinBoxFactory.h"
#include "doubleSpinBox.h"


// --------------------------------------------------------------------------
DoubleSpinBoxFactory::DoubleSpinBoxFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *DoubleSpinBoxFactory::createWidget(QWidget *_parent)
{
    DoubleSpinBox* widget = new DoubleSpinBox(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString DoubleSpinBoxFactory::domXml() const
{
  return "<widget class=\"DoubleSpinBox\" name=\"doubleSpinBox\">\n"
    "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon DoubleSpinBoxFactory::icon() const
{
  return QIcon(":/icons/doublespinbox.png");
}

// --------------------------------------------------------------------------
QString DoubleSpinBoxFactory::includeFile() const
{
    return "doubleSpinBox.h";
}

// --------------------------------------------------------------------------
bool DoubleSpinBoxFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString DoubleSpinBoxFactory::name() const
{
    return "DoubleSpinBox";
}

//-----------------------------------------------------------------------------
QString DoubleSpinBoxFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString DoubleSpinBoxFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString DoubleSpinBoxFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void DoubleSpinBoxFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
