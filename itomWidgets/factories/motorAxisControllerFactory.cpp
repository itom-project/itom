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
#include "motorAxisControllerFactory.h"
#include "motorAxisController.h"


// --------------------------------------------------------------------------
MotorAxisControllerFactory::MotorAxisControllerFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *MotorAxisControllerFactory::createWidget(QWidget *_parent)
{
    MotorAxisController* widget = new MotorAxisController(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString MotorAxisControllerFactory::domXml() const
{
  return "<widget class=\"MotorAxisController\" \
          name=\"MotorAxisController\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon MotorAxisControllerFactory::icon() const
{
  return QIcon(":/icons/motorAxisController.png");
}

// --------------------------------------------------------------------------
QString MotorAxisControllerFactory::includeFile() const
{
    return "motorAxisController.h";
}

// --------------------------------------------------------------------------
bool MotorAxisControllerFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString MotorAxisControllerFactory::name() const
{
    return "MotorAxisController";
}

//-----------------------------------------------------------------------------
QString MotorAxisControllerFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString MotorAxisControllerFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString MotorAxisControllerFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void MotorAxisControllerFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
