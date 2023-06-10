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
#include "statusLedFactory.h"
#include "statusLed.h"


// --------------------------------------------------------------------------
StatusLedFactory::StatusLedFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *StatusLedFactory::createWidget(QWidget *_parent)
{
    StatusLed* widget = new StatusLed(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString StatusLedFactory::domXml() const
{
  return "<widget class=\"StatusLed\" \
          name=\"StatusLed\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon StatusLedFactory::icon() const
{
  return QIcon(":/icons/statusLed.png");
}

// --------------------------------------------------------------------------
QString StatusLedFactory::includeFile() const
{
    return "statusLed.h";
}

// --------------------------------------------------------------------------
bool StatusLedFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString StatusLedFactory::name() const
{
    return "StatusLed";
}

//-----------------------------------------------------------------------------
QString StatusLedFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString StatusLedFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString StatusLedFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void StatusLedFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
