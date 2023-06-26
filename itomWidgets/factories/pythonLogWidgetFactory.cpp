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
#include "pythonLogWidgetFactory.h"
#include "pythonLogWidget.h"


// --------------------------------------------------------------------------
PythonLogWidgetFactory::PythonLogWidgetFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *PythonLogWidgetFactory::createWidget(QWidget *_parent)
{
    PythonLogWidget* widget = new PythonLogWidget(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString PythonLogWidgetFactory::domXml() const
{
  return "<widget class=\"PythonLogWidget\" name=\"pythonLogWidget\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon PythonLogWidgetFactory::icon() const
{
  return QIcon(":/icons/widget.png");
}

// --------------------------------------------------------------------------
QString PythonLogWidgetFactory::includeFile() const
{
    return "pythonLogWidget.h";
}

// --------------------------------------------------------------------------
bool PythonLogWidgetFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString PythonLogWidgetFactory::name() const
{
    return "PythonLogWidget";
}

//-----------------------------------------------------------------------------
QString PythonLogWidgetFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString PythonLogWidgetFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString PythonLogWidgetFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void PythonLogWidgetFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
