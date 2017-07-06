/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2016, Institut fuer Technische Optik (ITO), 
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
#include "pythonMessageBoxFactory.h"
#include "pythonMessageBox.h"


// --------------------------------------------------------------------------
PythonMessageBoxFactory::PythonMessageBoxFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *PythonMessageBoxFactory::createWidget(QWidget *_parent)
{
    PythonMessageBox* widget = new PythonMessageBox(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString PythonMessageBoxFactory::domXml() const
{
  return "<widget class=\"PythonMessageBox\" name=\"pythonMessageBox\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon PythonMessageBoxFactory::icon() const
{
  return QIcon(":/icons/widget.png");
}

// --------------------------------------------------------------------------
QString PythonMessageBoxFactory::includeFile() const
{
    return "pythonMessageBox.h";
}

// --------------------------------------------------------------------------
bool PythonMessageBoxFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString PythonMessageBoxFactory::name() const
{
    return "PythonMessageBox";
}

//-----------------------------------------------------------------------------
QString PythonMessageBoxFactory::group() const
{ 
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString PythonMessageBoxFactory::toolTip() const
{ 
    return QString(); 
}

//-----------------------------------------------------------------------------
QString PythonMessageBoxFactory::whatsThis() const
{
    return QString(); 
}

//-----------------------------------------------------------------------------
void PythonMessageBoxFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}