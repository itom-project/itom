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
#include "pathLineEditFactory.h"
#include "pathLineEdit.h"


// --------------------------------------------------------------------------
PathLineEditFactory::PathLineEditFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *PathLineEditFactory::createWidget(QWidget *_parent)
{
    PathLineEdit* widget = new PathLineEdit(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString PathLineEditFactory::domXml() const
{
  return "<widget class=\"PathLineEdit\" \
          name=\"pathLineEdit\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon PathLineEditFactory::icon() const
{
  return QIcon(":/icons/pushbutton.png");
}

// --------------------------------------------------------------------------
QString PathLineEditFactory::includeFile() const
{
    return "pathLineEdit.h";
}

// --------------------------------------------------------------------------
bool PathLineEditFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString PathLineEditFactory::name() const
{
    return "PathLineEdit";
}

//-----------------------------------------------------------------------------
QString PathLineEditFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString PathLineEditFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString PathLineEditFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void PathLineEditFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
