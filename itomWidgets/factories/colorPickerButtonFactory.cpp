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
#include "colorPickerButtonFactory.h"
#include "colorPickerButton.h"


// --------------------------------------------------------------------------
ColorPickerButtonFactory::ColorPickerButtonFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *ColorPickerButtonFactory::createWidget(QWidget *_parent)
{
    ColorPickerButton* widget = new ColorPickerButton(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString ColorPickerButtonFactory::domXml() const
{
  return "<widget class=\"ColorPickerButton\" \
          name=\"ColorPickerButton\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon ColorPickerButtonFactory::icon() const
{
  return QIcon(":/icons/pushbutton.png");
}

// --------------------------------------------------------------------------
QString ColorPickerButtonFactory::includeFile() const
{
    return "colorPickerButton.h";
}

// --------------------------------------------------------------------------
bool ColorPickerButtonFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString ColorPickerButtonFactory::name() const
{
    return "ColorPickerButton";
}

//-----------------------------------------------------------------------------
QString ColorPickerButtonFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString ColorPickerButtonFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString ColorPickerButtonFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void ColorPickerButtonFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
