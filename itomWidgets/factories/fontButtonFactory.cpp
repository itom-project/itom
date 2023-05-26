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
#include "fontButtonFactory.h"
#include "fontButton.h"


// --------------------------------------------------------------------------
FontButtonFactory::FontButtonFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *FontButtonFactory::createWidget(QWidget *_parent)
{
	FontButton* widget = new FontButton(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString FontButtonFactory::domXml() const
{
  return "<widget class=\"FontButton\" \
          name=\"FontButton\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon FontButtonFactory::icon() const
{
  return QIcon(":/icons/pushbutton.png");
}

// --------------------------------------------------------------------------
QString FontButtonFactory::includeFile() const
{
    return "fontButton.h";
}

// --------------------------------------------------------------------------
bool FontButtonFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString FontButtonFactory::name() const
{
    return "FontButton";
}

//-----------------------------------------------------------------------------
QString FontButtonFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString FontButtonFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString FontButtonFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void FontButtonFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
