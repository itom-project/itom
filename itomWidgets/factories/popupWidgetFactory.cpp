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
#include "popupWidgetFactory.h"
#include "popupWidget.h"

// --------------------------------------------------------------------------
PopupWidgetFactory::PopupWidgetFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *PopupWidgetFactory::createWidget(QWidget *_parent)
{
    PopupWidget* newWidget = new PopupWidget(_parent);
    // if the widget is a tooltip, it wouldn't accept children
    newWidget->setWindowFlags(Qt::WindowType());
    // if the widget auto hides, it disappear from the workplace and don't allow
    // children anymore.
    newWidget->setAutoHide(false);
    return newWidget;
}

// --------------------------------------------------------------------------
QString PopupWidgetFactory::domXml() const
{
  return "<widget class=\"PopupWidget\" name=\"popupWidget\">\n</widget>\n";
}

// --------------------------------------------------------------------------
QIcon PopupWidgetFactory::icon() const
{
  return QIcon(":/icons/widget.png");
}

// --------------------------------------------------------------------------
QString PopupWidgetFactory::includeFile() const
{
    return "popupWidget.h";
}

// --------------------------------------------------------------------------
bool PopupWidgetFactory::isContainer() const
{
    return true;
}

// --------------------------------------------------------------------------
QString PopupWidgetFactory::name() const
{
    return "PopupWidget";
}

//-----------------------------------------------------------------------------
QString PopupWidgetFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString PopupWidgetFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString PopupWidgetFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void PopupWidgetFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
