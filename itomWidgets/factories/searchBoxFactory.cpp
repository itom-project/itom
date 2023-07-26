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
#include "searchBoxFactory.h"
#include "searchBox.h"


// --------------------------------------------------------------------------
SearchBoxFactory::SearchBoxFactory(QObject *_parent)
  : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *SearchBoxFactory::createWidget(QWidget *_parent)
{
    SearchBox* widget = new SearchBox(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString SearchBoxFactory::domXml() const
{
  return "<widget class=\"SearchBox\" \
          name=\"SearchBox\">\n"
          "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon SearchBoxFactory::icon() const
{
  return QIcon(":/icons/search.svg");
}

// --------------------------------------------------------------------------
QString SearchBoxFactory::includeFile() const
{
    return "searchBox.h";
}

// --------------------------------------------------------------------------
bool SearchBoxFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString SearchBoxFactory::name() const
{
    return "SearchBox";
}

//-----------------------------------------------------------------------------
QString SearchBoxFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString SearchBoxFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString SearchBoxFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void SearchBoxFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
    return;
    }
    initialized = true;
}
