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
#include "brushCreatorButtonFactory.h"
#include "brushCreatorButton.h"


// --------------------------------------------------------------------------
brushCreatorButtonFactory::brushCreatorButtonFactory(QObject *_parent)
    : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *brushCreatorButtonFactory::createWidget(QWidget *_parent)
{
    BrushCreatorButton* widget = new BrushCreatorButton(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString brushCreatorButtonFactory::domXml() const
{
    return "<widget class=\"BrushCreatorButton\" \
                           name=\"BrushCreatorButton\">\n"
                           "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon brushCreatorButtonFactory::icon() const
{
    return QIcon(":/icons/bucket.png");
}

// --------------------------------------------------------------------------
QString brushCreatorButtonFactory::includeFile() const
{
    return "brushCreatorButton.h";
}

// --------------------------------------------------------------------------
bool brushCreatorButtonFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString brushCreatorButtonFactory::name() const
{
    return "BrushCreatorButton";
}

//-----------------------------------------------------------------------------
QString brushCreatorButtonFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString brushCreatorButtonFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString brushCreatorButtonFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void brushCreatorButtonFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
        return;
    }
    initialized = true;
}
