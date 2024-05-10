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
#include "penCreatorButtonFactory.h"
#include "penCreatorButton.h"


// --------------------------------------------------------------------------
PenCreatorButtonFactory::PenCreatorButtonFactory(QObject *_parent)
    : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *PenCreatorButtonFactory::createWidget(QWidget *_parent)
{
    PenCreatorButton* widget = new PenCreatorButton(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString PenCreatorButtonFactory::domXml() const
{
    return "<widget class=\"PenCreatorButton\" \
                name=\"PenCreatorButton\">\n"
             "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon PenCreatorButtonFactory::icon() const
{
    return QIcon(":/icons/pen.png");
}

// --------------------------------------------------------------------------
QString PenCreatorButtonFactory::includeFile() const
{
    return "penCreatorButton.h";
}

// --------------------------------------------------------------------------
bool PenCreatorButtonFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString PenCreatorButtonFactory::name() const
{
    return "PenCreatorButton";
}

//-----------------------------------------------------------------------------
QString PenCreatorButtonFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString PenCreatorButtonFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString PenCreatorButtonFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void PenCreatorButtonFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
        return;
    }
    initialized = true;
}
