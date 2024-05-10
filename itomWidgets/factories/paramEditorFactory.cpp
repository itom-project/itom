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
#include "paramEditorFactory.h"
#include "paramEditorWidget.h"


// --------------------------------------------------------------------------
ParamEditorFactory::ParamEditorFactory(QObject *_parent)
    : QObject(_parent)
{
}

// --------------------------------------------------------------------------
QWidget *ParamEditorFactory::createWidget(QWidget *_parent)
{
    ParamEditorWidget* widget = new ParamEditorWidget(_parent);
    return widget;
}

// --------------------------------------------------------------------------
QString ParamEditorFactory::domXml() const
{
    return "<widget class=\"ParamEditorWidget\" \
                           name=\"ParamEditorWidget\">\n"
                           "</widget>\n";
}

// --------------------------------------------------------------------------
QIcon ParamEditorFactory::icon() const
{
    return QIcon(":/icons/paramEditorWidget.png");
}

// --------------------------------------------------------------------------
QString ParamEditorFactory::includeFile() const
{
    return "paramEditorWidget.h";
}

// --------------------------------------------------------------------------
bool ParamEditorFactory::isContainer() const
{
    return false;
}

// --------------------------------------------------------------------------
QString ParamEditorFactory::name() const
{
    return "ParamEditorWidget";
}

//-----------------------------------------------------------------------------
QString ParamEditorFactory::group() const
{
    return "itom [widgets]";
}

//-----------------------------------------------------------------------------
QString ParamEditorFactory::toolTip() const
{
    return QString();
}

//-----------------------------------------------------------------------------
QString ParamEditorFactory::whatsThis() const
{
    return QString();
}

//-----------------------------------------------------------------------------
void ParamEditorFactory::initialize(QDesignerFormEditorInterface *formEditor)
{
    Q_UNUSED(formEditor);
    if (initialized)
    {
        return;
    }
    initialized = true;
}
