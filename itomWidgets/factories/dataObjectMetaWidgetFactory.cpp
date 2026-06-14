/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#include "dataObjectMetaWidget.h"

#include <QtCore/QtPlugin>
#include "dataObjectMetaWidgetFactory.h"


DataObjectMetaWidgetFactory::DataObjectMetaWidgetFactory(QObject* parent)
    : QObject(parent)
{
    initialized = false;
}

void DataObjectMetaWidgetFactory::initialize(QDesignerFormEditorInterface* /*core*/)
{
    if (initialized)
        return;

    initialized = true;
}

bool DataObjectMetaWidgetFactory::isInitialized() const
{
    return initialized;
}

QWidget* DataObjectMetaWidgetFactory::createWidget(QWidget* parent)
{
    return new DataObjectMetaWidget(parent);
}

QString DataObjectMetaWidgetFactory::name() const
{
    return "DataObjectMetaWidget";
}

QString DataObjectMetaWidgetFactory::group() const
{
    return "itom [widgets]";
}

QIcon DataObjectMetaWidgetFactory::icon() const
{
    return QIcon(":/icons/paramEditorWidget.png");
}

QString DataObjectMetaWidgetFactory::toolTip() const
{
    return QString();
}

QString DataObjectMetaWidgetFactory::whatsThis() const
{
    return QObject::tr("itom widget to show the dataObject meta information.");
}

bool DataObjectMetaWidgetFactory::isContainer() const
{
    return false;
}

QString DataObjectMetaWidgetFactory::domXml() const
{
    return "<widget class=\"DataObjectMetaWidget\" name=\"dataObjectMetaWidget\">\n"
        " <attribute name=\"verticalHeaderDefaultSectionSize\">\n \
            <number>20</number>\n \
          </attribute>\n"
        " <attribute name=\"horizontalHeaderDefaultSectionSize\">\n \
            <number>100</number>\n \
          </attribute>\n"
        "<property name=\"rowCount\">\n \
            <number>3</number>\n \
           </property>\n \
         <property name=\"columnCount\">\n \
            <number>3</number>\n \
           </property>\n \
        </widget>\n";
}

QString DataObjectMetaWidgetFactory::includeFile() const
{
    return "dataObjectMetaWidget.h";
}
