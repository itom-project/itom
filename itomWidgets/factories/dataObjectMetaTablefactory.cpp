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

#include "dataObjectMetaTable.h"

#include <QtCore/QtPlugin>
#include "dataObjectMetaTablefactory.h"


DataObjectMetaTableFactory::DataObjectMetaTableFactory(QObject* parent)
    : QObject(parent)
{
    initialized = false;
}

void DataObjectMetaTableFactory::initialize(QDesignerFormEditorInterface* /*core*/)
{
    if (initialized)
        return;

    initialized = true;
}

bool DataObjectMetaTableFactory::isInitialized() const
{
    return initialized;
}

QWidget* DataObjectMetaTableFactory::createWidget(QWidget* parent)
{
    return new DataObjectMetaTable(parent);
}

QString DataObjectMetaTableFactory::name() const
{
    return "DataObjectMetaTable";
}

QString DataObjectMetaTableFactory::group() const
{
    return "itom Plugins";
}

QIcon DataObjectMetaTableFactory::icon() const
{
    return QIcon(":/itomDesignerPlugins/itom/icons/q_itoM32.png");
}

QString DataObjectMetaTableFactory::toolTip() const
{
    return QString();
}

QString DataObjectMetaTableFactory::whatsThis() const
{
    return QObject::tr("itom widget to interprete a dataObject as a table.");
}

bool DataObjectMetaTableFactory::isContainer() const
{
    return false;
}

QString DataObjectMetaTableFactory::domXml() const
{
    return "<widget class=\"DataObjectMetaTable\" name=\"DataObjectMetaTable\">\n"
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

QString DataObjectMetaTableFactory::includeFile() const
{
    return "dataObjectMetaTable.h";
}