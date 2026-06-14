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

#include "dataObjectTable.h"

#include <QtCore/QtPlugin>
#include "dataObjectTableFactory.h"


DataObjectTableFactory::DataObjectTableFactory(QObject *parent)
    : QObject(parent)
{
    initialized = false;
}

void DataObjectTableFactory::initialize(QDesignerFormEditorInterface * /*core*/)
{
    if (initialized)
        return;

    initialized = true;
}

bool DataObjectTableFactory::isInitialized() const
{
    return initialized;
}

QWidget *DataObjectTableFactory::createWidget(QWidget *parent)
{
    return new DataObjectTable(parent);
}

QString DataObjectTableFactory::name() const
{
    return "DataObjectTable";
}

QString DataObjectTableFactory::group() const
{
    return "itom [widgets]";
}

QIcon DataObjectTableFactory::icon() const
{
    return QIcon(":/icons/dataObjectTable.png");
}

QString DataObjectTableFactory::toolTip() const
{
    return QString();
}

QString DataObjectTableFactory::whatsThis() const
{
    return QObject::tr("itom widget to interpret a dataObject as a table.");
}

bool DataObjectTableFactory::isContainer() const
{
    return false;
}

QString DataObjectTableFactory::domXml() const
{
    return "<widget class=\"DataObjectTable\" name=\"dataObjectTable\">\n"
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
         <property name=\"alignment\">\n \
           <set>Qt::AlignRight | Qt::AlignTrailing | Qt::AlignVCenter</set>\n \
           </property>\n \
        </widget>\n";
}

QString DataObjectTableFactory::includeFile() const
{
    return "dataObjectTable.h";
}
