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

#include "dObMetaDataTable.h"

#include <QtCore/QtPlugin>
#include "dObMetaDataTablefactory.h"


dObMetaDataTableFactory::dObMetaDataTableFactory(QObject *parent)
    : QObject(parent)
{
    initialized = false;
}

void dObMetaDataTableFactory::initialize(QDesignerFormEditorInterface * /*core*/)
{
    if (initialized)
        return;

    initialized = true;
}

bool dObMetaDataTableFactory::isInitialized() const
{
    return initialized;
}

QWidget *dObMetaDataTableFactory::createWidget(QWidget *parent)
{
    return new dObMetaDataTable(parent);
}

QString dObMetaDataTableFactory::name() const
{
    return "dObMetaDataTable";
}

QString dObMetaDataTableFactory::group() const
{
    return "itom Plugins";
}

QIcon dObMetaDataTableFactory::icon() const
{
    return QIcon(":/itomDesignerPlugins/itom/icons/q_itoM32.png");
}

QString dObMetaDataTableFactory::toolTip() const
{
    return QString();
}

QString dObMetaDataTableFactory::whatsThis() const
{
    return QObject::tr("itom widget to interprete a dataObject as a table.");
}

bool dObMetaDataTableFactory::isContainer() const
{
    return false;
}

QString dObMetaDataTableFactory::domXml() const
{
    return "<widget class=\"dObMetaDataTable\" name=\"dObMetaDataTable\">\n"
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

QString dObMetaDataTableFactory::includeFile() const
{
    return "dObMetaDataTable.h";
}