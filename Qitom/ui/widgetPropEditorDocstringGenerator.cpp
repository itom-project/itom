/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "widgetPropEditorDocstringGenerator.h"

#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorDocstringGenerator::WidgetPropEditorDocstringGenerator(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);

    ui.comboDocstringStyle->clear();
    ui.comboDocstringStyle->addItem(tr("Google Style"), "googleStyle");
    ui.comboDocstringStyle->addItem(tr("Numpy Style"), "numpyStyle");

    ui.comboDocstringQuote->clear();
    ui.comboDocstringQuote->addItem("\"\"\"...\"\"\"", "doubleQuotes");
    ui.comboDocstringQuote->addItem("'''...'''", "apostrophe");
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorDocstringGenerator::~WidgetPropEditorDocstringGenerator()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorDocstringGenerator::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    QString style = settings.value("docstringGeneratorStyle", "googleStyle").toString();
    QString quotes = settings.value("docstringGeneratorQuotes", "doubleQuotes").toString();

    for (int i = 0; i < ui.comboDocstringStyle->count(); ++i)
    {
        if (ui.comboDocstringStyle->itemData(i, Qt::UserRole).toString() == style)
        {
            ui.comboDocstringStyle->setCurrentIndex(i);
            break;
        }
    }

    for (int i = 0; i < ui.comboDocstringQuote->count(); ++i)
    {
        if (ui.comboDocstringQuote->itemData(i, Qt::UserRole).toString() == quotes)
        {
            ui.comboDocstringQuote->setCurrentIndex(i);
            break;
        }
    }

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorDocstringGenerator::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    settings.setValue("docstringGeneratorStyle", ui.comboDocstringStyle->currentData(Qt::UserRole));
    settings.setValue("docstringGeneratorQuotes", ui.comboDocstringQuote->currentData(Qt::UserRole));

    settings.endGroup();
}

} //end namespace ito
