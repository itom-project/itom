/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#include "widgetPropGeneralLanguage.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>
#include <qcoreapplication.h>
#include <qfiledialog.h>
#include <qstringlist.h>
#include <qdir.h>
#include <qfileinfo.h>
#include <qlocale.h>
#include <qregexp.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropGeneralLanguage::WidgetPropGeneralLanguage(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);

    ui.listWidget->setSelectionMode(QAbstractItemView::SingleSelection);
    QLocale loc;
    QListWidgetItem *lwi;
    QString lang;
    QString lang1, lang2;
    QString langID;
    QFileInfo finfo;
    QString baseName;

    //add default language
    loc = QLocale(QLocale::English, QLocale::UnitedStates); //English/United States
    lang1 = QLocale::languageToString(loc.language());
    lang2 = QLocale::countryToString(loc.country());
    lang = QString("%1 - %2 (%3) [Default]").arg(lang1).arg(lang2).arg(loc.name());
    lwi = new QListWidgetItem(lang, ui.listWidget);
    lwi->setData(Qt::UserRole + 1, loc.name());

    QDir languageDir;
    languageDir.cd(QCoreApplication::applicationDirPath() + "/translation");
    languageDir.setNameFilters(QStringList("qitom_*.qm"));

    foreach (const QString &fileName, languageDir.entryList(QDir::Files))
    {
        finfo = QFileInfo(fileName);
        langID = finfo.baseName().mid(6); //split "qitom_"
        loc = QLocale(langID); //is for example de_DE or de...
        lang1 = QLocale::languageToString(loc.language());
        lang2 = QLocale::countryToString(loc.country());
        lang = QString("%1 - %2 (%3)").arg(lang1).arg(lang2).arg(loc.name());
        lwi = new QListWidgetItem(lang, ui.listWidget);
        lwi->setData(Qt::UserRole + 1, loc.name());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropGeneralLanguage::~WidgetPropGeneralLanguage()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralLanguage::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("Language");

    QString lang = settings.value("language", "en").toString();
    QLocale loc(lang);

    if (loc.language() == QLocale::C) //the language could not be detected, use the default one as selected langauge
    {
        loc = QLocale(QLocale::English, QLocale::UnitedStates);
    }

    //foreach(const QListWidgetItem &item, ui.listWidget->items())
    //{
    //    if(item->data(Qt::UserRole+1).toString() == loc.name())
    //    {
    //        ui.listWidget->setCurrentIndex(item);

    //        ui.lblCurrentLanguage->setText(tr("Current Language: ") + loc.name());
    //        break;
    //    }
    //}

    for (int i = 0; i < ui.listWidget->count(); i++)
    {
        if (ui.listWidget->item(i)->data(Qt::UserRole + 1).toString() == loc.name())
        {
            ui.listWidget->setCurrentRow(i);

            ui.lblCurrentLanguage->setText(tr("Current Language: ") + ui.listWidget->currentItem()->text());
            break;
        }
    }

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralLanguage::writeSettings()
{
    if (ui.listWidget->currentItem())
    {
        QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
        settings.beginGroup("Language");

        settings.setValue("language", ui.listWidget->currentItem()->data(Qt::UserRole + 1).toString());

        settings.endGroup();

        ui.lblCurrentLanguage->setText(tr("Current Language: ") + ui.listWidget->currentItem()->text());
    }
}

} //end namespace ito