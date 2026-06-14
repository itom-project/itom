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

#include "widgetPropGeneralStyles.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>
#include <qdir.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropGeneralStyles::WidgetPropGeneralStyles(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropGeneralStyles::~WidgetPropGeneralStyles()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralStyles::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("ApplicationStyle");
    QString qssFile = settings.value("cssFile", "").toString();
    QString rccFile = settings.value("rccFile", "").toString();
    QDir appPath(QCoreApplication::applicationDirPath());
    QDir qssDir(qssFile);
    if (qssFile != "" && qssDir.isRelative())
    {
        qssFile = QDir::cleanPath(appPath.absoluteFilePath(qssFile));
    }

    QDir rccDir(rccFile);
    if (rccFile != "" && rccDir.isRelative())
    {
        rccFile = QDir::cleanPath(appPath.absoluteFilePath(rccFile));
    }

    QFileInfo qss(qssFile);
    QFileInfo rcc(rccFile);
    QDir stylePath = appPath;
    stylePath.cd("styles");
    stylePath.cd("stylesheets");

    ui.comboPredefinedStyle->clear();
    QStringList styleFolders = stylePath.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    QString absPath;
    QString rccAbsPath = rccFile != "" ? rcc.absoluteFilePath() : "";
    QString qssAbsPath = qssFile != "" ? qss.absoluteFilePath() : "";
    int foundIdx = -1;

    foreach(const QString &styleFolder, styleFolders)
    {
        if (stylePath.exists(QString("%1/%1.qss").arg(styleFolder)))
        {
            absPath = stylePath.absoluteFilePath(QString("%1/%1.qss").arg(styleFolder));

            ui.comboPredefinedStyle->addItem(styleFolder, styleFolder);

            if (qssAbsPath == absPath)
            {
                if (rccFile == "" || !rcc.exists())
                {
                    foundIdx = ui.comboPredefinedStyle->count() - 1;
                }
                else if ((rcc.baseName() == qss.baseName()) && stylePath.exists(QString("%1/%1.rcc").arg(styleFolder)))
                {
                    foundIdx = ui.comboPredefinedStyle->count() - 1;
                }
            }
        }
    }

    if (!qss.exists() || !qss.isFile() || qss.suffix() != "qss")
    {
        ui.radioNoStyle->setChecked(true);
    }
    else if (foundIdx >= 0)
    {
        ui.radioPredefinedStyle->setChecked(true);
        ui.comboPredefinedStyle->setCurrentIndex(foundIdx);
    }
    else
    {
        ui.radioUserdefinedStyle->setChecked(true);
        ui.pathResource->setCurrentPath(rccAbsPath);
        ui.pathStylesheet->setCurrentPath(qssAbsPath);
        ui.comboPredefinedStyle->setCurrentIndex(-1);
    }

    QString iconTheme = settings.value("iconTheme", "auto").toString();

    if (iconTheme.compare("auto", Qt::CaseInsensitive) == 0)
    {
        ui.comboIconTheme->setCurrentIndex(0);
    }
    else if (iconTheme.compare("bright", Qt::CaseInsensitive) == 0)
    {
        ui.comboIconTheme->setCurrentIndex(1);
    }
    else
    {
        ui.comboIconTheme->setCurrentIndex(2);
    }

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralStyles::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("ApplicationStyle");

    if (ui.radioNoStyle->isChecked())
    {
        settings.setValue("cssFile", "");
        settings.setValue("rccFile", "");
    }
    else if (ui.radioPredefinedStyle->isChecked())
    {
        QDir appPath(QCoreApplication::applicationDirPath());
        QString name = ui.comboPredefinedStyle->currentData().toString();
        QFileInfo qss(appPath.absoluteFilePath(QString("styles/stylesheets/%1/%1.qss").arg(name)));

        if (qss.exists())
        {
            settings.setValue("cssFile", appPath.relativeFilePath(QString("styles/stylesheets/%1/%1.qss").arg(name)));
            QFileInfo rcc(appPath.absoluteFilePath(QString("styles/stylesheets/%1/%1.rcc").arg(name)));

            if (rcc.exists())
            {
                settings.setValue("rccFile", appPath.relativeFilePath(QString("styles/stylesheets/%1/%1.rcc").arg(name)));
            }
            else
            {
                settings.setValue("rccFile", "");
            }
        }
        else
        {
            settings.setValue("cssFile", "");
            settings.setValue("rccFile", "");
        }
    }
    else
    {
        QDir appPath(QCoreApplication::applicationDirPath());

        if (ui.pathStylesheet->currentPath() != "")
        {
            settings.setValue("cssFile", appPath.relativeFilePath(ui.pathStylesheet->currentPath()));
            if (ui.pathResource->currentPath() != "")
            {
                settings.setValue("rccFile", appPath.relativeFilePath(ui.pathResource->currentPath()));
            }
            else
            {
                settings.setValue("rccFile", "");
            }
        }
        else
        {
            settings.setValue("cssFile", "");
            settings.setValue("rccFile", "");
        }
    }

    switch (ui.comboIconTheme->currentIndex())
    {
    default:
        settings.setValue("iconTheme", "auto");
        break;
    case 1:
        settings.setValue("iconTheme", "bright");
        break;
    case 2:
        settings.setValue("iconTheme", "dark");
        break;
    }

    settings.endGroup();
}

} //end namespace ito
