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

#include "dialogAbout.h" 

#include "../global.h"

#include <QClipboard>
#include <qmessagebox.h>

namespace ito
{

//-------------------------------------------------------------------------------------------------------------------------------------------------
DialogAboutQItom::DialogAboutQItom(const QMap<QString, QString> &versionMap)
{
    QString tabText;

    m_VersionString.clear();
    QFile file(":/license/about.html"); //:/license/about.html");
    if (!file.open(QIODevice::ReadOnly)) 
    {
        tabText = tr("Could not load file %1. Reason: %2.").arg("about.html").arg(file.errorString());
    }

    QTextStream in(&file);
    tabText = in.readAll();
    file.close();

    bool hasGIT = false;
    bool hasSVN = false;

    ui.setupUi(this);
    ui.itomLogo->setPixmap(QPixmap(QString::fromUtf8(":/application/icons/itomicon/itomIcon64.png")));
    ui.ITOLogo->setPixmap(QPixmap(QString::fromUtf8(":/application/icons/itomicon/itologo64.png")));

    QMapIterator<QString, QString> i(versionMap);
    while (i.hasNext()) 
    {
        i.next();

        m_VersionString.append(QString("%1: %2\n").arg(i.key(), i.value()));
        QString keyWord = QString("$%1$").arg(i.key());
        if (tabText.contains(keyWord))
        {
            tabText = tabText.replace(keyWord, QString(i.value()).replace('\n', "<p>"));
        }
        if (i.key() == "itom_GIT_Rev" && i.value() != "")
        {
            hasGIT = true;
        }
    }

    int x0 = tabText.indexOf("$USINGGIT$");
    int x1 = tabText.lastIndexOf("$USINGGIT$");
    if (!hasGIT)
    {
        tabText.remove(x0, x1 - x0 + 10);
    }
    else
    {
        tabText.remove(x0, 10);
        tabText.remove(x1 - 10, 10);    
    }

    x0 = tabText.indexOf("$WITHPCL$");
    x1 = tabText.lastIndexOf("$WITHPCL$");
#if ITOM_POINTCLOUDLIBRARY > 0
    tabText.remove(x0, 9);
    tabText.remove(x1 - 9, 9);
#else
    tabText.remove(x0, x1 - x0 + 9);
#endif


    ui.txtBasic->setHtml(tabText);


    //contributors
    file.setFileName(":/license/contributors.html");
    if (!file.open(QIODevice::ReadOnly)) 
    {
        tabText = tr("Could not load file %1. Reason: %2.").arg("contributors.html").arg(file.errorString());
    }
    else
    {
        tabText = QTextStream(&file).readAll();
        file.close();
    }

    ui.txtContributors->setHtml(tabText);

    //license
    file.setFileName(":/license/COPYING.txt");
    if (!file.open(QIODevice::ReadOnly)) 
    {
        tabText = tr("Could not load file %1. Reason: %2.").arg("COPYING.txt").arg(file.errorString());
    }
    else
    {
        tabText = QTextStream(&file).readAll();
        file.close();
    }

    ui.txtLicense->setText(tabText);

};

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DialogAboutQItom::on_pushButtonCopy_clicked()
{
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(m_VersionString, QClipboard::Clipboard);
    QMessageBox::information(this, tr("copy"), tr("The version string has been copied to the clipboard"));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DialogAboutQItom::on_pushButton_close_clicked()
{
    close();
}

} //end namespace ito