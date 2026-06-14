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

#include "dialogAbout.h"

#include "../global.h"

#include <QClipboard>
#include <qmessagebox.h>

namespace ito
{

//-------------------------------------------------------------------------------------------------------------------------------------------------
DialogAboutQItom::DialogAboutQItom(const QMap<QString, QString> &versionMap, QWidget *parent /*= NULL*/) :
    QDialog(parent),
    m_textColor(Qt::black),
    m_linkColor(Qt::blue)
{
    setWindowFlags(windowFlags() | Qt::WindowMaximizeButtonHint);

    m_VersionString.clear();
    QFile file(":/license/about.html"); //:/license/about.html");
    if (!file.open(QIODevice::ReadOnly))
    {
        m_aboutText = tr("Could not load file %1. Reason: %2.").arg("about.html").arg(file.errorString());
    }

    QTextStream in(&file);
    m_aboutText = in.readAll();

    file.close();

    bool hasGIT = false;
    bool hasSVN = false;
    bool hasAdditionalEdition = false;

    ui.setupUi(this);
    ui.itomLogo->setPixmap(QPixmap(QString::fromUtf8(":/application/icons/itomicon/itomLogo3_64.png")));
    ui.ITOLogo->setPixmap(QPixmap(QString::fromUtf8(":/application/icons/itomicon/itologo64.png")));

    QMapIterator<QString, QString> i(versionMap);
    while (i.hasNext())
    {
        i.next();

        m_VersionString.append(QString("%1: %2\n").arg(i.key(), i.value()));
        QString keyWord = QString("$%1$").arg(i.key());

        if (m_aboutText.contains(keyWord))
        {
            m_aboutText = m_aboutText.replace(keyWord, QString(i.value()).replace('\n', "<p>"));
        }

        if (i.key() == "itom_GIT_Rev" && i.value() != "")
        {
            hasGIT = true;
        }
        else if (i.key() == "itom_EditionName" && i.value() != "")
        {
            hasAdditionalEdition = true;
        }
    }

    int x0 = m_aboutText.indexOf("$USINGGIT$");
    int x1 = m_aboutText.lastIndexOf("$USINGGIT$");

    if (!hasGIT)
    {
        m_aboutText.remove(x0, x1 - x0 + 10);
    }
    else
    {
        m_aboutText.remove(x0, 10);
        m_aboutText.remove(x1 - 10, 10);
    }

    x0 = m_aboutText.indexOf("$WITHPCL$");
    x1 = m_aboutText.lastIndexOf("$WITHPCL$");
#if ITOM_POINTCLOUDLIBRARY > 0
    m_aboutText.remove(x0, 9);
    m_aboutText.remove(x1 - 9, 9);
#else
    m_aboutText.remove(x0, x1 - x0 + 9);
#endif

    x0 = m_aboutText.indexOf("$HASITOMEDITION$");
    x1 = m_aboutText.lastIndexOf("$HASITOMEDITION$");

    if (!hasAdditionalEdition)
    {
        m_aboutText.remove(x0, x1 - x0 + 16);
    }
    else
    {
        m_aboutText.remove(x0, 16);
        m_aboutText.remove(x1 - 16, 16);
    }


    ui.txtBasic->setHtml(m_aboutText);


    //contributors
    file.setFileName(":/license/contributors.html");
    if (!file.open(QIODevice::ReadOnly))
    {
        m_contributorsText = tr("Could not load file %1. Reason: %2.").arg("contributors.html").arg(file.errorString());
    }
    else
    {
        m_contributorsText = QTextStream(&file).readAll();
        file.close();
    }

    ui.txtContributors->setHtml(m_contributorsText);

    QString licenseText;
    //license
    file.setFileName(":/license/COPYING.txt");
    if (!file.open(QIODevice::ReadOnly))
    {
        licenseText = tr("Could not load file %1. Reason: %2.").arg("COPYING.txt").arg(file.errorString());
    }
    else
    {
        licenseText = QTextStream(&file).readAll();
        file.close();
    }

    ui.txtLicense->setText(licenseText);

    //address
    file.setFileName(":/license/address.html");
    if (!file.open(QIODevice::ReadOnly))
    {
        m_addressText = tr("Could not load file %1. Reason: %2.").arg("address.html").arg(file.errorString());
    }
    else
    {
        m_addressText = QTextStream(&file).readAll();
        file.close();
    }

    ui.itoText->setText(m_addressText);
};

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DialogAboutQItom::styleTexts()
{
    QString cssCode = QString("p, li, b { color: %1 }; \n a { color: %2; }\n").arg(m_textColor.name(), m_linkColor.name());

    QString temp;
    temp = m_aboutText;
    temp.replace("/*styleinclude*/", cssCode);
    temp.replace("#textColor", m_textColor.name());
    temp.replace("#linkColor", m_linkColor.name());
    ui.txtBasic->setHtml(temp);

    temp = m_contributorsText;
    temp.replace("/*styleinclude*/", cssCode);
    temp.replace("#textColor", m_textColor.name());
    temp.replace("#linkColor", m_linkColor.name());
    ui.txtContributors->setHtml(temp);

    temp = m_addressText;
    temp.replace("#textColor", m_textColor.name());
    temp.replace("#linkColor", m_linkColor.name());
    ui.itoText->setHtml(temp);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DialogAboutQItom::setLinkColor(const QColor &color)
{
    m_linkColor = color;
    styleTexts();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DialogAboutQItom::setTextColor(const QColor &color)
{
    m_textColor = color;
    styleTexts();
}

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
