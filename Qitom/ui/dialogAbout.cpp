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
#include <QClipboard>

DialogAboutQItom::DialogAboutQItom(QList<QPair<QString, QString> > versionMap)
{
    m_VersionString.clear();
    QFile file(":/license/about.html");
    if(!file.open(QIODevice::ReadOnly)) 
    {
        QMessageBox::information(0, "error", file.errorString());
    }

    QTextStream in(&file);
    QString currTxt(in.readAll());
    file.close();

    bool hasGIT = false;
    bool hasSVN = false;

    ui.setupUi(this);
    ui.itomLogo->setPixmap( QPixmap(QString::fromUtf8(":/application/icons/itomicon/q_itoM64.png")));
    ui.ITOLogo->setPixmap( QPixmap(QString::fromUtf8(":/application/icons/itomicon/itologo64.png")));

    for( int i = 0; i < versionMap.size(); i++)
    {
        m_VersionString.append(QString("%1: %2\n").arg(versionMap[i].first, versionMap[i].second));
        QString keyWord(QString("$%1$").arg(versionMap[i].first));
        if(currTxt.contains(keyWord))
        {
            currTxt = currTxt.replace(keyWord, versionMap[i].second.replace('\n', "<p>"));
        }
        if(versionMap[i].first.compare("itom_GITHASH") == 0)
        {
            hasGIT = true;
        }
        if(versionMap[i].first.compare("itom_SVNRevision") == 0)
        {
            hasSVN = true;
        }
    }

    int x0 = currTxt.indexOf("$USINGSVN$");
    int x1 = currTxt.lastIndexOf("$USINGSVN$");
    if(!hasSVN)
    {
        currTxt.remove(x0, x1-x0+10);
    }
    else
    {
        currTxt.remove(x0, 10);
        x1 = currTxt.lastIndexOf("$USINGSVN$");
        currTxt.remove(x1-10, 10);
    }

    x0 = currTxt.indexOf("$USINGGIT$");
    x1 = currTxt.lastIndexOf("$USINGGIT$");
    if(!hasGIT)
    {
        currTxt.remove(x0, x1-x0+10);
    }
    else
    {
        currTxt.remove(x0, 10);
        currTxt.remove(x1-10, 10);    
    }

    ui.infoText->setHtml(currTxt);

};

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DialogAboutQItom::on_pushButtonCopy_clicked()
{
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(m_VersionString, QClipboard::Clipboard);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DialogAboutQItom::on_pushButton_close_clicked()
{
    close();
}