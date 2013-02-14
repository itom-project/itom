/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

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

    ui.setupUi(this);
    ui.itomLogo->setPixmap( QPixmap(QString::fromUtf8(":/application/icons/itomicon/q_itoM64.png")));
    ui.ITOLogo->setPixmap( QPixmap(QString::fromUtf8(":/application/icons/itomicon/itologo64.png")));

    QString currTxt = ui.infoText->toHtml();

    for( int i = 0; i < versionMap.size(); i++)
    {
        m_VersionString.append(QString("%1: %2\n").arg(versionMap[i].first, versionMap[i].second));
        QString keyWord(QString("[%1]").arg(versionMap[i].first));
        if(currTxt.contains(keyWord))
        {
            currTxt = currTxt.replace(keyWord, versionMap[i].second.replace('\n', "<p>"));
        }
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