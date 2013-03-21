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

#include "userManagement.h"

#include "../AppManagement.h"
#include <QSettings>
#include <QDir>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
DialogUserManagement::DialogUserManagement(QWidget *parent, Qt::WindowFlags f) :
    QDialog(parent),
    m_userModel(NULL)
{
    ui.setupUi(this);

    m_userModel = new UserModel();
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, "itomSettings");
    QSettings::setDefaultFormat(QSettings::IniFormat);

    QString settingsFile;
    QDir appDir(QCoreApplication::applicationDirPath());
    if (!appDir.cd("itomSettings"))
    {
        appDir.mkdir("itomSettings");
        appDir.cd("itomSettings");
    }

    QStringList iniList = appDir.entryList(QStringList("itom_*.ini"));

    int nUser = 0;
    foreach(QString iniFile, iniList) 
    {
        QSettings settings(QDir::cleanPath(appDir.absoluteFilePath(iniFile)), QSettings::IniFormat);

        settings.beginGroup("ITOMIniFile");
        if (settings.contains("name"))
        {
            qDebug() << "found user ini file: " << iniFile;
            m_userModel->addUser(UserInfoStruct(QString(settings.value("name").toString()), iniFile.mid(5, iniFile.length() - 9), QDir::cleanPath(appDir.absoluteFilePath(iniFile)), QString(settings.value("role").toString())));
        }
        settings.endGroup();
    }

    ui.userList->setModel(m_userModel);
    QItemSelectionModel *selModel = ui.userList->selectionModel();
    QObject::connect(selModel, SIGNAL(currentChanged (const QModelIndex &, const QModelIndex &)), this, SLOT(userListCurrentChanged(const QModelIndex &, const QModelIndex &))); 
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogUserManagement::~DialogUserManagement()
{
    m_userModel->deleteLater();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::userListCurrentChanged(const QModelIndex &current, const QModelIndex &previous)
{
    QModelIndex curIdx = ui.userList->currentIndex();
    if (curIdx.isValid())
    {
        QModelIndex midx = m_userModel->index(curIdx.row(), 0);
        ui.lineEdit_name->setText(midx.data().toString());
        midx = m_userModel->index(curIdx.row(), 2);
        ui.lineEdit_group->setText(midx.data().toString());
        midx = m_userModel->index(curIdx.row(), 3);
        ui.lineEdit_iniFile->setText(midx.data().toString());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------

} //end namespace ito