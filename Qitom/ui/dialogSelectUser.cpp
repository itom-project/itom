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

#include "dialogSelectUser.h"

#include "../AppManagement.h"

namespace ito {

DialogSelectUser::DialogSelectUser(QWidget *parent) :
    QDialog(parent),
    m_userModel(NULL)
{
    ui.setupUi(this);
}

DialogSelectUser::~DialogSelectUser()
{
}

void DialogSelectUser::DialogInit(UserModel *model)
{
    m_userModel = model;
    QItemSelectionModel *selModel = ui.userList->selectionModel();
    QObject::connect(selModel, SIGNAL(currentChanged (const QModelIndex &, const QModelIndex &)), this, SLOT(userListCurrentChanged(const QModelIndex &, const QModelIndex &))); 
}

void DialogSelectUser::userListCurrentChanged(const QModelIndex &current, const QModelIndex &previous)
{
    if (m_userModel)
    {
        QModelIndex curIdx = ui.userList->currentIndex();
        if (curIdx.isValid())
        {
            QModelIndex midx = m_userModel->index(curIdx.row(), 0);
            ui.lineEdit_name->setText(midx.data().toString());
            midx = m_userModel->index(curIdx.row(), 2);
            ui.lineEdit_role->setText(midx.data().toString());
            midx = m_userModel->index(curIdx.row(), 3);
            ui.lineEdit_iniFile->setText(midx.data().toString());
        }
    }
}


} //end namespace ito