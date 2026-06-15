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

#ifndef USERMANAGEMENTWIDGET_H
#define USERMANAGEMENTWIDGET_H

#include "../models/UserModel.h"

#include <qdialog.h>

#include "ui_userManagement.h"

namespace ito {

class DialogUserManagement : public QDialog
{
    Q_OBJECT

    public:

        DialogUserManagement(QWidget* parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());

        ~DialogUserManagement();
        Ui::userManagement ui;

    private:
        UserModel *m_userModel;
        QString m_currentUserName;
        void readModel(const QModelIndex &index);
        void loadUserList();

    protected:
        void init();

    private slots:
        void userListCurrentChanged(const QModelIndex &current, const QModelIndex &previous);
        void on_pushButton_newUser_clicked();
        void on_pushButton_delUser_clicked();
        void on_pushButton_editUser_clicked();
        void on_userList_doubleClicked(const QModelIndex & index);
};

} //end namespace ito

#endif //USERMANAGEMENTWIDGET_H
