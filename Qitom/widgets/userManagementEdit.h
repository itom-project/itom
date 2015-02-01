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

#ifndef USERMANAGEMENTEDITWIDGET_H
#define USERMANAGEMENTEDITWIDGET_H

#include "../models/UserModel.h"

#include <qdialog.h>

#include "ui_userManagementEdit.h"

namespace ito {

class DialogUserManagementEdit : public QDialog
{
    Q_OBJECT

    public:
        DialogUserManagementEdit(const QString &filename, UserModel *userModel, QWidget * parent = 0, Qt::WindowFlags f = 0);
        ~DialogUserManagementEdit();
        Ui::userManagementEdit ui;

    private:
        UserModel *m_userModel;
        bool saveUser();
        QString m_fileName;

    protected:
//        void init();

    private slots:
        void on_buttonBox_clicked(QAbstractButton* btn);
};

} //end namespace ito

#endif //USERMANAGEMENTEDITWIDGET_H