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
        DialogUserManagementEdit(
            const QString& filename,
            UserModel* userModel,
            QWidget* parent = nullptr,
            Qt::WindowFlags f = Qt::WindowFlags(),
            bool isStandardUser = false);

        ~DialogUserManagementEdit();
        Ui::userManagementEdit ui;

    private:
        void updateScriptButtons();
		void enableWidgetsByUserRole(const UserRole currentUserRole, const UserFeatures &currentFeatures, const UserRole userRole, const UserFeatures &features);

        UserModel *m_userModel;
        bool saveUser();
        QString m_fileName;
        QByteArray m_oldPassword;
        QString m_osUser;
        bool m_showsStandardUser;

    protected:
        QString clearName(const QString &name);

    private slots:
        void on_buttonBox_clicked(QAbstractButton* btn);
        void on_lv_startUpScripts_currentRowChanged(int row);
        void on_lineEdit_name_textChanged(const QString &text);
        void on_cmdAutoID_toggled(bool checked);

    public slots:
        void on_pb_addScript_clicked();
        void on_pb_removeScript_clicked();
        void on_pb_downScript_clicked();
        void on_pb_upScript_clicked();
        void on_cmdUseWindowsUser_clicked();
};

} //end namespace ito

#endif //USERMANAGEMENTEDITWIDGET_H
