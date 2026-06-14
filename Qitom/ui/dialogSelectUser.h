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

#ifndef DIALOGSELECTUSER_H
#define DIALOGSELECTUSER_H

#include "../global.h"
#include "models/UserModel.h"

#include <qdialog.h>
#include <qlist.h>

#include "ui_dialogSelectUser.h"

namespace ito {

class DialogSelectUser : public QDialog
{
    Q_OBJECT

public:
    DialogSelectUser(UserModel *model, QWidget *parent = NULL);
    ~DialogSelectUser();

    bool selectUser(const QString &id);
    QModelIndex selectedIndex() const { return ui.userList->currentIndex(); }


protected:
    void init();

    Ui::DialogSelectUser ui;
    UserModel *m_userModel;

private slots:
    void userListCurrentChanged(const QModelIndex &current, const QModelIndex &previous);
    void on_userList_doubleClicked(const QModelIndex current);
    void on_buttonBox_clicked(QAbstractButton* btn);

private:
    int checkPassword();
};

} //end namespace ito

#endif
