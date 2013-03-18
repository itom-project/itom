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

#ifndef DIALOGSELECTUSER_H
#define DIALOGSELECTUSER_H

#include "../global.h"
#include "./models/UserModel.h"

#include <qdialog.h>
#include <qlist.h>

#include "ui_dialogSelectUser.h"

namespace ito {

class DialogSelectUser : public QDialog 
{
    Q_OBJECT

public:
    DialogSelectUser(QWidget *parent = NULL);
    ~DialogSelectUser();
    void DialogInit(UserModel *model);
    Ui::DialogSelectUser ui;

private:
    UserModel *m_userModel;

protected:
    void init();

private slots:
    void userListCurrentChanged(const QModelIndex &current, const QModelIndex &previous);
};

} //end namespace ito

#endif