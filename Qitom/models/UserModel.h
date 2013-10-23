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

#ifndef USERMODEL_H
#define USERMODEL_H

#include <qabstractitemmodel.h>

namespace ito 
{
    /*!
        \class UserInfo
        \brief holds the relevant user information
    */
    struct UserInfoStruct
    {
        UserInfoStruct(QString sname, QString sid, QString siniFile, QString srole) : name(sname), id(sid), iniFile(siniFile), role(srole) {}
        QString name;
        QString id;
        QString iniFile;
        QString role;
    };

    /** @class UserModel
    *   @brief class for for visualizing the available users
    *   
    *   The UserModel is used in the initially shown user list. It contains the userId (which is the user name part of the ini-file name),
    *   the plain text user name and the ini-file.
    */
    class UserModel : public QAbstractItemModel
    {
        Q_OBJECT

        public:
            UserModel(/*const QString &data, QObject *parent = 0*/);
            ~UserModel();

            QVariant data(const QModelIndex &index, int role) const;
            QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
            QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
            QModelIndex parent(const QModelIndex &index) const;
            int rowCount(const QModelIndex &parent = QModelIndex()) const;
            int columnCount(const QModelIndex &parent = QModelIndex()) const;
            int addUser(const UserInfoStruct &newUser);

        private:
            QList<QString> m_headers;               //!<  string list of names of column headers
            QList<QVariant> m_alignment;            //!<  list of alignments for the corresponding headers
            QList<UserInfoStruct> m_userInfo;     //!<  list with user information
    };
}

#endif //USERMODEL_H
