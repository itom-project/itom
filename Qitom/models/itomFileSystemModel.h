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

#ifndef ITOMFILESYSTEMMODEL_H
#define ITOMFILESYSTEMMODEL_H

#include <qfilesystemmodel.h>

class QMimeData;

namespace ito
{

class ItomFileSystemModel : public QFileSystemModel
{
Q_OBJECT
public:
    explicit ItomFileSystemModel(QObject *parent = 0);

signals:

public slots:

public:
    bool dropMimeData(const QMimeData *data, Qt::DropAction action, int row, int column, const QModelIndex &parent);

};

} //end namespace ito

#endif // ITOMFILESYSTEMMODEL_H
