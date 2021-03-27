/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2018, Institut fuer Technische Optik (ITO), 
   Universitaet Stuttgart, Germany 
 
   This file is part of itom.

   itom is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   itom is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef DATAOBJECTDELEGATE_H
#define DATAOBJECTDELEGATE_H

#include "DataObject/dataobj.h"

#include <qtableview.h>
#include <qabstractitemmodel.h>
#include <qitemdelegate.h>
#include <qsharedpointer.h>
#include <qheaderview.h>
#include <qstringlist.h>

class DataObjectDelegate : public QItemDelegate
{
    Q_OBJECT

public:
    DataObjectDelegate(QObject *parent = 0);
    virtual ~DataObjectDelegate();

    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,  const QModelIndex &index) const;
    void setEditorData(QWidget *editor, const QModelIndex &index) const;
    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
    void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;

    friend class DataObjectTable;

private:
    double m_min;
    double m_max;
    int m_editorDecimals;
    QStringList m_suffixes;

};

#endif //DATAOBJECTDELEGATE_H