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

#ifndef DATAOBJECTMODEL_H
#define DATAOBJECTMODEL_H

#include "DataObject/dataobj.h"

#include <qabstractitemmodel.h>
#include <qsharedpointer.h>
#include <qstringlist.h>

#ifndef DATAOBJECTMODEL_TYPEDEFINED
#define DATAOBJECTMODEL_TYPEDEFINED
    Q_DECLARE_METATYPE(ito::complex64);
    Q_DECLARE_METATYPE(ito::complex128);
    Q_DECLARE_METATYPE(ito::Rgba32);
#endif

class DataObjectModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    DataObjectModel();
    ~DataObjectModel();

    QVariant data(const QModelIndex &index, int role) const;
    bool setData ( const QModelIndex & index, const QVariant & value, int role = Qt::EditRole );

    QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
    QModelIndex parent(const QModelIndex &index) const;
    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;
    Qt::ItemFlags flags ( const QModelIndex & index ) const;

    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;

    void setHeaderLabels(Qt::Orientation orientation, const QStringList &labels);
    inline QStringList getHorizontalHeaderLabels() const { return m_horizontalHeader; }
    inline QStringList getVerticalHeaderLabels() const { return m_verticalHeader; }

    void setDataObject(QSharedPointer<ito::DataObject> dataObj);
    inline QSharedPointer<ito::DataObject> getDataObject() const { return m_sharedDataObj; };

    inline int getType() const { return m_sharedDataObj.data() ? m_sharedDataObj->getType() : ito::tUInt8; }

    void setReadOnly(bool value);
    inline bool getReadOnly() const { return m_readOnly; }

    void setDefaultGrid(int rows, int cols);
    inline int getDefaultRows() const { return m_defaultRows; }
    inline int getDefaultCols() const { return m_defaultCols; }

    void setDecimals(const int decimals);
    inline int getDecimals() const { return m_decimals; }

    void setSuffixes(const QStringList &suffixes);
    inline QStringList getSuffixes() const { return m_suffixes; }

    void setAlignment(const Qt::Alignment &alignment);
    inline Qt::Alignment getAlignment() const { return m_alignment; }

    static int displayRoleWithoutSuffix;
    static int preciseDisplayRoleWithoutSuffix;

protected:
    bool setValue(const int &row, const int &column, const QVariant &value);

    QString getDisplayNumber(const unsigned int &number, const int column) const;
    QString getDisplayNumber(const int &number, const int column) const;
    QString getDisplayNumber(const ito::float64 &number, const int column, int decimals = -1) const;
    QString getDisplayNumber(const ito::float32 &number, const int column, int decimals = -1) const;
    QString getDisplayNumber(const ito::complex64 &number, const int column, int decimals = -1) const;
    QString getDisplayNumber(const ito::complex128 &number, const int column, int decimals = -1) const;

    QSharedPointer<ito::DataObject> m_sharedDataObj;

private:
    QStringList m_verticalHeader;
    QStringList m_horizontalHeader;
    int m_defaultRows;
    int m_defaultCols;
    bool m_readOnly;
    int m_decimals;
    QStringList m_suffixes;
    Qt::Alignment m_alignment;
    
    bool m_dummyData;
};

#endif //DATAOBJECTMODEL_H