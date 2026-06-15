/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2021, Institut fuer Technische Optik (ITO),
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

   In addition, as a special exception, the Institut fuer Technische
   Optik (ITO) gives you certain additional rights.
   These rights are described in the ITO LGPL Exception version 1.0,
   which can be found in the file LGPL_EXCEPTION.txt in this package.
*********************************************************************** */

#pragma once

#include "DataObject/dataobj.h"
#include "common/interval.h"

#include <qabstractitemmodel.h>
#include <qsharedpointer.h>
#include <qstringlist.h>
#include <qcolor.h>
#include <qlocale.h>

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

    QVariant data(const QModelIndex& index, int role) const;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole);

    QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const;
    QModelIndex parent(const QModelIndex& index) const;
    int rowCount(const QModelIndex& parent = QModelIndex()) const;
    int columnCount(const QModelIndex& parent = QModelIndex()) const;
    Qt::ItemFlags flags(const QModelIndex& index) const;

    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;

    void setHeaderLabels(Qt::Orientation orientation, const QStringList& labels);
    inline QStringList getHorizontalHeaderLabels() const
    {
        return m_horizontalHeader;
    }
    inline QStringList getVerticalHeaderLabels() const
    {
        return m_verticalHeader;
    }

    void setDataObject(QSharedPointer<ito::DataObject> dataObj);
    inline QSharedPointer<ito::DataObject> getDataObject() const
    {
        return m_sharedDataObj;
    };

    inline int getType() const
    {
        return m_sharedDataObj.data() ? m_sharedDataObj->getType() : ito::tUInt8;
    }

    void setReadOnly(bool value);
    inline bool getReadOnly() const
    {
        return m_readOnly;
    }

    void setDefaultGrid(int rows, int cols);
    inline int getDefaultRows() const
    {
        return m_defaultRows;
    }
    inline int getDefaultCols() const
    {
        return m_defaultCols;
    }

    void setDecimals(const int decimals);
    inline int getDecimals() const
    {
        return m_decimals;
    }

    void setSuffixes(const QStringList& suffixes);
    inline QStringList getSuffixes() const
    {
        return m_suffixes;
    }

    void setAlignment(const Qt::Alignment& alignment);
    inline Qt::Alignment getAlignment() const
    {
        return m_alignment;
    }

    void setNumberFormat(const char& format);
    inline char getNumberFormat() const
    {
        return m_numberFormat;
    }

    void setHeatmapType(int type);

    void setHeatmapInterval(const ito::AutoInterval &interval);
    ito::AutoInterval getHeatmapInterval() const
    {
        return m_heatmapInterval;
    }

    static int displayRoleWithoutSuffix;
    static int preciseDisplayRoleWithoutSuffix;
    static int longlongDoubleOrStringRoleWithoutSuffix;

protected:
    bool setValue(const int& row, const int& column, const QVariant& value);

    QString getDisplayNumber(const unsigned int& number, const int column) const;
    QString getDisplayNumber(const int& number, const int column) const;
    QString getDisplayNumber(const ito::float64& number, const int column, int decimals = -1) const;
    QString getDisplayNumber(const ito::float32& number, const int column, int decimals = -1) const;
    QString getDisplayNumber(
        const ito::complex64& number, const int column, int decimals = -1) const;
    QString getDisplayNumber(
        const ito::complex128& number, const int column, int decimals = -1) const;
    QString getDisplayNumber(
        const ito::DateTime& number, const int column, bool longDate) const;
    QString getDisplayNumber(
        const ito::TimeDelta& number, const int column, bool longDate) const;

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
    char m_numberFormat;
    ito::AutoInterval m_heatmapInterval;
    bool m_enableHeatmap;
    QColor m_colorStopLow;
    QColor m_colorStopMiddle;
    QColor m_colorStopHigh;
    QLocale m_locale;

    bool m_dummyData;
};
