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

#include "dataObjectModel.h"
#include <qcolor.h>
#include <qnumeric.h>

#include "common/typeDefs.h"

int DataObjectModel::displayRoleWithoutSuffix = Qt::UserRole + 1;
int DataObjectModel::preciseDisplayRoleWithoutSuffix = Qt::UserRole + 2;

//----------------------------------------------------------------------------------------------------------------------------------
DataObjectModel::DataObjectModel() :
    m_readOnly(false), m_defaultRows(3), m_defaultCols(3), m_decimals(2), m_dummyData(true),
    m_alignment(Qt::AlignLeft)
{
    m_sharedDataObj = QSharedPointer<ito::DataObject>(new ito::DataObject());
    m_sharedDataObj->zeros(m_defaultRows, m_defaultCols, ito::tFloat32);
}

//----------------------------------------------------------------------------------------------------------------------------------
DataObjectModel::~DataObjectModel()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(const unsigned int& number, const int column) const
{
    QString suffix;
    if (m_suffixes.size() > 0 && column >= 0)
    {
        suffix = m_suffixes[std::min(column, m_suffixes.size() - 1)];
    }

    if (column >= 0)
    {
        // local style (dot might be replaced by dot...)
        return QString("%L1%2").arg(number).arg(suffix);
    }
    else
    {
        // programm style, dot remains dot... (for copy to clipboard operations)
        return QString("%1%2").arg(number).arg(suffix);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(const int& number, const int column) const
{
    QString suffix;
    if (m_suffixes.size() > 0 && column >= 0)
    {
        suffix = m_suffixes[std::min(column, m_suffixes.size() - 1)];
    }

    if (column >= 0)
    {
        // local style (dot might be replaced by dot...)
        return QString("%L1%2").arg(number).arg(suffix);
    }
    else
    {
        // programm style, dot remains dot... (for copy to clipboard operations)
        return QString("%1%2").arg(number).arg(suffix);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(
    const ito::float64& number, const int column, int decimals /*= -1*/) const
{
    QString suffix;
    if (m_suffixes.size() > 0 && column >= 0)
    {
        suffix = m_suffixes[std::min(column, m_suffixes.size() - 1)];
    }

    if (decimals < 0)
        decimals = m_decimals;

    if (qIsNaN(number))
    {
        return QString("NaN");
    }
    else if (std::numeric_limits<ito::float64>::infinity() == number)
    {
        return QString("Inf");
    }
    else if (std::numeric_limits<ito::float64>::infinity() == -number)
    {
        return QString("-Inf");
    }
    else
    {
        if (column >= 0)
        {
            // local style (dot might be replaced by dot...)
            return QString("%L1%2").arg(number, 0, 'f', decimals).arg(suffix);
        }
        else
        {
            // programm style, dot remains dot... (for copy to clipboard operations)
            return QString("%1%2").arg(number, 0, 'f', decimals).arg(suffix);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(
    const ito::float32& number, const int column, int decimals /*= -1*/) const
{
    QString suffix;
    if (m_suffixes.size() > 0 && column >= 0)
    {
        suffix = m_suffixes[std::min(column, m_suffixes.size() - 1)];
    }

    if (decimals < 0)
        decimals = m_decimals;

    if (qIsNaN(number))
    {
        return QString("NaN");
    }
    else if (std::numeric_limits<ito::float32>::infinity() == number)
    {
        return QString("Inf");
    }
    else if (std::numeric_limits<ito::float32>::infinity() == -number)
    {
        return QString("-Inf");
    }
    else
    {
        if (column >= 0)
        {
            // local style (dot might be replaced by dot...)
            return QString("%L1%2").arg(number, 0, 'f', decimals).arg(suffix);
        }
        else
        {
            // programm style, dot remains dot... (for copy to clipboard operations)
            return QString("%1%2").arg(number, 0, 'f', decimals).arg(suffix);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(
    const ito::complex64& number, const int column, int decimals /*= -1*/) const
{
    QString suffix;
    if (m_suffixes.size() > 0 && column >= 0)
    {
        suffix = m_suffixes[std::min(column, m_suffixes.size() - 1)];
    }

    if (decimals < 0)
        decimals = m_decimals;

    if (qIsNaN(number.real()))
    {
        return QString("NaN");
    }
    else if (std::numeric_limits<ito::float32>::infinity() == number.real())
    {
        return QString("Inf");
    }
    else if (std::numeric_limits<ito::float32>::infinity() == -number.real())
    {
        return QString("-Inf");
    }
    else
    {
        if (column >= 0)
        {
            // local style (dot might be replaced by dot...)
            if (number.imag() >= 0)
            {
                return QString("%L1+%L2i%3")
                    .arg(number.real(), 0, 'f', decimals)
                    .arg(number.imag(), 0, 'f', decimals)
                    .arg(suffix);
            }
            return QString("%L1-%L2i%3")
                .arg(number.real(), 0, 'f', decimals)
                .arg(-number.imag(), 0, 'f', decimals)
                .arg(suffix);
        }
        else
        {
            // programm style, dot remains dot... (for copy to clipboard operations)
            if (number.imag() >= 0)
            {
                return QString("%1+%2i%3")
                    .arg(number.real(), 0, 'f', decimals)
                    .arg(number.imag(), 0, 'f', decimals)
                    .arg(suffix);
            }
            return QString("%1-%2i%3")
                .arg(number.real(), 0, 'f', decimals)
                .arg(-number.imag(), 0, 'f', decimals)
                .arg(suffix);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(
    const ito::complex128& number, const int column, int decimals /*= -1*/) const
{
    QString suffix;
    if (m_suffixes.size() > 0 && column >= 0)
    {
        suffix = m_suffixes[std::min(column, m_suffixes.size() - 1)];
    }

    if (decimals < 0)
        decimals = m_decimals;

    if (qIsNaN(number.real()))
    {
        return QString("NaN");
    }
    else if (std::numeric_limits<ito::float64>::infinity() == number.real())
    {
        return QString("Inf");
    }
    else if (std::numeric_limits<ito::float64>::infinity() == -number.real())
    {
        return QString("-Inf");
    }
    else
    {
        if (column >= 0)
        {
            // local style (dot might be replaced by dot...)
            if (number.imag() >= 0)
            {
                return QString("%L1+%L2i%3")
                    .arg(number.real(), 0, 'f', decimals)
                    .arg(number.imag(), 0, 'f', decimals)
                    .arg(suffix);
            }
            return QString("%L1-%L2i%3")
                .arg(number.real(), 0, 'f', decimals)
                .arg(-number.imag(), 0, 'f', decimals)
                .arg(suffix);
        }
        else
        {
            // programm style, dot remains dot... (for copy to clipboard operations)
            if (number.imag() >= 0)
            {
                return QString("%1+%2i%3")
                    .arg(number.real(), 0, 'f', decimals)
                    .arg(number.imag(), 0, 'f', decimals)
                    .arg(suffix);
            }
            return QString("%1-%2i%3")
                .arg(number.real(), 0, 'f', decimals)
                .arg(-number.imag(), 0, 'f', decimals)
                .arg(suffix);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QVariant DataObjectModel::data(const QModelIndex& index, int role) const
{
    if (index.row() < 0 || index.column() < 0 ||
        (m_sharedDataObj->getDims() > 0 &&
         (index.row() >= m_sharedDataObj->getSize(0) ||
          index.column() >= m_sharedDataObj->getSize(1))) ||
        !index.isValid())
    {
        return QVariant();
    }

    int row = index.row();
    int column = index.column();

    if (role == Qt::DisplayRole || role == Qt::ToolTipRole)
    {
        int decimals = (role == Qt::DisplayRole)
            ? m_decimals
            : 2 * m_decimals; // show the tooltip text more precise than the display

        switch (m_sharedDataObj->getDims())
        {
        case 0:
            return getDisplayNumber(
                0.0, column); // default case (for designer, adjustment can be done using the
                              // defaultRow and defaultCol property)
        case 1:
        case 2: {
            switch (m_sharedDataObj->getType())
            {
            case ito::tInt8:
                return getDisplayNumber(m_sharedDataObj->at<ito::int8>(row, column), column);
            case ito::tUInt8:
                return getDisplayNumber(m_sharedDataObj->at<ito::uint8>(row, column), column);
            case ito::tInt16:
                return getDisplayNumber(m_sharedDataObj->at<ito::int16>(row, column), column);
            case ito::tUInt16:
                return getDisplayNumber(m_sharedDataObj->at<ito::uint16>(row, column), column);
            case ito::tInt32:
                return getDisplayNumber(m_sharedDataObj->at<ito::int32>(row, column), column);
            case ito::tUInt32:
                return getDisplayNumber(m_sharedDataObj->at<ito::uint32>(row, column), column);
            case ito::tFloat32:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::float32>(row, column), column, decimals);
            case ito::tFloat64:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::float64>(row, column), column, decimals);
            case ito::tComplex64:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::complex64>(row, column), column, decimals);
            case ito::tComplex128:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::complex128>(row, column), column, decimals);
            case ito::tRGBA32: {
                ito::Rgba32 c = m_sharedDataObj->at<ito::Rgba32>(row, column);
                return QColor::fromRgba(c.argb());
            }
            }
        }
        default:
            return QVariant();
        }
    }
    else if (role == Qt::EditRole)
    {
        switch (m_sharedDataObj->getDims())
        {
        case 0:
            return (double)0.0; // default case (for designer, adjustment can be done using the
                                // defaultRow and defaultCol property)
        case 1:
        case 2:
            switch (m_sharedDataObj->getType())
            {
            case ito::tInt8:
                return (int)m_sharedDataObj->at<ito::int8>(row, column);
            case ito::tUInt8:
                return (uint)m_sharedDataObj->at<ito::uint8>(row, column);
            case ito::tInt16:
                return (int)m_sharedDataObj->at<ito::int16>(row, column);
            case ito::tUInt16:
                return (uint)m_sharedDataObj->at<ito::uint16>(row, column);
            case ito::tInt32:
                return (int)m_sharedDataObj->at<ito::int32>(row, column);
            case ito::tUInt32:
                return (uint)m_sharedDataObj->at<ito::uint32>(row, column);
            case ito::tFloat32:
                return (float)m_sharedDataObj->at<ito::float32>(row, column);
            case ito::tFloat64:
                return (double)m_sharedDataObj->at<ito::float64>(row, column);
            case ito::tComplex64:
                return QVariant::fromValue(m_sharedDataObj->at<ito::complex64>(row, column));
            case ito::tComplex128:
                return QVariant::fromValue(m_sharedDataObj->at<ito::complex128>(row, column));
            case ito::tRGBA32:
                return QColor(m_sharedDataObj->at<ito::Rgba32>(row, column).argb());
            }
        default:
            return QVariant();
        }
    }
    else if (role == Qt::TextAlignmentRole)
    {
        return (int)m_alignment;
    }
    else if (role == displayRoleWithoutSuffix || role == preciseDisplayRoleWithoutSuffix)
    {
        int decimals = (role == Qt::DisplayRole)
            ? m_decimals
            : 2 * m_decimals; // show the tooltip text more precise than the display

        switch (m_sharedDataObj->getDims())
        {
        case 0:
            return getDisplayNumber(0.0, -1); // default case (for designer, adjustment can be done
                                              // using the defaultRow and defaultCol property)
        case 1:
        case 2: {
            switch (m_sharedDataObj->getType())
            {
            case ito::tInt8:
                return getDisplayNumber(m_sharedDataObj->at<ito::int8>(row, column), -1);
            case ito::tUInt8:
                return getDisplayNumber(m_sharedDataObj->at<ito::uint8>(row, column), -1);
            case ito::tInt16:
                return getDisplayNumber(m_sharedDataObj->at<ito::int16>(row, column), -1);
            case ito::tUInt16:
                return getDisplayNumber(m_sharedDataObj->at<ito::uint16>(row, column), -1);
            case ito::tInt32:
                return getDisplayNumber(m_sharedDataObj->at<ito::int32>(row, column), -1);
            case ito::tUInt32:
                return getDisplayNumber(m_sharedDataObj->at<ito::uint32>(row, column), -1);
            case ito::tFloat32:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::float32>(row, column), -1, decimals);
            case ito::tFloat64:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::float64>(row, column), -1, decimals);
            case ito::tComplex64:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::complex64>(row, column), -1, decimals);
            case ito::tComplex128:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::complex128>(row, column), -1, decimals);
            case ito::tRGBA32: {
                ito::Rgba32 c = m_sharedDataObj->at<ito::Rgba32>(row, column);
                return QColor::fromRgba(c.argb());
            }
            }
        }
        default:
            return QVariant();
        }
    }
    else
    {
        return QVariant();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
bool DataObjectModel::setData(
    const QModelIndex& index, const QVariant& value, int role /* = Qt::EditRole */)
{
    if (index.row() < 0 || index.row() >= m_sharedDataObj->getSize(0) || index.column() < 0 ||
        index.column() >= m_sharedDataObj->getSize(1) || !index.isValid())
    {
        return false;
    }

    if (role == Qt::EditRole)
    {
        switch (m_sharedDataObj->getDims())
        {
        /*case 0:
            return QVariant();*/
        case 1:
            if (index.column() == 0 && index.row() >= 0 &&
                index.row() < (int)m_sharedDataObj->getSize(0))
            {
                return setValue(index.row(), index.column(), value);
            }
            return false;
        case 2:
            if (index.column() >= 0 && index.column() < (int)m_sharedDataObj->getSize(1) &&
                index.row() >= 0 && index.row() < (int)m_sharedDataObj->getSize(0))
            {
                return setValue(index.row(), index.column(), value);
            }
            return false;
        }
    }

    return false;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool DataObjectModel::setValue(const int& row, const int& column, const QVariant& value)
{
    Q_ASSERT(row >= 0);
    Q_ASSERT(column >= 0);
    Q_ASSERT(m_sharedDataObj->getDims() == 1 || m_sharedDataObj->getDims() == 2);
    Q_ASSERT(row < (int)m_sharedDataObj->getSize(0));
    Q_ASSERT(column < (int)m_sharedDataObj->getSize(1));

    QModelIndex i = createIndex(row, column);
    bool ok = false;

    switch (m_sharedDataObj->getType())
    {
    case ito::tInt8: {
        int val = value.toInt(&ok);
        if (!ok)
            return false;
        m_sharedDataObj->at<ito::int8>(row, column) = cv::saturate_cast<ito::int8>(val);
        emit dataChanged(i, i);
        return true;
    }
    case ito::tUInt8: {
        uint val = value.toUInt(&ok);
        if (!ok)
            return false;
        m_sharedDataObj->at<ito::uint8>(row, column) = cv::saturate_cast<ito::uint8>(val);
        emit dataChanged(i, i);
        return true;
    }
    case ito::tInt16: {
        int val = value.toInt(&ok);
        if (!ok)
            return false;
        m_sharedDataObj->at<ito::int16>(row, column) = cv::saturate_cast<ito::int16>(val);
        emit dataChanged(i, i);
        return true;
    }
    case ito::tUInt16: {
        uint val = value.toUInt(&ok);
        if (!ok)
            return false;
        m_sharedDataObj->at<ito::uint16>(row, column) = cv::saturate_cast<ito::uint16>(val);
        emit dataChanged(i, i);
        return true;
    }
    case ito::tInt32: {
        int val = value.toInt(&ok);
        if (!ok)
            return false;
        m_sharedDataObj->at<ito::uint32>(row, column) = cv::saturate_cast<ito::int32>(val);
        emit dataChanged(i, i);
        return true;
    }
    case ito::tUInt32: {
        uint val = value.toUInt(&ok);
        if (!ok)
            return false;
        m_sharedDataObj->at<ito::uint32>(row, column) = cv::saturate_cast<ito::uint32>(val);
        emit dataChanged(i, i);
        return true;
    }
    case ito::tFloat32: {
        double val = value.toDouble(&ok);
        if (!ok)
            return false;
        m_sharedDataObj->at<ito::float32>(row, column) = cv::saturate_cast<ito::float32>(val);
        emit dataChanged(i, i);
        return true;
    }
    case ito::tFloat64: {
        double val = value.toDouble(&ok);
        if (!ok)
            return false;
        m_sharedDataObj->at<ito::float64>(row, column) = cv::saturate_cast<ito::float64>(val);
        emit dataChanged(i, i);
        return true;
    }
    case ito::tComplex64:
        if (value.canConvert<ito::complex64>())
        {
            m_sharedDataObj->at<ito::complex64>(row, column) = value.value<ito::complex64>();
            emit dataChanged(i, i);
            return true;
        }
        return false;
    case ito::tComplex128:
        if (value.canConvert<ito::complex128>())
        {
            m_sharedDataObj->at<ito::complex128>(row, column) = value.value<ito::complex128>();
            emit dataChanged(i, i);
            return true;
        }
        return false;
    case ito::tRGBA32:
        if (value.canConvert<QColor>())
        {
            QColor c = value.value<QColor>();
            m_sharedDataObj->at<ito::Rgba32>(row, column) =
                ito::Rgba32(c.alpha(), c.red(), c.green(), c.blue());
            emit dataChanged(i, i);
            return true;
        }
        return false;
    }
    return false;
}

//----------------------------------------------------------------------------------------------------------------------------------
QModelIndex DataObjectModel::index(int row, int column, const QModelIndex& parent) const
{
    if (parent.isValid() == false)
    {
        return createIndex(row, column);
    }
    return QModelIndex();
}

//----------------------------------------------------------------------------------------------------------------------------------
QModelIndex DataObjectModel::parent(const QModelIndex& /*index*/) const
{
    return QModelIndex();
}

//----------------------------------------------------------------------------------------------------------------------------------
int DataObjectModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid() == false && m_sharedDataObj->getDims() > 0)
    {
        return m_sharedDataObj->getSize(0);
    }
    else if (parent.isValid() == false) // default case
    {
        return m_defaultRows;
    }
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
int DataObjectModel::columnCount(const QModelIndex& parent) const
{
    if (parent.isValid() == false)
    {
        if (m_sharedDataObj->getDims() > 1)
        {
            return m_sharedDataObj->getSize(1);
        }
        else if (m_sharedDataObj->getDims() == 0) // default case
        {
            return m_defaultCols;
        }
        return 1;
    }
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
QVariant DataObjectModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole)
    {
        if (orientation == Qt::Horizontal)
        {
            switch (m_sharedDataObj->getDims())
            {
            case 0:
                if (m_horizontalHeader.count() > section) // default case
                {
                    return m_horizontalHeader[section];
                }
                else
                {
                    return section;
                }
            case 1:
                return 0;
            default: {
                if (section >= 0 && section < (int)m_sharedDataObj->getSize(1))
                {
                    if (m_horizontalHeader.count() > section)
                    {
                        return m_horizontalHeader[section];
                    }
                    else
                    {
                        return section;
                    }
                }
                return QVariant();
            }
            }
        }
        else // vertical
        {
            switch (m_sharedDataObj->getDims())
            {
            case 0:
                if (m_verticalHeader.count() > section) // default case
                {
                    return m_verticalHeader[section];
                }
                else
                {
                    return section;
                }
            case 1:
                return 1;
            default: {
                if (section >= 0 && section < (int)m_sharedDataObj->getSize(0))
                {
                    if (m_verticalHeader.count() > section)
                    {
                        return m_verticalHeader[section];
                    }
                    else
                    {
                        return section;
                    }
                }
                return QVariant();
            }
            }
        }
    }

    return QVariant();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectModel::setHeaderLabels(Qt::Orientation orientation, const QStringList& labels)
{
    beginResetModel();
    if (orientation == Qt::Horizontal)
    {
        m_horizontalHeader = labels;
    }
    else
    {
        m_verticalHeader = labels;
    }
    endResetModel();
    emit headerDataChanged(orientation, 0, labels.count() - 1);
}

//----------------------------------------------------------------------------------------------------------------------------------
Qt::ItemFlags DataObjectModel::flags(const QModelIndex& index) const
{
    if (m_readOnly || m_sharedDataObj->getDims() == 0)
    {
        return Qt::ItemIsSelectable | Qt::ItemIsEnabled;
    }
    return Qt::ItemIsEditable | Qt::ItemIsSelectable | Qt::ItemIsEnabled;
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectModel::setDataObject(QSharedPointer<ito::DataObject> dataObj)
{
    m_dummyData = false;

    beginResetModel();
    m_sharedDataObj = dataObj;
    endResetModel();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectModel::setReadOnly(bool value)
{
    beginResetModel();
    m_readOnly = value;
    endResetModel();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectModel::setDefaultGrid(int rows, int cols)
{
    if (m_defaultRows != rows || m_defaultCols != cols)
    {
        beginResetModel();

        if (m_dummyData)
        {
            QSharedPointer<ito::DataObject> newObj(new ito::DataObject());
            if (m_decimals == 0)
            {
                newObj->zeros(rows, cols, ito::tInt16);
            }
            else
            {
                newObj->zeros(rows, cols, ito::tFloat32);
            }
            m_sharedDataObj = newObj;
        }

        m_defaultRows = rows;
        m_defaultCols = cols;
        endResetModel();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectModel::setAlignment(const Qt::Alignment& alignment)
{
    beginResetModel();
    m_alignment = alignment;
    endResetModel();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectModel::setDecimals(const int decimals)
{
    beginResetModel();
    m_decimals = decimals;
    endResetModel();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectModel::setSuffixes(const QStringList& suffixes)
{
    beginResetModel();
    m_suffixes = suffixes;
    endResetModel();
}
