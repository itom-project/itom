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
#include "dataObjectTable.h"
#include <qcolor.h>
#include <qnumeric.h>

#include "DataObject/dataObjectFuncs.h"
#include "common/helperDatetime.h"
#include "common/typeDefs.h"

int DataObjectModel::displayRoleWithoutSuffix = Qt::UserRole + 1;
int DataObjectModel::preciseDisplayRoleWithoutSuffix = Qt::UserRole + 2;
int DataObjectModel::longlongDoubleOrStringRoleWithoutSuffix = Qt::UserRole + 3;

//-------------------------------------------------------------------------------------
DataObjectModel::DataObjectModel() :
    m_readOnly(false), m_defaultRows(0), m_defaultCols(0), m_decimals(2), m_dummyData(true),
    m_alignment(Qt::AlignLeft), m_numberFormat('f'), m_heatmapInterval(0, 0, true),
    m_enableHeatmap(true), m_colorStopLow(122, 190, 2), m_colorStopMiddle(255, 218, 0),
    m_colorStopHigh(254, 5, 0)
{
    m_sharedDataObj = QSharedPointer<ito::DataObject>(new ito::DataObject());
    // m_sharedDataObj->zeros(m_defaultRows, m_defaultCols, ito::tFloat32);
    setAlignment(Qt::AlignVCenter);
    setHeatmapType(DataObjectTable::Off);
}

//-------------------------------------------------------------------------------------
DataObjectModel::~DataObjectModel()
{
}

//-------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(const unsigned int& number, const int column) const
{
    QString suffix;
    if (m_suffixes.size() > 0 && column >= 0)
    {
        suffix = m_suffixes[std::min(column, (int)m_suffixes.size() - 1)];
    }

    if (column >= 0)
    {
        // local style (dot might be replaced by dot...)
        return QString("%L1%2").arg(number).arg(suffix);
    }
    else
    {
        // program style, dot remains dot... (for copy to clipboard operations)
        return QString("%1%2").arg(number).arg(suffix);
    }
}

//-------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(const int& number, const int column) const
{
    QString suffix;
    if (m_suffixes.size() > 0 && column >= 0)
    {
        suffix = m_suffixes[std::min(column, (int)m_suffixes.size() - 1)];
    }

    if (column >= 0)
    {
        // local style (dot might be replaced by dot...)
        return QString("%L1%2").arg(number).arg(suffix);
    }
    else
    {
        // program style, dot remains dot... (for copy to clipboard operations)
        return QString("%1%2").arg(number).arg(suffix);
    }
}

//-------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(
    const ito::float64& number, const int column, int decimals /*= -1*/) const
{
    QString suffix;
    if (m_suffixes.size() > 0 && column >= 0)
    {
        suffix = m_suffixes[std::min(column, (int)m_suffixes.size() - 1)];
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
            return QString("%L1%2").arg(number, 0, m_numberFormat, decimals).arg(suffix);
        }
        else
        {
            // programm style, dot remains dot... (for copy to clipboard operations)
            return QString("%1%2").arg(number, 0, m_numberFormat, decimals).arg(suffix);
        }
    }
}

//-------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(
    const ito::float32& number, const int column, int decimals /*= -1*/) const
{
    QString suffix;
    if (m_suffixes.size() > 0 && column >= 0)
    {
        suffix = m_suffixes[std::min(column, (int)m_suffixes.size() - 1)];
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
            return QString("%L1%2").arg(number, 0, m_numberFormat, decimals).arg(suffix);
        }
        else
        {
            // programm style, dot remains dot... (for copy to clipboard operations)
            return QString("%1%2").arg(number, 0, m_numberFormat, decimals).arg(suffix);
        }
    }
}

//-------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(
    const ito::complex64& number, const int column, int decimals /*= -1*/) const
{
    QString suffix;
    if (m_suffixes.size() > 0 && column >= 0)
    {
        suffix = m_suffixes[std::min(column, (int)m_suffixes.size() - 1)];
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
                    .arg(number.real(), 0, m_numberFormat, decimals)
                    .arg(number.imag(), 0, m_numberFormat, decimals)
                    .arg(suffix);
            }
            return QString("%L1-%L2i%3")
                .arg(number.real(), 0, m_numberFormat, decimals)
                .arg(-number.imag(), 0, m_numberFormat, decimals)
                .arg(suffix);
        }
        else
        {
            // programm style, dot remains dot... (for copy to clipboard operations)
            if (number.imag() >= 0)
            {
                return QString("%1+%2i%3")
                    .arg(number.real(), 0, m_numberFormat, decimals)
                    .arg(number.imag(), 0, m_numberFormat, decimals)
                    .arg(suffix);
            }
            return QString("%1-%2i%3")
                .arg(number.real(), 0, m_numberFormat, decimals)
                .arg(-number.imag(), 0, m_numberFormat, decimals)
                .arg(suffix);
        }
    }
}

//-------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(
    const ito::complex128& number, const int column, int decimals /*= -1*/) const
{
    QString suffix;
    if (m_suffixes.size() > 0 && column >= 0)
    {
        suffix = m_suffixes[std::min(column, (int)m_suffixes.size() - 1)];
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
                    .arg(number.real(), 0, m_numberFormat, decimals)
                    .arg(number.imag(), 0, m_numberFormat, decimals)
                    .arg(suffix);
            }
            return QString("%L1-%L2i%3")
                .arg(number.real(), 0, m_numberFormat, decimals)
                .arg(-number.imag(), 0, m_numberFormat, decimals)
                .arg(suffix);
        }
        else
        {
            // programm style, dot remains dot... (for copy to clipboard operations)
            if (number.imag() >= 0)
            {
                return QString("%1+%2i%3")
                    .arg(number.real(), 0, m_numberFormat, decimals)
                    .arg(number.imag(), 0, m_numberFormat, decimals)
                    .arg(suffix);
            }
            return QString("%1-%2i%3")
                .arg(number.real(), 0, m_numberFormat, decimals)
                .arg(-number.imag(), 0, m_numberFormat, decimals)
                .arg(suffix);
        }
    }
}

//-------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(
    const ito::DateTime& number, const int column, bool longDate) const
{
    auto dt = ito::datetime::toQDateTime(number);

    if (longDate || dt.time().msec() != 0)
    {
        return dt.toString(Qt::DateFormat::ISODateWithMs);
    }
    else if (dt.time().second() != 0)
    {
        return dt.toString(Qt::DateFormat::ISODate);
    }
    else
    {
        QString datestring = m_locale.toString(dt, QLocale::ShortFormat);

        // deprecated in Qt6, however the method above should be equal.
        //QString datestring = dt.toString(Qt::DateFormat::DefaultLocaleShortDate);

        return datestring;
    }
}

//-------------------------------------------------------------------------------------
QString DataObjectModel::getDisplayNumber(
    const ito::TimeDelta& number, const int column, bool /*longDate*/) const
{
    int days, seconds, useconds;
    ito::timedelta::toDSU(number, days, seconds, useconds);

    int sec = seconds % 60;
    seconds -= sec;
    int minutes = seconds / 60;
    int min = minutes % 60;
    minutes -= min;
    int hour = minutes / 60;
    QLatin1Char fill('0');

    QString result;

    if (days != 0)
    {
        result = QObject::tr("%1 days ").arg(days);
    }

    if ((days >= 0) && (sec < 0 || min < 0 || hour < 0 || useconds < 0))
    {
        result += "-";
    }

    result += QObject::tr("%1:%2:%3")
                  .arg(std::abs(hour), 2, 10, fill)
                  .arg(std::abs(min), 2, 10, fill)
                  .arg(std::abs(sec), 2, 10, fill);

    if (useconds != 0)
    {
        if (useconds % 1000 == 0)
        {
            result += QString(".%1").arg(std::abs(useconds / 1000), 3, 10, fill);
        }
        else
        {
            result += QString(".%1").arg(std::abs(useconds), 6, 10, fill);
        }
    }

    return result;
}

//-------------------------------------------------------------------------------------
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
        // show the tooltip text more precise than the display
        int decimals = (role == Qt::DisplayRole) ? m_decimals : 2 * m_decimals;

        switch (m_sharedDataObj->getDims())
        {
        case 0:
            // default case (for designer, adjustment can be done using the
            // defaultRow and defaultCol property)
            return getDisplayNumber(0.0, column);
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
            case ito::tDateTime:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::DateTime>(row, column),
                    column,
                    role == Qt::ToolTipRole);
            case ito::tTimeDelta:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::TimeDelta>(row, column), column, false);
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
            // default case (for designer, adjustment can be done using the
            // defaultRow and defaultCol property)
            return (double)0.0;
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
            {
                ito::Rgba32 c = m_sharedDataObj->at<ito::Rgba32>(row, column);
                return QColor(c.r, c.g, c.b, c.a);
            }
            case ito::tDateTime:
                return ito::datetime::toQDateTime(m_sharedDataObj->at<ito::DateTime>(row, column));
            }
        default:
            return QVariant();
        }
    }
    else if (role == longlongDoubleOrStringRoleWithoutSuffix)
    {
        const int decimals = 8;

        switch (m_sharedDataObj->getDims())
        {
        case 0:
            // default case (for designer, adjustment can be done using the
            // defaultRow and defaultCol property)
            return (qlonglong)0;
        case 1:
        case 2:
            switch (m_sharedDataObj->getType())
            {
            case ito::tInt8:
                return (qlonglong)m_sharedDataObj->at<ito::int8>(row, column);
            case ito::tUInt8:
                return (qlonglong)m_sharedDataObj->at<ito::uint8>(row, column);
            case ito::tInt16:
                return (qlonglong)m_sharedDataObj->at<ito::int16>(row, column);
            case ito::tUInt16:
                return (qlonglong)m_sharedDataObj->at<ito::uint16>(row, column);
            case ito::tInt32:
                return (qlonglong)m_sharedDataObj->at<ito::int32>(row, column);
            case ito::tUInt32:
                return (qlonglong)m_sharedDataObj->at<ito::uint32>(row, column);
            case ito::tFloat32:
                return (double)m_sharedDataObj->at<ito::float32>(row, column);
            case ito::tFloat64:
                return (double)m_sharedDataObj->at<ito::float64>(row, column);
            case ito::tComplex64:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::complex64>(row, column), -1, decimals);
            case ito::tComplex128:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::complex128>(row, column), -1, decimals);
            case ito::tRGBA32: {
                ito::Rgba32 c = m_sharedDataObj->at<ito::Rgba32>(row, column);

                if (c.alpha() < 255)
                {
                    return QColor::fromRgba(c.argb()).name(QColor::HexArgb);
                }
                else
                {
                    return QColor::fromRgba(c.argb()).name(QColor::HexRgb);
                }
            }
            case ito::tTimeDelta:
                return getDisplayNumber(m_sharedDataObj->at<ito::TimeDelta>(row, column), -1, true);
            case ito::tDateTime:
                return getDisplayNumber(m_sharedDataObj->at<ito::DateTime>(row, column), -1, true);
            }
        default:
            return QVariant();
        }
    }
    else if (role == Qt::BackgroundRole)
    {
        if (m_enableHeatmap)
        {
            double value;

            switch (m_sharedDataObj->getType())
            {
            case ito::tInt8:
                value = m_sharedDataObj->at<ito::int8>(row, column);
                break;
            case ito::tUInt8:
                value = m_sharedDataObj->at<ito::uint8>(row, column);
                break;
            case ito::tInt16:
                value = m_sharedDataObj->at<ito::int16>(row, column);
                break;
            case ito::tUInt16:
                value = m_sharedDataObj->at<ito::uint16>(row, column);
                break;
            case ito::tInt32:
                value = m_sharedDataObj->at<ito::int32>(row, column);
                break;
            case ito::tUInt32:
                value = m_sharedDataObj->at<ito::uint32>(row, column);
                break;
            case ito::tFloat32:
                value = m_sharedDataObj->at<ito::float32>(row, column);
                break;
            case ito::tFloat64:
                value = m_sharedDataObj->at<ito::float64>(row, column);
                break;
            case ito::tRGBA32: {
                ito::Rgba32 c(m_sharedDataObj->at<ito::Rgba32>(row, column));

                if (c.alpha() == 0)
                {
                    return QVariant();
                }

                return QColor(c.argb());
            }
            case ito::tDateTime: {
                const auto dt = m_sharedDataObj->at<ito::DateTime>(row, column);
                // seconds since epoch in utc
                value = dt.datetime / 1000000.0 + dt.utcOffset;
                break;
            }
            case ito::tTimeDelta:
                // seconds
                value = m_sharedDataObj->at<ito::TimeDelta>(row, column).delta / 1000000.0;
                break;
            default:
                return QVariant();
            }

            if (qIsFinite(value))
            {
                double factor = (value - m_heatmapInterval.minimum()) /
                    (m_heatmapInterval.maximum() - m_heatmapInterval.minimum());
                const QColor& c1 = factor < 0.5 ? m_colorStopLow : m_colorStopMiddle;
                const QColor& c2 = factor < 0.5 ? m_colorStopMiddle : m_colorStopHigh;
                factor = factor < 0.5 ? factor : factor - 0.5;
                factor = qBound(0.0, factor * 2, 1.0);

                int r = c1.red() * (1.0 - factor) + c2.red() * factor;
                int g = c1.green() * (1.0 - factor) + c2.green() * factor;
                int b = c1.blue() * (1.0 - factor) + c2.blue() * factor;
                int a = c1.alpha() * (1.0 - factor) + c2.alpha() * factor;

                return QColor(r, g, b, a);
            }
            else
            {
                return QVariant();
            }
        }
        else
        {
            return QVariant();
        }
    }
    else if (role == Qt::TextAlignmentRole)
    {
        return (int)m_alignment;
    }
    else if (role == displayRoleWithoutSuffix || role == preciseDisplayRoleWithoutSuffix)
    {
        int decimals = (role == displayRoleWithoutSuffix) ? 4 : 8;

        switch (m_sharedDataObj->getDims())
        {
        case 0:
            // default case (for designer, adjustment can be done
            // using the defaultRow and defaultCol property)
            return getDisplayNumber(0.0, -1);
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
            case ito::tDateTime:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::DateTime>(row, column),
                    -1,
                    role == preciseDisplayRoleWithoutSuffix);
            case ito::tTimeDelta:
                return getDisplayNumber(
                    m_sharedDataObj->at<ito::TimeDelta>(row, column),
                    -1,
                    role == preciseDisplayRoleWithoutSuffix);
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

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
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
    case ito::tDateTime:
        if (value.canConvert<QDateTime>())
        {
            QDateTime dt = value.value<QDateTime>();
            ito::DateTime dt2 = ito::datetime::toDateTime(dt);
            m_sharedDataObj->at<ito::DateTime>(row, column) = dt2;
            emit dataChanged(i, i);
            return true;
        }
        return false;
    }

    return false;
}

//-------------------------------------------------------------------------------------
QModelIndex DataObjectModel::index(int row, int column, const QModelIndex& parent) const
{
    if (parent.isValid() == false)
    {
        return createIndex(row, column);
    }

    return QModelIndex();
}

//-------------------------------------------------------------------------------------
QModelIndex DataObjectModel::parent(const QModelIndex& /*index*/) const
{
    return QModelIndex();
}

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
Qt::ItemFlags DataObjectModel::flags(const QModelIndex& index) const
{
    if (m_readOnly || m_sharedDataObj->getDims() == 0)
    {
        return Qt::ItemIsSelectable | Qt::ItemIsEnabled;
    }

    return Qt::ItemIsEditable | Qt::ItemIsSelectable | Qt::ItemIsEnabled;
}

//-------------------------------------------------------------------------------------
void DataObjectModel::setDataObject(QSharedPointer<ito::DataObject> dataObj)
{
    m_dummyData = false;

    beginResetModel();
    m_sharedDataObj = dataObj;

    int type = dataObj->getType();

    if (m_heatmapInterval.isAuto() && type != ito::tRGBA32 && type != ito::tComplex64 &&
        type != ito::tComplex128)
    {
        ito::uint32 temp[] = {0, 0, 0};
        ito::dObjHelper::minMaxValue(
            dataObj.data(), m_heatmapInterval.rmin(), temp, m_heatmapInterval.rmax(), temp, true);
    }

    endResetModel();
}

//-------------------------------------------------------------------------------------
void DataObjectModel::setReadOnly(bool value)
{
    beginResetModel();
    m_readOnly = value;
    endResetModel();
}

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
void DataObjectModel::setAlignment(const Qt::Alignment& alignment)
{
    if (alignment != m_alignment)
    {
        beginResetModel();
        m_alignment = alignment;
        endResetModel();
    }
}

//-------------------------------------------------------------------------------------
void DataObjectModel::setDecimals(const int decimals)
{
    if (m_decimals != decimals)
    {
        beginResetModel();
        m_decimals = decimals;
        endResetModel();
    }
}

//-------------------------------------------------------------------------------------
void DataObjectModel::setHeatmapType(int type)
{
    beginResetModel();

    switch ((DataObjectTable::HeatmapType)type)
    {
    case DataObjectTable::Off:
        m_enableHeatmap = false;
        break;
    case DataObjectTable::RealColor:
        m_enableHeatmap = true;
        break;
    case DataObjectTable::RedYellowGreen:
        m_enableHeatmap = true;
        m_colorStopLow = QColor(122, 190, 2);
        m_colorStopMiddle = QColor(255, 218, 0);
        m_colorStopHigh = QColor(254, 5, 0);
        break;
    case DataObjectTable::GreenYellowRed:
        m_enableHeatmap = true;
        m_colorStopHigh = QColor(122, 190, 2);
        m_colorStopMiddle = QColor(255, 218, 0);
        m_colorStopLow = QColor(254, 5, 0);
        break;
    case DataObjectTable::RedWhiteGreen:
        m_enableHeatmap = true;
        m_colorStopLow = QColor(122, 190, 2);
        m_colorStopMiddle = QColor(255, 255, 255);
        m_colorStopHigh = QColor(254, 5, 0);
        break;
    case DataObjectTable::GreenWhiteRed:
        m_enableHeatmap = true;
        m_colorStopHigh = QColor(122, 190, 2);
        m_colorStopMiddle = QColor(255, 255, 255);
        m_colorStopLow = QColor(254, 5, 0);
        break;
    }

    endResetModel();
}

//-------------------------------------------------------------------------------------
void DataObjectModel::setNumberFormat(const char& format)
{
    if (format != m_numberFormat)
    {
        beginResetModel();
        m_numberFormat = format;
        endResetModel();
    }
}

//-------------------------------------------------------------------------------------
void DataObjectModel::setSuffixes(const QStringList& suffixes)
{
    beginResetModel();
    m_suffixes = suffixes;
    endResetModel();
}

//-------------------------------------------------------------------------------------
void DataObjectModel::setHeatmapInterval(const ito::AutoInterval& interval)
{
    beginResetModel();
    m_heatmapInterval = interval;

    if (m_sharedDataObj)
    {
        int type = m_sharedDataObj->getType();

        if (m_heatmapInterval.isAuto() && type != ito::tRGBA32 && type != ito::tComplex64 &&
            type != ito::tComplex128)
        {
            ito::uint32 temp[] = {0, 0, 0};
            ito::dObjHelper::minMaxValue(
                m_sharedDataObj.data(),
                m_heatmapInterval.rmin(),
                temp,
                m_heatmapInterval.rmax(),
                temp,
                true);
        }
    }

    endResetModel();
}
