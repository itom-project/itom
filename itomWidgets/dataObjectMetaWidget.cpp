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
*********************************************************************** */

#include "dataObjectMetaWidget.h"
#include <complex>
#include <qspinbox.h>

#include "opencv2/imgproc/imgproc.hpp"
#include <QWidgetItem>

#include "common/numeric.h"
#include "common/sharedStructuresGraphics.h"
//#include "common/apiFunctionsGraphInc.h"
//#include "common/apiFunctionsInc.h"

//----------------------------------------------------------------------------------------------------------------------------------

template <typename _Tp> void convertComplexMat(const cv::Mat& complexIn, cv::Mat& matOut)
{
    ito::float64 absVal = 0;
    ito::float64 absValMax = -std::numeric_limits<ito::float64>::max();
    ito::float64 absValMin = std::numeric_limits<ito::float64>::max();
    ito::float64 phaVal = 0;
    ito::float64 phaValMax = -std::numeric_limits<ito::float64>::max();
    ito::float64 phaValMin = std::numeric_limits<ito::float64>::max();

    ito::float64 deltaAbs = 0;
    ito::float64 deltaPha = 0;

    for (int y = 0; y < complexIn.rows; y++)
    {
        const _Tp* ptr = complexIn.ptr<_Tp>(y);
        for (int x = 0; x < complexIn.cols; x++)
        {
            absVal = abs(ptr[x]);

            if (absVal < absValMin)
            {
                absValMin = absVal;
            }
            if (absVal > absValMax)
            {
                absValMax = absVal;
            }

            phaVal = arg(ptr[x]);

            if (phaVal < phaValMin)
            {
                phaValMin = phaVal;
            }
            if (phaVal > phaValMax)
            {
                phaValMax = phaVal;
            }
        }
    }

    if (ito::isNotZero(absValMax - absValMin))
        deltaAbs = 255.0 / (absValMax - absValMin);
    else
        deltaAbs = 255.0;

    if (ito::isNotZero(phaValMax - phaValMin))
        deltaPha = 255.0 / (phaValMax - phaValMin);
    else
        deltaPha = 255.0;


    for (int y = 0; y < complexIn.rows; y++)
    {
        const _Tp* ptr = complexIn.ptr<_Tp>(y);
        ito::uint8* dstPtr = matOut.ptr<ito::uint8>(y);
        for (int x = 0; x < complexIn.cols; x++)
        {
            absVal = abs(ptr[x]);
            dstPtr[x] = cv::saturate_cast<ito::uint8>(deltaAbs * (absVal - absValMin));

            phaVal = arg(ptr[x]);
            dstPtr[x + complexIn.cols + 2] = cv::saturate_cast<ito::uint8>(phaVal * deltaPha);
        }
        dstPtr[complexIn.cols] = 255;
        dstPtr[complexIn.cols + 1] = 255;
    }

    return;
}

//----------------------------------------------------------------------------------------------------------------------------------
DataObjectMetaWidget::DataObjectMetaWidget(QWidget* parent /*= 0*/) :
    QTreeWidget(parent), m_readOnly(false), m_decimals(3), m_preview(true), m_previewSize(256),
    m_detailedStatus(false)
{
    m_data = ito::DataObject();

    m_colorBarName = "";

    this->setHeaderHidden(true);
    this->setColumnCount(2);
    addTopLevelItem(
        new QTreeWidgetItem(this, QStringList(tr("Object details").toLatin1().data()), 0));
    addTopLevelItem(
        new QTreeWidgetItem(this, QStringList(tr("Axes details").toLatin1().data()), 0));
    addTopLevelItem(
        new QTreeWidgetItem(this, QStringList(tr("Value details").toLatin1().data()), 0));
    addTopLevelItem(new QTreeWidgetItem(this, QStringList(tr("Tags").toLatin1().data()), 0));
    addTopLevelItem(new QTreeWidgetItem(this, QStringList(tr("Protocol").toLatin1().data()), 0));
    setIconSize(QSize(m_previewSize, m_previewSize));

    m_colorTable.reserve(256);
    m_colorTable.resize(256);

    for (int i = 0; i < 256; i++)
    {
        m_colorTable[i] = 0xFF000000 + (i << 16) + (i << 8) + i;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
DataObjectMetaWidget::~DataObjectMetaWidget()
{
    m_data = ito::DataObject();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectMetaWidget::setData(QSharedPointer<ito::DataObject> dataObj)
{
    m_data = *dataObj;

    // clear old object data
    this->clear();

    bool checker;

    QStringList data;
    data.reserve(2);
    // write new object data

    int dims = m_data.getDims();

    // write header
    QTreeWidgetItem* objTree =
        new QTreeWidgetItem(this, QStringList(tr("Object details").toLatin1().data()), 0);
    QTreeWidgetItem* axesTree =
        new QTreeWidgetItem(this, QStringList(tr("Axes details").toLatin1().data()), 0);
    QTreeWidgetItem* valueTree =
        new QTreeWidgetItem(this, QStringList(tr("Value details").toLatin1().data()), 0);
    QTreeWidgetItem* tagsTree =
        new QTreeWidgetItem(this, QStringList(tr("Tags").toLatin1().data()), 0);
    QTreeWidgetItem* protocolTree =
        new QTreeWidgetItem(this, QStringList(tr("Protocol").toLatin1().data()), 0);

    QString typeStr;
    switch (m_data.getType())
    {
    case ito::tUInt8:
        typeStr = "uint8";
        break;
    case ito::tInt8:
        typeStr = "int8";
        break;
    case ito::tUInt16:
        typeStr = "uint16";
        break;
    case ito::tInt16:
        typeStr = "int16";
        break;
    case ito::tUInt32:
        typeStr = "uint32";
        break;
    case ito::tInt32:
        typeStr = "int32";
        break;
    case ito::tFloat32:
        typeStr = "float32";
        break;
    case ito::tFloat64:
        typeStr = "float64";
        break;
    case ito::tComplex64:
        typeStr = "complex64";
        break;
    case ito::tComplex128:
        typeStr = "complex128";
        break;
    case ito::tRGBA32:
        typeStr = "rgba32";
        break;
    default:
        typeStr = "";
        break;
    }

    data.append("dtype");
    data.append(typeStr);
    objTree->addChild(new QTreeWidgetItem(objTree, data, 0));

    data[0] = "continuous";
    data[1] = m_data.getContinuous() == 0 ? "no" : "yes";
    objTree->addChild(new QTreeWidgetItem(objTree, data, 0));

    data[0] = "own data";
    data[1] = m_data.getOwnData() == 0 ? "no" : "yes";
    objTree->addChild(new QTreeWidgetItem(objTree, data, 0));

    data[0] = "dimensions";
    data[1] = QString::number(dims);
    objTree->addChild(new QTreeWidgetItem(objTree, data, 0));

    // pixel size
    data[0] = "shape";
    data[1].reserve(2 + 2 + 12 * dims);
    data[1] = "[";
    for (int dim = 0; dim < dims; dim++)
    {
        data[1].append(QString::number(m_data.getSize(dim)));
        if (dim < (dims - 1))
        {
            data[1].append(", ");
        }
    }
    data[1].append("]");
    objTree->addChild(new QTreeWidgetItem(objTree, data, 0));

    // physical size
    data[0] = "physical shape";
    data[1] = "[";
    double val;
    std::string unit;

    for (int dim = 0; dim < dims; dim++)
    {
        val = m_data.getPixToPhys(dim, m_data.getSize(dim), checker) -
            m_data.getPixToPhys(dim, 0, checker);
        data[1].append(QString::number(val));
        unit = m_data.getAxisUnit(dim, checker);

        if (unit != "")
        {
            data[1].append(" " + QString::fromLocal8Bit(unit.data()));
        }

        if (dim < (dims - 1))
        {
            data[1].append(", ");
        }
    }

    data[1].append("]");
    objTree->addChild(new QTreeWidgetItem(objTree, data, 0));

    // physical origin
    data[0] = "physical origin";
    data[1] = "[";

    for (int dim = 0; dim < dims; dim++)
    {
        val = m_data.getPixToPhys(dim, 0, checker);
        data[1].append(QString::number(val));
        unit = m_data.getAxisUnit(dim, checker);

        if (unit != "")
        {
            data[1].append(" " + QString::fromLocal8Bit(unit.data()));
        }

        if (dim < (dims - 1))
        {
            data[1].append(", ");
        }
    }
    data[1].append("]");
    objTree->addChild(new QTreeWidgetItem(objTree, data, 0));

    // write axes tree
    data[0] = "descriptions";
    data[1].reserve(2 + 2 + 12 * dims);
    data[1] = "[";

    for (int dim = 0; dim < dims; dim++)
    {
        data[1].append(QString::fromLocal8Bit(m_data.getAxisDescription(dim, checker).data()));

        if (dim < (dims - 1))
        {
            data[1].append(", ");
        }
    }

    data[1].append("]");
    axesTree->addChild(new QTreeWidgetItem(axesTree, data, 0));

    data[0] = "units";
    data[1].reserve(2 + 2 + 12 * dims);
    data[1] = "[";

    for (int dim = 0; dim < dims; dim++)
    {
        data[1].append(QString::fromLocal8Bit(m_data.getAxisUnit(dim, checker).data()));
        if (dim < (dims - 1))
        {
            data[1].append(", ");
        }
    }

    data[1].append("]");
    axesTree->addChild(new QTreeWidgetItem(axesTree, data, 0));

    data[0] = "scales";
    data[1].reserve(2 + 2 + 12 * dims);
    data[1] = "[";

    for (int dim = 0; dim < dims; dim++)
    {
        val = m_data.getAxisScale(dim);
        data[1].append(QString::number(val));

        if (dim < (dims - 1))
        {
            data[1].append(", ");
        }
    }

    data[1].append("]");
    axesTree->addChild(new QTreeWidgetItem(axesTree, data, 0));

    data[0] = "offsets";
    data[1] = "[";

    for (int dim = 0; dim < dims; dim++)
    {
        val = m_data.getAxisOffset(dim);
        data[1].append(QString::number(val));

        if (dim < (dims - 1))
        {
            data[1].append(", ");
        }
    }

    data[1].append("]");
    axesTree->addChild(new QTreeWidgetItem(axesTree, data, 0));
    addTopLevelItem(axesTree);

    // value tree
    data[0] = "description";
    data[1] = QString::fromLocal8Bit(m_data.getValueDescription().data());
    valueTree->addChild(new QTreeWidgetItem(valueTree, data, 0));

    data[0] = "unit";
    data[1] = QString::fromLocal8Bit(m_data.getValueUnit().data());
    valueTree->addChild(new QTreeWidgetItem(valueTree, data, 0));

    data[0] = "scale";
    data[1] = QString::number(m_data.getValueScale());
    valueTree->addChild(new QTreeWidgetItem(valueTree, data, 0));

    data[0] = "offset";
    data[1] = QString::number(m_data.getValueOffset());
    valueTree->addChild(new QTreeWidgetItem(valueTree, data, 0));

    addTopLevelItem(valueTree);

    // tags tree
    int tagNumber = m_data.getTagListSize();
    std::string key;
    ito::DataObjectTagType type;

    for (int i = 0; i < tagNumber; i++)
    {
        m_data.getTagByIndex(i, key, type);

        if (key == "protocol")
            continue;

        if (type.getType() == ito::DataObjectTagType::typeDouble)
        {
            data[0] = key.data();
            data[1] = QString::number(type.getVal_ToDouble(), 'g', m_decimals);
            tagsTree->addChild(new QTreeWidgetItem(tagsTree, data, 0));
        }
        else if (type.getType() == ito::DataObjectTagType::typeString)
        {
            data[0] = key.data();
            data[1] = type.getVal_ToString().data();
            tagsTree->addChild(new QTreeWidgetItem(tagsTree, data, 0));
        }
    }

    addTopLevelItem(tagsTree);

    //  protocol tree
    type = m_data.getTag("protocol", checker);

    if (checker)
    {
        QString temp = QString::fromLocal8Bit(type.getVal_ToString().data());
#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
        QStringList tempList = temp.split('\n', Qt::SkipEmptyParts);
#else
        QStringList tempList = temp.split('\n', QString::SkipEmptyParts);
#endif
        for (int i = 0; i < tempList.size(); i++)
        {
            data[0] = "";
            data[1] = tempList[i];
            protocolTree->addChild(new QTreeWidgetItem(protocolTree, data, 0));
        }
    }
    addTopLevelItem(protocolTree);

    expandAll();

    resizeColumnToContents(0);
    resizeColumnToContents(1);
}

//----------------------------------------------------------------------------------------------------------------------------------
QSharedPointer<ito::DataObject> DataObjectMetaWidget::getData() const
{
    return QSharedPointer<ito::DataObject>(new ito::DataObject(m_data));
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectMetaWidget::setReadOnly(const bool value)
{
    m_readOnly = value;
}
//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectMetaWidget::setDecimals(const int value)
{
    m_decimals = value > 0 ? value : 0;
}
//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectMetaWidget::setPreviewSize(const int value)
{
    m_previewSize = value > 0 ? (value < 512 ? value : 512) : 0;
}
//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectMetaWidget::setPreviewStatus(const bool value)
{
    m_preview = value;
}
//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectMetaWidget::setDetailedStatus(const bool value)
{
    m_detailedStatus = value;
}
//----------------------------------------------------------------------------------------------------------------------------------
/*
void DataObjectMetaWidget::setColorMap(const QString &name)
{
    ito::RetVal retval(ito::retOk);
    int numPalettes = 1;

    if(ITOM_API_FUNCS_GRAPH == NULL)
    {
        return;
    }

    retval += apiPaletteGetNumberOfColorBars(numPalettes);

    if (numPalettes == 0 || retval.containsError())
    {
        return;
    }

    ito::ItomPalette newPalette;

    retval += apiPaletteGetColorBarName(name, newPalette);

    if (retval.containsError())
    {
        return;
    }

    if(newPalette.type == ito::tPaletteNoType)
    {
        return;
    }

    m_colorBarName = name;
    m_colorTable = newPalette.colorVector256;

    // replot


    return;
}
*/

//----------------------------------------------------------------------------------------------------------------------------------
QSize DataObjectMetaWidget::sizeHint() const
{
    int h = 25;
    int w = 15;

    return QSize(w, h);
}
