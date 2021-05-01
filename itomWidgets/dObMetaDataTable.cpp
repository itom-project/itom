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

#include "dObMetaDataTable.h"
#include <qspinbox.h>
#include <complex>

#include <QWidgetItem>
#include "opencv2/imgproc/imgproc.hpp"

#include "common/sharedStructuresGraphics.h"
#include "common/numeric.h"
//#include "common/apiFunctionsGraphInc.h"
//#include "common/apiFunctionsInc.h"

//----------------------------------------------------------------------------------------------------------------------------------

template<typename _Tp> void convertComplexMat(const cv::Mat &complexIn, cv::Mat &matOut)
{
    ito::float64 absVal = 0;
    ito::float64 absValMax = -std::numeric_limits<ito::float64>::max();
    ito::float64 absValMin = std::numeric_limits<ito::float64>::max();
    ito::float64 phaVal = 0;
    ito::float64 phaValMax = -std::numeric_limits<ito::float64>::max();
    ito::float64 phaValMin = std::numeric_limits<ito::float64>::max();

    ito::float64 deltaAbs = 0;
    ito::float64 deltaPha = 0;

    for(int y = 0; y < complexIn.rows; y++)
    {
        const _Tp* ptr = complexIn.ptr<_Tp>(y);
        for(int x = 0; x < complexIn.cols; x++)
        {
            absVal = abs(ptr[x]);

            if(absVal < absValMin)
            {
                absValMin = absVal; 
            }
            if(absVal > absValMax)
            {
                absValMax = absVal; 
            }

            phaVal = arg(ptr[x]);

            if(phaVal < phaValMin)
            {
                phaValMin = phaVal; 
            }
            if(phaVal > phaValMax)
            {
                phaValMax = phaVal; 
            }
        }
    }

    if(ito::isNotZero(absValMax-absValMin)) deltaAbs = 255.0 / (absValMax-absValMin);
    else deltaAbs = 255.0;
                
    if(ito::isNotZero(phaValMax - phaValMin)) deltaPha = 255.0 / (phaValMax - phaValMin);
    else deltaPha = 255.0;
                

    for(int y = 0; y < complexIn.rows; y++)
    {
        const _Tp* ptr = complexIn.ptr<_Tp>(y);
        ito::uint8 * dstPtr = matOut.ptr<ito::uint8>(y);
        for(int x = 0; x < complexIn.cols; x++)
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
dObMetaDataTable::dObMetaDataTable(QWidget *parent /*= 0*/) : QTreeWidget(parent), 
    m_readOnly(false),
    m_decimals(3),
    m_preview(true),
    m_previewSize(256),
    m_detailedStatus(false)
{
    m_data = ito::DataObject();

    m_colorBarName = "";

    this->setHeaderHidden(true);
    this->setColumnCount(2);
    addTopLevelItem(new QTreeWidgetItem(this, QStringList(tr("no preview available")), 0));
    addTopLevelItem(new QTreeWidgetItem(this, QStringList(tr("header")), 0));
    addTopLevelItem(new QTreeWidgetItem(this, QStringList(tr("tag Space")), 0));
    addTopLevelItem(new QTreeWidgetItem(this, QStringList(tr("protocol")), 0));
    setIconSize(QSize(m_previewSize, m_previewSize));
    
    m_colorTable.reserve(256);
    m_colorTable.resize(256);
    
    for(int i = 0; i < 256; i++)
    {
        m_colorTable[i] = 0xFF000000 + (i << 16) + (i << 8) + i; 
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
dObMetaDataTable::~dObMetaDataTable()
{
    m_data = ito::DataObject();
}

//----------------------------------------------------------------------------------------------------------------------------------
void dObMetaDataTable::setData(QSharedPointer<ito::DataObject> dataObj)
{
    m_data = *dataObj;

    // clear old object data
    this->clear();

    bool checker;

    QStringList data;
    data.reserve(2);
    // write new object data
    // add preview is ask

    int dims = m_data.getDims();

    if(m_preview)
    {
        
        if(dims > 0)
        {
            setIconSize(QSize(m_previewSize, m_previewSize));

            bool setThis = true;
            QIcon newPreview;
            cv::Mat tempMat;
            cv::resize(*((cv::Mat*)(m_data.get_mdata()[m_data.seekMat(0)])), tempMat, cv::Size(64, 64), 0, 0, cv::INTER_NEAREST);
            if(m_data.getType() == ito::tRGBA32)
            {
                newPreview = QIcon(QPixmap::fromImage(QImage(tempMat.ptr<ito::uint8>(0),64,64, 64 *4, QImage::Format_ARGB32), Qt::ColorOnly));
            }
            else if(m_data.getType() == ito::tUInt8)
            {
                QImage retImage(tempMat.ptr<ito::uint8>(0), 64,64, 64, QImage::Format_Indexed8);
                retImage.setColorTable(m_colorTable);
                newPreview = QIcon(QPixmap::fromImage(retImage));
            }
            else if(m_data.getType() == ito::tUInt16)
            {
                double minVal = 0.0;
                double maxVal = 1.0;
                cv::minMaxIdx(tempMat, &minVal, &maxVal);

                tempMat.convertTo(tempMat, CV_8U, 256.0 / maxVal);

                QImage retImage(tempMat.ptr<ito::uint8>(0), 64,64, 64, QImage::Format_Indexed8);
                retImage.setColorTable(m_colorTable);
                newPreview = QIcon(QPixmap::fromImage(retImage));
            }
            else if(m_data.getType() == ito::tComplex64 || m_data.getType() == ito::tComplex128)
            {
                addTopLevelItem(new QTreeWidgetItem(this, QStringList(tr("no preview available")), 0));

                cv::Mat tarMat(64, 130, CV_8U);

                if (m_data.getType() == ito::tComplex64)
                {
                    convertComplexMat<ito::complex64>(tempMat, tarMat);
                }
                else
                {
                    convertComplexMat<ito::complex128>(tempMat, tarMat);
                }

                QImage retImage(tarMat.ptr<ito::uint8>(0), 130,64, 130, QImage::Format_Indexed8);
                retImage.setColorTable(m_colorTable);
                newPreview = QIcon(QPixmap::fromImage(retImage));
            }
            else
            {
                double minVal = 0.0;
                double maxVal = 1.0;
                cv::minMaxIdx(tempMat, &minVal, &maxVal);

                if(!ito::isFinite(minVal))
                {
                    minVal = 0.0;
                }

                if(!ito::isFinite(maxVal))
                {
                    minVal = 1.0;
                }

                tempMat.convertTo(tempMat, CV_8U, 255.0 / (maxVal - minVal), - 255.0 * minVal / (maxVal - minVal));
                
                QImage retImage(tempMat.ptr<ito::uint8>(0), 64,64, 64, QImage::Format_Indexed8);
                retImage.setColorTable(m_colorTable);
                newPreview = QIcon(QPixmap::fromImage(retImage));
            }

            if(setThis)
            {
                QTreeWidgetItem *preview = new QTreeWidgetItem(this, QStringList(tr("preview")), 0);
                preview->addChild(new QTreeWidgetItem(preview, QStringList(tr("")), 0));
                preview->child(0)->setIcon(1, newPreview);
            }
        }
        else
        {
            addTopLevelItem(new QTreeWidgetItem(this, QStringList(tr("no preview available")), 0));
        }
    }

    // write header
    QTreeWidgetItem *header = new QTreeWidgetItem(this, QStringList(tr("header")), 0);

    data.append("type");
    data.append(QString::number(m_data.getType()));
    header->addChild(new QTreeWidgetItem(header, data, 0));

    if(m_detailedStatus)
    {
        data[0] = "continuous";
        data[1] = m_data.getContinuous() == 0 ? "no" : "yes";
        header->addChild(new QTreeWidgetItem(header, data, 0));

        data[0] = "pwn Data";
        data[1] = m_data.getOwnData() == 0 ? "no" : "yes";
        header->addChild(new QTreeWidgetItem(header, data, 0));

        data[0] = "dimensions";
        data[1] = QString::number(dims);
        header->addChild(new QTreeWidgetItem(header, data, 0));
    }

#if 0
    for(int dim = 0; dim < dims - 3; dim++)
    {
        data[0] = (QString::number(dim));
        data[1] = (QString::number(m_data.getSize(dim)));
        header->addChild(new QTreeWidgetItem(header, data, 0));
    }

    if(dims > 2)
    {
        data[0] = 'z';
        data[1] = (QString::number(m_data.getSize(dims - 3))); 
        header->addChild(new QTreeWidgetItem(header, data, 0));
    }

    if(dims > 0)
    {
        data[0] = 'y';
        data[1] = (QString::number(m_data.getSize(dims - 2))); 
        header->addChild(new QTreeWidgetItem(header, data, 0));

        data[0] = 'x';
        data[1] = (QString::number(m_data.getSize(dims - 1))); 
        header->addChild(new QTreeWidgetItem(header, data, 0));
    }
#else

    data[0] = "pixel size";
    data[1].reserve(2 + 2 + 12 * dims);
    data[1] = "[";
    for(int dim = 0; dim < dims; dim++)
    {
        data[1].append(QString::number(m_data.getSize(dim)));
        data[1].append(", ");
    }
    data[1].append("]");

/*
    if(data[1].length() > columnWidth(1))
    {
        setColumnWidth(1, data[1].length());
    }
*/

    header->addChild(new QTreeWidgetItem(header, data, 0));
    data[0] = "physical size";
    data[1] = "[";
    double val;
    for(int dim = 0; dim < dims; dim++)
    {
        val = m_data.getPixToPhys(dim, m_data.getSize(dim), checker) - m_data.getPixToPhys(dim, 0, checker);
        data[1].append(QString::number(val));
        data[1].append(m_data.getAxisUnit(dim, checker).data());
        data[1].append(", ");
    }
    data[1].append("]");

/*
    if(data[1].length() > columnWidth(1))
    {
        setColumnWidth(1, data[1].length());
    }
*/

    header->addChild(new QTreeWidgetItem(header, data, 0));
#endif

    if(m_detailedStatus)
    {
        data[0] = "scales";
        data[1].reserve(2 + 2 + 12 * dims);
        data[1] = "[";
        for(int dim = 0; dim < dims; dim++)
        {
            val = m_data.getAxisScale(dim);
            data[1].append(QString::number(val));
            data[1].append(", ");
        }
        data[1].append("]");
        header->addChild(new QTreeWidgetItem(header, data, 0));

        data[0] = "offsets";
        data[1] = "[";
        for(int dim = 0; dim < dims; dim++)
        {
            val = m_data.getAxisOffset(dim);
            data[1].append(QString::number(val));
            data[1].append(", ");
        }
        data[1].append("]");
        header->addChild(new QTreeWidgetItem(header, data, 0));
    }

    data[0] = "value unit";
    data[1] = m_data.getValueUnit().data();
    header->addChild(new QTreeWidgetItem(header, data, 0));
    addTopLevelItem(header);
    
    // write tagSpace

    QTreeWidgetItem *tagSpace = new QTreeWidgetItem(this, QStringList(tr("Tag Space")), 0);

    int tagNumber = m_data.getTagListSize();
    std::string key;
    ito::DataObjectTagType type;

    for(int i = 0; i < tagNumber; i++)
    {
        m_data.getTagByIndex(i, key, type);

        if(key == "protocol")
            continue;

        if(type.getType() == ito::DataObjectTagType::typeDouble)
        {
            data[0] = key.data();
            data[1] = QString::number(type.getVal_ToDouble(), 'g', m_decimals);
            tagSpace->addChild(new QTreeWidgetItem(tagSpace, data, 0));
        }
        else if(type.getType() == ito::DataObjectTagType::typeString)
        {
            data[0] = key.data();
            data[1] = type.getVal_ToString().data();
            tagSpace->addChild(new QTreeWidgetItem(tagSpace, data, 0));        
        }

/*
        if(data[1].length() > columnWidth(1))
        {
            setColumnWidth(1, data[1].length());
        }
*/

    }

    addTopLevelItem(tagSpace);

    // write protocol

    QTreeWidgetItem *protocol = new QTreeWidgetItem(this, QStringList(tr("Protocol")), 0);

    
    type = m_data.getTag("protocol", checker);

    if(checker)
    {
        QString temp = type.getVal_ToString().data();
        QStringList tempList = temp.split('\n', QString::SkipEmptyParts);
        for(int i = 0; i < tempList.size(); i++)
        {
            data[0] = "";
            data[1] = tempList[i];
            protocol->addChild(new QTreeWidgetItem(protocol, data, 0)); 
/*
            if(data[1].length() > columnWidth(1))
            {
                setColumnWidth(1, data[1].length());
            }
*/
        }

    }
    addTopLevelItem(protocol);

    expandAll();

    resizeColumnToContents(0);
    resizeColumnToContents(1);
}

//----------------------------------------------------------------------------------------------------------------------------------
QSharedPointer<ito::DataObject> dObMetaDataTable::getData() const
{
    return QSharedPointer<ito::DataObject>(new ito::DataObject(m_data));
}

//----------------------------------------------------------------------------------------------------------------------------------
void dObMetaDataTable::setReadOnly(const bool value)
{
    m_readOnly = value;
}
//----------------------------------------------------------------------------------------------------------------------------------
void dObMetaDataTable::setDecimals(const int value)
{
    m_decimals = value > 0 ? value : 0;
}
//----------------------------------------------------------------------------------------------------------------------------------
void dObMetaDataTable::setPreviewSize(const int value)
{
   m_previewSize = value > 0 ? (value < 512 ? value : 512) : 0;
}
//----------------------------------------------------------------------------------------------------------------------------------
void dObMetaDataTable::setPreviewStatus(const bool value)
{
    m_preview = value;
}
//----------------------------------------------------------------------------------------------------------------------------------
void dObMetaDataTable::setDetailedStatus(const bool value)
{
    m_detailedStatus = value;
}
//----------------------------------------------------------------------------------------------------------------------------------
/*
void dObMetaDataTable::setColorMap(const QString &name)
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
QSize dObMetaDataTable::sizeHint() const
{

    int h = 25;
    int w = 15;

    return QSize(w,h);
}

