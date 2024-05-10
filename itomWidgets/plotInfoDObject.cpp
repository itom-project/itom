/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    University of Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "plotInfoDObject.h"

#include <QtGui/qpainter.h>

//---------------------------------------------------------------------------------------------------------
PlotInfoDObject::PlotInfoDObject(QWidget* parent /*= NULL*/) : QPlainTextEdit(parent)
{
	clearObjectInfo();
}
//---------------------------------------------------------------------------------------------------------
void PlotInfoDObject::updateInfoHeader(const QString newString)
{
	m_infoHeader = newString;
	QString baseText = m_infoHeader;
	baseText.append(m_infoDetail);
	setPlainText(baseText);
}
//---------------------------------------------------------------------------------------------------------
void PlotInfoDObject::updateInfoHeader(const QString typeString, const int dType, const int dims, const int sizes[])
{
	m_valid = true;
	QString baseText = typeString;
	switch (dType)
	{
	case ito::tUInt8:
		baseText.append("\n\nType: uint8\n");
		break;
	case ito::tInt8:
		baseText.append("\n\nType: int8\n");
		break;
	case ito::tUInt16:
		baseText.append("\n\nType: uint16\n");
		break;
	case ito::tInt16:
		baseText.append("\n\nType: int16\n");
		break;
	case ito::tUInt32:
		baseText.append("\n\nType: uint32\n");
		break;
	case ito::tInt32:
		baseText.append("\n\nType: int32\n");
		break;
	case ito::tFloat32:
		baseText.append("\n\nType: float32\n");
		break;
	case ito::tFloat64:
		baseText.append("\n\nType: float64\n");
		break;
	case ito::tComplex64:
		baseText.append("\n\nType: complex64\n");
		break;
	case ito::tComplex128:
		baseText.append("\n\nType: complex128\n");
		break;
    case ito::tRGBA32:
        baseText.append("\n\nType: rgba32\n");
        break;
	default:
		baseText.append("\n\nUndefined type!");
		m_valid = false;
		break;
	}
	if (dims < 2)
	{
		baseText.append("\nempty object container");
	}
	else if (m_valid == true)
	{
		baseText.append(QString("Dimensions: %1\n").arg(QString::number(dims)));
		baseText.append(QString("Plainsize: %1px x %2px\n\n").arg(QString::number(sizes[dims - 1]), QString::number(sizes[dims - 2])));
	}

	if (m_valid == false)
	{
		m_addDetailInfo = false;
		updateInfoDetail("");
	}
	updateInfoHeader(baseText);
}
//---------------------------------------------------------------------------------------------------------
void PlotInfoDObject::updateInfoDetail(const QString newString)
{
	m_infoDetail = newString;
	QString baseText = m_infoHeader;
	baseText.append(m_infoDetail);
	setPlainText(baseText);
}
//---------------------------------------------------------------------------------------------------------
void PlotInfoDObject::updateInfoDetail(const double minVal, const double maxVal, const double meanVal, const double devVal)
{
	QString baseText = "";
	if (m_valid)
	{
		baseText.append(QString("Maximum: %1\n").arg(QString::number(maxVal)));
		baseText.append(QString("Minimum: %1\n").arg(QString::number(minVal)));
		baseText.append(QString("Mean Value: %1\n").arg(QString::number(meanVal)));
		baseText.append(QString("Dev Value: %1\n").arg(QString::number(devVal)));
	}
	updateInfoDetail(baseText);
}
//---------------------------------------------------------------------------------------------------------
void PlotInfoDObject::clearObjectInfo()
{
	m_valid = false;
	m_infoHeader = "No object info available\n";
	m_infoDetail = "";
	setPlainText(m_infoHeader);
	m_addDetailInfo = false;
}
//---------------------------------------------------------------------------------------------------------
void PlotInfoDObject::setUseDetailInfo(const bool state)
{
	m_addDetailInfo = state;
	if (!state)
	{
		updateInfoDetail("");
	}
	return;
}
//---------------------------------------------------------------------------------------------------------
QPainterPath PlotInfoDObject::renderToPainterPath(const int xsize, const int ysize, const int fontSize)
{
	QPainterPath destinationPath(QPoint(0, 0));

	destinationPath.addText(QPoint(0, 0), font(), toPlainText());

	return destinationPath;
}
//---------------------------------------------------------------------------------------------------------
