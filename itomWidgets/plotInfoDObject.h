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

#ifndef OBJLEGENDWIDGET_H
#define OBJLEGENDWIDGET_H

#ifdef __APPLE__
extern "C++" {
#endif

#include "../common/commonGlobal.h"
#include "../common/typeDefs.h"

#include "commonWidgets.h"

#include <QtWidgets/qplaintextedit.h>
#include <QtGui/qpixmap.h>
#include <QtGui/qpainterpath.h>

class ITOMWIDGETS_EXPORT PlotInfoDObject : public QPlainTextEdit
{
    Q_OBJECT

    public:
		PlotInfoDObject(QWidget* parent = NULL);
		bool useDetailInfo() const { return m_addDetailInfo; }

    private:
		QString m_infoHeader;
		QString m_infoDetail;
		bool m_valid;
		bool m_addDetailInfo;

    public slots:

		void updateInfoHeader(const QString newString);
		void updateInfoHeader(const QString typeString, const int dType, const int dims, const int sizes[]);
		void updateInfoDetail(const QString newString);
		void updateInfoDetail(const double minVal, const double maxVal, const double meanVal, const double devVal);
		void clearObjectInfo();
		void setUseDetailInfo(const bool state);
		QPainterPath renderToPainterPath(const int xsize, const int ysize, const int fontSize);

    private slots:

};

#ifdef __APPLE__
}
#endif

#endif // OBJLEGENDWIDGET_H
