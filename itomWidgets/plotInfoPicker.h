/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2022, Institut fuer Technische Optik (ITO),
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

#pragma once

#ifdef __APPLE__
extern "C++"
{
#endif

#include "../common/commonGlobal.h"
#include "../common/shape.h"
#include "../common/typeDefs.h"

#include "commonWidgets.h"

#include <QtGui/qpainterpath.h>
#include <QtGui/qpixmap.h>
#include <QtWidgets/qtreewidget.h>
#include <qpoint.h>
#include <qvector3d.h>
#include <qvector4d.h>
#include <qdatetime.h>

    class ITOMWIDGETS_EXPORT PlotInfoPicker : public QTreeWidget
    {
        Q_OBJECT

    public:
        PlotInfoPicker(QWidget* parent = NULL);

    private:
        QHash<int, QPair<int, int>> m_relationHash;

    public slots:
        void updatePicker(const int index, const QPointF position);
        void updatePickers(const QVector<int> indices, const QVector<QPointF> positions);
        void updatePickers(
            const QVector<int> indices,
            const QVector<QDateTime>& xpositions,
            const QVector<qreal>& ypositions);
        void updatePicker(const int index, const QVector3D position);
        void updatePickers(const QVector<int> indices, const QVector<QVector3D> positions);

        void updateChildPlot(const int index, int type, const QVector4D positionAndDirection);
        void updateChildPlots(
            const QVector<int> indices,
            const QVector<int> type,
            const QVector<QVector4D> positionAndDirection);
        void removeChildPlot(int index);
        void removeChildPlots();

        void removePicker(int index);
        void removePickers();

        QPainterPath renderToPainterPath(const int xsize, const int ysize, const int fontSize);

    private slots:
    };

#ifdef __APPLE__
}
#endif
