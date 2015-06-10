/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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

#ifndef MARKERLEGENDWIDGET_H
#define MARKERLEGENDWIDGET_H

#ifdef __APPLE__
extern "C++" {
#endif

#include "defines.h"
#include "../common/typeDefs.h"

#include <qvector3d.h>
#include <qvector4d.h>
#if QT_VERSION < 0x050000
#include <qtreewidget.h>
#include <qhash.h>
#else
#include <QtWidgets/qtreewidget.h>
//
#endif

class relation
{
    enum tRelationType
    {
        tLength,
        tRadius,
        tDiameter,
        tDistance,
        tHeightDistance,
        tInPlaneDistance
    };
    int firstIndex;
    int secondIndex;
    tRelationType type;
};

class MARKERLEGEND_EXPORT MarkerLegendWidget : public QTreeWidget
{
    Q_OBJECT
        
    public:        
        MarkerLegendWidget(QWidget* parent = NULL);

    private:
        QHash< int, relation> m_relationHash;

        bool m_onlyTwoDims;

    public slots:
        void updatePicker(const int index, const QVector3D position);
        void updatePickers(const QVector<int> indices, const QVector< QVector3D> positions);
        void updateGeometry(const int index, const ito::GeometricPrimitive element);
        void updateGeometries(const QVector< ito::GeometricPrimitive > elements);
        void updateLinePlot(const int type, const QVector4D positionAndDirection);

        void removePicker(int index);
        void removePickers();

        void removeGeometry(int index);
        void removeGeometries();

        void removeLinePlot();

    private slots:
};

#ifdef __APPLE__
}
#endif

#endif // MARKERLEGENDWIDGET_H