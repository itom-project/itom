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

class MARKERLEGEND_EXPORT MarkerLegend : public QTreeWidget
{
    Q_OBJECT
        
    public:        
        MarkerLegend(QWidget* parent = NULL);

    private:
        QHash< int, relation> m_relationHash;

    public slots:
        void updatePicker(int index, QVector<float>  position);
        void updatePickers(QVector<int> indices, QVector< QVector<float> > positions);
        void updateGeometry(int index, QPair< int, QVector<float> > element);
        void updateGeometries(QVector<int> index, QVector< QPair <int,  QVector<float> > > elements);
        void updateLinePlot(int type, QVector<QPointF > positions);

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