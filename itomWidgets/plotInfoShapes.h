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

#ifndef PLOTINFOSHAPES_H
#define PLOTINFOSHAPES_H

#ifdef __APPLE__
extern "C++" {
#endif

#include "../common/commonGlobal.h"
#include "../common/typeDefs.h"
#include "../common/shape.h"

#include "commonWidgets.h"

#include <QtWidgets/qtreewidget.h>
#include <QtGui/qpainterpath.h>

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

class ITOMWIDGETS_EXPORT PlotInfoShapes : public QTreeWidget
{
    Q_OBJECT

    public:
		PlotInfoShapes(QWidget* parent = NULL);

    private:
		void setItem2Point(QTreeWidgetItem* curItem, const ito::Shape &element);
		void setItem2Line(QTreeWidgetItem* curItem, const ito::Shape &element);
		void setItem2Circle(QTreeWidgetItem* curItem, const ito::Shape &element);
		void setItem2Ellipse(QTreeWidgetItem* curItem, const ito::Shape &element);
		void setItem2Square(QTreeWidgetItem* curItem, const ito::Shape &element);
		void setItem2Rect(QTreeWidgetItem* curItem, const ito::Shape &element);
		void setItem2Poly(QTreeWidgetItem* curItem, const ito::Shape &element);

        void adjustNumberOfChildItems(QTreeWidgetItem* curItem, int count);

        QHash< int, relation> m_relationHash;


        bool m_onlyTwoDims;

    public slots:
        void updateShape(const ito::Shape element);
        void updateShapes(const QVector< ito::Shape > elements);

        void removeShape(int index);
        void removeShapes();

		void addRelation(const int index1, const int index2, const int relationType);
		void removeRelation(const int index1, const int index2);
		void removeRelations(const int index1, const int index2);
		void removeRelations(const int index);
		void removeRelations();

		QPainterPath renderToPainterPath(const int xsize, const int ysize, const int fontSize);

    private slots:
};

#ifdef __APPLE__
}
#endif

#endif // SHAPELEGENDWIDGET_H
