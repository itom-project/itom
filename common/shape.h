/* ********************************************************************
itom software
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2016, Institut fuer Technische Optik (ITO),
Universitaet Stuttgart, Germany

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

#ifndef SHAPE_H
#define SHAPE_H

#include "typeDefs.h"
#include "commonGlobal.h"

#include <qpolygon.h>
#include <qtransform.h>
#include <qregion.h>
#include <qdatastream.h>

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib


namespace ito
{
    class ShapePrivate;
    class Shape;

    QDataStream ITOMCOMMONQT_EXPORT &operator<<(QDataStream &out, const ito::Shape &shape);

    QDataStream ITOMCOMMONQT_EXPORT &operator>>(QDataStream &in, ito::Shape &shape);


    class ITOMCOMMONQT_EXPORT Shape
    {
    public:

        /** \enum ShapeType
        */
        enum ShapeType
        {
            Invalid = 0,
            MultiPointPick = 6,            /**! Multi point pick*/
            Point = 101,          /**! Element is tPoint or order to pick points*/
            Line = 102,          /**! Element is tLine or order to pick lines*/
            Rectangle = 103,          /**! Element is tRectangle or order to pick rectangles*/
            Square = 104,          /**! Element is tSquare or order to pick squares*/
            Ellipse = 105,          /**! Element is tEllipse or order to pick ellipses*/
            Circle = 106,          /**! Element is tCircle or order to pick circles*/
            Polygon = 110,          /**! Element is tPolygon or order to pick polygon*/
            MoveLock = 0x00010000,   /**! Element can not be moved */
            RotateLock = 0x00020000,   /**! Element can not be rotated */
            ResizeLock = 0x00040000,   /**! Element can not be resized */
            TypeMask = 0x0000FFFF,   /**! Mask for the type space */
            FlagMask = 0xFFFF0000    /**! Mask for the flag space */
        };

        explicit Shape();
        explicit Shape(int type, int flags, const QPolygonF &basePoints, const QTransform &transform = QTransform());
        explicit Shape(int type, int flags, const QPolygonF &basePoints, int index, const QTransform &transform = QTransform());
        explicit Shape(int type, int flags, const QPolygonF &basePoints, int index, const QString &name, const QTransform &transform = QTransform());
        explicit Shape(int type, int flags, const QPolygonF &basePoints, const QString &name, const QTransform &transform = QTransform());
        Shape(const Shape &other);
        virtual ~Shape();

        Shape& operator =(const Shape &other);

        bool isValid() const;

        int flags() const;
        void setFlags(const int &flags);

        int index() const;
        void setIndex(const int &index);

        QString name() const;
        void setName(const QString &name);

        int type() const;

        QTransform transform() const;
        QTransform &rtransform() const;
        void setTransform(const QTransform &trafo);

        QPolygonF basePoints() const; /*!< base points are various points that help to define the geometry in a precise description. */
        QPolygonF &rbasePoints();
        const QPolygonF &rbasePoints() const;
        QPolygonF contour(bool applyTrafo = true, qreal tol = -1.0) const; /*!< returns the enclosing contour as polygon. If the shape is elliptic, an approximation is applied, where tol is the maximum distance between real contour and a line segment of the polygon (if -1.0, the tolerance is defined to be 1% of the smaller diameter of the ellise*/
        QRegion   region() const;

        void point1MoveTo(const QPointF &newPoint1);

        double area() const;

        static Shape fromRectangle(const QRectF &rect, int index = -1, QString name = "", const QTransform &trafo = QTransform());
        static Shape fromRectangle(qreal x1, qreal y1, qreal x2, qreal y2, int index = -1, QString name = "", const QTransform &trafo = QTransform());
        static Shape fromSquare(const QPointF &center, qreal sideLength, int index = -1, QString name = "", const QTransform &trafo = QTransform());
        static Shape fromEllipse(const QRectF &rect, int index = -1, QString name = "", const QTransform &trafo = QTransform());
        static Shape fromEllipse(qreal x1, qreal y1, qreal x2, qreal y2, int index = -1, QString name = "", const QTransform &trafo = QTransform());
        static Shape fromCircle(const QPointF &center, qreal radius, int index = -1, QString name = "", const QTransform &trafo = QTransform());
        static Shape fromLine(const QPointF &p1, const QPointF &p2, int index = -1, QString name = "", const QTransform &trafo = QTransform());
        static Shape fromLine(qreal x1, qreal y1, qreal x2, qreal y2, int index = -1, QString name = "", const QTransform &trafo = QTransform());
        static Shape fromPoint(const QPointF &point, int index = -1, QString name = "", const QTransform &trafo = QTransform());
        static Shape fromPoint(qreal x, qreal y, int index = -1, QString name = "", const QTransform &trafo = QTransform());
        static Shape fromPolygon(const QPolygonF &polygon, int index = -1, QString name = "", const QTransform &trafo = QTransform());
        static Shape fromMultipoint(const QPolygonF &polygon, int index = -1, QString name = "", const QTransform &trafo = QTransform());

    private:
        ShapePrivate *d;

        QPolygonF ramerDouglasPeucker(qreal tol) const;

    };


}

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif //SHAPE_H