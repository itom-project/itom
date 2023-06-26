/* ********************************************************************
itom software
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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
#include "../shape/shapeCommon.h"

#include <qpolygon.h>
#include <qtransform.h>
#include <qregion.h>
#include <qcolor.h>
#include <qdatastream.h>

#if !defined(Q_MOC_RUN) || defined(ITOMSHAPE_MOC) //only moc this file in itomShapeLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{
    class ShapePrivate;
    class Shape;

    QDataStream ITOMSHAPE_EXPORT &operator<<(QDataStream &out, const ito::Shape &shape);

	QDataStream ITOMSHAPE_EXPORT &operator>>(QDataStream &in, ito::Shape &shape);


	class ITOMSHAPE_EXPORT Shape
    {

    public:

        /** \enum ShapeType
        */
        enum ShapeType
        {
            Invalid = 0,
            MultiPointPick = 0x00000001,    /**! Multi point pick*/
            Point = 0x00000002,             /**! Element is point in order to pick points*/
            Line = 0x00000004,              /**! Element is line in order to pick lines*/
            Rectangle = 0x00000008,         /**! Element is rectangle in order to pick rectangles*/
            Square = 0x00000010,            /**! Element is square in order to pick squares*/
            Ellipse = 0x00000020,           /**! Element is ellipse in order to pick ellipses*/
            Circle = 0x00000040,            /**! Element is circle in order to pick circles*/
            Polygon = 0x00000080,           /**! Element is polygon in order to pick polygon*/

            //REMARK: If this enumeration is changed, please identically change ItomQwtPlotEnums::ShapeType in the designer plugins!
        };

        enum ShapeFlag
        {
            MoveLock = 0x00010000,   /**! Element can not be moved */
            RotateLock = 0x00020000, /**! Element can not be rotated */
            ResizeLock = 0x00040000, /**! Element can not be resized */
        };

        enum ShapeMask
        {
            TypeMask = 0x0000FFFF,  /**! Mask for the type space */
            FlagMask = 0xFFFF0000   /**! Mask for the flag space */
        };

        explicit Shape();
        explicit Shape(unsigned int type, unsigned int flags, const QPolygonF &basePoints, const QTransform &transform = QTransform());
        explicit Shape(unsigned int type, unsigned int flags, const QPolygonF &basePoints, int index, const QTransform &transform = QTransform());
        explicit Shape(unsigned int type, unsigned int flags, const QPolygonF &basePoints, int index, const QString &name, const QTransform &transform = QTransform());
        explicit Shape(unsigned int type, unsigned int flags, const QPolygonF &basePoints, const QString &name, const QTransform &transform = QTransform());
        Shape(const Shape &other);
        virtual ~Shape();

        Shape& operator =(const Shape &other);

        /* Returns a normalized shape; i.e., a rectangle, square, circle or ellipse that has a non-negative width and height. */
        Shape normalized() const;

        bool isValid() const;

        unsigned int flags() const;
        void setFlags(const unsigned int &flags);

        bool unclosed() const; //!< return true if this shape (e.g. polygon) is currently being created and not closed, yet.
        void setUnclosed(bool unclosed);

        int index() const;
        void setIndex(const int &index);

        QString name() const;
        void setName(const QString &name);

        unsigned int type() const;
        void setType(const unsigned int &type);

		QColor color() const;
		void setColor(const QColor &color);

        QTransform transform() const;
        QTransform &rtransform() const;
        void setTransform(const QTransform &trafo);

        double rotationAngleDeg() const; /*!< return the current rotation angle (in degree, counterclockwise) of this shape (obtained by current transformation matrix) */
        double rotationAngleRad() const; /*!< return the current rotation angle (in radians, counterclockwise) of this shape (obtained by current transformation matrix) */

        void setRotationAngleDeg(double degree); /*!< set the current rotation angle (in degree, counterclockwise) of this shape without changing the translation values */
        void setRotationAngleRad(double radians); /*!< set the current rotation angle (in radians, counterclockwise) of this shape without changing the translation values */

        void rotateByCenterDeg(double degree); /*!< rotate this shape around its current center points by the given angle (in degree). This rotation changes the current transformation matrix, not the base points of the shape. */
        void rotateByCenterRad(double radians); /*!< rotate this shape around its current center points by the given angle (in radians). This rotation changes the current transformation matrix, not the base points of the shape. */

        void translate(const QPointF &delta); /*!< moves this shape by the given delta in the global coordinate system, which is the system where all base points are mapped using the current transformation matrix. This translation operation changes the current transformation matrix, not the base points of the shape. */

        ito::float64 userData1() const;
        void setUserData1(const ito::float64 &userData1);

        ito::float64 userData2() const;
        void setUserData2(const ito::float64 &userData2);

        QPolygonF basePoints() const; /*!< base points are various points that help to define the geometry in a precise description. */
        QPolygonF &rbasePoints();
        const QPolygonF &rbasePoints() const;
        QPolygonF contour(bool applyTrafo = true, qreal tol = -1.0) const; /*!< returns the enclosing contour as polygon. If the shape is elliptic, an approximation is applied, where tol is the maximum distance between real contour and a line segment of the polygon (if -1.0, the tolerance is defined to be 1% of the smaller diameter of the ellise*/
        QRegion   region() const;


        void point1MoveTo(const QPointF &newPoint1);

		QPointF centerPoint() const; /*!< center point of this shape (after applying the transformation assigned to this shape)*/
        QPointF baseCenterPoint() const; /*!< center point of this shape */

        double area() const; /*!< return the area of this shape, or zero if the shape is a point, multi-point or line */
        double circumference() const;
		double distance(const Shape &otherShape) const;
		double centerDistance(const Shape &otherShape) const;

		double radius() const;
		double radiusX() const;
		double radiusY() const;

        bool contains(const QPointF &point) const; /*!< returns true if shape contains the given point, or false if this is not the case. In case of shapes with an area of 0, this method always returns false.*/
        QVector<bool> contains(const QPolygonF &points) const; /*!< repeatedly calls contains(point) for each point in points and returns a vector of boolean values to tell for each point if it is contained in the shape or not. */

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

		static QString type2QString(const int type);

		//static ito::DataObject maskFromMultipleShapes(const ito::DataObject &dataObject, const QVector<ito::Shape> &shapes, bool inverse = false);
		//ito::DataObject mask(const ito::DataObject &dataObject, bool inverse = false) const;

    protected:

        ShapePrivate *d;

        QPolygonF ramerDouglasPeucker(qreal tol) const;
        //void maskHelper(const ito::DataObject &dataObject, ito::DataObject &mask, bool inverse = false) const;

		static double distanceLine2Point2D(const Shape &line, const QPointF &point);
		static double distanceLine2Line2D(const Shape &line1, const Shape &line2);
		static double distancePoint2Point2D(const QPointF &point1, const QPointF &point2);
    };
}


#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif //SHAPE_H
