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

#include "../shape.h"
#define _USE_MATH_DEFINES
#include "math.h"

namespace ito {

QDataStream &operator<<(QDataStream &out, const ito::Shape &obj)
{
    out << obj.type() << obj.flags() << obj.basePoints() << obj.transform();
    return out;
}

QDataStream &operator>>(QDataStream &in, ito::Shape &obj)
{
    int type, flags;
    QPolygonF polygons;
    QTransform transform;
    in >> type >> flags >> polygons >> transform;
    obj = Shape(type, flags, polygons, transform);
    return in;
}

class ShapePrivate
{
public:
    ShapePrivate() :
      m_type(Shape::Invalid)
    {
    }

    int m_type;

    /*!< 
    * point: one point
    * line: start pt, end pt
    * rectangle, square: top left, bottom right points
    * circle, ellipse: top left, bottom right points of bounding box
    * polygon: polygons
    */
    QPolygonF m_polygon; 
    QTransform m_transform;
};

//----------------------------------------------------------------------------------------------
Shape::Shape() : d(NULL)
{
    d = new ShapePrivate();
}

//----------------------------------------------------------------------------------------------
Shape::Shape(int type, int flags, const QPolygonF &basePoints, const QTransform &transform /*=QTransform()*/) : d(NULL)
{
    d = new ShapePrivate();
    d->m_type = (type & Shape::TypeMask) | (flags & Shape::FlagMask);
    d->m_polygon = basePoints;
    d->m_transform = transform;
}

//----------------------------------------------------------------------------------------------
Shape::Shape(const Shape &other) : d(NULL)
{
    d = new ShapePrivate();
    d->m_type = other.d->m_type;
    d->m_polygon = other.d->m_polygon;
    d->m_transform = other.d->m_transform;
}

//----------------------------------------------------------------------------------------------
Shape::~Shape()
{
    if (d)
    {
        delete d;
        d = NULL;
    }
}

//----------------------------------------------------------------------------------------------
Shape& Shape::operator =(const Shape &other)
{
    if (d)
    {
        delete d;
    }

    d = new ShapePrivate();
    d->m_type = other.d->m_type;
    d->m_polygon = other.d->m_polygon;
    d->m_transform = other.d->m_transform;

    return *this;
}

//----------------------------------------------------------------------------------------------
bool Shape::isValid() const
{
    return (d->m_type & Shape::Invalid) > 0;
}

//----------------------------------------------------------------------------------------------
int Shape::flags() const
{
    return d->m_type & Shape::FlagMask;
}

//----------------------------------------------------------------------------------------------
void Shape::setFlags(const int &flags)
{
    d->m_type = (d->m_type & Shape::TypeMask) | (flags & Shape::FlagMask);
}

//----------------------------------------------------------------------------------------------
int Shape::type() const
{
    return d->m_type & Shape::TypeMask;
}

//----------------------------------------------------------------------------------------------
QTransform Shape::transform() const
{
    return d->m_transform;
}

//----------------------------------------------------------------------------------------------
QTransform &Shape::rtransform() const
{
    return d->m_transform;
}

//----------------------------------------------------------------------------------------------
void Shape::setTransform(const QTransform &trafo)
{
    d->m_transform = trafo;
}

//----------------------------------------------------------------------------------------------
QPolygonF Shape::basePoints() const
{
    return d->m_polygon;
}

//----------------------------------------------------------------------------------------------
QPolygonF Shape::contour(bool applyTrafo /*= false*/, qreal tol /*= -1.0*/) const
{
    QPolygonF poly;
    if (applyTrafo)
    {
        poly = d->m_transform.map(d->m_polygon);
    }
    else
    {
        poly = d->m_polygon;
    }

    switch (type())
    {
    case Point:
    case Line:
    case Rectangle:
    case Square:
    case Polygon:
        return poly;
    case Ellipse:
    case Circle:
        //todo: Ramer-Douglas-Peucker algorithm (see http://stackoverflow.com/questions/22694850/approximating-an-ellipse-with-a-polygon)
        return QPolygonF();
    default:
        return QPolygonF();
    }
}

//----------------------------------------------------------------------------------------------
QRegion Shape::region() const
{
    return QRegion(contour(true).toPolygon());
}

//----------------------------------------------------------------------------------------------
double Shape::area() const
{
    switch (type())
    {
    case Point:
    case Line:
        return 0;
    case Rectangle:
    case Square:
        {
            QPointF size = d->m_polygon[1] - d->m_polygon[0];
            return (size.rx() * size.ry());
        }
    case Polygon:
        {
            //from http://www.mathopenref.com/coordpolygonarea.html
            qreal val = 0.0;
            for (int i = 0; i < d->m_polygon.size() - 1; ++i)
            {
                val += (d->m_polygon[i].rx()*d->m_polygon[i+1].ry() - d->m_polygon[i].ry()*d->m_polygon[i+1].rx());
            }

            if (d->m_polygon.size() > 0)
            {
                val += (d->m_polygon.last().rx()*d->m_polygon[0].ry() - d->m_polygon[0].ry()*d->m_polygon.last().rx());
            }

            return std::abs(val) / 2.0;
        }
    case Ellipse:
    case Circle:
        {
            QPointF size = d->m_polygon[1] - d->m_polygon[0];
            return (M_PI * size.rx() * size.ry() / 4);
        }
    default:
        return 0.0;
    }
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromRect(const QRectF &rect, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Rectangle;
    s.d->m_polygon << rect.topLeft() << rect.bottomRight();
    s.d->m_transform = trafo;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromRect(qreal x1, qreal y1, qreal x2, qreal y2, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Rectangle;
    s.d->m_polygon << QPointF(x1,y1) << QPointF(x2,y2);
    s.d->m_transform = trafo;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromEllipse(const QRectF &rect, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Ellipse;
    s.d->m_polygon << rect.topLeft() << rect.bottomRight();
    s.d->m_transform = trafo;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromEllipse(qreal x1, qreal y1, qreal x2, qreal y2, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Ellipse;
    s.d->m_polygon << QPointF(x1,y1) << QPointF(x2,y2);
    s.d->m_transform = trafo;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromLine(const QPointF &p1, const QPointF &p2, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Line;
    s.d->m_polygon << p1 << p2;
    s.d->m_transform = trafo;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromLine(qreal x1, qreal y1, qreal x2, qreal y2, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Line;
    s.d->m_polygon << QPointF(x1,y1) << QPointF(x2,y2);
    s.d->m_transform = trafo;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromPoint(const QPointF &point, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Point;
    s.d->m_polygon << point;
    s.d->m_transform = trafo;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromPoint(qreal x, qreal y, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Line;
    s.d->m_polygon << QPointF(x,y);
    s.d->m_transform = trafo;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromPolygon(const QPolygonF &polygon, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Line;
    s.d->m_polygon = polygon;
    s.d->m_transform = trafo;
    return s;
}




} //end namespace ito