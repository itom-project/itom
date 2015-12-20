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
    out << obj.type() << obj.flags() << obj.basePoints() << obj.transform() << obj.index() << obj.name();
    return out;
}

QDataStream &operator>>(QDataStream &in, ito::Shape &obj)
{
    int type, flags;
    QPolygonF polygons;
    QTransform transform;
    int index;
    QString name;
    in >> type >> flags >> polygons >> transform >> index >> name;
    obj = Shape(type, flags, polygons, index, name, transform);
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
    int m_index; /*!< index of shape, -1: no specific index*/
    QString m_name; /*!< name (label) of shape */
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
    d->m_index = -1; //no specific index
    d->m_name = "";
}

//----------------------------------------------------------------------------------------------
Shape::Shape(int type, int flags, const QPolygonF &basePoints, int index, const QTransform &transform /*=QTransform()*/) : d(NULL)
{
    d = new ShapePrivate();
    d->m_type = (type & Shape::TypeMask) | (flags & Shape::FlagMask);
    d->m_polygon = basePoints;
    d->m_transform = transform;
    d->m_index = index;
    d->m_name = "";
}

//----------------------------------------------------------------------------------------------
Shape::Shape(int type, int flags, const QPolygonF &basePoints, int index, const QString &name, const QTransform &transform /*=QTransform()*/) : d(NULL)
{
    d = new ShapePrivate();
    d->m_type = (type & Shape::TypeMask) | (flags & Shape::FlagMask);
    d->m_polygon = basePoints;
    d->m_transform = transform;
    d->m_index = index;
    d->m_name = name;
}

//----------------------------------------------------------------------------------------------
Shape::Shape(int type, int flags, const QPolygonF &basePoints, const QString &name, const QTransform &transform /*=QTransform()*/) : d(NULL)
{
    d = new ShapePrivate();
    d->m_type = (type & Shape::TypeMask) | (flags & Shape::FlagMask);
    d->m_polygon = basePoints;
    d->m_transform = transform;
    d->m_index = -1; //no specific index
    d->m_name = name;
}

//----------------------------------------------------------------------------------------------
Shape::Shape(const Shape &other) : d(NULL)
{
    d = new ShapePrivate();
    d->m_type = other.d->m_type;
    d->m_polygon = other.d->m_polygon;
    d->m_transform = other.d->m_transform;
    d->m_index = other.d->m_index;
    d->m_name = other.d->m_name;
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
    d->m_index = other.d->m_index;
    d->m_name = other.d->m_name;

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
int Shape::index() const
{
    return d->m_index;
}

//----------------------------------------------------------------------------------------------
void Shape::setIndex(const int &index)
{
    d->m_index = index;
}

//----------------------------------------------------------------------------------------------
QString Shape::name() const
{
    return d->m_name;
}

//----------------------------------------------------------------------------------------------
void Shape::setName(const QString &name)
{
    d->m_name = name;
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
QPolygonF &Shape::rbasePoints()
{
    return d->m_polygon;
}

//----------------------------------------------------------------------------------------------
QPolygonF circle2Polygon(const QPointF &center, qreal radius, qreal tol)
{
    if (tol < 0)
    {
        tol = qBound(1.0e-6, 0.01 * radius, 0.5 * radius);
    }

    qreal sideLength = 2 * std::sqrt((2 * radius - tol) * tol);
    qreal angle = 2 * std::asin(sideLength / (2 * radius));
    int segments = std::ceil(2 * M_PI / angle); //round up to the next higher number of segments to have an error of max. tol.
    angle = (2 * M_PI) / (qreal)segments;
    QPolygonF poly;

    for (int i = 0; i < segments; ++i)
    {
        poly << center + QPointF(radius * std::cos(angle * i), radius * std::sin(angle * i));
    }

    return poly;
}

//----------------------------------------------------------------------------------------------
QPolygonF Shape::contour(bool applyTrafo /*= true*/, qreal tol /*= -1.0*/) const
{
    switch (type())
    {
    case Point:
    case Line:
    {
        if (applyTrafo)
        {
            return d->m_transform.map(d->m_polygon);
        }
        else
        {
            return d->m_polygon;
        }
        break;
    }
    case Rectangle:
    case Square:
    {
        QRectF rect(d->m_polygon[0], d->m_polygon[1]);
        if (applyTrafo)
        {
            return d->m_transform.mapToPolygon(rect.toRect());
        }
        else
        {
            return QPolygonF() << rect.topLeft() << rect.bottomLeft() << rect.bottomRight() << rect.topRight();
        }
    }
    case Polygon:
    {
        if (applyTrafo)
        {
            return d->m_transform.map(d->m_polygon);
        }
        else
        {
            return d->m_polygon;
        }
        break;
    }
    case Circle:
    {
        QPointF center = (d->m_polygon[0] + d->m_polygon[1]) / 2.0;
        qreal radius = std::abs((d->m_polygon[1] - d->m_polygon[0]).x() / 2.0);
        if (radius > 0)
        {
            if (applyTrafo)
            {
                QPolygonF poly = circle2Polygon(center, radius, tol);
                poly.translate(d->m_transform.dx(), d->m_transform.dy()); //rotation is irrelevant
                return poly;
            }
            else
            {
                return circle2Polygon(center, radius, tol);
            }
        }
        else
        {
            QPolygonF poly;
            poly << center;
            if (applyTrafo)
            {
                poly.translate(d->m_transform.dx(), d->m_transform.dy()); //rotation is irrelevant
                return poly;
            }
            else
            {
                return poly;
            }
        }
        break; 
    }
    case Ellipse:
    {
        QPolygonF poly = ramerDouglasPeucker(tol);
        if (applyTrafo)
        {
            return d->m_transform.map(poly);
        }
        else
        {
            return poly;
        }
    }
    default:
        return QPolygonF();
    }
}

//----------------------------------------------------------------------------------------------
//this struct is used, since it is faster to pass one argument by-ref to the iterative function call instead of multiple variables
struct RamerDouglasPeuckerData
{
    QList<QLineF> edges;
    double a;
    double b;
    double tol;
};

int ramerDouglasPeuckerIter(RamerDouglasPeuckerData &data, int current_index)
{
    const QLineF &seg = data.edges[current_index];

    QPointF edge_center = (seg.p1() + seg.p2()) / 2.0;
    QPointF normal_vector(seg.dy(), -seg.dx()); //towards the right side (outer side)
    qreal b2 = data.b * data.b;
    qreal a2 = data.a * data.a;
    qreal p = b2 * normal_vector.rx() * normal_vector.rx() + a2 * normal_vector.ry() * normal_vector.ry();
    qreal q = 2 * b2 * edge_center.rx() * normal_vector.rx() + a2 * edge_center.ry() * normal_vector.ry();
    qreal r = b2 * edge_center.rx() * edge_center.rx() + a2 * edge_center.ry() * edge_center.ry() - a2 * b2;

    //solve for pm^2 + qm + r = 0 -> positive root of m only!
    qreal m = (-q + std::sqrt(q*q - 4.0 * p * r)) / (2.0 * p);

    QPointF ellipse_point = edge_center + m * normal_vector;

    //correct ellipse_point due to discretization and rounding errors:
    qreal y_new = data.b * std::sqrt(1 - ellipse_point.x() * ellipse_point.x() / a2);
    qreal x_new = data.a * std::sqrt(1 - ellipse_point.y() * ellipse_point.y() / b2);

    if (std::abs(y_new - ellipse_point.ry()) < std::abs(x_new - ellipse_point.rx()))
    {
        ellipse_point.ry() = (ellipse_point.y() * y_new) >= 0 ? y_new : -y_new;
    }
    else
    {
        ellipse_point.rx() = (ellipse_point.x() * x_new) >= 0 ? x_new : -x_new;
    }

    qreal length = (ellipse_point - edge_center).manhattanLength();
    if (length > data.tol)
    {
        QLineF seg1(seg.p1(), ellipse_point);
        QLineF seg2(ellipse_point, seg.p2());
        data.edges[current_index] = seg1;
        data.edges.insert(current_index + 1, seg2); //add one segment after the current one, but the current one must be checked in the next turn

        return current_index;
    }
    else
    {
        return current_index + 1; //handle next edge, this edge is short enough, no edge has been added
    }
}

//----------------------------------------------------------------------------------------------
QPolygonF Shape::ramerDouglasPeucker(qreal tol) const
{
    //Ramer-Douglas-Peucker algorithm (see http://stackoverflow.com/questions/22694850/approximating-an-ellipse-with-a-polygon)
    QPolygonF p;
    if (type() == Ellipse || type() == Circle)
    {
        QRectF rect(d->m_polygon[0], d->m_polygon[1]);
        int N = 0;

        if (tol < 0)
        {
            tol = 0.01 * std::min(rect.width(), rect.height());
        }
        
        if (!rect.isEmpty())
        {
            //move the shape such that the center is the origin, afterwards translate all points back to the real center
            QPointF p_left(-rect.width() / 2, 0.0);
            QPointF p_right(rect.width() / 2, 0.0);
            QPointF p_top(0.0, rect.height() / 2);
            QPointF p_bottom(0.0, -rect.height() / 2);
            QPointF p_center((rect.left() + rect.right()) / 2, (rect.top() + rect.bottom()) / 2);
            
            int next_index = 0;
            RamerDouglasPeuckerData data;
            data.edges << QLineF(p_left, p_bottom) << QLineF(p_bottom, p_right) << QLineF(p_right, p_top) << QLineF(p_top, p_left);
            data.a = rect.width() / 2;
            data.b = rect.height() / 2;
            data.tol = tol;
            while (next_index < data.edges.size())
            {
                next_index = ramerDouglasPeuckerIter(data, next_index);
            }

            p.reserve(data.edges.size());
            foreach(const QLineF &line, data.edges)
            {
                p.push_back(line.p1());
            }

            p.translate(p_center);
        }
        else
        {
            p.push_back(rect.topLeft());
        }
    }

    return p;
}

//----------------------------------------------------------------------------------------------
QRegion Shape::region() const
{
    QRegion region;

    switch (type())
    {
    case Shape::Point:
    case Shape::Line:
        break;
    case Shape::Rectangle:
    case Shape::Square:
    {
        region = QRegion(QRectF(d->m_polygon[0], d->m_polygon[1]).toRect(), QRegion::Rectangle);
        break;
    }
    case Shape::Polygon:
    {
        region = QRegion(d->m_polygon.toPolygon());
        break;
    }
    case Shape::Ellipse:
    case Shape::Circle:
    {
        region = QRegion(QRectF(d->m_polygon[0], d->m_polygon[1]).toRect(), QRegion::Ellipse);
        break;
    }
    default:
        break;
    }

    if (d->m_transform.isIdentity())
    {
        return region;
    }
    else
    {
        return d->m_transform.map(region); //time consuming
    }

    return region;
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
/*static*/ Shape Shape::fromRect(const QRectF &rect, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Rectangle;
    s.d->m_polygon << rect.topLeft() << rect.bottomRight();
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromRect(qreal x1, qreal y1, qreal x2, qreal y2, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Rectangle;
    s.d->m_polygon << QPointF(x1,y1) << QPointF(x2,y2);
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromEllipse(const QRectF &rect, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Ellipse;
    s.d->m_polygon << rect.topLeft() << rect.bottomRight();
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromEllipse(qreal x1, qreal y1, qreal x2, qreal y2, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Ellipse;
    s.d->m_polygon << QPointF(x1,y1) << QPointF(x2,y2);
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromLine(const QPointF &p1, const QPointF &p2, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Line;
    s.d->m_polygon << p1 << p2;
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromLine(qreal x1, qreal y1, qreal x2, qreal y2, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Line;
    s.d->m_polygon << QPointF(x1,y1) << QPointF(x2,y2);
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromPoint(const QPointF &point, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Point;
    s.d->m_polygon << point;
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromPoint(qreal x, qreal y, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Line;
    s.d->m_polygon << QPointF(x,y);
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromPolygon(const QPolygonF &polygon, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Line;
    s.d->m_polygon = polygon;
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromSquare(const QPointF &center, qreal sideLength, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    QPointF dist(sideLength / 2, sideLength / 2);
    s.d->m_type = Square;
    s.d->m_polygon << (center - dist) << (center + dist);
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromCircle(const QPointF &center, qreal radius, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    QPointF dist(radius, radius);
    s.d->m_type = Circle;
    s.d->m_polygon << (center - dist) << (center + dist);
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}




} //end namespace ito