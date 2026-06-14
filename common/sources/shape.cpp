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

#include "../shape.h"
#define _USE_MATH_DEFINES
#include "math.h"
#include <qrect.h>
#include <qcolor.h>

#include "numeric.h"

namespace ito {

    QDataStream &operator<<(QDataStream &out, const ito::Shape &obj)
    {
        out << obj.type() << obj.flags() << obj.basePoints() << obj.transform() << obj.index() << obj.name() << obj.color();
        return out;
    }

    QDataStream &operator>>(QDataStream &in, ito::Shape &obj)
    {
        unsigned int type, flags;
        QPolygonF polygons;
        QTransform transform;
		QColor color;
        int index;
        QString name;
        in >> type >> flags >> polygons >> transform >> index >> name >> color;
        obj = Shape(type, flags, polygons, index, name, transform);
		obj.setColor(color);
        return in;
    }

class ShapePrivate
{
public:
    ShapePrivate() :
      m_type(Shape::Invalid),
      m_unclosed(false)
    {
    }

    unsigned int m_type;

    /*!<
    * multipoint: like polygons, multiple points
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
    ito::float64 m_userData[2]; /*!< two user defined value for further meta information */
    bool m_unclosed; /*!< true if this shape is currently created and hence unclosed, e.g. an open polygon which is closed after the final corner point (default: true) */
	QColor m_color; /*!< color of shape, if the color is invalid (default), the standard shape color of plots is used for visualization */
};

//----------------------------------------------------------------------------------------------
Shape::Shape() : d(NULL)
{
    d = new ShapePrivate();
    d->m_type = Shape::Invalid;
    d->m_index = -1;
    d->m_name = "";
    memset(d->m_userData, 0, sizeof(d->m_userData));
}

//----------------------------------------------------------------------------------------------
Shape::Shape(unsigned int type, unsigned int flags, const QPolygonF &basePoints, const QTransform &transform /*=QTransform()*/) : d(NULL)
{
    d = new ShapePrivate();
    d->m_type = (type & Shape::TypeMask) | (flags & Shape::FlagMask);
    d->m_polygon = basePoints;
    d->m_transform = transform;
    d->m_index = -1; //no specific index
    d->m_name = "";
    memset(d->m_userData, 0, sizeof(d->m_userData));
    d->m_unclosed = true;
}

//----------------------------------------------------------------------------------------------
Shape::Shape(unsigned int type, unsigned int flags, const QPolygonF &basePoints, int index, const QTransform &transform /*=QTransform()*/) : d(NULL)
{
    d = new ShapePrivate();
    d->m_type = (type & Shape::TypeMask) | (flags & Shape::FlagMask);
    d->m_polygon = basePoints;
    d->m_transform = transform;
    d->m_index = index;
    d->m_name = "";
    memset(d->m_userData, 0, sizeof(d->m_userData));
    d->m_unclosed = true;
}

//----------------------------------------------------------------------------------------------
Shape::Shape(unsigned int type, unsigned int flags, const QPolygonF &basePoints, int index, const QString &name, const QTransform &transform /*=QTransform()*/) : d(NULL)
{
    d = new ShapePrivate();
    d->m_type = (type & Shape::TypeMask) | (flags & Shape::FlagMask);
    d->m_polygon = basePoints;
    d->m_transform = transform;
    d->m_index = index;
    d->m_name = name;
    memset(d->m_userData, 0, sizeof(d->m_userData));
    d->m_unclosed = true;
}

//----------------------------------------------------------------------------------------------
Shape::Shape(unsigned int type, unsigned int flags, const QPolygonF &basePoints, const QString &name, const QTransform &transform /*=QTransform()*/) : d(NULL)
{
    d = new ShapePrivate();
    d->m_type = (type & Shape::TypeMask) | (flags & Shape::FlagMask);
    d->m_polygon = basePoints;
    d->m_transform = transform;
    d->m_index = -1; //no specific index
    d->m_name = name;
    memset(d->m_userData, 0, sizeof(d->m_userData));
    d->m_unclosed = true;
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
    memcpy(d->m_userData, other.d->m_userData, sizeof(d->m_userData));
    d->m_unclosed = other.d->m_unclosed;
    d->m_color = other.d->m_color;
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
    if (&other == this)
    {
        return *this;
    }

    ShapePrivate *d_old = d;

    d = new ShapePrivate();
    d->m_type = other.d->m_type;
    d->m_polygon = other.d->m_polygon;
    d->m_transform = other.d->m_transform;
    d->m_index = other.d->m_index;
    d->m_name = other.d->m_name;
    memcpy(d->m_userData, other.d->m_userData, sizeof(d->m_userData));
    d->m_unclosed = other.d->m_unclosed;
    d->m_color = other.d->m_color;

    if (d_old)
    {
        delete d_old;
    }

    return *this;
}

//----------------------------------------------------------------------------------------------
bool Shape::isValid() const
{
    return (d->m_type & Shape::Invalid) > 0;
}

//----------------------------------------------------------------------------------------------
unsigned int Shape::flags() const
{
    return d->m_type & Shape::FlagMask;
}

//----------------------------------------------------------------------------------------------
void Shape::setFlags(const unsigned int &flags)
{
    d->m_type = (d->m_type & Shape::TypeMask) | (flags & Shape::FlagMask);
}

//----------------------------------------------------------------------------------------------
bool Shape::unclosed() const
{
    return d->m_unclosed;
}

//----------------------------------------------------------------------------------------------
void Shape::setUnclosed(bool unclosed)
{
    d->m_unclosed = unclosed;
}

//----------------------------------------------------------------------------------------------
unsigned int Shape::type() const
{
    return d->m_type & Shape::TypeMask;
}

//----------------------------------------------------------------------------------------------
void Shape::setType(const unsigned int &type)
{
    d->m_type = (d->m_type & Shape::FlagMask) | (type & Shape::TypeMask);
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
QColor Shape::color() const
{
	return d->m_color;
}

//----------------------------------------------------------------------------------------------
void Shape::setColor(const QColor &color)
{
	d->m_color = color;
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
double Shape::rotationAngleDeg() const
{
    //careful: m12 is the first column in the 2nd row (based on the QTransform definition)!
    return (std::atan2(d->m_transform.m12(), d->m_transform.m11()) * 180.0 / M_PI);
}

//----------------------------------------------------------------------------------------------
double Shape::rotationAngleRad() const
{
    //careful: m12 is the first column in the 2nd row (based on the QTransform definition)!
    return std::atan2(d->m_transform.m12(), d->m_transform.m11());
}

//----------------------------------------------------------------------------------------------
void Shape::setRotationAngleDeg(double degree)
{
    qreal dx = d->m_transform.dx(); //equal to transform.m31()
    qreal dy = d->m_transform.dy(); //equal to transform.m32()
    d->m_transform.reset();
    d->m_transform.translate(dx,dy);
    d->m_transform.rotate(degree);
}

//----------------------------------------------------------------------------------------------
void Shape::setRotationAngleRad(double radians)
{
    qreal dx = d->m_transform.dx(); //equal to transform.m31()
    qreal dy = d->m_transform.dy(); //equal to transform.m32()
    d->m_transform.reset();
    d->m_transform.translate(dx,dy);
    d->m_transform.rotateRadians(radians);
}

//----------------------------------------------------------------------------------------------
void Shape::rotateByCenterDeg(double degree)
{
    rotateByCenterRad(degree * M_PI / 180.0);
}

//----------------------------------------------------------------------------------------------
void Shape::rotateByCenterRad(double radians)
{
    //derived from https://de.serlo.org/44216/drehung-um-beliebigen-punkt-z
    QPointF center_base = baseCenterPoint();
    QPointF center = d->m_transform.map(center_base);

    double cosa = std::cos(radians);
    double sina = std::sin(radians);

    const QTransform &cur = d->m_transform;

    double m11 = cosa * cur.m11() - sina * cur.m12(); // = m22
    double m21 = cosa * cur.m21() - sina * cur.m22(); // = -m12

    QTransform rotation_only(m11, -m21, m21, m11, 0, 0);
    QPointF translate = center - rotation_only.map(center_base);

    d->m_transform = QTransform(m11, -m21, m21, m11, translate.x(), translate.y());

}

//----------------------------------------------------------------------------------------------
void Shape::translate(const QPointF &delta)
{
    QTransform &t = d->m_transform;
    QTransform new_trafo(t.m11(), t.m12(), t.m21(), t.m22(), t.dx() + delta.x(), t.dy() + delta.y());
    d->m_transform = new_trafo;
}

//----------------------------------------------------------------------------------------------
ito::float64 Shape::userData1() const
{
    return d->m_userData[0];
}

//----------------------------------------------------------------------------------------------
void Shape::setUserData1(const ito::float64 &userData1)
{
    d->m_userData[0] = userData1;
}

//----------------------------------------------------------------------------------------------
ito::float64 Shape::userData2() const
{
    return d->m_userData[1];
}

//----------------------------------------------------------------------------------------------
void Shape::setUserData2(const ito::float64 &userData2)
{
    d->m_userData[1] = userData2;
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
const QPolygonF &Shape::rbasePoints() const
{
    return d->m_polygon;
}

//----------------------------------------------------------------------------------------------
void Shape::point1MoveTo(const QPointF &newPoint1)
{
    switch (type())
    {
    case MultiPointPick:
    case Point:
    case Line:
    case Rectangle:
    case Square:
    case Circle:
    case Ellipse:
    case Polygon:
    {
        QPointF dist = d->m_transform.inverted().map(newPoint1) - d->m_polygon[0];
        for (int i = 0; i < d->m_polygon.size(); ++i)
        {
            d->m_polygon[i] += dist;
        }
    }
    break;
    default:
        break;
    }
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
QPolygonF Shape::contour(bool applyTrafo /*= false*/, qreal tol /*= -1.0*/) const
{
    switch (type())
    {
    case Point:
    case MultiPointPick:
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
            return d->m_transform.map(QPolygonF(rect));
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
bool Shape::contains(const QPointF &point) const /*!< returns true if shape contains the given point, or false if this is not the case. In case of shapes with an area of 0, this method always returns false.*/
{
    QTransform invTrafo = d->m_transform.inverted();
    QPointF pointInvTrafo = invTrafo.map(point);

    switch (type())
    {
        case Point:
        case MultiPointPick:
        case Line:
        default:
        {
            return false;
        }
        case Rectangle:
        case Square:
        {
            QRectF rect(d->m_polygon[0], d->m_polygon[1]);
            return rect.contains(pointInvTrafo);
        }
        case Polygon:
        {
            return d->m_polygon.containsPoint(pointInvTrafo, Qt::OddEvenFill);
        }
        case Circle:
        {
            QPointF center = (d->m_polygon[0] + d->m_polygon[1]) / 2.0;
            qreal radius = (d->m_polygon[1] - d->m_polygon[0]).x() / 2.0;
            QPointF dist = pointInvTrafo - center;
            return (dist.rx() * dist.rx() + dist.ry() * dist.ry()) <= (radius * radius);
        }
        case Ellipse:
        {
            QPointF center = (d->m_polygon[0] + d->m_polygon[1]) / 2.0;
            QPointF radius = (d->m_polygon[1] - d->m_polygon[0]) / 2.0;
            qreal a2inv = 1.0 / (radius.rx() * radius.rx());
            qreal b2inv = 1.0 / (radius.ry() * radius.ry());
            QPointF dist = pointInvTrafo - center;
            return (a2inv * dist.rx() * dist.rx() + b2inv * dist.ry() * dist.ry()) <= 1.0;
        }
    }
}

//----------------------------------------------------------------------------------------------
QVector<bool> Shape::contains(const QPolygonF &points) const /*!< repeatedly calls contains(point) for each point in points and returns a vector of boolean values to tell for each point if it is contained in the shape or not. */
{
    QVector<bool> result;

    if (points.size() > 0)
    {
        QTransform invTrafo = d->m_transform.inverted();
        QPolygonF pointsInvTrafo = invTrafo.map(points);
        result.resize(pointsInvTrafo.size());


        switch (type())
        {
            case Point:
            case MultiPointPick:
            case Line:
            default:
            {
                result.fill(false, pointsInvTrafo.size());
                break;
            }
            case Rectangle:
            case Square:
            {
                QRectF rect(d->m_polygon[0], d->m_polygon[1]);

                for (int i = 0; i < pointsInvTrafo.size(); ++i)
                {
                    result[i] = rect.contains(pointsInvTrafo[i]);
                }
                break;
            }
            case Polygon:
            {
                for (int i = 0; i < pointsInvTrafo.size(); ++i)
                {
                    result[i] = d->m_polygon.containsPoint(pointsInvTrafo[i], Qt::OddEvenFill);
                }
                break;
            }
            case Circle:
            {
                QPointF center = (d->m_polygon[0] + d->m_polygon[1]) / 2.0;
                qreal radius = (d->m_polygon[1] - d->m_polygon[0]).x() / 2.0;
                QPointF dist;

                for (int i = 0; i < pointsInvTrafo.size(); ++i)
                {
                    dist = pointsInvTrafo[i] - center;
                    result[i] = (dist.rx() * dist.rx() + dist.ry() * dist.ry()) <= (radius * radius);
                }
                break;
            }
            case Ellipse:
            {
                QPointF center = (d->m_polygon[0] + d->m_polygon[1]) / 2.0;
                QPointF radius = (d->m_polygon[1] - d->m_polygon[0]) / 2.0;
                qreal a2inv = 1.0 / (radius.rx() * radius.rx());
                qreal b2inv = 1.0 / (radius.ry() * radius.ry());
                QPointF dist;

                for (int i = 0; i < pointsInvTrafo.size(); ++i)
                {
                    dist = pointsInvTrafo[i] - center;
                    result[i] = (a2inv * dist.rx() * dist.rx() + b2inv * dist.ry() * dist.ry()) <= 1.0;
                }
                break;
            }
        }
    }

    return result;
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

    qreal length = QLineF(ellipse_point, edge_center).length();
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
        rect = rect.normalized(); // in case of a negative width or height
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
    case MultiPointPick:
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
QPointF Shape::centerPoint() const
{
    if (d->m_transform.isIdentity())
    {
        return baseCenterPoint();
    }

    switch (type())
    {
    case Point:
        return d->m_transform.map(d->m_polygon[0]);
    case Polygon:
    case MultiPointPick: {
        QPointF sum(0.0, 0.0);

        foreach (const QPointF &curPoint, d->m_polygon)
        {
            sum += d->m_transform.map(curPoint);
        }

        return sum / (qreal)d->m_polygon.size();
    }
    case Line:
    case Rectangle:
    case Square:
    case Ellipse:
    case Circle: {
        QPointF p1 = d->m_polygon[1];
        QPointF p2 = d->m_polygon[0];
        p1 = d->m_transform.map(p1);
        p2 = d->m_transform.map(p2);
        return (p1+p2) / 2.0;
    }
    default:
        return QPointF(0.0, 0.0);
    }
}

//----------------------------------------------------------------------------------------------
QPointF Shape::baseCenterPoint() const
{
	switch (type())
	{

	case Point:
		return d->m_polygon[0];
	case Polygon:
	case MultiPointPick:
	{
		QPointF sum(0.0, 0.0);

		foreach (const QPointF &curPoint, d->m_polygon)
		{
			sum += curPoint;
		}

		return sum / (qreal)d->m_polygon.size();
	}
	case Line:
	case Rectangle:
	case Square:
	case Ellipse:
	case Circle:
	{
		return (d->m_polygon[1] + d->m_polygon[0]) / 2.0;
	}
	default:
		return QPointF(0.0, 0.0);
	}

}

//----------------------------------------------------------------------------------------------
double Shape::area() const
{
	switch (type())
    {
    case MultiPointPick:
    case Point:
    case Line:
        return 0.0;
    case Rectangle:
    case Square:
        {
            QPointF size = d->m_polygon[1] - d->m_polygon[0];
            return (std::abs(size.rx()) * std::abs(size.ry()));
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
            return (M_PI * std::abs(size.rx()) * std::abs(size.ry()) / 4);
        }
    default:
        return 0.0;
    }
}

//----------------------------------------------------------------------------------------------
double Shape::circumference() const
{
    switch (type())
    {
    case MultiPointPick:
    case Point:
    case Line:
        return 0.0;
    case Rectangle:
    case Square:
        {
            QPointF size = d->m_polygon[1] - d->m_polygon[0];
            return (2 * std::abs(size.rx()) + 2 * std::abs(size.ry()));
        }
    case Polygon:
        {
            //from http://www.mathopenref.com/coordpolygonarea.html
            qreal val = 0.0;
            for (int i = 0; i < d->m_polygon.size() - 1; ++i)
            {
				val += std::sqrt(     std::pow(d->m_polygon[i + 1].rx() - d->m_polygon[i].rx(), 2)
							    	+ std::pow(d->m_polygon[i + 1].ry() - d->m_polygon[i].ry(), 2));
            }

            if (d->m_polygon.size() > 0)
            {
				val += std::sqrt(  std::pow(d->m_polygon[0].rx() - d->m_polygon.last().rx(), 2)
								 + std::pow(d->m_polygon[0].ry() - d->m_polygon.last().ry(), 2));
			}

            return val;
        }
	// N\E4hrung mittels N\E4herungsformel nach Ramanujan https://de.wikipedia.org/wiki/Ellipse
    case Ellipse:
		{
			QPointF size = d->m_polygon[1] - d->m_polygon[0];
			double a = std::abs(size.rx());
			double b = std::abs(size.ry());
			if (ito::isZeroValue(a, std::numeric_limits<double>::epsilon()))
			{
				return 0.0;
			}
			if (ito::isZeroValue(b, std::numeric_limits<double>::epsilon()))
			{
				return 0.0;
			}
			double lamdaSq = std::pow((a - b) / (a + b), 2);

			return M_PI * (a + b) * (  1.0 + (3.0 * lamdaSq) / (10.0 + std::sqrt(4.0 - 3.0 * lamdaSq ) ) );
		}
	case Circle:
        {
            QPointF size = d->m_polygon[1] - d->m_polygon[0];
            return ( M_PI * std::abs(size.rx()));
        }
    default:
        return 0.0;
    }
}

//----------------------------------------------------------------------------------------------
double Shape::distance(const Shape &otherShape) const
{
	if (type() == Line && otherShape.type() == Line)
	{
		return distanceLine2Line2D(*this, otherShape);
	}

	if (type() == Line)
	{
		return distanceLine2Point2D(*this, otherShape.centerPoint());
	}

	if (otherShape.type() == Line)
	{
		return distanceLine2Point2D(otherShape, this->centerPoint());
	}
	return 0.0;
}

//----------------------------------------------------------------------------------------------
double Shape::centerDistance(const Shape &otherShape) const
{
	return distancePoint2Point2D(this->centerPoint(), otherShape.centerPoint());
}

//----------------------------------------------------------------------------------------------
double Shape::radius() const
{
	/*
	QRectF rect(d->m_polygon[0], d->m_polygon[1]);
	if (applyTrafo)
	{
	d->m_transform.mapToPolygon(rect.toRect());
	}
	else
	{

	}
	*/

	if (this->rbasePoints().size() < 2)
	{
		return 0;
	}
	if (this->rbasePoints().size() > 2)
	{
		return std::numeric_limits<double>::signaling_NaN();
	}
	double res = std::fabs((this->rbasePoints()[0].x() - this->rbasePoints()[1].x()) / 2.0);
	res += std::fabs((this->rbasePoints()[0].y() - this->rbasePoints()[1].y()) / 2.0);
	res /= 2.0;
	return res;
}

//----------------------------------------------------------------------------------------------
double Shape::radiusX() const
{
	/*
	QRectF rect(d->m_polygon[0], d->m_polygon[1]);
	if (applyTrafo)
	{
	d->m_transform.mapToPolygon(rect.toRect());
	}
	else
	{

	}
	*/

	if (this->rbasePoints().size() < 2)
	{
		return 0;
	}
	if (this->rbasePoints().size() > 2)
	{
		return std::numeric_limits<double>::signaling_NaN();
	}

	double res = std::fabs((this->rbasePoints()[0].x() - this->rbasePoints()[1].x()) / 2.0);
	return res;
}

//----------------------------------------------------------------------------------------------
double Shape::radiusY() const
{

	/*
	QRectF rect(d->m_polygon[0], d->m_polygon[1]);
	if (applyTrafo)
	{
	d->m_transform.mapToPolygon(rect.toRect());
	}
	else
	{

	}
	*/

	if (this->rbasePoints().size() < 2)
	{
		return 0;
	}
	if (this->rbasePoints().size() > 2)
	{
		return std::numeric_limits<double>::signaling_NaN();
	}

	double res = std::fabs((this->rbasePoints()[0].y() - this->rbasePoints()[1].y()) / 2.0);
	return res;
}

//----------------------------------------------------------------------------------------------
double Shape::distanceLine2Point2D(const Shape &line, const QPointF &point)
{
	/*
	QRectF rect(d->m_polygon[0], d->m_polygon[1]);
	if (applyTrafo)
	{
	d->m_transform.mapToPolygon(rect.toRect());
	}
	else
	{

	}
	*/

	double result = 0.0;

	QPointF dirVec = line.d->m_polygon[1] - line.d->m_polygon[0];
	QPointF normVec(dirVec.y() * -1, dirVec.x());
	normVec /= std::sqrt(std::pow(normVec.x(), 2) + pow(normVec.y(), 2));
	QPointF baseVec = line.d->m_polygon[0];

	// m * nx + px = n * dx + bx
	// m * ny + py = n * dy + by

	// m * nx + px = n * dx + bx
	// n = (m * ny + py - by) / dy

	// m * nx + px = (m * ny + py - by) * dx / dy  + bx
	// m * nx + px = (m * ny * dx / dy) + (py - by) * dx / dy  + bx
	// m * nx - (m * ny * dx / dy) =  (py - by) * dx / dy  + bx - px
	// m * (nx - ny * dx / dy) =  (py - by) * dx / dy  + bx - px
	// m =  ((py - by) * dx / dy  + bx - px) / (nx - ny * dx / dy)

	if (ito::isZeroValue<double>(normVec.x() - normVec.y() * dirVec.x() / dirVec.y(), std::numeric_limits<double>::epsilon()))
	{
		return std::numeric_limits<double>::quiet_NaN();
	}
	result = ((point.y() - baseVec.y()) * dirVec.x() / dirVec.y() + baseVec.x() - point.x()) / (normVec.x() - normVec.y() * dirVec.x() / dirVec.y());
	return result;
}

//----------------------------------------------------------------------------------------------
double Shape::distanceLine2Line2D(const Shape &line1, const Shape &line2)
{
	/*
	QRectF rect(d->m_polygon[0], d->m_polygon[1]);
	if (applyTrafo)
	{
	d->m_transform.mapToPolygon(rect.toRect());
	}
	else
	{

	}
	*/

	double result = 0.0;
	QPointF dirVec1 = line1.d->m_polygon[1] - line1.d->m_polygon[0];
	dirVec1 /= std::sqrt(std::pow(dirVec1.x(), 2) + pow(dirVec1.y(), 2));
	QPointF baseVec1 = line1.d->m_polygon[0];

	QPointF dirVec2 = line2.d->m_polygon[1] - line2.d->m_polygon[0];
	double length2 = std::sqrt(std::pow(dirVec2.x(), 2) + pow(dirVec2.y(), 2));
	dirVec2 /= length2;
	QPointF baseVec2 = line2.d->m_polygon[0];

	if ((ito::isZeroValue<double>(dirVec1.x() - dirVec2.x(), std::numeric_limits<double>::epsilon()) &&
		ito::isZeroValue<double>(dirVec1.y() - dirVec2.y(), std::numeric_limits<double>::epsilon()))
		||
		(ito::isZeroValue<double>(dirVec1.x() + dirVec2.x(), std::numeric_limits<double>::epsilon()) &&
		ito::isZeroValue<double>(dirVec1.y() + dirVec2.y(), std::numeric_limits<double>::epsilon())))
	{
		// line must be parallel
		result = distanceLine2Point2D(line1, baseVec2);
	}
	else // lines are not parallel
	{

		// m * d2x + b2x = n * d1x + b1x
		// m * d2y + b2y = n * d1y + b1y

		// m * d2x + b2x = n * d1x + b1x
		// n = (m * d2y + b2y - b1y) / d1y

		// m * d2x + b2x = (m * d2y + b2y - b1y) * d1x / d1y  + b1x
		// m * d2x + b2x = (m * d2y * d1x / d1y) + (b2y - b1y) * d1x / d1y  + b1x
		// m * d2x - (m * d2y * d1x / d1y) =  (b2y - b1y) * d1x / d1y  + b1x - b2x
		// m * (d2x - d2y * d1x / d1y) =  (b2y - b1y) * d1x / d1y  + b1x - b2x
		// m =  ((b2y - b1y) * d1x / d1y  + b1x - b2x) / (d2x - d2y * d1x / d1y)

		// lines have an intersection
		result = ((baseVec2.y() - baseVec1.y()) * dirVec1.x() / dirVec1.y() + baseVec1.x() - baseVec2.x()) / (dirVec2.x() - dirVec2.y() * dirVec1.x() / dirVec1.y());
		// lines does not have an intersection

		if (result > length2 || result < 0.0)
		{
			result = distanceLine2Point2D(line1, baseVec2);
			double tmpRes = distanceLine2Point2D(line1, line2.d->m_polygon[1]);
			if (tmpRes < result)
			{
				result = tmpRes;
			}

			tmpRes = distanceLine2Point2D(line2, line1.d->m_polygon[1]);
			if (tmpRes < result)
			{
				result = tmpRes;
			}
			tmpRes = distanceLine2Point2D(line2, line1.d->m_polygon[0]);
			if (tmpRes < result)
			{
				result = tmpRes;
			}
		}
		else // both lines intersect
		{
			result = 0.0;
		}
	}

	return result;
}

//----------------------------------------------------------------------------------------------
double Shape::distancePoint2Point2D(const QPointF &point1, const QPointF &point2)
{
	/*
	QRectF rect(d->m_polygon[0], d->m_polygon[1]);
	if (applyTrafo)
	{
	d->m_transform.mapToPolygon(rect.toRect());
	}
	else
	{

	}
	*/

	QPointF val = point1 - point2;
	return std::sqrt(std::pow(val.rx(), 2) + std::pow(val.ry(), 2));
}

//----------------------------------------------------------------------------------------------
Shape Shape::normalized() const
{
    Shape out = *this;

    switch (type())
    {
    case Rectangle:
    case Square:
    case Ellipse:
    case Circle:
    {
        QRectF rect(basePoints()[0], basePoints()[1]);
        if (rect.width() < 0 || rect.height() < 0)
        {
            rect = rect.normalized();
            out.rbasePoints()[0] = rect.topLeft();
            out.rbasePoints()[1] = rect.bottomRight();
        }
        break;
    }
    default:
        break;
    }

    return out;
}

//----------------------------------------------------------------------------------------------
//ito::DataObject Shape::mask(const ito::DataObject &dataObject, bool inverse /*= false*/) const
//{
//    ito::DataObject mask;
//
//    if (dataObject.getTotal() > 0)
//    {
//        mask.zeros(dataObject.getDims(), dataObject.getSize(), ito::tUInt8);
//        dataObject.copyAxisTagsTo(mask);
//
//        if (inverse)
//        {
//            mask.setTo(255);
//        }
//
//        maskHelper(dataObject, mask, inverse);
//    }
//
//    return mask;
//}
//
////----------------------------------------------------------------------------------------------
//void Shape::maskHelper(const ito::DataObject &dataObject, ito::DataObject &mask, bool inverse /*= false*/) const
//{
//    //only call this via mask or maskFromMultipleShapes
//    int dims = dataObject.getDims();
//    int numPlanes = dataObject.getNumPlanes();
//
//    if (dataObject.getTotal() > 0)
//    {
//        int rows = dataObject.getSize(dims - 2);
//        int cols = dataObject.getSize(dims - 1);
//        cv::Mat *mat;
//        ito::uint8 *ptr;
//
//        switch (type())
//        {
//        case MultiPointPick:
//        case Point:
//        case Line:
//            break;
//        case Rectangle:
//        case Square:
//        {
//            QPointF p1 = d->m_transform.map(d->m_polygon[0]);
//            QPointF p2 = d->m_transform.map(d->m_polygon[1]);
//
//            p1.setX(dataObject.getPhysToPix(dims - 1, p1.x()));
//            p1.setY(dataObject.getPhysToPix(dims - 2, p1.y()));
//
//            p2.setX(dataObject.getPhysToPix(dims - 1, p2.x()));
//            p2.setY(dataObject.getPhysToPix(dims - 2, p2.y()));
//
//            int minRow = qBound(0, qRound(p1.y()), rows - 1);
//            int maxRow = qBound(0, qRound(p2.y()), rows - 1);
//            if (maxRow < minRow)
//            {
//                std::swap(minRow, maxRow);
//            }
//
//            int minCol = qBound(0, qRound(p1.x()), cols - 1);
//            int maxCol = qBound(0, qRound(p2.x()), cols - 1);
//            if (maxCol < minCol)
//            {
//                std::swap(minCol, maxCol);
//            }
//
//            for (int plane = 0; plane < numPlanes; ++plane)
//            {
//                mat = mask.getCvPlaneMat(0);
//                for (int row = minRow; row <= maxRow; ++row)
//                {
//                    ptr = mat->ptr<ito::uint8>(row);
//                    for (int col = minCol; col <= maxCol; ++col)
//                    {
//                        ptr[col] = (inverse ? 0 : 255);
//                    }
//                }
//            }
//        }
//        case Polygon:
//        {
//            QPolygonF cont = contour(true);
//            //trafo from phys coords to pixel coords
//            for (int i = 0; i < cont.size(); ++i)
//            {
//                cont[i].setX(dataObject.getPhysToPix(dims - 1, cont[i].x()));
//                cont[i].setY(dataObject.getPhysToPix(dims - 2, cont[i].y()));
//            }
//
//            QRectF boundingRect = cont.boundingRect();
//            QPointF p1 = boundingRect.topLeft();
//            QPointF p2 = boundingRect.bottomRight();
//            QPointF test;
//
//            int minRow = qBound(0, qRound(p1.y()), rows - 1);
//            int maxRow = qBound(0, qRound(p2.y()), rows - 1);
//            if (maxRow < minRow)
//            {
//                std::swap(minRow, maxRow);
//            }
//
//            int minCol = qBound(0, qRound(p1.x()), cols - 1);
//            int maxCol = qBound(0, qRound(p2.x()), cols - 1);
//            if (maxCol < minCol)
//            {
//                std::swap(minCol, maxCol);
//            }
//
//            for (int plane = 0; plane < numPlanes; ++plane)
//            {
//                mat = mask.getCvPlaneMat(0);
//                for (int row = minRow; row <= maxRow; ++row)
//                {
//                    ptr = mat->ptr<ito::uint8>(row);
//                    test.setY(row);
//                    for (int col = minCol; col <= maxCol; ++col)
//                    {
//                        test.setX(col);
//                        if (cont.containsPoint(test, Qt::OddEvenFill))
//                        {
//                            ptr[col] = (inverse ? 0 : 255);
//                        }
//                    }
//                }
//            }
//        }
//        case Ellipse:
//        case Circle:
//        {
//            QPointF p1 = d->m_transform.map(d->m_polygon[0]);
//            QPointF p2 = d->m_transform.map(d->m_polygon[1]);
//
//            p1.setX(dataObject.getPhysToPix(dims - 1, p1.x()));
//            p1.setY(dataObject.getPhysToPix(dims - 2, p1.y()));
//
//            p2.setX(dataObject.getPhysToPix(dims - 1, p2.x()));
//            p2.setY(dataObject.getPhysToPix(dims - 2, p2.y()));
//
//            QPointF c = 0.5 * (p1 + p2);
//            QPointF r = 0.5 * (p2 - p1);
//            double x, y;
//            double a = r.x();
//            double b = r.y();
//
//            int minRow = qBound(0, qRound(p1.y()), rows - 1);
//            int maxRow = qBound(0, qRound(p2.y()), rows - 1);
//            if (maxRow < minRow)
//            {
//                std::swap(minRow, maxRow);
//            }
//
//            int minCol = qBound(0, qRound(p1.x()), cols - 1);
//            int maxCol = qBound(0, qRound(p2.x()), cols - 1);
//            if (maxCol < minCol)
//            {
//                std::swap(minCol, maxCol);
//            }
//
//            for (int plane = 0; plane < numPlanes; ++plane)
//            {
//                mat = mask.getCvPlaneMat(0);
//                for (int row = minRow; row <= maxRow; ++row)
//                {
//                    ptr = mat->ptr<ito::uint8>(row);
//                    for (int col = minCol; col <= maxCol; ++col)
//                    {
//                        x = col - c.x();
//                        y = row - c.y();
//
//                        if ((((x*x) / (a*a)) + ((y*y) / (b*b))) <= 1.0)
//                        {
//                            ptr[col] = (inverse ? 0 : 255);
//                        }
//                    }
//                }
//            }
//        }
//        default:
//            break;
//        }
//    }
//}
//
////----------------------------------------------------------------------------------------------
///*static*/ ito::DataObject Shape::maskFromMultipleShapes(const ito::DataObject &dataObject, const QVector<ito::Shape> &shapes, bool inverse /*= false*/)
//{
//    ito::DataObject mask;
//
//    if (dataObject.getTotal() > 0)
//    {
//        mask.zeros(dataObject.getDims(), dataObject.getSize(), ito::tUInt8);
//        dataObject.copyAxisTagsTo(mask);
//
//        if (inverse)
//        {
//            mask.setTo(255);
//        }
//
//        foreach(const ito::Shape &shape, shapes)
//        {
//            shape.maskHelper(dataObject, mask, inverse);
//        }
//    }
//
//    return mask;
//}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromRectangle(const QRectF &rect, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
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
/*static*/ Shape Shape::fromRectangle(qreal x1, qreal y1, qreal x2, qreal y2, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Rectangle;
    s.d->m_polygon << QPointF(x1, y1) << QPointF(x2, y2);
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
    s.d->m_polygon << QPointF(x1, y1) << QPointF(x2, y2);
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
    s.d->m_polygon << QPointF(x1, y1) << QPointF(x2, y2);
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
    s.d->m_polygon << QPointF(x, y);
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromPolygon(const QPolygonF &polygon, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = Polygon;
    s.d->m_polygon = polygon;
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ Shape Shape::fromMultipoint(const QPolygonF &polygon, int index /*= -1*/, QString name /*= ""*/, const QTransform &trafo /*= QTransform()*/)
{
    Shape s;
    s.d->m_type = MultiPointPick;
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
    QPointF dist(std::abs(sideLength) / 2, std::abs(sideLength) / 2);
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
    QPointF dist(std::abs(radius), std::abs(radius));
    s.d->m_type = Circle;
    s.d->m_polygon << (center - dist) << (center + dist);
    s.d->m_transform = trafo;
    s.d->m_index = index;
    s.d->m_name = name;
    return s;
}

//----------------------------------------------------------------------------------------------
/*static*/ QString Shape::type2QString(const int type)
{
	switch (type & ito::Shape::TypeMask)
	{
		case MultiPointPick:
			return "Multipoint";
		case Point:
			return "Point";
		case Line:
			return "Line";
		case Rectangle:
			return "Rectangle";
		case Square:
			return "Square";
		case Circle:
			return "Circle";
		case Ellipse:
			return "Ellipse";
		case Polygon:
			return "Polygon";
		break;
		default:
			return "N.A.";
	}
	return "N.A.";
}


} //end namespace ito
