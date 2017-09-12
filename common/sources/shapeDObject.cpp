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

#include "../shapeDObject.h"
#include "opencv2/imgproc/imgproc.hpp"
#define _USE_MATH_DEFINES
#include "math.h"

#include "numeric.h"

namespace ito 
{
//----------------------------------------------------------------------------------------------
	/*static*/ ito::DataObject ShapeDObject::mask(const ito::DataObject &dataObject, const ito::Shape &shape, bool inverse /*= false*/)
{
    ito::DataObject mask;

    if (dataObject.getTotal() > 0)
    {
        mask.zeros(dataObject.getDims(), dataObject.getSize(), ito::tUInt8);
        dataObject.copyAxisTagsTo(mask);

        if (inverse)
        {
            mask.setTo(255);
        }

		maskHelper(dataObject, mask, shape, inverse);
    }

    return mask;
}

//----------------------------------------------------------------------------------------------
/*static*/ void ShapeDObject::maskHelper(const ito::DataObject &dataObject, ito::DataObject &mask, const ito::Shape &shape, bool inverse /*= false*/)
{
    //only call this via mask or maskFromMultipleShapes
    int dims = dataObject.getDims();
    int numPlanes = dataObject.getNumPlanes();

    if (dataObject.getTotal() > 0)
    {
        int rows = dataObject.getSize(dims - 2);
        int cols = dataObject.getSize(dims - 1);
        cv::Mat *mat;
        ito::uint8 *ptr;

		switch (shape.type())
        {
		case Shape::MultiPointPick:
		case Shape::Point:
		case Shape::Line:
            break;
		case Shape::Rectangle:
		case Shape::Square:
        {
            QPointF p1 = shape.rbasePoints()[0];
            QPointF p2 = shape.rbasePoints()[1];
            QPointF p3 = shape.centerPoint();

            cv::Scalar col;
            if (inverse)
                col = cv::Scalar(0, 0, 0);
            else
                col = cv::Scalar(255, 255, 255);

            cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f(p3.x(), p3.y()),
                cv::Size(fabs(p2.x() - p1.x()), fabs(p2.y() - p1.y())), -shape.rotationAngleDeg());
            cv::Rect brect = rRect.boundingRect();
            cv::rectangle(*mask.getCvPlaneMat(0), brect, col, -1);

/*
            QPointF p1 = shape.rtransform().map(shape.rbasePoints()[0]);
            QPointF p2 = shape.rtransform().map(shape.rbasePoints()[1]);

            //QPointF p1 = d->m_transform.map(d->m_polygon[0]);
            //QPointF p2 = d->m_transform.map(d->m_polygon[1]);

            p1.setX(dataObject.getPhysToPix(dims - 1, p1.x()));
            p1.setY(dataObject.getPhysToPix(dims - 2, p1.y()));

            p2.setX(dataObject.getPhysToPix(dims - 1, p2.x()));
            p2.setY(dataObject.getPhysToPix(dims - 2, p2.y()));

            int minRow = qBound(0, qRound(p1.y()), rows - 1);
            int maxRow = qBound(0, qRound(p2.y()), rows - 1);
            if (maxRow < minRow)
            {
                std::swap(minRow, maxRow);
            }

            int minCol = qBound(0, qRound(p1.x()), cols - 1);
            int maxCol = qBound(0, qRound(p2.x()), cols - 1);
            if (maxCol < minCol)
            {
                std::swap(minCol, maxCol);
            }

            for (int plane = 0; plane < numPlanes; ++plane)
            {
                mat = mask.getCvPlaneMat(0);
                for (int row = minRow; row <= maxRow; ++row)
                {
                    ptr = mat->ptr<ito::uint8>(row);
                    for (int col = minCol; col <= maxCol; ++col)
                    {
                        ptr[col] = (inverse ? 0 : 255);
                    }
                }
            }
*/
        }
		case Shape::Polygon:
        {
			QPolygonF cont = shape.contour(true);
            //trafo from phys coords to pixel coords
            for (int i = 0; i < cont.size(); ++i)
            {
                cont[i].setX(dataObject.getPhysToPix(dims - 1, cont[i].x()));
                cont[i].setY(dataObject.getPhysToPix(dims - 2, cont[i].y()));
            }

            QRectF boundingRect = cont.boundingRect();
            QPointF p1 = boundingRect.topLeft();
            QPointF p2 = boundingRect.bottomRight();
            QPointF test;

            int minRow = qBound(0, qRound(p1.y()), rows - 1);
            int maxRow = qBound(0, qRound(p2.y()), rows - 1);
            if (maxRow < minRow)
            {
                std::swap(minRow, maxRow);
            }

            int minCol = qBound(0, qRound(p1.x()), cols - 1);
            int maxCol = qBound(0, qRound(p2.x()), cols - 1);
            if (maxCol < minCol)
            {
                std::swap(minCol, maxCol);
            }

            for (int plane = 0; plane < numPlanes; ++plane)
            {
                mat = mask.getCvPlaneMat(0);
                for (int row = minRow; row <= maxRow; ++row)
                {
                    ptr = mat->ptr<ito::uint8>(row);
                    test.setY(row);
                    for (int col = minCol; col <= maxCol; ++col)
                    {
                        test.setX(col);
                        if (cont.containsPoint(test, Qt::OddEvenFill))
                        {
                            ptr[col] = (inverse ? 0 : 255);
                        }
                    }
                }
            }
        }
		case Shape::Ellipse:
		case Shape::Circle:
        {
            QPointF p1 = shape.rbasePoints()[0];
            QPointF p2 = shape.rbasePoints()[1];
            QPointF p3 = shape.centerPoint();

            cv::Scalar col;
            if (inverse)
                col = cv::Scalar(0, 0, 0);
            else
                col = cv::Scalar(255, 255, 255);

            cv::ellipse(*mask.getCvPlaneMat(0), cv::Point2f(p3.x(), p3.y()), 
                cv::Size(fabs(p2.x() - p1.x()), fabs(p2.y() - p1.y())), -shape.rotationAngleDeg(), 0, 360,
                col, -1);
/*
            //QPointF p1 = d->m_transform.map(d->m_polygon[0]);
            //QPointF p2 = d->m_transform.map(d->m_polygon[1]);
			QPointF p1 = shape.rtransform().map(shape.rbasePoints()[0]);
			QPointF p2 = shape.rtransform().map(shape.rbasePoints()[1]);

            p1.setX(dataObject.getPhysToPix(dims - 1, p1.x()));
            p1.setY(dataObject.getPhysToPix(dims - 2, p1.y()));

            p2.setX(dataObject.getPhysToPix(dims - 1, p2.x()));
            p2.setY(dataObject.getPhysToPix(dims - 2, p2.y()));

            QPointF c = 0.5 * (p1 + p2);
            QPointF r = 0.5 * (p2 - p1);
            double x, y;
            double a = r.x();
            double b = r.y();

            int minRow = qBound(0, qRound(p1.y()), rows - 1);
            int maxRow = qBound(0, qRound(p2.y()), rows - 1);
            if (maxRow < minRow)
            {
                std::swap(minRow, maxRow);
            }

            int minCol = qBound(0, qRound(p1.x()), cols - 1);
            int maxCol = qBound(0, qRound(p2.x()), cols - 1);
            if (maxCol < minCol)
            {
                std::swap(minCol, maxCol);
            }

            for (int plane = 0; plane < numPlanes; ++plane)
            {
                mat = mask.getCvPlaneMat(0);
                for (int row = minRow; row <= maxRow; ++row)
                {
                    ptr = mat->ptr<ito::uint8>(row);
                    for (int col = minCol; col <= maxCol; ++col)
                    {
                        x = col - c.x();
                        y = row - c.y();

                        if ((((x*x) / (a*a)) + ((y*y) / (b*b))) <= 1.0)
                        {
                            ptr[col] = (inverse ? 0 : 255);
                        }
                    }
                }
            }
*/
        }
        default:
            break;
        }
    }
}

//----------------------------------------------------------------------------------------------
/*static*/ ito::DataObject ShapeDObject::maskFromMultipleShapes(const ito::DataObject &dataObject, const QVector<ito::Shape> &shapes, bool inverse /*= false*/)
{
    ito::DataObject mask;
    
    if (dataObject.getTotal() > 0)
    {
        mask.zeros(dataObject.getDims(), dataObject.getSize(), ito::tUInt8);
        dataObject.copyAxisTagsTo(mask);

        if (inverse)
        {
            mask.setTo(255);
        }

        foreach(const ito::Shape &shape, shapes)
        {
			ShapeDObject::maskHelper(dataObject, mask, shape, inverse);
        }
    }

    return mask;
}
} //end namespace ito
