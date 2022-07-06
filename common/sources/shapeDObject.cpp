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

#include "../shapeDObject.h"
#include "opencv2/imgproc/imgproc.hpp"
#define _USE_MATH_DEFINES
#include "math.h"

#include "numeric.h"

namespace ito {
//----------------------------------------------------------------------------------------------
/*static*/ ito::DataObject ShapeDObject::mask(
    const ito::DataObject& dataObject, const ito::Shape& shape, bool inverse /*= false*/)
{
    ito::DataObject mask;

    if (dataObject.getTotal() > 0)
    {
        mask.zeros(
            dataObject.getSize()[dataObject.getDims() - 2],
            dataObject.getSize()[dataObject.getDims() - 1],
            ito::tUInt8);
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
/*static*/ void ShapeDObject::maskHelper(
    const ito::DataObject& dataObject,
    ito::DataObject& mask,
    const ito::Shape& shape,
    bool inverse /*= false*/)
{
    // only call this via mask or maskFromMultipleShapes
    int dims = dataObject.getDims();
    int numPlanes = dataObject.getNumPlanes();

    if (dataObject.getTotal() > 0)
    {
        int rows = dataObject.getSize(dims - 2);
        int cols = dataObject.getSize(dims - 1);

        switch (shape.type())
        {
        case Shape::MultiPointPick:
        case Shape::Point:
        case Shape::Line:
            break;

        case Shape::Rectangle:
        case Shape::Square: {
            QPointF p1 = shape.rbasePoints()[0];
            QPointF p2 = shape.rbasePoints()[1];
            QPointF pCenter = shape.centerPoint();

            p1.setX(dataObject.getPhysToPixUnclipped(dims - 1, p1.x()));
            p1.setY(dataObject.getPhysToPixUnclipped(dims - 2, p1.y()));

            p2.setX(dataObject.getPhysToPixUnclipped(dims - 1, p2.x()));
            p2.setY(dataObject.getPhysToPixUnclipped(dims - 2, p2.y()));

            pCenter.setX(dataObject.getPhysToPixUnclipped(dims - 1, pCenter.x()));
            pCenter.setY(dataObject.getPhysToPixUnclipped(dims - 2, pCenter.y()));

            cv::Scalar color;

            if (inverse)
            {
                color = cv::Scalar(0, 0, 0);
            }
            else
            {
                color = cv::Scalar(255, 255, 255);
            }

            double angle = shape.rotationAngleDeg();

            if (angle == 0.0)
            {
                cv::RotatedRect rRect = cv::RotatedRect(
                    cv::Point2f(pCenter.x(), pCenter.y()),
                    cv::Size(fabs(p2.x() - p1.x()), fabs(p2.y() - p1.y())),
                    0.0);
                cv::Rect brect = rRect.boundingRect();
                cv::rectangle(*mask.getCvPlaneMat(0), brect, color, -1);
            }
            else
            {
                cv::RotatedRect rRect = cv::RotatedRect(
                    cv::Point2f(pCenter.x(), pCenter.y()),
                    cv::Size(fabs(p2.x() - p1.x()), fabs(p2.y() - p1.y())),
                    angle);
                cv::Point2f vertices[4];
                rRect.points(vertices);
                cv::Point vertices_int[4];

                for (int i = 0; i < 4; ++i)
                {
                    vertices_int[i] = cv::Point(qRound(vertices[i].x), qRound(vertices[i].y));
                }
                cv::fillConvexPoly(*mask.getCvPlaneMat(0), vertices_int, 4, color);
            }
        }
        break;

        case Shape::Polygon: {
            QPointF p;
            std::vector<cv::Point> pts;
            std::vector<std::vector<cv::Point>> cnt;

            QPolygonF vertices = shape.transform().map(shape.basePoints());

            for (int np = 0; np < vertices.size(); np++)
            {
                p = vertices[np];
                p.setX(dataObject.getPhysToPixUnclipped(dims - 1, p.x()));
                p.setY(dataObject.getPhysToPixUnclipped(dims - 2, p.y()));
                pts.push_back(cv::Point(qRound(p.x()), qRound(p.y())));
            }

            cnt.push_back(pts);
            cv::Scalar color;

            if (inverse)
            {
                color = cv::Scalar(0, 0, 0);
            }
            else
            {
                color = cv::Scalar(255, 255, 255);
            }

            cv::fillPoly(*mask.getCvPlaneMat(0), cnt, color);
        }
        break;

        case Shape::Ellipse:
        case Shape::Circle: {
            QPointF p1 = shape.rbasePoints()[0];
            QPointF p2 = shape.rbasePoints()[1];
            QPointF pCenter = shape.centerPoint();

            p1.setX(dataObject.getPhysToPixUnclipped(dims - 1, p1.x()));
            p1.setY(dataObject.getPhysToPixUnclipped(dims - 2, p1.y()));

            p2.setX(dataObject.getPhysToPixUnclipped(dims - 1, p2.x()));
            p2.setY(dataObject.getPhysToPixUnclipped(dims - 2, p2.y()));

            pCenter.setX(dataObject.getPhysToPixUnclipped(dims - 1, pCenter.x()));
            pCenter.setY(dataObject.getPhysToPixUnclipped(dims - 2, pCenter.y()));

            cv::Scalar color;

            if (inverse)
            {
                color = cv::Scalar(0, 0, 0);
            }
            else
            {
                color = cv::Scalar(255, 255, 255);
            }

            cv::Size size =
                cv::Size(qRound((p2.x() - p1.x()) / 2.0), qRound((p2.y() - p1.y()) / 2.0));

            if (size.height < 0)
            {
                size.height *= -1.0;
            }

            if (size.width < 0)
            {
                size.width *= -1.0;
            }

            cv::ellipse(
                *mask.getCvPlaneMat(0),
                cv::Point2f(pCenter.x(), pCenter.y()),
                size,
                shape.rotationAngleDeg(),
                0,
                360,
                color,
                -1);
        }
        break;

        default:
            break;
        }
    }
}

//----------------------------------------------------------------------------------------------
/*static*/ ito::DataObject ShapeDObject::maskFromMultipleShapes(
    const ito::DataObject& dataObject, const QVector<ito::Shape>& shapes, bool inverse /*= false*/)
{
    ito::DataObject mask;

    if (dataObject.getTotal() > 0)
    {
        mask.zeros(
            dataObject.getSize()[dataObject.getDims() - 2],
            dataObject.getSize()[dataObject.getDims() - 1],
            ito::tUInt8); // 2d mask dataObject
        dataObject.copyAxisTagsTo(mask);

        if (inverse)
        {
            mask.setTo(255);
        }

        foreach (const ito::Shape& shape, shapes)
        {
            ShapeDObject::maskHelper(dataObject, mask, shape, inverse);
        }
    }

    return mask;
}

} // end namespace ito
