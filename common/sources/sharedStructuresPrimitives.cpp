/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut fr Technische
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

#include "sharedStructuresPrimitives.h"
#include <iostream>

namespace ito
{
    //----------------------------------------------------------------------------------------------------------------------------------
    PrimitiveBase::PrimitiveBase()
    {
        memset(cells, 0, sizeof(float32)*PRIM_ELEMENTLENGTH);
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    PrimitiveBase::PrimitiveBase(const PrimitiveBase &rhs)
    {
        memcpy(cells, rhs.cells, sizeof(float32)*PRIM_ELEMENTLENGTH);
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    PrimitiveBase::PrimitiveBase(const GeometricPrimitive &rhs)
    {
        memcpy(cells, rhs.cells, sizeof(float32)*PRIM_ELEMENTLENGTH);
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 PrimitiveBase::distanceTo(const GeometricPrimitive &comperator, const bool plaine /*= false*/) const
    {
        switch(getType())
        {
            default:
            case tNoType:
            case tMultiPointPick:
            case tPolygon:
                return std::numeric_limits<float32>::quiet_NaN();

            case tPoint:
                switch(extractType(comperator))
                {
                    default:
                    case tNoType:
                    case tMultiPointPick:
                    case tPolygon:
                        return std::numeric_limits<float32>::quiet_NaN();

                    case tPoint:
                    {
                        ito::float32 res = pow(comperator.cells[2] - cells[2], 2) - pow(comperator.cells[3] - cells[3], 2);
                        if(!plaine) res += pow(comperator.cells[4] - cells[4], 2);
                        return sqrt(res);
                    }

                    case tLine:
                    break;

                    case tRectangle:
                    break;

                    case tSquare:
                    break;

                    case tEllipse:
                    break;

                    case tCircle:
                    break;
                }
            break;

            case tLine:
                switch(extractType(comperator))
                {
                    default:
                    case tNoType:
                    case tMultiPointPick:
                    case tPolygon:
                        return std::numeric_limits<float32>::quiet_NaN();
                    break;

                    case tPoint:

                    break;

                    case tLine:
                    break;

                    case tRectangle:
                    break;

                    case tSquare:
                    break;

                    case tEllipse:
                    break;

                    case tCircle:
                    break;
                }
            break;

            case tRectangle:
                switch(extractType(comperator))
                {
                    default:
                    case tNoType:
                    case tMultiPointPick:
                    case tPolygon:
                        return std::numeric_limits<float32>::quiet_NaN();
                    break;

                    case tPoint:

                    break;

                    case tLine:
                    break;

                    case tRectangle:
                    break;

                    case tSquare:
                    break;

                    case tEllipse:
                    break;

                    case tCircle:
                    break;
                }
            break;

            case tSquare:
                switch(extractType(comperator))
                {
                    default:
                    case tNoType:
                    case tMultiPointPick:
                    case tPolygon:
                        return std::numeric_limits<float32>::quiet_NaN();
                    break;

                    case tPoint:

                    break;

                    case tLine:
                    break;

                    case tRectangle:
                    break;

                    case tSquare:
                    break;

                    case tEllipse:
                    break;

                    case tCircle:
                    break;
                }
            break;

            case tEllipse:
                switch(extractType(comperator))
                {
                    default:
                    case tNoType:
                    case tMultiPointPick:
                    case tPolygon:
                        return std::numeric_limits<float32>::quiet_NaN();
                    break;

                    case tPoint:

                    break;

                    case tLine:
                    break;

                    case tRectangle:
                    break;

                    case tSquare:
                    break;

                    case tEllipse:
                    break;

                    case tCircle:
                    break;
                }
            break;

            case tCircle:
                switch(extractType(comperator))
                {
                    default:
                    case tNoType:
                    case tMultiPointPick:
                    case tPolygon:
                        return std::numeric_limits<float32>::quiet_NaN();
                    break;

                    case tPoint:

                    break;

                    case tLine:
                    break;

                    case tRectangle:
                    break;

                    case tSquare:
                    break;

                    case tEllipse:
                    break;

                    case tCircle:
                    break;
                }
            break;
        }
        return std::numeric_limits<float32>::quiet_NaN();
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 PrimitiveBase::distanceToCenters(const GeometricPrimitive &comperator, const bool plaine /*= false*/) const
    {
        switch(getType())
        {
            default:
            case tNoType:
            case tMultiPointPick:
            case tPolygon:
                return std::numeric_limits<float32>::quiet_NaN();

            case tCircle:
            case tEllipse:
            case tSquare:
            case tPoint:
            {
                switch(PrimitiveBase::extractType(comperator))
                {
                    default:
                    case tNoType:
                    case tMultiPointPick:
                    case tPolygon:
                        return std::numeric_limits<float32>::quiet_NaN();

                    case tLine:
                        std::cout << "Not implemented yet" << std::endl;
                    return std::numeric_limits<float32>::quiet_NaN();
                    
                    case tRectangle:
                    {                      
                        ito::float32 res = pow(cells[2] - (comperator.cells[2] + comperator.cells[5]) / 2.0f, 2) - pow(cells[3] - (comperator.cells[3] + comperator.cells[6]) / 2.0f, 2);
                        if(!plaine) res += pow(cells[4] - (comperator.cells[4] + comperator.cells[7]) / 2.0f, 2);
                        return sqrt(res);
                    }

                    case tPoint:
                    case tEllipse:
                    case tCircle:
                    case tSquare:
                    {
                        ito::float32 res = pow(comperator.cells[2] - cells[2], 2) - pow(comperator.cells[3] - cells[3], 2);
                        if(!plaine) res += pow(comperator.cells[4] - cells[4], 2);
                        return sqrt(res);
                    }
                }
            }

            case tLine:
            return std::numeric_limits<float32>::quiet_NaN();

            case tRectangle:
            {
                switch(PrimitiveBase::extractType(comperator))
                {
                    default:
                    case tNoType:
                    case tMultiPointPick:
                    case tPolygon:
                        return std::numeric_limits<float32>::quiet_NaN();
                    break;

                    case tLine:
                        std::cout << "Not implemented yet" << std::endl;
                    return std::numeric_limits<float32>::quiet_NaN();
                    
                    case tRectangle:
                    {                      
                        ito::float32 res = pow((comperator.cells[2] + comperator.cells[4]) / 2.0f - (cells[2] + cells[5]) / 2.0f, 2) - pow((comperator.cells[3] + comperator.cells[6]) / 2.0f - (cells[3] + cells[6]) / 2.0f, 2);
                        if(!plaine) res += pow((comperator.cells[4] + comperator.cells[7]) / 2.0f - (cells[4] + cells[7]) / 2.0f, 2);
                        return sqrt(res);
                    }

                    case tPoint:
                    case tEllipse:
                    case tCircle:
                    case tSquare:
                    {
                        ito::float32 res = pow(comperator.cells[2] - (cells[2] + cells[5]) / 2.0f, 2) - pow(comperator.cells[3] - (cells[3] + cells[6]) / 2.0f, 2);
                        if(!plaine) res += pow(comperator.cells[4] - (cells[4] + cells[7]) / 2.0f, 2);
                        return sqrt(res);
                    }
                }
            }    
        }
    }
    //----------------------------------------------------------------------------------------------------------------------------------
}