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
#include "DataObject/dataObjectFuncs.h"
#include <iostream>

#ifdef _USE_MATH_DEFINES
    #include <math.h>
#else
    #define _USE_MATH_DEFINES
    #include <math.h>
    #undef _USE_MATH_DEFINES
#endif
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
    void PrimitiveBase::setIndex( const int idx ) 
    {
        cells[0] = (float32)idx;
    } 
    //----------------------------------------------------------------------------------------------------------------------------------
    void PrimitiveBase::setFlags( const int flags ) 
    {
        cells[1] = ( float32 )( ( ( (int)(cells[1]) ) & tGeoTypeMask ) | ( flags & tGeoFlagMask ));
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void PrimitiveBase::setType( const int type ) 
    {
        cells[1] = ( float32 )( ( ( (int)(cells[1]) ) & tGeoFlagMask ) | ( type & tGeoTypeMask ));
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void PrimitiveBase::setTypeAndFlags( const int val ) 
    {
        cells[1] = (float32)val;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32* PrimitiveBase::ptr_data() 
    {
        return cells;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32* PrimitiveBase::ptr_geo()
    {
        return &(cells[2]);
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 PrimitiveBase::distanceLineToPoint(const PrimitiveBase &line, const PrimitiveBase &point, const bool plaine /*= false*/)
    {

        float32 cSq = pow(point.cells[2] - line.cells[2], 2) + pow(point.cells[3] - line.cells[3], 2);
        float32 bSq = pow(line.cells[5] - line.cells[2], 2) + pow(line.cells[6] - line.cells[3], 2);
        if(!plaine)
        {
            cSq += pow(point.cells[4] - line.cells[4], 2);
            bSq += pow(line.cells[7] - line.cells[4], 2);
        }
        if(cSq - bSq < 0)
        {
            return std::numeric_limits<float32>::quiet_NaN();
        }
        return sqrt(cSq - bSq);
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 PrimitiveBase::distanceLineToPointRAW(const float32 *line, const float32 *point, const bool plaine /*= false*/)
    {

        float32 cSq = pow(point[0] - line[0], 2) + pow(point[1] - line[1], 2);
        float32 bSq = pow(line[0] - line[0], 2) + pow(line[3] - line[1], 2);
        if(!plaine)
        {
            cSq += pow(point[2] - line[2], 2);
            bSq += pow(line[5] - line[2], 2);
        }
        if(cSq - bSq < 0)
        {
            return std::numeric_limits<float32>::quiet_NaN();
        }
        return sqrt(cSq - bSq);
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 PrimitiveBase::distanceRectCenterToLine(const PrimitiveBase &rect, const PrimitiveBase &line, const bool plaine /*= false*/)
    {
        GeometricPrimitive point;
        memset(point.cells, 0, sizeof(float32)*PRIM_ELEMENTLENGTH);
        point.cells[2] = (rect.cells[2] + rect.cells[5]) / 2.0f;
        point.cells[3] = (rect.cells[3] + rect.cells[6]) / 2.0f;
        point.cells[4] = (rect.cells[4] + rect.cells[7]) / 2.0f;
        return distanceLineToPoint(line, point, plaine);
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 PrimitiveBase::distancePointToPoint(const PrimitiveBase &point1, const PrimitiveBase &point2, const bool plaine /*= false*/)
    {
        float32 res = pow(point1.cells[2] - point2.cells[2], 2) - pow(point1.cells[3] - point2.cells[3], 2);
        if(!plaine) res += pow(point1.cells[4] - point2.cells[4], 2);
        return sqrt(res);
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 PrimitiveBase::distancePointToPointRAW(const float32* point1, const float32* point2, const bool plaine /*= false*/)
    {
        float32 res = pow(point1[0] - point2[0], 2) - pow(point1[1] - point2[1], 2);
        if(!plaine) res += pow(point1[3] - point2[3], 2);
        return sqrt(res);
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 PrimitiveBase::distanceRectCenterToPoint(const PrimitiveBase &rect, const PrimitiveBase &point, const bool plaine /*= false*/)
    {
        GeometricPrimitive centerPoint;
        memset(centerPoint.cells, 0, sizeof(float32)*PRIM_ELEMENTLENGTH);
        centerPoint.cells[2] = (rect.cells[2] + rect.cells[5]) / 2.0f;
        centerPoint.cells[3] = (rect.cells[3] + rect.cells[6]) / 2.0f;
        centerPoint.cells[4] = (rect.cells[4] + rect.cells[7]) / 2.0f;
        return distancePointToPoint(centerPoint, point, plaine);
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void PrimitiveBase::normalVector(float32 direction[3]) const
    {
        direction[0] = 0;
        direction[1] = 0;
        direction[2] = 1;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 PrimitiveBase::area() const
    {
        return 0.0f;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitivePoint::normalVector(float32 direction[3]) const
    {
        direction[0] = 0;
        direction[1] = 0;
        direction[2] = 1;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 GeometricPrimitiveLine::length(const bool plaine /*= false*/) const
    {
        ito::float32 dir[3] = {0.0f,0.0f,1.0f};
        directionVector(dir);
        if(plaine)
        {
            return vectorLength2D(dir);
        }
        else
        {
            return vectorLength(dir);
        }
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitiveLine::normalVector(float32 direction[3]) const
    {
        ito::float32 base[3] = {0.0f,1.0f,0.0f};
        ito::float32 dir[3] = {1.0f,0.0f,0.0f};
        ito::float32 len = 1.0;
        directionVector(dir);
        vectorCross(dir, base, direction);

        if((len = vectorLength(direction)) < 0.001 )
        {
            base[0] = 1.0f;
            base[1] = 0.0f;
            base[2] = 0.0f;
            vectorCross(dir, base, direction);
            len = vectorLength(direction);           
        }
        if(ito::dObjHelper::isNotZero(len))
        {
            vectorScale(direction, 1.0f/len);
        }
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitiveCircle::normalVector(float32 direction[3]) const
    {
        direction[0] = 0.0f;
        direction[1] = 0.0f;
        direction[2] = 1.0f;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 GeometricPrimitiveCircle::area() const
    {
        return (float32)(M_PI * pow(cells[5], 2)); 
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitiveEllipse::normalVector(float32 direction[3]) const
    {
        direction[0] = 0.0f;
        direction[1] = 0.0f;
        direction[2] = 1.0f;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 GeometricPrimitiveEllipse::area() const
    {
        return (float32)(M_PI * (cells[5] / 2.0 * cells[6] / 2.0)); 
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitiveSquare::normalVector(float32 direction[3]) const
    {
        direction[0] = 0.0f;
        direction[1] = 0.0f;
        direction[2] = 1.0f;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitiveSquare::topLeft(float32 direction[3]) const
    {
#if _DEBUG
        throw std::invalid_argument("Calculation for alpha != 0 not implemented yet");
#endif
        direction[0] = 0.0f;
        direction[1] = 0.0f;
        direction[2] = 1.0f;           

    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitiveSquare::topRight(float32 direction[3]) const
    {
#if _DEBUG
        throw std::invalid_argument("Calculation for alpha != 0 not implemented yet");
#endif
        direction[0] = 0.0f;
        direction[1] = 0.0f;
        direction[2] = 1.0f;       

    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitiveSquare::bottomLeft(float32 direction[3]) const
    {
#if _DEBUG
        throw std::invalid_argument("Calculation for alpha != 0 not implemented yet");
#endif
        direction[0] = 0.0f;
        direction[1] = 0.0f;
        direction[2] = 1.0f;       

    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitiveSquare::bottomRight(float32 direction[3]) const
    {
#if _DEBUG
        throw std::invalid_argument("Calculation for alpha != 0 not implemented yet");
#endif
        direction[0] = 0.0f;
        direction[1] = 0.0f;
        direction[2] = 1.0f;       

    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 GeometricPrimitiveSquare::area() const
    {
        return cells[5] * cells[5]; 
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitiveRectangle::normalVector(float32 direction[3]) const
    {
        ito::float32 len = 1.0;
        if(ito::dObjHelper::isNotZero(alpha()))
        {
#if _DEBUG
            throw std::invalid_argument("Calculation for alpha != 0 not implemented yet");
#endif
            direction[0] = 0.0f;
            direction[1] = 0.0f;
            direction[2] = 1.0f;
        }
        else // if alpha is 0, the plane can be defined by dx/dz and dy/dz
        {

            float32 dir[3] = { cells[5] - cells[2], 0, cells[7] - cells[4]};
            float32 dir2[3] = { 0, cells[6] - cells[3], cells[7] - cells[4]};

            vectorCross(dir, dir2, direction);     
            len = vectorLength(direction); 
        }

        if(ito::dObjHelper::isNotZero(len))
        {
            vectorScale(direction, 1.0f/len);
        }
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitiveRectangle::topLeft(float32 direction[3]) const
    {
#if _DEBUG
        throw std::invalid_argument("Calculation for alpha != 0 not implemented yet");
#endif
        direction[0] = 0.0f;
        direction[1] = 0.0f;
        direction[2] = 1.0f;   

    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitiveRectangle::topRight(float32 direction[3]) const
    {
#if _DEBUG
        throw std::invalid_argument("Calculation for alpha != 0 not implemented yet");
#endif
        direction[0] = 0.0f;
        direction[1] = 0.0f;
        direction[2] = 1.0f;       

    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitiveRectangle::bottomLeft(float32 direction[3]) const
    {
#if _DEBUG
        throw std::invalid_argument("Calculation for alpha != 0 not implemented yet");
#endif
        direction[0] = 0.0f;
        direction[1] = 0.0f;
        direction[2] = 1.0f;       

    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void GeometricPrimitiveRectangle::bottomRight(float32 direction[3]) const
    {
#if _DEBUG
        throw std::invalid_argument("Calculation for alpha != 0 not implemented yet");
#endif
        direction[0] = 0.0f;
        direction[1] = 0.0f;
        direction[2] = 1.0f;       

    }
    //----------------------------------------------------------------------------------------------------------------------------------
    float32 GeometricPrimitiveRectangle::area() const
    {
        if(ito::dObjHelper::isNotZero(alpha()))
        {
#if _DEBUG
            throw std::invalid_argument("Calculation for alpha != 0 not implemented yet");
#endif
            return 0.0f;
        }
        else // if alpha is 0, the plane can be defined by dx/dz and dy/dz
        {

            float32 dir[3] = { cells[5] - cells[2], 0, cells[7] - cells[4]};
            float32 dir2[3] = { 0, cells[6] - cells[3], cells[7] - cells[4]};
  
            return vectorLength(dir) * vectorLength(dir2); 
        }
    }
}