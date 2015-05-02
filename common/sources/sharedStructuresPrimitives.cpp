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
#ifdef USE_DEPRECIATED_ITOM_GEOMETRIC_ELEMS
namespace ito
{
    //----------------------------------------------------------------------------------------------------------------------------------
    PrimitiveContainer::PrimitiveContainer(ito::DataObject primitives)
    {
        std::cout << "PrimitiveContainer class is depreciated and will be removed within the next version of itom" << std::endl;
        int newSize = 64;
        int cols = 11;

#if (defined _DEBUG) && (defined WIN32)
        if (primitives.getDims() != 0 && primitives.getSize(primitives.getDims()-1) < 11)
        {
            cv::error(cv::Exception(CV_StsAssert, "Error, primitives object not valid.", "", __FILE__, __LINE__));
        }
#endif

        if (primitives.getDims() == 2 || primitives.calcNumMats() == 1)
        {
            newSize = newSize * (primitives.getSize(primitives.getDims() - 2) / 64 + 1);

            m_primitives.zeros(newSize, 11, ito::tFloat32);

            cv::Mat* scr = (cv::Mat*)(primitives.get_mdata()[primitives.seekMat(0)]);
            cv::Mat* dst = (cv::Mat*)(m_primitives.get_mdata()[0]);

            for (int i = 0; i < scr->rows; i++)
            {
                memcpy(dst->ptr<ito::float32>(i), scr->ptr<ito::float32>(i), sizeof(ito::float32) * cols);
            }       
        }
        else
        {
            m_primitives.zeros(64, 11, ito::tFloat32);
        }

        m_internalMat = (cv::Mat*)(m_primitives.get_mdata()[0]);
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    PrimitiveContainer::~PrimitiveContainer()
    {
        m_internalMat = NULL;
        return;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    int PrimitiveContainer::getNumberOfElements(const int type) const
    {
        int val = 0;
        if (type == -1)
        {
            for (int i = 0; i < m_internalMat->rows; i++)
            {
                if (((int)(m_internalMat->ptr<float32>(i)[0]) & 0x0000FFFF) != 0)
                {
                    val++;
                }
            }
        }
        else
        {
            for (int i = 0; i < m_internalMat->rows; i++)
            {
                if (((int)(m_internalMat->ptr<float32>(i)[1]) & 0x0000FFFF) == type)
                {
                    val++;
                }
            }        
        }
        return val;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    int PrimitiveContainer::getFirstElementRow(const int type) const
    {
        for (int i = 0; i < m_internalMat->rows; i++)
        {
            if (((int)(m_internalMat->ptr<float32>(i)[1]) & 0x0000FFFF) == type)
            {
                return i;
            }
        }
        return -1;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::float32* PrimitiveContainer::getElementPtr(const int row)
    {
        if (row > -1 && row < m_internalMat->rows)
        {
            return m_internalMat->ptr<float32>(row);
        }
        return NULL;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    const ito::float32* PrimitiveContainer::getElementPtr(const int row) const
    {
        if (row > -1 && row < m_internalMat->rows)
        {
            return m_internalMat->ptr<const float32>(row);
        }
        return NULL;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    int PrimitiveContainer::getIndexFromRow(const int row) const
    {
        if (row > -1 && row < m_internalMat->rows)
        {
            return (int)(m_internalMat->ptr<float32>(row)[0]);       
        }
        return false;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    int PrimitiveContainer::getRowFromIndex(const int idx) const
    {
        for (int i = 0; i < m_internalMat->rows; i++)
        {
            if ((int)(m_internalMat->ptr<float32>(i)[0]) == idx)
            {
                return i;
            }
        }
        return -1;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    void PrimitiveContainer::clear(void)
    {
        memset(m_internalMat->ptr(), 0, m_internalMat->rows * 11);
        return;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    bool PrimitiveContainer::isElement(const int row) const
    {
        if (row > -1 && row < m_internalMat->rows)
        {
            if (m_internalMat->ptr<float32>(row)[1] > 0.0)
            {
                return true;
            }
        
        }
        return false;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal PrimitiveContainer::addElement(const int type, ito::float32 * cells)
    {
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal PrimitiveContainer::changeElement(const int type, ito::float32 * cells)
    {
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal PrimitiveContainer::removeElement(const int row)
    {
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal PrimitiveContainer::copyGeometricElements(const ito::DataObject &rhs)
    {
        bool newObject = false;
        int cols = 11;

        if (rhs.getDims() == 0 || rhs.calcNumMats() != 1)
        {
            return ito::retError;
        }

        if (rhs.getSize(rhs.getDims()-1) > m_primitives.getSize(1))
        {
            int newSize = 64 * (rhs.getSize(rhs.getDims() - 2) / 64 + 1);
            m_primitives.zeros(newSize, 11, ito::tFloat32);
            m_internalMat = (cv::Mat*)(m_primitives.get_mdata()[0]);
            newObject = true;
        }

        
        cols = cols > rhs.getSize(rhs.getDims()-1) ? cols : rhs.getSize(rhs.getDims() - 1);

        cv::Mat* scr = (cv::Mat*)(rhs.get_mdata()[rhs.seekMat(0)]);
        cv::Mat* dst = (cv::Mat*)(m_primitives.get_mdata()[0]);

        for (int i = 0; i < scr->rows; i++)
        {
            memcpy(dst->ptr<ito::float32>(i), scr->ptr<ito::float32>(i), sizeof(ito::float32) * cols);
        }

        if (!newObject)
        {
            for (int i = scr->rows; i < dst->rows; i++)
            {
                memset(dst->ptr<ito::float32>(i), 0, sizeof(ito::float32) * 11);
            }
        }
        return ito::retOk;
    }
}
#else
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
#endif