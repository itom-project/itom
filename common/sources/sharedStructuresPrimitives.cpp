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

namespace ito
{
    //----------------------------------------------------------------------------------------------------------------------------------
    PrimitiveContainer::PrimitiveContainer(ito::DataObject primitives)
    {
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