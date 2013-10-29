/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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
    PrimitiveContainer::PrimitiveContainer(ito::DataObject primitives)
    {
        int newSize = 64;

#ifdef _DEBUG
        if(primitives.getDims() != 0 && primitives.getSize(primitives.getDims()-1) < 11)
        {
            cv::error(cv::Exception(CV_StsAssert, "Error, primitives object not valid.", "", __FILE__, __LINE__));
        }

#endif

        if((primitives.getDims() == 2 || primitives.calcNumMats() == 1) && (primitives.getSize(primitives.getDims()-1) > 10))
        {
            newSize = newSize * (primitives.getSize(primitives.getDims() - 2) / 64 + 1);

            m_primitives.zeros(newSize, 11, ito::tFloat32);

            cv::Mat* scr = (cv::Mat*)(primitives.get_mdata()[primitives.seekMat(0)]);
            cv::Mat* dst = (cv::Mat*)(m_primitives.get_mdata()[0]);

            for(int i = 0; i < scr->rows; i++)
            {
                memcpy(dst->ptr<ito::float32>(i), scr->ptr<ito::float32>(i), sizeof(ito::float32) * 11);
            }       
        }
        else
        {
            m_primitives.zeros(64, 11, ito::tFloat32);
        }

        m_internalMat = (cv::Mat*)(m_primitives.get_mdata()[0]);
    }

    PrimitiveContainer::~PrimitiveContainer()
    {
        m_internalMat = NULL;
        return;
    }
}