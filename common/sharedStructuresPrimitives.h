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

#ifndef SHAREDSTRUCTURESPRIMITIVES_H
#define SHAREDSTRUCTURESPRIMITIVES_H

#include "typeDefs.h"
#include "../DataObject/dataobj.h"

#define PRIM_ELEMENTLENGTH 11

union geometricPrimitives
{
    struct point
    {
        ito::float32 idx;
        ito::float32 flags;
        ito::float32 x0;
        ito::float32 y0;
        ito::float32 z0;
    };

    struct line
    {
        ito::float32 idx;
        ito::float32 flags;
        ito::float32 x0;
        ito::float32 y0;
        ito::float32 z0;
        ito::float32 x1;
        ito::float32 y1;
        ito::float32 z1;
    };

    struct elipse
    {
        ito::float32 idx;
        ito::float32 flags;
        ito::float32 centerX;
        ito::float32 centerY;
        ito::float32 centerZ;
        ito::float32 r1;
        ito::float32 r2;
        ito::float32 alpha;
    };

    struct circle
    {
        ito::float32 idx;
        ito::float32 flags;
        ito::float32 centerX;
        ito::float32 centerY;
        ito::float32 centerZ;
        ito::float32 r1;
    };

    struct retangle
    {
        ito::float32 idx;
        ito::float32 flags;
        ito::float32 x0;
        ito::float32 y0;
        ito::float32 z0;
        ito::float32 x1;
        ito::float32 y1;
        ito::float32 z1;
        ito::float32 alpha;
    };

    struct square
    {
        ito::float32 idx;
        ito::float32 flags;
        ito::float32 centerX;
        ito::float32 centerY;
        ito::float32 centerZ;
        ito::float32 a;
        ito::float32 alpha;
    };

    struct polygoneElement
    {
        ito::float32 idx;
        ito::float32 flags;
        ito::float32 x0;
        ito::float32 y0;
        ito::float32 z0;
        ito::float32 directionX;
        ito::float32 directionY;
        ito::float32 directionZ;
        ito::float32 pointIdx;
        ito::float32 pointNumber;
    };

    ito::float32 cells[PRIM_ELEMENTLENGTH];
};

namespace ito
{
    class PrimitiveContainer 
    {
    public:
        PrimitiveContainer(DataObject primitives = DataObject());
        ~PrimitiveContainer();

        enum tPrimitive
        {
            tNoType     =   0,
            tPoint      =   1,
            tLine       =   2,
            tElipse     =   3,
            tCircle     =   4,
            tRetangle   =   5,
            tSquare     =   6,
            tPolygon    =   10
        };

        inline int getNumberOfRows() const {return m_primitives.getSize(0);};
        inline int getNumberOfElements(const int type) const;
        inline int getFirstElementRow(const int type) const;
        inline ito::float32* getElementPtr(const int row);
        inline const ito::float32* getElementPtr(const int row) const;
        inline int getIndexFromRow(const int row) const;
        inline int getRowFromIndex(const int idx) const;

        inline bool isElement(const int row) const;
        void clear(void);

        ito::RetVal addElement(const int type, ito::float32 * cells);
        ito::RetVal changeElement(const int type, ito::float32 * cells);
        ito::RetVal removeElement(const int row);

        ito::RetVal copyGeometricElements(const ito::DataObject &rhs);

    private:

        ito::DataObject m_primitives;
        cv::Mat * m_internalMat;
    };


}
#endif //SHAREDSTRUCTURESPRIMITIVES_H