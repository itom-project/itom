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

//#ifndef SHAREDSTRUCTURESPRIMITIVES_H
//#define SHAREDSTRUCTURESPRIMITIVES_H
//
//#include "typeDefs.h"
//#include "../DataObject/dataobj.h"
//
//#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib
//
//#define PRIM_ELEMENTLENGTH 11 /** \brief number of elements within the geometricPrimitives */
//
//
///** \struct geometricPrimitives
//*   \brief This union was defined for adressing geometricPrimitives.
//    \detail The union geometricPrimitives contains an array called cells with the size of PRIM_ELEMENTLENGTH.
//    The cells contain:
//    0. The unique index of the current primitive, castable to int32 with a maximum up to 16bit index values
//    1. Type flag 0000FFFF and further flags e.g. read&write only FFFF0000
//    2. First coordinate with x value
//    3. First coordinate with y value
//    4. First coordinate with z value
//    All other values depends on the primitiv type and may change between each type.
//    A point is defined as idx, flags, centerX0, centerY0, centerZ0
//    A line is defined as idx, flags, x0, y0, z0, x1, y1, z1
//    A ellipse is defined as idx, flags, centerX, centerY, centerZ, r1, r2
//    A circle is defined as idx, flags, centerX, centerY, centerZ, r
//    A rectangle is defined as idx, flags, x0, y0, z0, x1, y1, z1, alpha
//    A square is defined as idx, flags, centerX, centerY, centerZ, a, alpha
//    A polygon is defined as idx, flags, posX, posY, posZ, directionX, directionY, directionZ, idx, numIdx
//    \author Wolfram Lyda, twip optical solutions GmbH, Stuttgart
//    \date   12.2013
//*/
//
//struct ITOMCOMMONQT_EXPORT geometricPrimitives
//{
//
///*
//    struct point
//    {
//        ito::float32 idx;
//        ito::float32 flags;
//        ito::float32 x0;
//        ito::float32 y0;
//        ito::float32 z0;
//    };
//
//    struct line
//    {
//        ito::float32 idx;
//        ito::float32 flags;
//        ito::float32 x0;
//        ito::float32 y0;
//        ito::float32 z0;
//        ito::float32 x1;
//        ito::float32 y1;
//        ito::float32 z1;
//    };
//
//    struct elipse
//    {
//        ito::float32 idx;
//        ito::float32 flags;
//        ito::float32 centerX;
//        ito::float32 centerY;
//        ito::float32 centerZ;
//        ito::float32 r1;
//        ito::float32 r2;
//        ito::float32 alpha;
//    };
//
//    struct circle
//    {
//        ito::float32 idx;
//        ito::float32 flags;
//        ito::float32 centerX;
//        ito::float32 centerY;
//        ito::float32 centerZ;
//        ito::float32 r1;
//    };
//
//    struct retangle
//    {
//        ito::float32 idx;
//        ito::float32 flags;
//        ito::float32 x0;
//        ito::float32 y0;
//        ito::float32 z0;
//        ito::float32 x1;
//        ito::float32 y1;
//        ito::float32 z1;
//        ito::float32 alpha;
//    };
//
//    struct square
//    {
//        ito::float32 idx;
//        ito::float32 flags;
//        ito::float32 centerX;
//        ito::float32 centerY;
//        ito::float32 centerZ;
//        ito::float32 a;
//        ito::float32 alpha;
//    };
//
//    struct polygoneElement
//    {
//        ito::float32 idx;
//        ito::float32 flags;
//        ito::float32 x0;
//        ito::float32 y0;
//        ito::float32 z0;
//        ito::float32 directionX;
//        ito::float32 directionY;
//        ito::float32 directionZ;
//        ito::float32 pointIdx;
//        ito::float32 pointNumber;
//    };
//*/
//    ito::float32 cells[PRIM_ELEMENTLENGTH];
//};
//
//
//
//namespace ito
//{
//
//    /** \class PrimitiveContainer
//    *   \detail This is a container to store geometric primitives.
//                The enum tPrimitive of this file defines the geometric primitives for all plots.
//        \author Wolfram Lyda, twip optical solutions GmbH, Stuttgart
//        \date   12.2013
//    */
//    class ITOMCOMMONQT_EXPORT PrimitiveContainer 
//    {
//    public:
//
//        /** \enum tPrimitive
//        *   \detail Discribes the different primtive types
//            \sa itom1DQwtPlot, itom2DQwtPlot
//        */
//        enum tPrimitive
//        {
//            tNoType           =   0,            /**! NoType for pick*/
//            tMultiPointPick   =   6,            /**! Multi point pick*/
//            tPoint            =   101,          /**! Element is tPoint or order to pick points*/
//            tLine             =   102,          /**! Element is tLine or order to pick lines*/
//            tRectangle        =   103,          /**! Element is tRectangle or order to pick rectangles*/
//            tSquare           =   104,          /**! Element is tSquare or order to pick squares*/
//            tEllipse          =   105,          /**! Element is tEllipse or order to pick ellipses*/
//            tCircle           =   106,          /**! Element is tCircle or order to pick circles*/
//            tPolygon          =   110,          /**! Element is tPolygon or order to pick polygon*/
//            tMoveLock         =   0x00010000,   /**! Element is readOnly */
//            tRotateLock       =   0x00020000,   /**! Element can not be moved */
//            tResizeLock       =   0x00040000,   /**! Element can not be moved */
//            tTypeMask         =   0x0000FFFF,   /**! Mask for the type space */
//            tFlagMask         =   0xFFFF0000    /**! Mask for the flag space */
//        };
//
///** \cond HIDDEN_SYMBOLS */
//        PrimitiveContainer(DataObject primitives = DataObject());
//        ~PrimitiveContainer();
//
//
//
//        inline int getNumberOfRows() const {return m_primitives.getSize(0);};
//        inline int getNumberOfElements(const int type) const;
//        inline int getFirstElementRow(const int type) const;
//        inline ito::float32* getElementPtr(const int row);
//        inline const ito::float32* getElementPtr(const int row) const;
//        inline int getIndexFromRow(const int row) const;
//        inline int getRowFromIndex(const int idx) const;
//
//        inline bool isElement(const int row) const;
//        void clear(void);
//
//        ito::RetVal addElement(const int type, ito::float32 * cells);
//        ito::RetVal changeElement(const int type, ito::float32 * cells);
//        ito::RetVal removeElement(const int row);
//
//        ito::RetVal copyGeometricElements(const ito::DataObject &rhs);
//
//    private:
//
//        ito::DataObject m_primitives;
//        cv::Mat * m_internalMat;
///** \endcond */
//
//    };
//
//
//}
//
//#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)
//
//#endif //SHAREDSTRUCTURESPRIMITIVES_H