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

#ifndef SHAREDSTRUCTURESGRAPHICS_H
#define SHAREDSTRUCTURESGRAPHICS_H

#include "typeDefs.h"

#include <qstring.h>
#include <QtGui/qcolor.h>
#include <qpair.h>
#include <qvector.h>

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{
    //! enumeration that describes the type of input data that a designer plot widget accepts
    enum PlotDataType
    {
        DataObjLine         = 0x0001, /*!< 1xN or Nx1 data object */
        DataObjPlane        = 0x0002, /*!< 2D data object */
        DataObjPlaneStack   = 0x0004, /*!< data object with more than one plane */
        PointCloud          = 0x0008, /*!< point cloud */
        PolygonMesh         = 0x0010  /*!< polygon mesh */
    };
    Q_DECLARE_FLAGS(PlotDataTypes, PlotDataType)

    //! allowed data formats
    enum PlotDataFormat
    {
        Format_Gray8    = 0x0001,
        Format_Gray16   = 0x0002,
        Format_Gray32   = 0x0004,
        Format_RGB32    = 0x0008,
        Format_ARGB32   = 0x0010,
        Format_CMYK32   = 0x0020,
        Format_Float32  = 0x0040,
        Format_Float64  = 0x0080,
        Format_Complex  = 0x0100
    };
    Q_DECLARE_FLAGS(PlotDataFormats, PlotDataFormat)

    //! this enumeration describe the features a designer plot widget provides concerning the visualization of data objects, point clouds or polygonal meshes
    enum PlotFeature
    {
        Static      = 0x0001,
        Live        = 0x0002,
        Cartesian   = 0x0004,
        Polar       = 0x0008,
        Cylindrical = 0x0010,
        OpenGl      = 0x0020,
        Cuda        = 0x0040,
        X3D         = 0x0080,
        PlotLine    = 0x0100,
        PlotImage   = 0x0200,
        PlotISO     = 0x0400,
        Plot3D      = 0x0800
    };
    Q_DECLARE_FLAGS(PlotFeatures, PlotFeature)


    class PluginInfo
    {
        public:
            PluginInfo(void) : m_plotFeatures(Static) {}
            PluginInfo(PlotDataTypes plotDataTypes, PlotDataFormats plotDataFormats, PlotFeatures plotFeatures)
                : m_plotDataTypes(plotDataTypes),
                m_plotDataFormats(plotDataFormats),
                m_plotFeatures(plotFeatures)
            {}

            PlotDataTypes m_plotDataTypes;
            PlotDataFormats m_plotDataFormats;
            PlotFeatures m_plotFeatures;
    };

    enum tPalette
    {
        tPaletteNoType      = 0x00,
        tPaletteGray        = 0x01,
        tPaletteRGB         = 0x02,
        tPaletteFC          = 0x04,
        tPaletteIndexed     = 0x08,
        tPaletteLinear      = 0x10,
        tPaletteReadOnly    = 0x20
    };

    struct ItomPalette
    {
        ItomPalette() : type(0), name("") {}
        int type;
        QString name;
        QVector<QPair<double, QColor> > colorStops;
        QVector<ito::uint32> colorVector256; //corresponds to QVector<QRgb>
        QColor inverseColorOne;
        QColor inverseColorTwo;
        QColor invalidColor;
    };

}

Q_DECLARE_OPERATORS_FOR_FLAGS ( ito::PlotDataTypes )
Q_DECLARE_OPERATORS_FOR_FLAGS ( ito::PlotFeatures )
Q_DECLARE_OPERATORS_FOR_FLAGS ( ito::PlotDataFormats )

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif //SHAREDSTRUCTURESGRAPHICS_H
