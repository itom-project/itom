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

#ifndef SHAREDSTRUCTURESGRAPHICS_H
#define SHAREDSTRUCTURESGRAPHICS_H

#include "typeDefs.h"

#include <qstring.h>
#include <qcolor.h>
#include <qpair.h>
#include <qvector.h>


namespace ito
{
    enum PlotDataType
    {
        DataObjLine         = 0x0001,
        DataObjPlane        = 0x0002,
        DataObjPlaneStack   = 0x0004,
        PointCloud          = 0x0008,
        PolygonMesh         = 0x0010
    };
    Q_DECLARE_FLAGS(PlotDataTypes, PlotDataType)
    
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
        PluginInfo(PlotDataTypes plotDataTypes, PlotDataFormats plotDataFormats, PlotFeatures plotFeatures) : m_plotDataTypes(plotDataTypes), m_plotDataFormats(plotDataFormats), m_plotFeatures(plotFeatures) {}

        PlotDataTypes m_plotDataTypes;
        PlotDataFormats m_plotDataFormats;
        PlotFeatures m_plotFeatures;
};

class ItomPalette
{
    public:

        ItomPalette():m_name(""), m_type(0){colorStops.clear();};
        ItomPalette(const QString name, const char type): m_name(name), m_type(type) {colorStops.clear();};
        ItomPalette(const QString name, const char type, QColor start, QColor stop): m_name(name), m_type(type) 
        {
            colorStops.clear();
            colorStops.append(QPair<double, QColor>(0.0, start));
            colorStops.append(QPair<double, QColor>(1.0, stop));
        };

        ItomPalette(const ItomPalette & scr);
        ~ItomPalette() {colorStops.clear();};

        enum tPalette{
            NoType      = 0x00,
            GrayPalette = 0x01,
            RGBPalette  = 0x02,
            FCPalette  = 0x04,
            indexPalette = 0x08,
            LinearPalette = 0x10,
            ReadOnlyPalette = 0x20,
        };

        QString getName() const {return m_name;};
        inline int getSize() const {return colorStops.size();};
        inline int getType() const {return m_type;};

        double getPosFirst() const {return colorStops[0].first;};
        double getPosLast() const {return colorStops[colorStops.size()-1].first;};
        double getPos(unsigned int color) const;
        
        QColor getColorFirst() const {return colorStops[0].second;};
        QColor getColorLast() const {return colorStops[colorStops.size()-1].second;};
        QColor getColor(unsigned int color) const;

        inline void setWriteProtection() { m_type = m_type | ReadOnlyPalette; return;};
        void insertColorStop( double pos, const QColor color );
        //QColor getColor(double pos) const;

        QVector<ito::uint32> get256Colors() const;

    protected:
        inline void removeWriteProtection() { m_type = m_type & !ReadOnlyPalette; return;};

    private:
        QString m_name;
        char m_type; 
        
        QVector<QPair<double, QColor> > colorStops;

        int findUpper( double pos ) const;

    public slots:

    private slots:

    signals:

};

}

Q_DECLARE_OPERATORS_FOR_FLAGS ( ito::PlotDataTypes )
Q_DECLARE_OPERATORS_FOR_FLAGS ( ito::PlotFeatures )
Q_DECLARE_OPERATORS_FOR_FLAGS ( ito::PlotDataFormats )


#endif //SHAREDSTRUCTURESGRAPHICS_H
