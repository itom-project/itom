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

#include "sharedStructuresGraphics.h"

//----------------------------------------------------------------------------------------------------------------------------------
inline uchar saturate_cast(float v)
{ 
    int iv = (int)(v + (v >= 0 ? 0.5 : -0.5));
    return (uchar)((unsigned)iv <= UCHAR_MAX ? iv : iv > 0 ? UCHAR_MAX : 0);
}

namespace ito
{
//----------------------------------------------------------------------------------------------------------------------------------
ItomPalette::ItomPalette(const ItomPalette & scr)
{
    m_name = scr.m_name;
    m_type = scr.m_type;
    colorStops.clear();
    colorStops = scr.colorStops;
}
//----------------------------------------------------------------------------------------------------------------------------------
inline int ItomPalette::findUpper( double pos ) const
{
    // This code is copied from QWT-PLOT.
    int index = 0;

    //int n = _stops.size();
    int n = colorStops.size();

    //const ColorStop *stops = _stops.data();

    while ( n > 0 )
    {
        const int half = n >> 1;
        const int middle = index + half;

        if ( colorStops[middle].first <= pos )
        //if ( stops[middle].pos <= pos )
        {
            index = middle + 1;
            n -= half + 1;
        }
        else
            n = half;
    }

    return index;
}
//----------------------------------------------------------------------------------------------------------------------------------
void ItomPalette::insertColorStop( double pos, const QColor color )
{
    // This code is copied from QWT-PLOT.
    // Lookups need to be very fast, insertions are not so important.
    // Anyway, a balanced tree is what we need here. TODO ...

    if(m_type & ReadOnlyPalette)
    {
        //qDebug() << "ItomPalette insertColorStop. Tried to write to a readonly palette. ";
        return;
    }
    if ( pos < 0.0 || pos > 1.0 )
    {
        //qDebug() << "ItomPalette insertColorStop. Position out of range [0..1]. ";
        return;
    }
    int index;
    if ( colorStops.size() == 0 )
    //if ( _stops.size() == 0 )
    {
        index = 0;
        //_stops.resize( 1 );  
        colorStops.resize(1);
    }
    else
    {
        index = findUpper( pos );
        //if ( index == _stops.size() || qAbs( _stops[index].pos - pos ) >= 0.001 )
        if ( index == colorStops.size() || qAbs( colorStops[index].first - pos ) >= 0.001 )
        {
            //_stops.resize( _stops.size() + 1 );
            //for ( int i = _stops.size() - 1; i > index; i-- )
            //    _stops[i] = _stops[i-1];
            colorStops.resize( colorStops.size() + 1 );
            for ( int i = colorStops.size() - 1; i > index; i-- )
                colorStops[i] = colorStops[i-1];
        }   
    }
    //_stops[index] = ColorStop( pos, color );
    colorStops[index].first = pos;
    colorStops[index].second = color;
}
//----------------------------------------------------------------------------------------------------------------------------------
double ItomPalette::getPos(unsigned int color) const
{
    return colorStops[color].first;
}
//----------------------------------------------------------------------------------------------------------------------------------
QColor ItomPalette::getColor(unsigned int color) const
{
    return colorStops[color].second;
}
//----------------------------------------------------------------------------------------------------------------------------------
QVector<ito::uint32> ItomPalette::get256Colors() const
{
    QVector<ito::uint32> colors(256);
    
    int curIdx = 0;
    float pos = 0.0;

    float offsetR = colorStops[curIdx].second.red();
    float offsetG = colorStops[curIdx].second.green();
    float offsetB = colorStops[curIdx].second.blue();

    unsigned char rVal = 0.0;
    unsigned char gVal = 0.0;
    unsigned char bVal = 0.0;

    colors[0] = ((unsigned int)colorStops[curIdx].second.blue());
    colors[0] += ((unsigned int)colorStops[curIdx].second.green()) << 8;
    colors[0] += ((unsigned int)colorStops[curIdx].second.red()) <<16;

    colors[255] = ((unsigned int)colorStops[colorStops.size()-1].second.blue());
    colors[255] += ((unsigned int)colorStops[colorStops.size()-1].second.green()) << 8;
    colors[255] += ((unsigned int)colorStops[colorStops.size()-1].second.red()) <<16;

    for(int i = 1; i < 255; i++)
    {
        pos = i / 255.0;
        if((curIdx < colorStops.size()-2) && (pos > colorStops[curIdx+1].first))
        {
            curIdx++;
            offsetR = colorStops[curIdx].second.red();
            offsetG = colorStops[curIdx].second.green();
            offsetB = colorStops[curIdx].second.blue();
        }

        bVal = saturate_cast(((float)colorStops[curIdx+1].second.blue() - (float)colorStops[curIdx].second.blue())/(colorStops[curIdx+1].first - colorStops[curIdx].first) * (pos - colorStops[curIdx].first) + offsetB);
        gVal = saturate_cast(((float)colorStops[curIdx+1].second.green() - (float)colorStops[curIdx].second.green())/(colorStops[curIdx+1].first - colorStops[curIdx].first) * (pos - colorStops[curIdx].first) + offsetG);
        rVal = saturate_cast(((float)colorStops[curIdx+1].second.red() - (float)colorStops[curIdx].second.red())/(colorStops[curIdx+1].first - colorStops[curIdx].first) * (pos - colorStops[curIdx].first) + offsetR);

        colors[i] = ((unsigned int)bVal);
        colors[i] += ((unsigned int)gVal) << 8;
        colors[i] += ((unsigned int)rVal) <<16;
    }

    return colors;
}

//----------------------------------------------------------------------------------------------------------------------------------

}  // namespace ito