/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "paletteOrganizer.h"
#include "../AppManagement.h"
#include <qsettings.h>

//----------------------------------------------------------------------------------------------------------------------------------
inline uchar saturate_cast(float v)
{
    int iv = (int)(v + (v >= 0 ? 0.5 : -0.5));
    return (uchar)((unsigned)iv <= UCHAR_MAX ? iv : iv > 0 ? UCHAR_MAX : 0);
}

//----------------------------------------------------------------------------------------------------------------------------------
namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      Find the next color stop and its values within this palette

/*! \detail     The palette itself is based on a small set of color stops.
                Between this stops, the color is linear interpolated.
                This function is used to give the next valid color stop for a position to allow such an interpolation.

    \param      pos     the position within the palette
    \return     int     the index of the next color stop
*/
inline int ItomPaletteBase::findUpper( double pos ) const
{
    // This code is copied from QWT-PLOT.
    int index = 0;

    //int n = _stops.size();
    int n = m_paletteData.colorStops.size();

    //const ColorStop *stops = _stops.data();

    while ( n > 0 )
    {
        const int half = n >> 1;
        const int middle = index + half;

        if (m_paletteData.colorStops[middle].first <= pos )
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
//! \brief      Set the first inverse color for this color bar
/*! \detail     Each colorbar has 2 inverse colors to highlight lines, cursers ...

    \param      color   new color as QColor value
    \return     bool    true if success
*/
bool ItomPaletteBase::setInverseColorOne(const QColor &color)
{
    if((m_paletteData.type & ito::tPaletteReadOnly) && m_paletteData.inverseColorOne.isValid())
    {
        return false;
    }

    m_paletteData.inverseColorOne = color;

    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      Set the second inverse color for this color bar
/*! \detail     Each colorbar has 2 inverse colors to highlight lines, cursers ...

    \param      color   new color as QColor value
    \return     bool    true if success
*/
bool ItomPaletteBase::setInverseColorTwo(const QColor &color)
{
    if ((m_paletteData.type & ito::tPaletteReadOnly) && m_paletteData.inverseColorTwo.isValid())
    {
        return false;
    }

    m_paletteData.inverseColorTwo = color;

    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool ItomPaletteBase::setName(const QString &name)
{
    if ((m_paletteData.type & ito::tPaletteReadOnly))
    {
        return false;
    }

    m_paletteData.name = name;

    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      Set the invalid color for this color bar
/*! \detail     Each colorbar has a invalid color for inf and NaN values

    \param      color   new color as QColor value
    \return     bool    true if success
*/
bool ItomPaletteBase::setInvalidColor(const QColor &color)
{
    if ((m_paletteData.type & ito::tPaletteReadOnly) && m_paletteData.invalidColor.isValid())
    {
        //qDebug() << "ItomPalette setInversColorTwo. Tried to write to a readonly palette. ";
        return false;
    }

    m_paletteData.invalidColor = color;

    return true;
}
//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      Get the invalid color for this color bar
/*! \detail     Each colorbar has a invalid color for inf and NaN values

    \param      color   new color as QColor value
    \return     bool    the invalid color or if not defined the first colorStop or else black
*/
QColor ItomPaletteBase::getInvalidColor() const
{
    return m_paletteData.invalidColor.isValid() ? m_paletteData.invalidColor :
        (m_paletteData.colorStops.size() > 0 ? QColor(m_paletteData.colorStops[0].second) : QColor(0, 0, 0));
}
//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      Insert a new color stop into the palette defined by color and position

/*! \detail     The palette itself is based on a small set of color stops.
                Between this stops, the color is linear interpolated.

    \param      pos     the position within the palette
    \param      color   new color as QColor value
    \return     bool    true if success
*/
bool ItomPaletteBase::insertColorStop( double pos, const QColor &color )
{
    // This code is copied from QWT-PLOT.
    // Lookups need to be very fast, insertions are not so important.
    // Anyway, a balanced tree is what we need here. TODO ...

    if (m_paletteData.type & ito::tPaletteReadOnly)
    {
        return false;
    }
    if ( pos < 0.0 || pos > 1.0 )
    {
        return false;
    }
    int index;
    if (m_paletteData.colorStops.size() == 0 )
    {
        index = 0;
        m_paletteData.colorStops.resize(1);
    }
    else
    {
        index = findUpper( pos );
        if ( index == m_paletteData.colorStops.size() || qAbs(m_paletteData.colorStops[index].first - pos ) >= 0.001 )
        {
            m_paletteData.colorStops.resize(m_paletteData.colorStops.size() + 1 );
            for ( int i = m_paletteData.colorStops.size() - 1; i > index; i-- )
            {
                m_paletteData.colorStops[i] = m_paletteData.colorStops[i-1];
            }
        }
    }
    m_paletteData.colorStops[index].first = pos;
    m_paletteData.colorStops[index].second = color;
    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool ItomPaletteBase::setColorStops(const QVector<QPair<double, QColor> > &colorStops)
{
    if (m_paletteData.type & ito::tPaletteReadOnly)
    {
        return false;
    }

    m_paletteData.colorStops = colorStops;
    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      Get the position of the n-th color-stop in the palette
/*! \detail     This function returns the position (double value) of the color stop defined by int color

    \param      color     index of the color to retrieve
    \return     double    position of the color stop
*/
double ItomPaletteBase::getPos(unsigned int color) const
{
    if(color > (unsigned int) (m_paletteData.colorStops.size() - 1))
        return m_paletteData.colorStops[m_paletteData.colorStops.size() - 1].first;
    return m_paletteData.colorStops[color].first;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      Get the RGBA-Color of the n-th color-stop in the palette
/*! \detail     This function returns the position (double value) of the color stop defined by int color

    \param      index     index of the color stop, whose color is returned
    \return     QColor    the RGBA-Color of the color stop
*/
QColor ItomPaletteBase::getColor(unsigned int index) const
{
    if(index > (unsigned int)(m_paletteData.colorStops.size() - 1))
        return m_paletteData.colorStops[m_paletteData.colorStops.size() - 1].second;
    return m_paletteData.colorStops[index].second;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      This function returns the internal structure of the palette
/*! \detail     This function returns the internal structure of the palette
*/
ItomPalette ItomPaletteBase::getPalette() const
{
    ItomPalette paletteOut = m_paletteData;
    paletteOut.colorVector256 = QVector<ito::uint32>(get256Colors());
    return paletteOut;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      Transform the color stops to a 256 color vector
/*! \detail     Transform the color stops to a 256 color vector

    \param      updateInverseColors     recalculate the ideals inverse color
*/
QVector<ito::uint32> ItomPaletteBase::get256Colors(bool includeAlpha) const
{
    QVector<ito::uint32> colors(256);
    //colors are defined like QRgb

    int curIdx = 0;
    float pos = 0.0;

    if (m_paletteData.colorStops.size() > 0) //check if at least one color stop is defined
    {
        float offsetR = m_paletteData.colorStops[curIdx].second.red();
        float offsetG = m_paletteData.colorStops[curIdx].second.green();
        float offsetB = m_paletteData.colorStops[curIdx].second.blue();
        float offsetA = m_paletteData.colorStops[curIdx].second.alpha();

        unsigned char rVal = 0;
        unsigned char gVal = 0;
        unsigned char bVal = 0;
        unsigned char alphaVal = 0;

        colors[0] = ((unsigned int)m_paletteData.colorStops[curIdx].second.blue());
        colors[0] += ((unsigned int)m_paletteData.colorStops[curIdx].second.green()) << 8;
        colors[0] += ((unsigned int)m_paletteData.colorStops[curIdx].second.red()) << 16;

        colors[255] = ((unsigned int)m_paletteData.colorStops[m_paletteData.colorStops.size()-1].second.blue());
        colors[255] += ((unsigned int)m_paletteData.colorStops[m_paletteData.colorStops.size()-1].second.green()) << 8;
        colors[255] += ((unsigned int)m_paletteData.colorStops[m_paletteData.colorStops.size()-1].second.red()) << 16;

        if(includeAlpha)
        {
            colors[0] += ((unsigned int)m_paletteData.colorStops[curIdx].second.alpha()) << 24;
            colors[255] += ((unsigned int)m_paletteData.colorStops[m_paletteData.colorStops.size()-1].second.alpha()) << 24;
        }

        for(int i = 1; i < 255; i++)
        {
            pos = i / 255.0;
            if((curIdx < m_paletteData.colorStops.size()-2) && (pos > m_paletteData.colorStops[curIdx+1].first))
            {
                curIdx++;
                offsetR = m_paletteData.colorStops[curIdx].second.red();
                offsetG = m_paletteData.colorStops[curIdx].second.green();
                offsetB = m_paletteData.colorStops[curIdx].second.blue();
                offsetA = m_paletteData.colorStops[curIdx].second.alpha();
            }

            bVal = saturate_cast(((float)m_paletteData.colorStops[curIdx+1].second.blue()
                - (float)m_paletteData.colorStops[curIdx].second.blue())
                / (m_paletteData.colorStops[curIdx+1].first - m_paletteData.colorStops[curIdx].first)
                * (pos - m_paletteData.colorStops[curIdx].first) + offsetB);
            gVal = saturate_cast(((float)m_paletteData.colorStops[curIdx+1].second.green()
                - (float)m_paletteData.colorStops[curIdx].second.green())
                / (m_paletteData.colorStops[curIdx+1].first
                - m_paletteData.colorStops[curIdx].first)
                * (pos - m_paletteData.colorStops[curIdx].first) + offsetG);
            rVal = saturate_cast(((float)m_paletteData.colorStops[curIdx+1].second.red()
                - (float)m_paletteData.colorStops[curIdx].second.red())
                / (m_paletteData.colorStops[curIdx+1].first
                - m_paletteData.colorStops[curIdx].first)
                * (pos - m_paletteData.colorStops[curIdx].first) + offsetR);

            alphaVal = saturate_cast(((float)m_paletteData.colorStops[curIdx+1].second.alpha()
                - (float)m_paletteData.colorStops[curIdx].second.alpha())
                / (m_paletteData.colorStops[curIdx+1].first
                - m_paletteData.colorStops[curIdx].first)
                * (pos - m_paletteData.colorStops[curIdx].first) + offsetA);

            colors[i] = ((unsigned int)bVal);
            colors[i] += ((unsigned int)gVal) << 8;
            colors[i] += ((unsigned int)rVal) <<16;

            if(includeAlpha)
            {
                colors[i] += ((unsigned int)alphaVal) <<24;
            }
        }
    }
    else
    {
        colors.fill(0);
    }

    return colors;
}

//----------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \return PaletteOrganizer
*/
PaletteOrganizer::PaletteOrganizer()
{
    m_colorPalettes.clear();
    m_restrictedKeyWords.clear();
    m_colorPaletteLUT.clear();

    m_restrictedKeyWords.append("");
    m_restrictedKeyWords.append("none");

    noPalette = ItomPaletteBase("none", 0);

    ItomPaletteBase newPalette;
    //QColor inv1, inv2;
    //declare "gray"
    newPalette = ItomPaletteBase("gray", ito::tPaletteGray | ito::tPaletteRGB | ito::tPaletteLinear | ito::tPaletteIndexed, QColor::fromRgb(0, 0, 0), QColor::fromRgb(255, 255, 255));
    newPalette.setInverseColorOne(Qt::blue);
    newPalette.setInverseColorTwo("#e31a1c"); //dark red
    newPalette.setWriteProtection();
    //set "gray"
    m_colorPalettes.append(newPalette);
    m_restrictedKeyWords.append("gray");
    m_builtInPalettes << "gray";
    //------------

    //declare "grayMarked"
    newPalette = ItomPaletteBase("grayMarked", ito::tPaletteGray | ito::tPaletteLinear | ito::tPaletteIndexed);

    // https://github.com/itom-project/designerPlugins/issues/39
    const double tol = 0.0000; //5;
    const double eps = std::numeric_limits<double>::epsilon();
    newPalette.insertColorStop(0.0, Qt::magenta);
    newPalette.insertColorStop(eps, Qt::magenta);
    newPalette.insertColorStop(eps, Qt::black);
    newPalette.insertColorStop(1.0 - eps, Qt::white);
    newPalette.insertColorStop(1.0 - eps, Qt::red);
    newPalette.insertColorStop(1.0, Qt::red);
    newPalette.setInverseColorOne(Qt::blue);
    newPalette.setInverseColorTwo(Qt::green);
    newPalette.setWriteProtection();
    //set "grayMarked"
    m_colorPalettes.append(newPalette);
    m_restrictedKeyWords.append("grayMarked");
    m_builtInPalettes << "grayMarked";
    //------------

    //declare "falseColor"
    newPalette = ItomPaletteBase("falseColor", ito::tPaletteFC | ito::tPaletteLinear | ito::tPaletteIndexed, Qt::red, QColor::fromRgb(165, 30, 165));
    newPalette.insertColorStop(0.2, Qt::yellow);
    newPalette.insertColorStop(0.4, Qt::green);
    newPalette.insertColorStop(0.6, Qt::cyan);
    newPalette.insertColorStop(0.8, Qt::blue);
    //newPalette.calculateInverseColors(inv1, inv2);
    newPalette.setInverseColorOne(Qt::black);
    newPalette.setInverseColorTwo(Qt::gray);
    newPalette.setWriteProtection();
    //set "falseColor"
    m_colorPalettes.append(newPalette);
    m_restrictedKeyWords.append("falseColor");
    m_builtInPalettes << "falseColor";
    //------------

    //declare "falseColorIR"
    newPalette = ItomPaletteBase("falseColorIR", ito::tPaletteFC | ito::tPaletteLinear | ito::tPaletteIndexed, QColor::fromRgb(165, 30, 165), Qt::red);
    newPalette.insertColorStop(0.8, Qt::yellow);
    newPalette.insertColorStop(0.6, Qt::green);
    newPalette.insertColorStop(0.4, Qt::cyan);
    newPalette.insertColorStop(0.2, Qt::blue);
    newPalette.setInverseColorOne(Qt::black);
    newPalette.setInverseColorTwo(Qt::gray);
    newPalette.setWriteProtection();
    //set "falseColorIR"
    m_colorPalettes.append(newPalette);
    m_restrictedKeyWords.append("falseColorIR");
    m_builtInPalettes << "falseColorIR";
    //------------

    //declare "hotIron"
    newPalette = ItomPaletteBase("hotIron", ito::tPaletteFC | ito::tPaletteLinear | ito::tPaletteIndexed, Qt::black, Qt::white);
    newPalette.insertColorStop(0.33, Qt::red);
    newPalette.insertColorStop(0.67, QColor::fromRgb(255, 129, 0));
    newPalette.setInverseColorOne(Qt::blue);
    newPalette.setInverseColorTwo(Qt::green);
    newPalette.setWriteProtection();
    //set "hotIron"
    m_colorPalettes.append(newPalette);
    m_restrictedKeyWords.append("hotIron");
    m_builtInPalettes << "hotIron";
    //------------

    //declare "red"
    newPalette = ItomPaletteBase(ItomPaletteBase("red", ito::tPaletteGray | ito::tPaletteLinear | ito::tPaletteIndexed, QColor::fromRgb(0, 0, 0), QColor::fromRgb(255, 0, 0)));
    newPalette.setInverseColorOne(Qt::blue);
    newPalette.setInverseColorTwo(Qt::green);
    newPalette.setWriteProtection();
    //set "red"
    m_colorPalettes.append(newPalette);
    m_restrictedKeyWords.append("red");
    m_builtInPalettes << "red";
    //------------

    //declare "blue"
    newPalette = ItomPaletteBase(ItomPaletteBase("blue", ito::tPaletteGray | ito::tPaletteLinear | ito::tPaletteIndexed, QColor::fromRgb(0, 0, 0), QColor::fromRgb(0, 0, 255)));
    newPalette.setInverseColorOne("#e31a1c"); //dark red
    newPalette.setInverseColorTwo(Qt::green);
    newPalette.setWriteProtection();
    //set "blue"
    m_colorPalettes.append(newPalette);
    m_restrictedKeyWords.append("blue");
    m_builtInPalettes << "blue";
    //------------

    //declare "green"
    newPalette = ItomPaletteBase(ItomPaletteBase("green", ito::tPaletteGray | ito::tPaletteLinear | ito::tPaletteIndexed, QColor::fromRgb(0, 0, 0), QColor::fromRgb(0, 255, 0)));
    newPalette.setInverseColorOne(Qt::blue);
    newPalette.setInverseColorTwo("#e31a1c"); //dark red
    newPalette.setWriteProtection();
    //set "green"
    m_colorPalettes.append(newPalette);
    m_restrictedKeyWords.append("green");
    m_builtInPalettes << "green";
    //------------

    //declare "viridis" as new default map of Matplotlib (only approximated, since non linear-map is currently not supported in itom.
    newPalette = ItomPaletteBase("viridis", ito::tPaletteFC | ito::tPaletteLinear | ito::tPaletteIndexed, QColor::fromRgb(68, 1, 84), QColor::fromRgb(253, 231, 37));
    newPalette.insertColorStop(0.05, QColor::fromRgb(72, 20, 103));
    newPalette.insertColorStop(0.10, QColor::fromRgb(72, 37, 118));
    newPalette.insertColorStop(0.15, QColor::fromRgb(70, 52, 128));
    newPalette.insertColorStop(0.20, QColor::fromRgb(65, 68, 135));
    newPalette.insertColorStop(0.25, QColor::fromRgb(59, 82, 139));
    newPalette.insertColorStop(0.30, QColor::fromRgb(52, 96, 141));
    newPalette.insertColorStop(0.35, QColor::fromRgb(47, 108, 142));
    newPalette.insertColorStop(0.40, QColor::fromRgb(42, 120, 142));
    newPalette.insertColorStop(0.45, QColor::fromRgb(37, 132, 142));
    newPalette.insertColorStop(0.50, QColor::fromRgb(33, 145, 140));
    newPalette.insertColorStop(0.55, QColor::fromRgb(30, 156, 137));
    newPalette.insertColorStop(0.60, QColor::fromRgb(34, 168, 132));
    newPalette.insertColorStop(0.65, QColor::fromRgb(47, 180, 124));
    newPalette.insertColorStop(0.70, QColor::fromRgb(68, 191, 112));
    newPalette.insertColorStop(0.75, QColor::fromRgb(92, 200, 99));
    newPalette.insertColorStop(0.80, QColor::fromRgb(122, 209, 81));
    newPalette.insertColorStop(0.85, QColor::fromRgb(155, 217, 60));
    newPalette.insertColorStop(0.90, QColor::fromRgb(189, 223, 38));
    newPalette.insertColorStop(0.95, QColor::fromRgb(221, 227, 24));
    newPalette.setInverseColorOne("#ff7f00"); //dark orange
    newPalette.setInverseColorTwo("#e31a1c"); //dark red
    newPalette.setWriteProtection();
    //set "viridis"
    m_colorPalettes.append(newPalette);
    m_restrictedKeyWords.append("viridis");
    m_builtInPalettes << "viridis";

    //Google turbo map, extracted from https://github.com/cleterrier/ChrisLUTs, Apache 2.0 license
    newPalette = ItomPaletteBase("turbo", ito::tPaletteFC | ito::tPaletteLinear | ito::tPaletteIndexed, QColor::fromRgb(48, 18, 59), QColor::fromRgb(122, 4, 3));
    // newPalette.insertColorStop(1.0 / 255.0, QColor::fromRgb(50, 21, 67));
    newPalette.insertColorStop(2.0 / 255.0, QColor::fromRgb(51, 24, 74));
    // newPalette.insertColorStop(3.0 / 255.0, QColor::fromRgb(52, 27, 81));
    newPalette.insertColorStop(4.0 / 255.0, QColor::fromRgb(53, 30, 88));
    // newPalette.insertColorStop(5.0 / 255.0, QColor::fromRgb(54, 33, 95));
    newPalette.insertColorStop(6.0 / 255.0, QColor::fromRgb(55, 36, 102));
    // newPalette.insertColorStop(7.0 / 255.0, QColor::fromRgb(56, 39, 109));
    newPalette.insertColorStop(8.0 / 255.0, QColor::fromRgb(57, 42, 115));
    // newPalette.insertColorStop(9.0 / 255.0, QColor::fromRgb(58, 45, 121));
    newPalette.insertColorStop(10.0 / 255.0, QColor::fromRgb(59, 47, 128));
    // newPalette.insertColorStop(11.0 / 255.0, QColor::fromRgb(60, 50, 134));
    newPalette.insertColorStop(12.0 / 255.0, QColor::fromRgb(61, 53, 139));
    // newPalette.insertColorStop(13.0 / 255.0, QColor::fromRgb(62, 56, 145));
    newPalette.insertColorStop(14.0 / 255.0, QColor::fromRgb(63, 59, 151));
    // newPalette.insertColorStop(15.0 / 255.0, QColor::fromRgb(63, 62, 156));
    newPalette.insertColorStop(16.0 / 255.0, QColor::fromRgb(64, 64, 162));
    // newPalette.insertColorStop(17.0 / 255.0, QColor::fromRgb(65, 67, 167));
    newPalette.insertColorStop(18.0 / 255.0, QColor::fromRgb(65, 70, 172));
    // newPalette.insertColorStop(19.0 / 255.0, QColor::fromRgb(66, 73, 177));
    newPalette.insertColorStop(20.0 / 255.0, QColor::fromRgb(66, 75, 181));
    // newPalette.insertColorStop(21.0 / 255.0, QColor::fromRgb(67, 78, 186));
    newPalette.insertColorStop(22.0 / 255.0, QColor::fromRgb(68, 81, 191));
    // newPalette.insertColorStop(23.0 / 255.0, QColor::fromRgb(68, 84, 195));
    newPalette.insertColorStop(24.0 / 255.0, QColor::fromRgb(68, 86, 199));
    // newPalette.insertColorStop(25.0 / 255.0, QColor::fromRgb(69, 89, 203));
    newPalette.insertColorStop(26.0 / 255.0, QColor::fromRgb(69, 92, 207));
    // newPalette.insertColorStop(27.0 / 255.0, QColor::fromRgb(69, 94, 211));
    newPalette.insertColorStop(28.0 / 255.0, QColor::fromRgb(70, 97, 214));
    // newPalette.insertColorStop(29.0 / 255.0, QColor::fromRgb(70, 100, 218));
    newPalette.insertColorStop(30.0 / 255.0, QColor::fromRgb(70, 102, 221));
    // newPalette.insertColorStop(31.0 / 255.0, QColor::fromRgb(70, 105, 224));
    newPalette.insertColorStop(32.0 / 255.0, QColor::fromRgb(70, 107, 227));
    // newPalette.insertColorStop(33.0 / 255.0, QColor::fromRgb(71, 110, 230));
    newPalette.insertColorStop(34.0 / 255.0, QColor::fromRgb(71, 113, 233));
    // newPalette.insertColorStop(35.0 / 255.0, QColor::fromRgb(71, 115, 235));
    newPalette.insertColorStop(36.0 / 255.0, QColor::fromRgb(71, 118, 238));
    // newPalette.insertColorStop(37.0 / 255.0, QColor::fromRgb(71, 120, 240));
    newPalette.insertColorStop(38.0 / 255.0, QColor::fromRgb(71, 123, 242));
    // newPalette.insertColorStop(39.0 / 255.0, QColor::fromRgb(70, 125, 244));
    newPalette.insertColorStop(40.0 / 255.0, QColor::fromRgb(70, 128, 246));
    // newPalette.insertColorStop(41.0 / 255.0, QColor::fromRgb(70, 130, 248));
    newPalette.insertColorStop(42.0 / 255.0, QColor::fromRgb(70, 133, 250));
    // newPalette.insertColorStop(43.0 / 255.0, QColor::fromRgb(70, 135, 251));
    newPalette.insertColorStop(44.0 / 255.0, QColor::fromRgb(69, 138, 252));
    // newPalette.insertColorStop(45.0 / 255.0, QColor::fromRgb(69, 140, 253));
    newPalette.insertColorStop(46.0 / 255.0, QColor::fromRgb(68, 143, 254));
    // newPalette.insertColorStop(47.0 / 255.0, QColor::fromRgb(67, 145, 254));
    newPalette.insertColorStop(48.0 / 255.0, QColor::fromRgb(66, 148, 255));
    // newPalette.insertColorStop(49.0 / 255.0, QColor::fromRgb(65, 150, 255));
    newPalette.insertColorStop(50.0 / 255.0, QColor::fromRgb(64, 153, 255));
    // newPalette.insertColorStop(51.0 / 255.0, QColor::fromRgb(62, 155, 254));
    newPalette.insertColorStop(52.0 / 255.0, QColor::fromRgb(61, 158, 254));
    // newPalette.insertColorStop(53.0 / 255.0, QColor::fromRgb(59, 160, 253));
    newPalette.insertColorStop(54.0 / 255.0, QColor::fromRgb(58, 163, 252));
    // newPalette.insertColorStop(55.0 / 255.0, QColor::fromRgb(56, 165, 251));
    newPalette.insertColorStop(56.0 / 255.0, QColor::fromRgb(55, 168, 250));
    // newPalette.insertColorStop(57.0 / 255.0, QColor::fromRgb(53, 171, 248));
    newPalette.insertColorStop(58.0 / 255.0, QColor::fromRgb(51, 173, 247));
    // newPalette.insertColorStop(59.0 / 255.0, QColor::fromRgb(49, 175, 245));
    newPalette.insertColorStop(60.0 / 255.0, QColor::fromRgb(47, 178, 244));
    // newPalette.insertColorStop(61.0 / 255.0, QColor::fromRgb(46, 180, 242));
    newPalette.insertColorStop(62.0 / 255.0, QColor::fromRgb(44, 183, 240));
    // newPalette.insertColorStop(63.0 / 255.0, QColor::fromRgb(42, 185, 238));
    newPalette.insertColorStop(64.0 / 255.0, QColor::fromRgb(40, 188, 235));
    // newPalette.insertColorStop(65.0 / 255.0, QColor::fromRgb(39, 190, 233));
    newPalette.insertColorStop(66.0 / 255.0, QColor::fromRgb(37, 192, 231));
    // newPalette.insertColorStop(67.0 / 255.0, QColor::fromRgb(35, 195, 228));
    newPalette.insertColorStop(68.0 / 255.0, QColor::fromRgb(34, 197, 226));
    // newPalette.insertColorStop(69.0 / 255.0, QColor::fromRgb(32, 199, 223));
    newPalette.insertColorStop(70.0 / 255.0, QColor::fromRgb(31, 201, 221));
    // newPalette.insertColorStop(71.0 / 255.0, QColor::fromRgb(30, 203, 218));
    newPalette.insertColorStop(72.0 / 255.0, QColor::fromRgb(28, 205, 216));
    // newPalette.insertColorStop(73.0 / 255.0, QColor::fromRgb(27, 208, 213));
    newPalette.insertColorStop(74.0 / 255.0, QColor::fromRgb(26, 210, 210));
    // newPalette.insertColorStop(75.0 / 255.0, QColor::fromRgb(26, 212, 208));
    newPalette.insertColorStop(76.0 / 255.0, QColor::fromRgb(25, 213, 205));
    // newPalette.insertColorStop(77.0 / 255.0, QColor::fromRgb(24, 215, 202));
    newPalette.insertColorStop(78.0 / 255.0, QColor::fromRgb(24, 217, 200));
    // newPalette.insertColorStop(79.0 / 255.0, QColor::fromRgb(24, 219, 197));
    newPalette.insertColorStop(80.0 / 255.0, QColor::fromRgb(24, 221, 194));
    // newPalette.insertColorStop(81.0 / 255.0, QColor::fromRgb(24, 222, 192));
    newPalette.insertColorStop(82.0 / 255.0, QColor::fromRgb(24, 224, 189));
    // newPalette.insertColorStop(83.0 / 255.0, QColor::fromRgb(25, 226, 187));
    newPalette.insertColorStop(84.0 / 255.0, QColor::fromRgb(25, 227, 185));
    // newPalette.insertColorStop(85.0 / 255.0, QColor::fromRgb(26, 228, 182));
    newPalette.insertColorStop(86.0 / 255.0, QColor::fromRgb(28, 230, 180));
    // newPalette.insertColorStop(87.0 / 255.0, QColor::fromRgb(29, 231, 178));
    newPalette.insertColorStop(88.0 / 255.0, QColor::fromRgb(31, 233, 175));
    // newPalette.insertColorStop(89.0 / 255.0, QColor::fromRgb(32, 234, 172));
    newPalette.insertColorStop(90.0 / 255.0, QColor::fromRgb(34, 235, 170));
    // newPalette.insertColorStop(91.0 / 255.0, QColor::fromRgb(37, 236, 167));
    newPalette.insertColorStop(92.0 / 255.0, QColor::fromRgb(39, 238, 164));
    // newPalette.insertColorStop(93.0 / 255.0, QColor::fromRgb(42, 239, 161));
    newPalette.insertColorStop(94.0 / 255.0, QColor::fromRgb(44, 240, 158));
    // newPalette.insertColorStop(95.0 / 255.0, QColor::fromRgb(47, 241, 155));
    newPalette.insertColorStop(96.0 / 255.0, QColor::fromRgb(50, 242, 152));
    // newPalette.insertColorStop(97.0 / 255.0, QColor::fromRgb(53, 243, 148));
    newPalette.insertColorStop(98.0 / 255.0, QColor::fromRgb(56, 244, 145));
    // newPalette.insertColorStop(99.0 / 255.0, QColor::fromRgb(60, 245, 142));
    newPalette.insertColorStop(100.0 / 255.0, QColor::fromRgb(63, 246, 138));
    // newPalette.insertColorStop(101.0 / 255.0, QColor::fromRgb(67, 247, 135));
    newPalette.insertColorStop(102.0 / 255.0, QColor::fromRgb(70, 248, 132));
    // newPalette.insertColorStop(103.0 / 255.0, QColor::fromRgb(74, 248, 128));
    newPalette.insertColorStop(104.0 / 255.0, QColor::fromRgb(78, 249, 125));
    // newPalette.insertColorStop(105.0 / 255.0, QColor::fromRgb(82, 250, 122));
    newPalette.insertColorStop(106.0 / 255.0, QColor::fromRgb(85, 250, 118));
    // newPalette.insertColorStop(107.0 / 255.0, QColor::fromRgb(89, 251, 115));
    newPalette.insertColorStop(108.0 / 255.0, QColor::fromRgb(93, 252, 111));
    // newPalette.insertColorStop(109.0 / 255.0, QColor::fromRgb(97, 252, 108));
    newPalette.insertColorStop(110.0 / 255.0, QColor::fromRgb(101, 253, 105));
    // newPalette.insertColorStop(111.0 / 255.0, QColor::fromRgb(105, 253, 102));
    newPalette.insertColorStop(112.0 / 255.0, QColor::fromRgb(109, 254, 98));
    // newPalette.insertColorStop(113.0 / 255.0, QColor::fromRgb(113, 254, 95));
    newPalette.insertColorStop(114.0 / 255.0, QColor::fromRgb(117, 254, 92));
    // newPalette.insertColorStop(115.0 / 255.0, QColor::fromRgb(121, 254, 89));
    newPalette.insertColorStop(116.0 / 255.0, QColor::fromRgb(125, 255, 86));
    // newPalette.insertColorStop(117.0 / 255.0, QColor::fromRgb(128, 255, 83));
    newPalette.insertColorStop(118.0 / 255.0, QColor::fromRgb(132, 255, 81));
    // newPalette.insertColorStop(119.0 / 255.0, QColor::fromRgb(136, 255, 78));
    newPalette.insertColorStop(120.0 / 255.0, QColor::fromRgb(139, 255, 75));
    // newPalette.insertColorStop(121.0 / 255.0, QColor::fromRgb(143, 255, 73));
    newPalette.insertColorStop(122.0 / 255.0, QColor::fromRgb(146, 255, 71));
    // newPalette.insertColorStop(123.0 / 255.0, QColor::fromRgb(150, 254, 68));
    newPalette.insertColorStop(124.0 / 255.0, QColor::fromRgb(153, 254, 66));
    // newPalette.insertColorStop(125.0 / 255.0, QColor::fromRgb(156, 254, 64));
    newPalette.insertColorStop(126.0 / 255.0, QColor::fromRgb(159, 253, 63));
    // newPalette.insertColorStop(127.0 / 255.0, QColor::fromRgb(161, 253, 61));
    newPalette.insertColorStop(128.0 / 255.0, QColor::fromRgb(164, 252, 60));
    // newPalette.insertColorStop(129.0 / 255.0, QColor::fromRgb(167, 252, 58));
    newPalette.insertColorStop(130.0 / 255.0, QColor::fromRgb(169, 251, 57));
    // newPalette.insertColorStop(131.0 / 255.0, QColor::fromRgb(172, 251, 56));
    newPalette.insertColorStop(132.0 / 255.0, QColor::fromRgb(175, 250, 55));
    // newPalette.insertColorStop(133.0 / 255.0, QColor::fromRgb(177, 249, 54));
    newPalette.insertColorStop(134.0 / 255.0, QColor::fromRgb(180, 248, 54));
    // newPalette.insertColorStop(135.0 / 255.0, QColor::fromRgb(183, 247, 53));
    newPalette.insertColorStop(136.0 / 255.0, QColor::fromRgb(185, 246, 53));
    // newPalette.insertColorStop(137.0 / 255.0, QColor::fromRgb(188, 245, 52));
    newPalette.insertColorStop(138.0 / 255.0, QColor::fromRgb(190, 244, 52));
    // newPalette.insertColorStop(139.0 / 255.0, QColor::fromRgb(193, 243, 52));
    newPalette.insertColorStop(140.0 / 255.0, QColor::fromRgb(195, 241, 52));
    // newPalette.insertColorStop(141.0 / 255.0, QColor::fromRgb(198, 240, 52));
    newPalette.insertColorStop(142.0 / 255.0, QColor::fromRgb(200, 239, 52));
    // newPalette.insertColorStop(143.0 / 255.0, QColor::fromRgb(203, 237, 52));
    newPalette.insertColorStop(144.0 / 255.0, QColor::fromRgb(205, 236, 52));
    // newPalette.insertColorStop(145.0 / 255.0, QColor::fromRgb(208, 234, 52));
    newPalette.insertColorStop(146.0 / 255.0, QColor::fromRgb(210, 233, 53));
    // newPalette.insertColorStop(147.0 / 255.0, QColor::fromRgb(212, 231, 53));
    newPalette.insertColorStop(148.0 / 255.0, QColor::fromRgb(215, 229, 53));
    // newPalette.insertColorStop(149.0 / 255.0, QColor::fromRgb(217, 228, 54));
    newPalette.insertColorStop(150.0 / 255.0, QColor::fromRgb(219, 226, 54));
    // newPalette.insertColorStop(151.0 / 255.0, QColor::fromRgb(221, 224, 55));
    newPalette.insertColorStop(152.0 / 255.0, QColor::fromRgb(223, 223, 55));
    // newPalette.insertColorStop(153.0 / 255.0, QColor::fromRgb(225, 221, 55));
    newPalette.insertColorStop(154.0 / 255.0, QColor::fromRgb(227, 219, 56));
    // newPalette.insertColorStop(155.0 / 255.0, QColor::fromRgb(229, 217, 56));
    newPalette.insertColorStop(156.0 / 255.0, QColor::fromRgb(231, 215, 57));
    // newPalette.insertColorStop(157.0 / 255.0, QColor::fromRgb(233, 213, 57));
    newPalette.insertColorStop(158.0 / 255.0, QColor::fromRgb(235, 211, 57));
    // newPalette.insertColorStop(159.0 / 255.0, QColor::fromRgb(236, 209, 58));
    newPalette.insertColorStop(160.0 / 255.0, QColor::fromRgb(238, 207, 58));
    // newPalette.insertColorStop(161.0 / 255.0, QColor::fromRgb(239, 205, 58));
    newPalette.insertColorStop(162.0 / 255.0, QColor::fromRgb(241, 203, 58));
    // newPalette.insertColorStop(163.0 / 255.0, QColor::fromRgb(242, 201, 58));
    newPalette.insertColorStop(164.0 / 255.0, QColor::fromRgb(244, 199, 58));
    // newPalette.insertColorStop(165.0 / 255.0, QColor::fromRgb(245, 197, 58));
    newPalette.insertColorStop(166.0 / 255.0, QColor::fromRgb(246, 195, 58));
    // newPalette.insertColorStop(167.0 / 255.0, QColor::fromRgb(247, 193, 58));
    newPalette.insertColorStop(168.0 / 255.0, QColor::fromRgb(248, 190, 57));
    // newPalette.insertColorStop(169.0 / 255.0, QColor::fromRgb(249, 188, 57));
    newPalette.insertColorStop(170.0 / 255.0, QColor::fromRgb(250, 186, 57));
    // newPalette.insertColorStop(171.0 / 255.0, QColor::fromRgb(251, 184, 56));
    newPalette.insertColorStop(172.0 / 255.0, QColor::fromRgb(251, 182, 55));
    // newPalette.insertColorStop(173.0 / 255.0, QColor::fromRgb(252, 179, 54));
    newPalette.insertColorStop(174.0 / 255.0, QColor::fromRgb(252, 177, 54));
    // newPalette.insertColorStop(175.0 / 255.0, QColor::fromRgb(253, 174, 53));
    newPalette.insertColorStop(176.0 / 255.0, QColor::fromRgb(253, 172, 52));
    // newPalette.insertColorStop(177.0 / 255.0, QColor::fromRgb(254, 169, 51));
    newPalette.insertColorStop(178.0 / 255.0, QColor::fromRgb(254, 167, 50));
    // newPalette.insertColorStop(179.0 / 255.0, QColor::fromRgb(254, 164, 49));
    newPalette.insertColorStop(180.0 / 255.0, QColor::fromRgb(254, 161, 48));
    // newPalette.insertColorStop(181.0 / 255.0, QColor::fromRgb(254, 158, 47));
    newPalette.insertColorStop(182.0 / 255.0, QColor::fromRgb(254, 155, 45));
    // newPalette.insertColorStop(183.0 / 255.0, QColor::fromRgb(254, 153, 44));
    newPalette.insertColorStop(184.0 / 255.0, QColor::fromRgb(254, 150, 43));
    // newPalette.insertColorStop(185.0 / 255.0, QColor::fromRgb(254, 147, 42));
    newPalette.insertColorStop(186.0 / 255.0, QColor::fromRgb(254, 144, 41));
    // newPalette.insertColorStop(187.0 / 255.0, QColor::fromRgb(253, 141, 39));
    newPalette.insertColorStop(188.0 / 255.0, QColor::fromRgb(253, 138, 38));
    // newPalette.insertColorStop(189.0 / 255.0, QColor::fromRgb(252, 135, 37));
    newPalette.insertColorStop(190.0 / 255.0, QColor::fromRgb(252, 132, 35));
    // newPalette.insertColorStop(191.0 / 255.0, QColor::fromRgb(251, 129, 34));
    newPalette.insertColorStop(192.0 / 255.0, QColor::fromRgb(251, 126, 33));
    // newPalette.insertColorStop(193.0 / 255.0, QColor::fromRgb(250, 123, 31));
    newPalette.insertColorStop(194.0 / 255.0, QColor::fromRgb(249, 120, 30));
    // newPalette.insertColorStop(195.0 / 255.0, QColor::fromRgb(249, 117, 29));
    newPalette.insertColorStop(196.0 / 255.0, QColor::fromRgb(248, 114, 28));
    // newPalette.insertColorStop(197.0 / 255.0, QColor::fromRgb(247, 111, 26));
    newPalette.insertColorStop(198.0 / 255.0, QColor::fromRgb(246, 108, 25));
    // newPalette.insertColorStop(199.0 / 255.0, QColor::fromRgb(245, 105, 24));
    newPalette.insertColorStop(200.0 / 255.0, QColor::fromRgb(244, 102, 23));
    // newPalette.insertColorStop(201.0 / 255.0, QColor::fromRgb(243, 99, 21));
    newPalette.insertColorStop(202.0 / 255.0, QColor::fromRgb(242, 96, 20));
    // newPalette.insertColorStop(203.0 / 255.0, QColor::fromRgb(241, 93, 19));
    newPalette.insertColorStop(204.0 / 255.0, QColor::fromRgb(240, 91, 18));
    // newPalette.insertColorStop(205.0 / 255.0, QColor::fromRgb(239, 88, 17));
    newPalette.insertColorStop(206.0 / 255.0, QColor::fromRgb(237, 85, 16));
    // newPalette.insertColorStop(207.0 / 255.0, QColor::fromRgb(236, 83, 15));
    newPalette.insertColorStop(208.0 / 255.0, QColor::fromRgb(235, 80, 14));
    // newPalette.insertColorStop(209.0 / 255.0, QColor::fromRgb(234, 78, 13));
    newPalette.insertColorStop(210.0 / 255.0, QColor::fromRgb(232, 75, 12));
    // newPalette.insertColorStop(211.0 / 255.0, QColor::fromRgb(231, 73, 12));
    newPalette.insertColorStop(212.0 / 255.0, QColor::fromRgb(229, 71, 11));
    // newPalette.insertColorStop(213.0 / 255.0, QColor::fromRgb(228, 69, 10));
    newPalette.insertColorStop(214.0 / 255.0, QColor::fromRgb(226, 67, 10));
    // newPalette.insertColorStop(215.0 / 255.0, QColor::fromRgb(225, 65, 9));
    newPalette.insertColorStop(216.0 / 255.0, QColor::fromRgb(223, 63, 8));
    // newPalette.insertColorStop(217.0 / 255.0, QColor::fromRgb(221, 61, 8));
    newPalette.insertColorStop(218.0 / 255.0, QColor::fromRgb(220, 59, 7));
    // newPalette.insertColorStop(219.0 / 255.0, QColor::fromRgb(218, 57, 7));
    newPalette.insertColorStop(220.0 / 255.0, QColor::fromRgb(216, 55, 6));
    // newPalette.insertColorStop(221.0 / 255.0, QColor::fromRgb(214, 53, 6));
    newPalette.insertColorStop(222.0 / 255.0, QColor::fromRgb(212, 51, 5));
    // newPalette.insertColorStop(223.0 / 255.0, QColor::fromRgb(210, 49, 5));
    newPalette.insertColorStop(224.0 / 255.0, QColor::fromRgb(208, 47, 5));
    // newPalette.insertColorStop(225.0 / 255.0, QColor::fromRgb(206, 45, 4));
    newPalette.insertColorStop(226.0 / 255.0, QColor::fromRgb(204, 43, 4));
    // newPalette.insertColorStop(227.0 / 255.0, QColor::fromRgb(202, 42, 4));
    newPalette.insertColorStop(228.0 / 255.0, QColor::fromRgb(200, 40, 3));
    // newPalette.insertColorStop(229.0 / 255.0, QColor::fromRgb(197, 38, 3));
    newPalette.insertColorStop(230.0 / 255.0, QColor::fromRgb(195, 37, 3));
    // newPalette.insertColorStop(231.0 / 255.0, QColor::fromRgb(193, 35, 2));
    newPalette.insertColorStop(232.0 / 255.0, QColor::fromRgb(190, 33, 2));
    // newPalette.insertColorStop(233.0 / 255.0, QColor::fromRgb(188, 32, 2));
    newPalette.insertColorStop(234.0 / 255.0, QColor::fromRgb(185, 30, 2));
    // newPalette.insertColorStop(235.0 / 255.0, QColor::fromRgb(183, 29, 2));
    newPalette.insertColorStop(236.0 / 255.0, QColor::fromRgb(180, 27, 1));
    // newPalette.insertColorStop(237.0 / 255.0, QColor::fromRgb(178, 26, 1));
    newPalette.insertColorStop(238.0 / 255.0, QColor::fromRgb(175, 24, 1));
    // newPalette.insertColorStop(239.0 / 255.0, QColor::fromRgb(172, 23, 1));
    newPalette.insertColorStop(240.0 / 255.0, QColor::fromRgb(169, 22, 1));
    // newPalette.insertColorStop(241.0 / 255.0, QColor::fromRgb(167, 20, 1));
    newPalette.insertColorStop(242.0 / 255.0, QColor::fromRgb(164, 19, 1));
    // newPalette.insertColorStop(243.0 / 255.0, QColor::fromRgb(161, 18, 1));
    newPalette.insertColorStop(244.0 / 255.0, QColor::fromRgb(158, 16, 1));
    // newPalette.insertColorStop(245.0 / 255.0, QColor::fromRgb(155, 15, 1));
    newPalette.insertColorStop(246.0 / 255.0, QColor::fromRgb(152, 14, 1));
    // newPalette.insertColorStop(247.0 / 255.0, QColor::fromRgb(149, 13, 1));
    newPalette.insertColorStop(248.0 / 255.0, QColor::fromRgb(146, 11, 1));
    // newPalette.insertColorStop(249.0 / 255.0, QColor::fromRgb(142, 10, 1));
    newPalette.insertColorStop(250.0 / 255.0, QColor::fromRgb(139, 9, 2));
    // newPalette.insertColorStop(251.0 / 255.0, QColor::fromRgb(136, 8, 2));
    newPalette.insertColorStop(252.0 / 255.0, QColor::fromRgb(133, 7, 2));
    // newPalette.insertColorStop(253.0 / 255.0, QColor::fromRgb(129, 6, 2));
    newPalette.insertColorStop(254.0 / 255.0, QColor::fromRgb(126, 5, 2));

    newPalette.setInverseColorOne(Qt::black);
    newPalette.setInverseColorTwo(Qt::gray);
    newPalette.setInvalidColor(Qt::red);
    newPalette.setWriteProtection();
    //set "turbo"
    m_colorPalettes.append(newPalette);
    m_restrictedKeyWords.append("turbo");
    m_builtInPalettes << "turbo";
    //------------

    // load saved palettes from settings file
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("ColorPalettes");
    foreach(QString child, settings.childGroups())
    {
        ItomPaletteBase newPal;
        loadColorPaletteFromSettings(child, newPal, settings);

        int existingIndex = getColorPaletteList().indexOf(newPal.getName());
        if (existingIndex < 0)
        {
            m_colorPalettes.append(newPal);
        }
        else if ((m_colorPalettes[existingIndex].getType() & ito::tPaletteReadOnly) == 0)
        {
            m_colorPalettes[existingIndex] = newPal;
        }
    }
    settings.endGroup();

    calcColorPaletteLut();
}

//----------------------------------------------------------------------------------------------------------------------------------
/* save the given color palette to the settings. Settings must already be opened in the group,
where the palette should be saved. Each palette is stored as a subgroup of the current group. */
ito::RetVal PaletteOrganizer::saveColorPaletteToSettings(const ItomPaletteBase &palette, QSettings &settings) const
{
    settings.beginGroup(palette.getName());
    settings.setValue("name", palette.getName());
    settings.setValue("type", palette.getType());
    settings.setValue("invalidColor", palette.getInvalidColor());
    settings.setValue("inverseColor1", palette.getInverseColorOne());
    settings.setValue("inverseColor2", palette.getInverseColorTwo());
    settings.setValue("numColorStops", palette.getNumColorStops());

    const QVector< QPair<double, QColor> > &colorStops = palette.getColorStops();
    for (int idx = 0; idx < colorStops.size(); ++idx)
    {
        settings.setValue(QString("cs%1_1").arg(idx), colorStops[idx].first);
        settings.setValue(QString("cs%1_2").arg(idx), colorStops[idx].second);
    }

    settings.endGroup();

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
/* load a color palette from the settings. Settings must already be opened in the group,
where the palette should be loaded from. The current group must hereby consist of a subgroup with the group name 'paletteName'. */
ito::RetVal PaletteOrganizer::loadColorPaletteFromSettings(const QString &paletteName, ItomPaletteBase &palette, QSettings &settings) const
{
    if (! settings.childGroups().contains(paletteName))
    {
        return ito::RetVal::format(ito::retError, 0, tr("Settings do not contain a color palette entry for the palette name '%s'").toLatin1().data(), paletteName.toLatin1().constData());
    }

    settings.beginGroup(paletteName);
    const QString name = settings.value("name").toString();
    int type = settings.value("type").toInt();
    QColor invalidCol = settings.value("invalidColor").value<QColor>();
    QColor invCol1 = settings.value("inverseColor1").value<QColor>();
    QColor invCol2 = settings.value("inverseColor2").value<QColor>();
    QVariant numColStops = settings.value("numColorStops");
    QVector<QGradientStop> colorStops;
    for (int ns = 0; ns < numColStops.toInt(); ns++)
    {
        float pos = settings.value(QString("cs%1_1").arg(ns)).toFloat();
        QColor color = settings.value(QString("cs%1_2").arg(ns)).value<QColor>();
        colorStops.append(QGradientStop(pos, color));
    }

    ItomPaletteBase newPal(name, type, invCol1, invCol2, invalidCol, colorStops);

    settings.endGroup();

    palette = newPal;

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param index
    \return ItomPaletteBase
*/
ItomPaletteBase PaletteOrganizer::getColorPalette(const int index) const
{
    if(index < 0 || index >= m_colorPalettes.length())
        return noPalette;

    return m_colorPalettes[index];
}


//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param name
    \param found
    \return ItomPaletteBase
*/
ItomPaletteBase PaletteOrganizer::getColorPalette(const QString &name, bool *found /*= NULL*/) const
{
    for(int i = 0; i < m_colorPalettes.length(); i++)
    {
        if(!m_colorPalettes[i].getName().compare(name, Qt::CaseSensitive))
        {
            if (found) *found = true;
            return m_colorPalettes[i];
        }
    }
    if (found) *found = false;
    return noPalette;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param name
    \param found
    \return int
*/
int PaletteOrganizer::getColorBarIndex(const QString &name, bool *found /*= NULL*/) const
{
    if(m_colorPaletteLUT.contains(name))
    {
        if (found) *found = true;
        return m_colorPaletteLUT[name];
    }
    if (found) *found = false;
    return -1;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param type
    \return QList<QString>
*/
QList<QString> PaletteOrganizer::getColorPaletteList(const int type) const
{
    QList<QString> outPut;
    outPut.clear();

    for(int i = 0; i < m_colorPalettes.length(); i++)
    {
        if((type != ito::tPaletteNoType))
        {
            if(type & m_colorPalettes[i].getType())
                outPut.append(m_colorPalettes[i].getName());
        }
        else
        {
            outPut.append(m_colorPalettes[i].getName());
        }
    }
    return outPut;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool PaletteOrganizer::removeColorPalette(const int index)
{
    if (index >= 0 && index < m_colorPalettes.length() && !(m_colorPalettes[index].getType() & ito::tPaletteReadOnly))
    {
        m_colorPalettes.removeAt(index);
        calcColorPaletteLut();
        return true;
    }
    return false;
}

//----------------------------------------------------------------------------------------------------------------------------------
void PaletteOrganizer::calcColorPaletteLut()
{
    m_colorPaletteLUT.clear();
    for (int i = 0; i < m_colorPalettes.size(); ++i)
    {
        m_colorPaletteLUT.insert(m_colorPalettes[i].getName(), i);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PaletteOrganizer::setColorBarThreaded(QString name, ito::ItomPaletteBase newPalette, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval = ito::retOk;

    int idx = -1;

    if(m_restrictedKeyWords.contains(name))
    {
        retval += ito::RetVal(ito::retError, 0, tr("Palette %1 has a restricted access.").arg(name).toLatin1().constData());
    }
    else if(m_colorPaletteLUT.contains(name))
    {
        idx = m_colorPaletteLUT[name];
        if(idx < m_colorPalettes.length() && m_colorPalettes[idx].isWriteProtected())
        {
            retval += ito::RetVal(ito::retError, 0, tr("Palette %1 has a write protection.").arg(name).toLatin1().constData());
        }
        else
        {
            m_colorPalettes[idx] = newPalette;
        }
    }
    else
    {
        m_colorPalettes.append(newPalette);
        calcColorPaletteLut();
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }
    return retval;

}
//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PaletteOrganizer::getColorBarThreaded(QString name, QSharedPointer<ito::ItomPaletteBase> palette, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval = ito::retOk;

    bool found = false;

    *palette = getColorPalette(name, &found);

    if(!found)
    {
        retval += ito::RetVal(ito::retError, 0, tr("Palette %1 not found within palette list").arg(name).toLatin1().constData());
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }
    return retval;
}
//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PaletteOrganizer::getColorBarListThreaded(QSharedPointer<QStringList> palettes, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval = ito::retOk;

    bool found = false;

    if(!palettes.isNull())
    {
        palettes->clear();
        QList<QString> curList = this->getColorPaletteList(0);

        for(int i = 0; i < curList.size(); i++)
        {
            palettes->append(curList[i]);
        }

    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("Destination vector not initialized").toLatin1().constData());
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }
    return retval;
}
//----------------------------------------------------------------------------------------------------------------------------------
}//namespace ito
