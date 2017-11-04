/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

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
bool ItomPaletteBase::setInverseColorOne(const QColor color)
{
    if((m_paletteData.type & ito::tPaletteReadOnly) && m_paletteData.inverseColorOne.isValid())
    {
        //qDebug() << "ItomPalette setInversColorTwo. Tried to write to a readonly palette. ";
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
bool ItomPaletteBase::setInverseColorTwo(const QColor color)
{
    if ((m_paletteData.type & ito::tPaletteReadOnly) && m_paletteData.inverseColorTwo.isValid())
    {
        //qDebug() << "ItomPalette setInversColorTwo. Tried to write to a readonly palette. ";
        return false;
    }

    m_paletteData.inverseColorTwo = color;

    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      Set the invalid color for this color bar
/*! \detail     Each colorbar has a invalid color for inf and NaN values

    \param      color   new color as QColor value
    \return     bool    true if success
*/
bool ItomPaletteBase::setInvalidColor(const QColor color)
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
bool ItomPaletteBase::insertColorStop( double pos, const QColor color )
{
    // This code is copied from QWT-PLOT.
    // Lookups need to be very fast, insertions are not so important.
    // Anyway, a balanced tree is what we need here. TODO ...

    if (m_paletteData.type & ito::tPaletteReadOnly)
    {
        //qDebug() << "ItomPalette insertColorStop. Tried to write to a readonly palette. ";
        return false;
    }
    if ( pos < 0.0 || pos > 1.0 )
    {
        //qDebug() << "ItomPalette insertColorStop. Position out of range [0..1]. ";
        return false;
    }
    int index;
    if (m_paletteData.colorStops.size() == 0 )
    //if ( _stops.size() == 0 )
    {
        index = 0;
        //_stops.resize( 1 );  
        m_paletteData.colorStops.resize(1);
    }
    else
    {
        index = findUpper( pos );
        //if ( index == _stops.size() || qAbs( _stops[index].pos - pos ) >= 0.001 )
        if ( index == m_paletteData.colorStops.size() || qAbs(m_paletteData.colorStops[index].first - pos ) >= 0.001 )
        {
            //_stops.resize( _stops.size() + 1 );
            //for ( int i = _stops.size() - 1; i > index; i-- )
            //    _stops[i] = _stops[i-1];
            m_paletteData.colorStops.resize(m_paletteData.colorStops.size() + 1 );
            for ( int i = m_paletteData.colorStops.size() - 1; i > index; i-- )
                m_paletteData.colorStops[i] = m_paletteData.colorStops[i-1];
        }   
    }
    //_stops[index] = ColorStop( pos, color );
    m_paletteData.colorStops[index].first = pos;
    m_paletteData.colorStops[index].second = color;
    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      Get the position of the n-th color-stop in the palette
/*! \detail     This function returns the position (doubel value) of the color stop devined by int color

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
/*! \detail     This function returns the position (doubel value) of the color stop devined by int color

    \param      color     index of the color to retrieve
    \return     QColor    the RGBA-Color of the color stop
*/
QColor ItomPaletteBase::getColor(unsigned int color) const
{
    if(color > (unsigned int)(m_paletteData.colorStops.size() - 1)) 
        return m_paletteData.colorStops[m_paletteData.colorStops.size() - 1].second;
    return m_paletteData.colorStops[color].second;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      Function which shall calculate the ideal inverse colors
/*! \detail     Not inplemented yet

    \param      inv1     first invalid color
    \param      inv2     second invalid color
*/
void ItomPaletteBase::calculateInverseColors(QColor &inv1, QColor &inv2)
{
    return;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      This function updates the internal structur of the palette after modifications
/*! \detail     This function updates the internal structur of the palette after modifications

    \param      updateInverseColors     recalculate the ideals inverse color
*/
void ItomPaletteBase::update(const bool updateInverseColors)
{
    m_paletteData.colorVector256 = get256Colors();

    if(updateInverseColors || !m_paletteData.inverseColorOne.isValid() || !m_paletteData.inverseColorTwo.isValid())
    {
        QColor color1, color2;
        calculateInverseColors(color1, color2);
        if(updateInverseColors || !m_paletteData.inverseColorOne.isValid())
        {
            m_paletteData.inverseColorOne = color1;
        }
        if(updateInverseColors || !m_paletteData.inverseColorTwo.isValid())
        {
            m_paletteData.inverseColorTwo = color2;
        }
    }
    return;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \brief      This function returns the internal structur of the palett
/*! \detail     This function returns the internal structur of the palett
*/
ItomPalette ItomPaletteBase::getPalette()
{
    if(m_paletteData.colorStops.isEmpty() || m_paletteData.colorVector256.isEmpty() || m_paletteData.type == ito::tPaletteNoType)
    {
        update(false);
    }
    return m_paletteData;
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
    m_colorBars.clear();
    restrictedKeyWords.clear();
    m_colorBarLookUp.clear();

    restrictedKeyWords.append("");
    restrictedKeyWords.append("none");

    noPalette = ItomPaletteBase("none", 0); 

    ItomPaletteBase newPalette;
    //QColor inv1, inv2;
    //declare "gray"
    newPalette = ItomPaletteBase("gray", ito::tPaletteGray | ito::tPaletteRGB | ito::tPaletteLinear | ito::tPaletteIndexed, QColor::fromRgb(0, 0, 0), QColor::fromRgb(255, 255, 255));
    newPalette.setInverseColorOne(Qt::blue);
    newPalette.setInverseColorTwo("#e31a1c"); //dark red
    newPalette.setWriteProtection();
    //set "gray"
    m_colorBars.append(newPalette);
    restrictedKeyWords.append("gray");
    m_colorBarLookUp.insert("gray", m_colorBars.size() - 1);
    m_builtInPalettes << "gray";
    //------------

    //declare "grayMarked"
    newPalette = ItomPaletteBase("grayMarked", ito::tPaletteGray | ito::tPaletteLinear | ito::tPaletteIndexed, Qt::magenta, Qt::white);
    newPalette.insertColorStop(0.0, Qt::black);
    newPalette.insertColorStop(1.0, Qt::red);
    newPalette.setInverseColorOne(Qt::blue);
    newPalette.setInverseColorTwo(Qt::green);
    newPalette.setWriteProtection();
    //set "grayMarked"
    m_colorBars.append(newPalette);
    restrictedKeyWords.append("grayMarked");
    m_colorBarLookUp.insert("grayMarked", m_colorBars.size() - 1);
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
    m_colorBars.append(newPalette);
    restrictedKeyWords.append("falseColor");
    m_colorBarLookUp.insert("falseColor", m_colorBars.size() - 1);
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
    m_colorBars.append(newPalette);
    restrictedKeyWords.append("falseColorIR");
    m_colorBarLookUp.insert("falseColorIR", m_colorBars.size() - 1);
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
    m_colorBars.append(newPalette);
    restrictedKeyWords.append("hotIron");
    m_colorBarLookUp.insert("hotIron", m_colorBars.size() - 1);
    m_builtInPalettes << "hotIron";
    //------------

    //declare "red"
    newPalette = ItomPaletteBase(ItomPaletteBase("red", ito::tPaletteGray | ito::tPaletteLinear | ito::tPaletteIndexed, QColor::fromRgb(0, 0, 0), QColor::fromRgb(255, 0, 0)));
    newPalette.setInverseColorOne(Qt::blue);
    newPalette.setInverseColorTwo(Qt::green);
    newPalette.setWriteProtection();
    //set "red"
    m_colorBars.append(newPalette);
    restrictedKeyWords.append("red");
    m_colorBarLookUp.insert("red", m_colorBars.size() - 1);
    m_builtInPalettes << "red";
    //------------

    //declare "blue"
    newPalette = ItomPaletteBase(ItomPaletteBase("blue", ito::tPaletteGray | ito::tPaletteLinear | ito::tPaletteIndexed, QColor::fromRgb(0, 0, 0), QColor::fromRgb(0, 0, 255)));
    newPalette.setInverseColorOne("#e31a1c"); //dark red
    newPalette.setInverseColorTwo(Qt::green);
    newPalette.setWriteProtection();
    //set "blue"
    m_colorBars.append(newPalette);
    restrictedKeyWords.append("blue");
    m_colorBarLookUp.insert("blue", m_colorBars.size() - 1);
    m_builtInPalettes << "blue";
    //------------

    //declare "green"
    newPalette = ItomPaletteBase(ItomPaletteBase("green", ito::tPaletteGray | ito::tPaletteLinear | ito::tPaletteIndexed, QColor::fromRgb(0, 0, 0), QColor::fromRgb(0, 255, 0)));
    newPalette.setInverseColorOne(Qt::blue);
    newPalette.setInverseColorTwo("#e31a1c"); //dark red
    newPalette.setWriteProtection();
    //set "green"
    m_colorBars.append(newPalette);
    restrictedKeyWords.append("green");
    m_colorBarLookUp.insert("green", m_colorBars.size() - 1);
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
    m_colorBars.append(newPalette);
    restrictedKeyWords.append("viridis");
    m_colorBarLookUp.insert("viridis", m_colorBars.size() - 1);
    m_builtInPalettes << "viridis";
    //------------

    // load saved palettes from settings file
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("ito::Palettes");
    foreach(QString child, settings.childGroups())
    {
        settings.beginGroup(child);
        const QString name = settings.value("name").toString();
        int type = settings.value("type").toInt();
        uint uinvalidCol = settings.value("invalidColor").toUInt();
        uint uinvCol1 = settings.value("inverseColor1").toUInt();
        uint uinvCol2 = settings.value("inverseColor2").toUInt();
        QColor invalidCol(uinvalidCol & 0xFF, (uinvalidCol >> 8) & 0xFF, (uinvalidCol >> 16) & 0xFF);
        QColor invCol1(uinvCol1 & 0xFF, (uinvCol1 >> 8) & 0xFF, (uinvCol1 >> 16) & 0xFF);
        QColor invCol2(uinvCol2 & 0xFF, (uinvCol2 >> 8) & 0xFF, (uinvCol2 >> 16) & 0xFF);
        QVariant numColStops = settings.value("numColorStops");
        QVector<QGradientStop> colorStops;
        for (int ns = 0; ns < numColStops.toInt(); ns++)
        {
            QVariant val = settings.value(QString("cs%1_1").arg(ns));
            QVariant col = settings.value(QString("cs%1_2").arg(ns));
            uint ucol = val.toUInt();
            colorStops.append(QGradientStop(val.toFloat(), QColor(ucol & 0xFF, (ucol >> 8) & 0xFF, (ucol >> 16) & 0xFF)));
        }
        ItomPaletteBase newPal(name, type, invCol1, invCol2, invalidCol, colorStops);

        int existingIndex = getColorBarList().indexOf(newPal.getName());
        if (existingIndex < 0)
        {
            m_colorBars.append(newPal);
        }
        else if ((m_colorBars[existingIndex].getType() & ito::tPalette::tPaletteReadOnly) == 0)
        {
            m_colorBars[existingIndex] = newPal;
        }

        settings.endGroup();
    }
    settings.endGroup();

 //   m_colorBars.append(ItomPaletteBase("RGB", ItomPalette::RGBPalette | ItomPalette::ReadOnlyPalette, QColor::fromRgb(0, 0, 0), QColor::fromRgb(255, 255, 255)));
 //   restrictedKeyWords.append("256Colors");
 //   m_colorBarLookUp.insert("256Colors", 7);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param index
    \return ItomPaletteBase
*/
ItomPaletteBase PaletteOrganizer::getColorBar(const int index) const
{
    if(index < 0 || index >= m_colorBars.length())
        return noPalette;

    return m_colorBars[index];
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param curindex
    \param type
    \return ItomPaletteBase
*/
ItomPaletteBase PaletteOrganizer::getNextColorBar(const int curindex, const int type) const
{
    int nextIndex = (curindex + 1) % m_colorBars.length();

    if((type != ito::tPaletteNoType) && (type & m_colorBars[nextIndex].getType()))
    {
        int temIndex;

        for(int i = 0; i < m_colorBars.length(); i++)
        {
            temIndex = (curindex + i + 2) % m_colorBars.length();

            if(temIndex == curindex)
                return m_colorBars[curindex];
                
            if(type & m_colorBars[temIndex].getType())
                return m_colorBars[temIndex];
        }
        return m_colorBars[curindex];
    }
    else
    {
        return m_colorBars[nextIndex];
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param name
    \param found
    \return ItomPaletteBase
*/
ItomPaletteBase PaletteOrganizer::getColorBar(const QString name, bool *found /*= NULL*/) const
{
    for(int i = 0; i < m_colorBars.length(); i++)
    {
        if(!m_colorBars[i].getName().compare(name, Qt::CaseSensitive))
        {
            if (found) *found = true;
            return m_colorBars[i]; 
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
int PaletteOrganizer::getColorBarIndex(const QString name, bool *found /*= NULL*/) const
{
    if(m_colorBarLookUp.contains(name))
    {
        if (found) *found = true;
        return m_colorBarLookUp[name];
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
QList<QString> PaletteOrganizer::getColorBarList(const int type) const
{
    QList<QString> outPut;
    outPut.clear();

    for(int i = 0; i < m_colorBars.length(); i++)
    {
        if((type != ito::tPaletteNoType))
        {
            if(type & m_colorBars[i].getType())
                outPut.append(m_colorBars[i].getName());
        }
        else
        {
            outPut.append(m_colorBars[i].getName());
        }
    }
    return outPut;
}
//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PaletteOrganizer::setColorBarThreaded(QString name, ito::ItomPaletteBase newPalette, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval = ito::retOk;

    int idx = -1;

    if(restrictedKeyWords.contains(name))
    {
        retval += ito::RetVal(ito::retError, 0, tr("Palette %1 has a restricted access.").arg(name).toLatin1().data());
    }
    else if(m_colorBarLookUp.contains(name))
    {
        idx = m_colorBarLookUp[name];
        if(idx < m_colorBars.length() && m_colorBars[idx].getType() & ito::tPaletteReadOnly)
        {
            retval += ito::RetVal(ito::retError, 0, tr("Palette %1 has a write protection.").arg(name).toLatin1().data());
        }
        else
        {
            m_colorBars[idx] = newPalette;
        }
    }
    else
    {
        m_colorBars.append(newPalette);
        m_colorBarLookUp.insert(name, m_colorBars.size() - 1);
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

    *palette = this->getColorBar(name, &found);

    if(!found)
    {
        retval += ito::RetVal(ito::retError, 0, tr("Palette %1 not found within palette list").arg(name).toLatin1().data());
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }
    return retval;
}
//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PaletteOrganizer::getColorBarListThreaded(int types, QSharedPointer<QStringList> palettes, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval = ito::retOk;

    bool found = false;

    if(!palettes.isNull())
    {
        palettes->clear();
        QList<QString> curList = this->getColorBarList(0);

        for(int i = 0; i < curList.size(); i++)
        {
            palettes->append(curList[i]);
        }
        
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("Destination vector not initialized").toLatin1().data());
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