/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
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

//#include "opencv2/core/core.hpp"
//#include "opencv2/core/operations.hpp"

using namespace ito;
//----------------------------------------------------------------------------------------------------------------------------------
PaletteOrganizer::PaletteOrganizer()
{
    m_colorBars.clear();
    restrictedKeyWords.clear();
    m_colorBarLookUp.clear();

    restrictedKeyWords.append("");
    restrictedKeyWords.append("none");

    noPalette = ItomPalette("none", 0); 

    m_colorBars.append(ItomPalette("gray", ItomPalette::GrayPalette | ItomPalette::RGBPalette | ItomPalette::LinearPalette | ItomPalette::ReadOnlyPalette | ItomPalette::indexPalette, QColor::fromRgb(0, 0, 0), QColor::fromRgb(255, 255, 255)));
    restrictedKeyWords.append("gray");
    m_colorBarLookUp.insert("gray", 0);

    // Modified gray scale for ...
    ItomPalette newPalette("grayMarked", ItomPalette::GrayPalette | ItomPalette::LinearPalette | ItomPalette::indexPalette, Qt::magenta, Qt::white);
    newPalette.insertColorStop(0.0, Qt::black);
    newPalette.insertColorStop(1.0, Qt::red);
    newPalette.setWriteProtection();
   
    m_colorBars.append(newPalette);
    restrictedKeyWords.append("grayMarked");
    m_colorBarLookUp.insert("grayMarked", 1);

    // Add false Colors
    newPalette = ItomPalette("falseColor", ItomPalette::FCPalette | ItomPalette::LinearPalette | ItomPalette::indexPalette, Qt::red, Qt::magenta);
    newPalette.insertColorStop(0.2, Qt::yellow);
    newPalette.insertColorStop(0.4, Qt::green);
    newPalette.insertColorStop(0.6, Qt::cyan);
    newPalette.insertColorStop(0.8, Qt::blue);
    newPalette.setWriteProtection();

    m_colorBars.append(newPalette);
    restrictedKeyWords.append("falseColor");
    m_colorBarLookUp.insert("falseColor", 2);

    // Add false Colors
    newPalette = ItomPalette("falseColorIR", ItomPalette::FCPalette | ItomPalette::LinearPalette | ItomPalette::indexPalette, QColor::fromRgb(165, 30, 165), Qt::white);
    newPalette.insertColorStop(0.15, Qt::blue);
    newPalette.insertColorStop(0.35, Qt::cyan);
    newPalette.insertColorStop(0.55, Qt::green);
    newPalette.insertColorStop(0.75, Qt::yellow);
    newPalette.insertColorStop(0.97, Qt::red);
    newPalette.setWriteProtection();

    m_colorBars.append(newPalette);
    restrictedKeyWords.append("falseColorIR");
    m_colorBarLookUp.insert("falseColorIR", 3);

    m_colorBars.append(ItomPalette("red", ItomPalette::GrayPalette | ItomPalette::LinearPalette | ItomPalette::indexPalette | ItomPalette::ReadOnlyPalette, QColor::fromRgb(0, 0, 0), QColor::fromRgb(255, 0, 0)));
    restrictedKeyWords.append("red");
    m_colorBarLookUp.insert("red", 4);

    m_colorBars.append(ItomPalette("blue", ItomPalette::GrayPalette | ItomPalette::LinearPalette | ItomPalette::indexPalette | ItomPalette::ReadOnlyPalette, QColor::fromRgb(0, 0, 0), QColor::fromRgb(0, 255, 0)));
    restrictedKeyWords.append("blue");
    m_colorBarLookUp.insert("blue", 5);

    m_colorBars.append(ItomPalette("green", ItomPalette::GrayPalette | ItomPalette::LinearPalette | ItomPalette::indexPalette | ItomPalette::ReadOnlyPalette, QColor::fromRgb(0, 0, 0), QColor::fromRgb(0, 0, 255)));
    restrictedKeyWords.append("green");
    m_colorBarLookUp.insert("green", 6);

 //   m_colorBars.append(ItomPalette("RGB", ItomPalette::RGBPalette | ItomPalette::ReadOnlyPalette, QColor::fromRgb(0, 0, 0), QColor::fromRgb(255, 255, 255)));
 //   restrictedKeyWords.append("256Colors");
 //   m_colorBarLookUp.insert("256Colors", 7);
}
//----------------------------------------------------------------------------------------------------------------------------------
ItomPalette PaletteOrganizer::getColorBar(const int index) const
{
    if(index < 0 || index >= m_colorBars.length())
        return noPalette;

    return m_colorBars[index];
}

//----------------------------------------------------------------------------------------------------------------------------------
ItomPalette PaletteOrganizer::getNextColorBar(const int curindex, const int type) const
{
    int nextIndex = (curindex + 1) % m_colorBars.length();

    if((type != ItomPalette::NoType) && (type & m_colorBars[nextIndex].getType()))
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
        return m_colorBars[nextIndex];
}
//----------------------------------------------------------------------------------------------------------------------------------
ItomPalette PaletteOrganizer::getColorBar(const QString name) const
{
    for(int i = 0; i < m_colorBars.length(); i++)
    {
        if(!m_colorBars[i].getName().compare(name, Qt::CaseSensitive))
        {
            return m_colorBars[i]; 
        }
    }
    return noPalette;
}
//----------------------------------------------------------------------------------------------------------------------------------
int PaletteOrganizer::getColorBarIndex(const QString name) const
{
    if(m_colorBarLookUp.contains(name))
        return m_colorBarLookUp[name];
    return 0;
}
//----------------------------------------------------------------------------------------------------------------------------------
QList<QString> PaletteOrganizer::getColorBarList(const int type) const
{
    QList<QString> outPut;
    outPut.clear();

    for(int i = 0; i < m_colorBars.length(); i++)
    {
        if((type != ItomPalette::NoType))
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
