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

#ifndef PALETTEORGANIZER_H
#define PALETTEORGANIZER_H

#include "../../common/sharedStructures.h"
#include "../../common/sharedStructuresQt.h"
#include "../../common/sharedStructuresGraphics.h"

#include <qhash.h>
#include <qsharedpointer.h>
#include <qbrush.h>
#include <qsettings.h>

namespace ito
{

class ItomPaletteBase
{
    public:
        ItomPaletteBase() 
        {
            m_paletteData.name = ""; 
            m_paletteData.type = 0; 
            m_paletteData.inverseColorOne = QColor();         
            m_paletteData.inverseColorTwo = QColor();
            m_paletteData.invalidColor = QColor();
        }
        ItomPaletteBase(const QString name, const char type) 
        {
            m_paletteData.name = name;
            m_paletteData.type = type;
            m_paletteData.inverseColorOne = QColor();
            m_paletteData.inverseColorTwo = QColor();
            m_paletteData.invalidColor = QColor();
        }
        ItomPaletteBase(const QString name, const char type, QColor invCol1, QColor invCol2, QColor invalCol, QVector<QGradientStop> colStops) 
        { 
            m_paletteData.name = name;
            m_paletteData.type = type;
            m_paletteData.inverseColorOne = invCol1;
            m_paletteData.inverseColorTwo = invCol2;
            m_paletteData.invalidColor = invalCol;
            m_paletteData.colorStops = colStops;
        }
        ItomPaletteBase(const QString name, const char type, QColor start, QColor stop)
        {
            m_paletteData.name = name;
            m_paletteData.type = type;
            m_paletteData.colorStops.append(QPair<double, QColor>(0.0, start));
            m_paletteData.colorStops.append(QPair<double, QColor>(1.0, stop));
        }

        ItomPaletteBase(const ItomPaletteBase & src) : m_paletteData(src.m_paletteData)
        { }

        ~ItomPaletteBase() { 
            m_paletteData.colorStops.clear();
        }

        QString getName() const { return m_paletteData.name; }
        inline int getSize() const { return m_paletteData.colorStops.size(); }
        inline int getType() const { return m_paletteData.type; }

        double getPosFirst() const { return m_paletteData.colorStops[0].first; }
        double getPosLast() const { return m_paletteData.colorStops[m_paletteData.colorStops.size()-1].first; }
        double getPos(unsigned int color) const;

        bool   setInverseColorOne(const QColor color);
        QColor getInverseColorOne() const { return m_paletteData.inverseColorOne; }
        bool   setInverseColorTwo(const QColor color);
        QColor getInverseColorTwo() const { return m_paletteData.inverseColorTwo; }
        
        bool setInvalidColor(const QColor color);
        QColor getInvalidColor() const;

        QColor getColorFirst() const { return m_paletteData.colorStops[0].second; }
        QColor getColorLast() const { return m_paletteData.colorStops[m_paletteData.colorStops.size() - 1].second; }
        int findUpper(double pos) const;
        inline QVector<QPair<double, QColor> > getColorStops(void) const { return m_paletteData.colorStops; }
        QColor getColor(unsigned int index) const;

        void update(const bool updateInverseColors);
        ItomPalette getPalette();

        inline void setWriteProtection() { m_paletteData.type = m_paletteData.type | ito::tPaletteReadOnly; return; }
        bool insertColorStop( double pos, const QColor color );
        void calculateInverseColors(QColor &inv1, QColor &inv2);

        QVector<ito::uint32> get256Colors(bool includeAlpha = false) const;

    private:
        ItomPalette m_paletteData;
};

//----------------------------------------------------------------------------------
class PaletteOrganizer : public QObject
{
    Q_OBJECT

    public:
        PaletteOrganizer();
        ~PaletteOrganizer(){};

        /* save the given color palette to the settings. Settings must already be opened in the group, 
        where the palette should be saved. Each palette is stored as a subgroup of the current group. */
        ito::RetVal saveColorPaletteToSettings(const ItomPaletteBase &palette, QSettings &settings); 

        /* load a color palette from the settings. Settings must already be opened in the group, 
        where the palette should be loaded from. The current group must hereby consist of a subgroup with the group name 'name'. */
        ito::RetVal loadColorPaletteFromSettings(const QString &paletteName, ItomPaletteBase &palette, QSettings &settings); 

    private:        
        void calcColorBarLut();

        QList<QString> restrictedKeyWords;
        QList<ItomPaletteBase> m_colorBars;
        QList<QString> m_builtInPalettes;
        QHash<QString,int> m_colorBarLookUp;

        ItomPaletteBase noPalette;

    public slots:
        ItomPaletteBase getColorBar(const int index) const;
        ItomPaletteBase getNextColorBar(const int curindex, const int type = ito::tPaletteNoType) const;
        int getColorBarIndex(const QString& name, bool *found = NULL) const;
        ItomPaletteBase getColorBar(const QString &name, bool *found = NULL) const;
        QList<QString> getColorBarList(const int type = ito::tPaletteNoType) const;
        QList<QString> getBuiltInPaletteNames() const { return m_builtInPalettes; }
        int numberOfColorBars() const { return m_colorBars.length(); }
        bool removeColorbar(const int index);

        ito::RetVal setColorBarThreaded(QString name, ito::ItomPaletteBase newPalette, ItomSharedSemaphore *waitCond = NULL);
        ito::RetVal getColorBarThreaded(QString name, QSharedPointer<ito::ItomPaletteBase> palette, ItomSharedSemaphore *waitCond = NULL);
        ito::RetVal getColorBarListThreaded(int types, QSharedPointer<QStringList> palettes, ItomSharedSemaphore *waitCond = NULL);
}; 
} //namespace íto
#endif
