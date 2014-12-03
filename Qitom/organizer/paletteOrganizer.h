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

#ifndef PALETTEORGANIZER_H
#define PALETTEORGANIZER_H

#include "../../common/sharedStructures.h"
#include "../../common/sharedStructuresQt.h"
#include "../../common/sharedStructuresGraphics.h"

#include <qhash.h>
#include <qsharedpointer.h>
namespace ito
{

class ItomPaletteBase
{
    public:

        ItomPaletteBase():m_name(""), m_type(0), m_inverseColorOne(), m_inverseColorTwo() {m_colorStops.clear();};
        ItomPaletteBase(const QString name, const char type): m_name(name), m_type(type), m_inverseColorOne(), m_inverseColorTwo() {m_colorStops.clear();};
        ItomPaletteBase(const QString name, const char type, QColor start, QColor stop): m_name(name), m_type(type), m_inverseColorOne(), m_inverseColorTwo() 
        {
            m_colorStops.clear();
            m_colorStops.append(QPair<double, QColor>(0.0, start));
            m_colorStops.append(QPair<double, QColor>(1.0, stop));
        };

        ItomPaletteBase(const ItomPaletteBase & scr)
        {
            m_inverseColorOne = scr.m_inverseColorOne;
            m_inverseColorTwo = scr.m_inverseColorTwo;
            m_name = scr.m_name;
            m_type = scr.m_type;
            m_colorStops.clear();
            m_colorStops = scr.m_colorStops;
            m_paletteStucture = scr.m_paletteStucture;
        }

        ~ItomPaletteBase() {m_colorStops.clear(); m_paletteStucture.colorStops.clear(); m_paletteStucture.colorVector256.clear();};
/*
        enum tPalette{
            NoType      = 0x00,
            GrayPalette = 0x01,
            RGBPalette  = 0x02,
            FCPalette  = 0x04,
            indexPalette = 0x08,
            LinearPalette = 0x10,
            ReadOnlyPalette = 0x20,
        };
*/
        QString getName() const {return m_name;};
        inline int getSize() const {return m_colorStops.size();};
        inline int getType() const {return m_type;};

        double getPosFirst() const {return m_colorStops[0].first;};
        double getPosLast() const {return m_colorStops[m_colorStops.size()-1].first;};
        double getPos(unsigned int color) const;

        bool   setInversColorOne(const QColor color);
        QColor getInversColorOne() const {return m_inverseColorOne;};
        bool   setInversColorTwo(const QColor color);
        QColor getInversColorTwo() const {return m_inverseColorTwo;};
        
        QColor getColorFirst() const {return m_colorStops[0].second;};
        QColor getColorLast() const {return m_colorStops[m_colorStops.size()-1].second;};
        QColor getColor(unsigned int color) const;

        void update(const bool updateInverseColors);
        ItomPalette getPalette();

        inline void setWriteProtection() { m_type = m_type | ito::tPaletteReadOnly; return;};
        bool insertColorStop( double pos, const QColor color );
        void calculateInverseColors(QColor &inv1, QColor &inv2);
        //QColor getColor(double pos) const;

        QVector<ito::uint32> get256Colors() const;

    protected:
        inline void removeWriteProtection() { m_type = m_type & !ito::tPaletteReadOnly; return;};

    private:
        QString m_name;
        char m_type; 
        
        ItomPalette m_paletteStucture;

        QColor m_inverseColorTwo;
        QColor m_inverseColorOne;

        QVector<QPair<double, QColor> > m_colorStops;

        int findUpper( double pos ) const;

    public slots:

    private slots:

    signals:

};

class PaletteOrganizer : public QObject
{
    Q_OBJECT

    public:

        PaletteOrganizer();
        ~PaletteOrganizer(){};

    protected:


    private:
        
        QList<QString> restrictedKeyWords;
        QList<ItomPaletteBase> m_colorBars;
        QHash<QString,int> m_colorBarLookUp;

        ItomPaletteBase noPalette;

    public slots:
        ItomPaletteBase getColorBar(const int index) const;
        ItomPaletteBase getNextColorBar(const int curindex, const int type = ito::tPaletteNoType) const;
        int getColorBarIndex(const QString name, bool *found = NULL) const;
        ItomPaletteBase getColorBar(const QString name, bool *found = NULL) const;
        QList<QString> getColorBarList(const int type = ito::tPaletteNoType) const;
        int numberOfColorBars() const {return m_colorBars.length();};

        ito::RetVal setColorBarThreaded(QString name, ito::ItomPaletteBase newPalette, ItomSharedSemaphore *waitCond = NULL);
        ito::RetVal getColorBarThreaded(QString name, QSharedPointer<ito::ItomPaletteBase> palette, ItomSharedSemaphore *waitCond = NULL);
        ito::RetVal getColorBarListThreaded(int types, QSharedPointer<QStringList> palettes, ItomSharedSemaphore *waitCond = NULL);
    private slots:

    signals:

}; 
} //namespace íto
#endif
