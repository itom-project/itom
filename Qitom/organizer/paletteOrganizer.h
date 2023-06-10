/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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
        ItomPaletteBase(const QString &name, const char type)
        {
            m_paletteData.name = name;
            m_paletteData.type = type;
            m_paletteData.inverseColorOne = QColor();
            m_paletteData.inverseColorTwo = QColor();
            m_paletteData.invalidColor = QColor();
        }
        ItomPaletteBase(const QString &name, const char type, const QColor &invCol1, const QColor &invCol2,  const QColor &invalCol,  const QVector<QGradientStop> &colStops)
        {
            m_paletteData.name = name;
            m_paletteData.type = type;
            m_paletteData.inverseColorOne = invCol1;
            m_paletteData.inverseColorTwo = invCol2;
            m_paletteData.invalidColor = invalCol;
            m_paletteData.colorStops = colStops;
        }
        ItomPaletteBase(const QString &name, const char type,  const QColor &start,  const QColor &stop)
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
        bool setName(const QString &name);
        inline int getNumColorStops() const { return m_paletteData.colorStops.size(); }
        inline int getType() const { return m_paletteData.type; }

        double getPosFirst() const { return m_paletteData.colorStops[0].first; }
        double getPosLast() const { return m_paletteData.colorStops[m_paletteData.colorStops.size()-1].first; }
        double getPos(unsigned int color) const;

        bool   setInverseColorOne(const QColor &color);
        QColor getInverseColorOne() const { return m_paletteData.inverseColorOne; }
        bool   setInverseColorTwo(const QColor &color);
        QColor getInverseColorTwo() const { return m_paletteData.inverseColorTwo; }

        bool setInvalidColor(const QColor &color);
        QColor getInvalidColor() const;

        int findUpper(double pos) const;
        inline QVector<QPair<double, QColor> > getColorStops(void) const { return m_paletteData.colorStops; }
        bool setColorStops(const QVector<QPair<double, QColor> > &colorStops);
        QColor getColor(unsigned int index) const;

        ItomPalette getPalette() const;

        bool isWriteProtected() const { return m_paletteData.type & ito::tPaletteReadOnly; }
        inline void setWriteProtection() { m_paletteData.type = m_paletteData.type | ito::tPaletteReadOnly; }
        void removeWriteProtection() { m_paletteData.type = m_paletteData.type & (~ito::tPaletteReadOnly); }
        bool insertColorStop( double pos, const QColor &color );

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
        ito::RetVal saveColorPaletteToSettings(const ItomPaletteBase &palette, QSettings &settings) const;

        /* load a color palette from the settings. Settings must already be opened in the group,
        where the palette should be loaded from. The current group must hereby consist of a subgroup with the group name 'name'. */
        ito::RetVal loadColorPaletteFromSettings(const QString &paletteName, ItomPaletteBase &palette, QSettings &settings) const;

        ItomPaletteBase getColorPalette(const int index) const;
        ItomPaletteBase getColorPalette(const QString &name, bool *found = NULL) const;
        int getColorBarIndex(const QString& name, bool *found = NULL) const;

        QList<QString> getColorPaletteList(const int type = ito::tPaletteNoType) const;
        QList<QString> getBuiltInPaletteNames() const { return m_builtInPalettes; }
        int numberOfColorPalettes() const { return m_colorPalettes.length(); }
        bool removeColorPalette(const int index);

    private:
        void calcColorPaletteLut();

        QList<QString>         m_restrictedKeyWords;
        QList<ItomPaletteBase> m_colorPalettes;
        QList<QString>         m_builtInPalettes;
        QHash<QString,int>     m_colorPaletteLUT;

        ItomPaletteBase noPalette;

    public slots:
        ito::RetVal setColorBarThreaded(QString name, ito::ItomPaletteBase newPalette, ItomSharedSemaphore *waitCond = NULL);
        ito::RetVal getColorBarThreaded(QString name, QSharedPointer<ito::ItomPaletteBase> palette, ItomSharedSemaphore *waitCond = NULL);
        ito::RetVal getColorBarListThreaded(QSharedPointer<QStringList> palettes, ItomSharedSemaphore *waitCond = NULL);
};
} //namespace ito
#endif
