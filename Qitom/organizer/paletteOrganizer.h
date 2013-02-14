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
#include "../../common/sharedStructuresGraphics.h"

#include <qhash.h>

namespace ito
{

class PaletteOrganizer : public QObject
{
    Q_OBJECT

    public:

        PaletteOrganizer();
        ~PaletteOrganizer(){};

    protected:


    private:
        
        QList<QString> restrictedKeyWords;
        QList<ItomPalette> m_colorBars;
        QHash<QString,int> m_colorBarLookUp;

        ItomPalette noPalette;

    public slots:
        ItomPalette getColorBar(const int index) const;
        ItomPalette getNextColorBar(const int curindex, const int type = ItomPalette::NoType) const;
        int getColorBarIndex(const QString name) const;
        ItomPalette getColorBar(const QString name) const;
        QList<QString> getColorBarList(const int type = ItomPalette::NoType) const;
        int numberOfColorBars() const {return m_colorBars.length();};

    private slots:

    signals:

}; 
} //namespace íto
#endif
