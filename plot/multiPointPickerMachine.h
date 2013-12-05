/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2012, Institut für Technische Optik (ITO), 
   Universität Stuttgart, Germany 
 
   This file is part of itom.

   itom is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   itom is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef MULTIPOINTPICKERMACHINE_H
#define MULTIPOINTPICKERMACHINE_H

#include <qwt_picker_machine.h>
#include <qwt_plot_picker.h>

class MultiPointPickerMachine: public QwtPickerPolygonMachine
{
public:
    MultiPointPickerMachine();

    int maxNrItems() const { return m_maxNrItems; }
    void setMaxNrItems(int value = -1);

    virtual QList<Command> transition(
        const QwtEventPattern &, const QEvent * );

protected:
    int m_maxNrItems;
    int m_currentNrItems;
};

#endif