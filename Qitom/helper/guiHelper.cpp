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

#include "guiHelper.h"

#include <qscreen.h>
#include <qapplication.h>
#include <qglobal.h>

namespace ito
{
    int GuiHelper::dpi = 0;

    //-----------------------------------------------------------------------------------
    /*
    returns the number of dots per inch of the desktop screen or 96 in case that it can not be determined!
    */
    int GuiHelper::getScreenLogicalDpi(const QPoint *pos /*= nullptr*/)
    {
        QScreen *currentScreen;
        if (pos)
        {
            currentScreen = QApplication::screenAt(*pos);
        }
        else
        {
            currentScreen = QApplication::primaryScreen();
        }

        if (currentScreen)
        {
            dpi = currentScreen->logicalDotsPerInch();            
        }
        else
        {
            dpi = 96;
        }

        return dpi;
    }

    //-------------------------------------------------------------------------------------
    /* returns a screen dpi scaling factor >= 1. A factor of 1 is related to a default
       screen resolution of 96dpi. A factor higher than 1 is the factor between 
       real screen resolution and 96dpi. 
    */
    float GuiHelper::screenDpiFactor(const QPoint* pos /*= nullptr*/)
    {
        int dpi = getScreenLogicalDpi(pos);
        return qBound(1.0, (float)dpi / 96.0, 1.e10);
    }
};
