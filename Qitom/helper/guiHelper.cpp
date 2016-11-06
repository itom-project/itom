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

#include "guiHelper.h"

#if QT_VERSION < 0x050000
    #include <qdesktopwidget.h>
#else
    #include <qscreen.h>
    #include <qapplication.h>
#endif

#include <qglobal.h>

namespace ito
{
    int GuiHelper::dpi = 0;

    //-----------------------------------------------------------------------------------
    /*
    returns the number of dots per inch of the desktop screen or 96 in case that it can not be determined!
    */
    int GuiHelper::getScreenLogicalDpi(bool *ok /*= NULL*/)
    {
        if (dpi > 0)
        {
            if (ok)
            {
                *ok = true;
            }

            return dpi;
        }

#if QT_VERSION < 0x050000
        QDesktopWidget *dw = qApp->desktop();
        if (dw)
        {
            dpi = dw->logicalDpiX();
        }
        
        if (!dw || dpi <= 0)
        {
            if (ok)
            {
                *ok = false;
            }
            return 96;
        }
#else
        QList<QScreen*> screens = QApplication::screens();
        if (screens.size() > 0)
        {
            dpi = screens[0]->logicalDotsPerInch();
        }

        if (screens.size() == 0 || dpi <= 0)
        {
            if (ok)
            {
                *ok = false;
            }
            return 96;
        }
#endif

        if (ok)
        {
            *ok = true;
        }

        return dpi;
    }

    //-------------------------------------------------------------------------------------
    /* returns a screen dpi scaling factor >= 1. A factor of 1 is related to a default
       screen resolution of 96dpi. A factor higher than 1 is the factor between 
       real screen resolution and 96dpi. 
    */
    float GuiHelper::screenDpiFactor()
    {
        int dpi = getScreenLogicalDpi();
        return qBound(1.0, (float)dpi / 96.0, 1.e10);
    }
};
