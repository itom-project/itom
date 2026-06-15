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

#include <iostream>
#include <fstream>
#include "guiHelper.h"

#include <qscreen.h>
#include <qapplication.h>
#include <qglobal.h>
#include <qdir.h>
#include <qdebug.h>

#ifdef _WIN32
#include <io.h>
#define access _access_s
#else
#include <unistd.h>
#endif

namespace ito
{
    int GuiHelper::dpi = 0;
    QList<int> GuiHelper::screensDPI;

    //-----------------------------------------------------------------------------------
    /*
    returns the number of dots per inch of the desktop screen or 96 in case that it can not be determined!
    */
    int GuiHelper::getScreenLogicalDpi(const QPoint *pos /*= nullptr*/)
    {
#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
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
            //qDebug() << "logical" << dpi << "physical" << currentScreen->physicalDotsPerInch();
        }
        else
        {
            dpi = 96;
        }

        /*QList<QScreen*> screens = qApp->screens();

        for (int ii = 0; ii < screens.length(); ++ii)

        {
            QSize pixelSize = screens[ii]->size();

            QSizeF physicalSize = screens[ii]->physicalSize();

            double devicePixelRatio = screens[ii]->devicePixelRatio();

            double logicalDPIX = screens[ii]->logicalDotsPerInchX();

            double logicalDPIY = screens[ii]->logicalDotsPerInchY();

            double logicalDPI = screens[ii]->logicalDotsPerInch();

            double physicalDPIX = screens[ii]->physicalDotsPerInchX();

            double physicalDPIY = screens[ii]->physicalDotsPerInchY();

            double physicalDPI = screens[ii]->physicalDotsPerInch();


            double pixelValX = pixelSize.width();

            double pixelValY = pixelSize.height();

            double physicalSizeX_cm = physicalSize.width() / 10.0;

            double physicalSizeY_cm = physicalSize.height() / 10.0;

            double calcPixelPerCMX = pixelValX / physicalSizeX_cm;

            double calcPixelPerCMY = pixelValY / physicalSizeY_cm;


            double givenLogicalDotsPerCMX = logicalDPIX * 2.54;

            double givenLogicalDotsPerCMY = logicalDPIY * 2.54;

            double givenLogicalDotsPerCM = logicalDPI * 2.54;


            double givenPhysicalDotsPerCMX = physicalDPIX * 2.54;

            double givenPhysicalDotsPerCMY = physicalDPIY * 2.54;

            double givenPhysicalDotsPerCM = physicalDPI * 2.54;


            double ratioLogicalDPCMvsPPCMX = givenLogicalDotsPerCMX / calcPixelPerCMX;

            double ratioLogicalDPCMvsPPCMY = givenLogicalDotsPerCMY / calcPixelPerCMY;

            double ratioPhysicalDPCMvsPPCMX = givenPhysicalDotsPerCMX / calcPixelPerCMX;

            double ratioPhysicalDPCMvsPPCMY = givenPhysicalDotsPerCMY / calcPixelPerCMY;


            qDebug() << "\n\nScreen; " << ii;

            qDebug() << "logicalDPI; " << logicalDPI;

            qDebug() << "physicalDPI; " << physicalDPI;

            qDebug() << "Device Pixel Ratio; " << devicePixelRatio;

            qDebug() << "Pixel in X-Direction; " << pixelValX;

            qDebug() << "Pixel in Y-Direction; " << pixelValY;

            qDebug() << "Physical Size X-Direction in CM; " << physicalSizeX_cm;

            qDebug() << "Physical Size Y-Direction in CM; " << physicalSizeY_cm;

            qDebug() << "Calculated Pixel Per CM in X-Direction; " << calcPixelPerCMX;

            qDebug() << "Calculated Pixel Per CM in Y-Direction; " << calcPixelPerCMY;

            qDebug() << "Qt Logical Dots Per CM in X-Direction; " << givenLogicalDotsPerCMX;

            qDebug() << "Qt Logical Dots Per CM in Y-Direction; " << givenLogicalDotsPerCMY;

            qDebug() << "Qt Logical Dots Per CM Average; " << givenLogicalDotsPerCM;

            qDebug() << "Qt Physical Dots Per CM in X-Direction; " << givenPhysicalDotsPerCMX;

            qDebug() << "Qt Physical Dots Per CM in Y-Direction; " << givenPhysicalDotsPerCMY;

            qDebug() << "Qt Physical Dots Per CM Average; " << givenPhysicalDotsPerCM;

            qDebug() << "Ratio of Logical Dots Per CM vs Pixel Per CM in X-Direction; "

                     << ratioLogicalDPCMvsPPCMX;

            qDebug() << "Ratio of Logical Dots Per CM vs Pixel Per CM in Y-Direction; "

                     << ratioLogicalDPCMvsPPCMY;

            qDebug() << "Ratio of Physical Dots Per CM vs Pixel Per CM in X-Direction; "

                     << ratioPhysicalDPCMvsPPCMX;

            qDebug() << "Ratio of Physical Dots Per CM vs Pixel Per CM in Y-Direction; "

                     << ratioPhysicalDPCMvsPPCMY;
        }*/
#else
          QList<QScreen*> screens = QApplication::screens();
          if (screens.size() > 0)
          {
              dpi = screens[0]->logicalDotsPerInch();
          }

          if (screens.size() == 0 || dpi <= 0)
          {
              return 96;
          }
#endif

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
