/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#ifndef WINMATPLOTLIB_H
#define WINMATPLOTLIB_H

#include "../widgets/matplotlibWidget.h"

#include "../ui/matplotlibSubfigConfig.h"

#include <qmainwindow.h>
#include <qwidget.h>
#include <qaction.h>
#include <qtoolbar.h>
#include <qlabel.h>
#include <qlayout.h>



class WinMatplotlib : public QMainWindow
{
    Q_OBJECT

public:
    WinMatplotlib();
    ~WinMatplotlib() 
    {
        if(m_pMatplotlibSubfigConfig)
        {
            m_pMatplotlibSubfigConfig->deleteLater();
            m_pMatplotlibSubfigConfig = NULL;
        }
    };

private:
    MatplotlibWidget *m_pContent;

    QAction *m_actHome;
    QAction *m_actForward;
    QAction *m_actBack;
    QAction *m_actPan;
    QAction *m_actZoomToRect;
    QAction *m_actSubplotConfig;
    QAction *m_actSave;
    QAction *m_actMarker;

    QLabel *m_lblCoordinates;

    QToolBar *m_toolbar;

    MatplotlibSubfigConfig *m_pMatplotlibSubfigConfig;

signals:
    void subplotConfigSliderChanged(int type, int value);

private slots:
    void mnuMarkerClick(bool checked)
    {
        if(m_pContent)
        {
            m_pContent->setMouseTracking(checked);
        }
    }

    void subplotConfigSliderLeftChanged(int value) { emit subplotConfigSliderChanged(0, value); };
    void subplotConfigSliderTopChanged(int value) { emit subplotConfigSliderChanged(1, value); };
    void subplotConfigSliderRightChanged(int value) { emit subplotConfigSliderChanged(2, value); };
    void subplotConfigSliderBottomChanged(int value) { emit subplotConfigSliderChanged(3, value); };
    void subplotConfigSliderWSpaceChanged(int value) { emit subplotConfigSliderChanged(4, value); };
    void subplotConfigSliderHSpaceChanged(int value) { emit subplotConfigSliderChanged(5, value); };

public slots:
    void showSubplotConfig(int valLeft, int valTop, int valRight, int valBottom, int valWSpace, int valHSpace);
};

#endif