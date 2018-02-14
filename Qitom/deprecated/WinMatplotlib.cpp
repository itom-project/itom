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

#include "winMatplotlib.h"

#include <qicon.h>

WinMatplotlib::WinMatplotlib() :
    m_pContent(NULL),
    QMainWindow(),
    m_actHome(NULL),
    m_actForward(NULL),
    m_actBack(NULL),
    m_actPan(NULL),
    m_actZoomToRect(NULL),
    m_actSubplotConfig(NULL),
    m_actSave(NULL),
    m_toolbar(NULL),
    m_lblCoordinates(NULL),
    m_pMatplotlibSubfigConfig(NULL)
{  
    setWindowTitle("itom - matplotlib");
    setAttribute(Qt::WA_DeleteOnClose, true);

    m_actHome = new QAction(QIcon(":/matplotlib/icons/matplotlib/home.png"), tr("home"), this);
    m_actHome->setObjectName("actionHome");
    m_actHome->setToolTip("Reset original view");

    m_actForward = new QAction(QIcon(":/matplotlib/icons/matplotlib/forward.png"), tr("forward"), this);
    m_actForward->setObjectName("actionForward");
    m_actForward->setToolTip("Forward to next view");

    m_actBack = new QAction(QIcon(":/matplotlib/icons/matplotlib/back.png"), tr("back"), this);
    m_actBack->setObjectName("actionBack");
    m_actBack->setToolTip("Back to previous view");

    m_actPan = new QAction(QIcon(":/matplotlib/icons/matplotlib/move.png"), tr("move"), this);
    m_actPan->setObjectName("actionPan");
    m_actPan->setCheckable(true);
    m_actPan->setChecked(false);
    m_actPan->setToolTip("Pan axes with left mouse, zoom with right");

    m_actZoomToRect = new QAction(QIcon(":/matplotlib/icons/matplotlib/zoom_to_rect.png"), tr("zoom to rectangle"), this);
    m_actZoomToRect->setObjectName("actionZoomToRect");
    m_actZoomToRect->setCheckable(true);
    m_actZoomToRect->setChecked(false);
    m_actZoomToRect->setToolTip("Zoom to rectangle");

    m_actMarker = new QAction(QIcon(":/plots/icons_m/marker.png"), tr("marker"), this);
    m_actMarker->setObjectName("actionMarker");
    m_actMarker->setCheckable(true);
    m_actMarker->setChecked(false);
    m_actMarker->connect(m_actMarker, SIGNAL(toggled(bool)), this, SLOT(mnuMarkerClick(bool)));

    m_actSubplotConfig = new QAction(QIcon(":/matplotlib/icons/matplotlib/subplots.png"), tr("subplot configuration"), this);
    m_actSubplotConfig->setObjectName("actionSubplotConfig");
    m_actSubplotConfig->setToolTip("Configure subplots");

    m_actSave = new QAction(QIcon(":/files/icons/fileSave.png"), tr("save"), this);
    m_actSave->setObjectName("actionSave");
    m_actSave->setToolTip("Save the figure");

    m_lblCoordinates = new QLabel("",this);
    m_lblCoordinates->setAlignment( Qt::AlignRight | Qt::AlignTop);
    m_lblCoordinates->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Ignored);
    m_lblCoordinates->setObjectName("lblCoordinates");


    m_toolbar = new QToolBar(tr("toolbar"), this);
    m_toolbar->setObjectName("toolbar");
    m_toolbar->addAction(m_actHome);
    m_toolbar->addAction(m_actBack);
    m_toolbar->addAction(m_actForward);
    m_toolbar->addSeparator();
    m_toolbar->addAction(m_actPan);
    m_toolbar->addAction(m_actZoomToRect);
    m_toolbar->addAction(m_actMarker);
    m_toolbar->addSeparator();
    m_toolbar->addAction(m_actSubplotConfig);
    m_toolbar->addAction(m_actSave);

    QAction *lblAction = m_toolbar->addWidget(m_lblCoordinates);
    lblAction->setVisible(true);

    addToolBar(m_toolbar);



    m_pContent = new MatplotlibWidget(this);
    m_pContent->setObjectName("canvasWidget");


    setCentralWidget(m_pContent);
}

void WinMatplotlib::showSubplotConfig(int valLeft, int valTop, int valRight, int valBottom, int valWSpace, int valHSpace)
{
    if(m_pMatplotlibSubfigConfig == NULL)
    {
        m_pMatplotlibSubfigConfig = new MatplotlibSubfigConfig(valLeft, valTop, valRight, valBottom, valWSpace, valHSpace, this);

        connect(m_pMatplotlibSubfigConfig->sliderLeft(), SIGNAL(valueChanged(int)), this, SLOT(subplotConfigSliderLeftChanged(int)));
        connect(m_pMatplotlibSubfigConfig->sliderTop(), SIGNAL(valueChanged(int)), this, SLOT(subplotConfigSliderTopChanged(int)));
        connect(m_pMatplotlibSubfigConfig->sliderRight(), SIGNAL(valueChanged(int)), this, SLOT(subplotConfigSliderRightChanged(int)));
        connect(m_pMatplotlibSubfigConfig->sliderBottom(), SIGNAL(valueChanged(int)), this, SLOT(subplotConfigSliderBottomChanged(int)));
        connect(m_pMatplotlibSubfigConfig->sliderWSpace(), SIGNAL(valueChanged(int)), this, SLOT(subplotConfigSliderWSpaceChanged(int)));
        connect(m_pMatplotlibSubfigConfig->sliderHSpace(), SIGNAL(valueChanged(int)), this, SLOT(subplotConfigSliderHSpaceChanged(int)));
    }

    m_pMatplotlibSubfigConfig->setModal(true);
    m_pMatplotlibSubfigConfig->show();
}