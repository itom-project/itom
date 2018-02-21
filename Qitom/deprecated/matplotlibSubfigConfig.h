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

#ifndef MATPLOTLIBSUBFIGCONFIG_H
#define MATPLOTLIBSUBFIGCONFIG_H

#include <QtGui>
#include <qdialog.h>

#include "../GeneratedFiles/ui_matplotlibSubfigConfig.h"

class MatplotlibSubfigConfig : public QDialog 
{
    Q_OBJECT
public:
    MatplotlibSubfigConfig(int valLeft, int valTop, int valRight, int valBottom, int valWSpace, int valHSpace, QWidget *parent = 0) :
        QDialog(parent)
    {
        ui.setupUi(this);

        ui.sliderLeft->setSliderPosition(valLeft);
        ui.sliderTop->setSliderPosition(valTop);
        ui.sliderRight->setSliderPosition(valRight);
        ui.sliderBottom->setSliderPosition(valBottom);
        ui.sliderWSpace->setSliderPosition(valWSpace);
        ui.sliderHSpace->setSliderPosition(valHSpace);
    }

    ~MatplotlibSubfigConfig() {};

    QSlider *sliderLeft()   { return ui.sliderLeft;   };
    QSlider *sliderTop()    { return ui.sliderTop;    };
    QSlider *sliderRight()  { return ui.sliderRight;  };
    QSlider *sliderBottom() { return ui.sliderBottom; };
    QSlider *sliderHSpace() { return ui.sliderHSpace; };
    QSlider *sliderWSpace() { return ui.sliderWSpace; };

private:
    Ui::frmMatplotlibSubfigConfig ui;

private slots:

};

#endif