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

#ifndef USERINTERACTIONPLOTPICKER_H
#define USERINTERACTIONPLOTPICKER_H


#include <qwt_plot_picker.h>
#include <qpainter.h>
#include <qbrush.h>

class UserInteractionPlotPicker: public QwtPlotPicker
{
    Q_OBJECT

public:
    explicit UserInteractionPlotPicker( QWidget *canvas ) : QwtPlotPicker(canvas) {}
    virtual ~UserInteractionPlotPicker() {}

    explicit UserInteractionPlotPicker( int xAxis, int yAxis, QWidget *widget ) : QwtPlotPicker(xAxis,yAxis,widget) {}

    explicit UserInteractionPlotPicker( int xAxis, int yAxis,
        RubberBand rubberBand, DisplayMode trackerMode, QWidget *widget ) :
        QwtPlotPicker(xAxis,yAxis,rubberBand,trackerMode,widget) {}

    void setBackgroundFillBrush( const QBrush &brush );


protected:
    void reset();

    void drawTracker( QPainter *painter ) const;
    void drawRubberBand( QPainter *painter ) const;

private:
    QBrush m_rectFillBrush;

};

#endif
