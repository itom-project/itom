/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.

    This file is a port and modified version of the 
    LGPL library QSint (https://sourceforge.net/p/qsint)
*********************************************************************** */

#include "statusLed.h"

#include <QtGui/QPainter>

//---------------------------------------------------------------------------
StatusLed::StatusLed(QWidget *parent) :
    QWidget(parent)
{
    setColor(Qt::gray);
}

//---------------------------------------------------------------------------
void StatusLed::setColor(const QColor &ledColor)
{
    m_gradient.setColorAt(0.0, Qt::white);
    m_gradient.setColorAt(1.0, ledColor);
}

//---------------------------------------------------------------------------
void StatusLed::setColors(const QColor &ledColor, const QColor &highlightColor)
{
    m_gradient.setColorAt(0.0, highlightColor);
    m_gradient.setColorAt(1.0, ledColor);
}

//---------------------------------------------------------------------------
void StatusLed::paintEvent(QPaintEvent * /*event*/)
{
    QPainter p(this);

    p.setPen(QPen(Qt::black));
    p.setRenderHint(QPainter::Antialiasing);

    int radius = qMin(rect().width(), rect().height()) / 2 - 2;

    m_gradient.setCenter(rect().center());
    m_gradient.setFocalPoint(rect().center() - QPoint(radius / 2, radius / 2));
    m_gradient.setRadius(radius);

    p.setBrush(m_gradient);

    p.drawEllipse(rect().center(), radius, radius);
}

