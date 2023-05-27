/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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

*********************************************************************** */

#include "statusLed.h"

#include <QtGui/QPainter>

class StatusLedPrivate
{
  Q_DECLARE_PUBLIC(StatusLed);
protected:
  StatusLed* const q_ptr;
public:

  StatusLedPrivate(StatusLed& object);

  QRadialGradient m_gradientOn;
  QRadialGradient m_gradientOff;
  QColor m_colorsOn[2];
  QColor m_colorsOff[2];
  bool m_status;

private:
  Q_DISABLE_COPY(StatusLedPrivate);
};

// --------------------------------------------------------------------------
StatusLedPrivate::StatusLedPrivate(StatusLed& object)
  :q_ptr(&object)
{
    m_colorsOn[0] = Qt::white;
    m_colorsOff[0] = Qt::white;
    m_colorsOn[1] = Qt::green;
    m_colorsOff[1] = Qt::red;
    m_gradientOn.setColorAt(0.0, m_colorsOn[0]);
    m_gradientOn.setColorAt(1.0, m_colorsOn[1]);
    m_gradientOff.setColorAt(0.0, m_colorsOff[0]);
    m_gradientOff.setColorAt(1.0, m_colorsOff[1]);
    m_status = false; //Off
}

//---------------------------------------------------------------------------
StatusLed::StatusLed(QWidget *parent) :
    QWidget(parent),
    d_ptr(new StatusLedPrivate(*this))
{
    Q_D(StatusLed);
}

//---------------------------------------------------------------------------
StatusLed::~StatusLed()
{
}

//---------------------------------------------------------------------------
bool StatusLed::checked() const
{
    Q_D(const StatusLed);
    return d->m_status;
}

//---------------------------------------------------------------------------
QColor StatusLed::colorOnEdge() const
{
    Q_D(const StatusLed);
    return d->m_colorsOn[1];
}

//---------------------------------------------------------------------------
QColor StatusLed::colorOnCenter() const
{
    Q_D(const StatusLed);
    return d->m_colorsOn[0];
}

//---------------------------------------------------------------------------
QColor StatusLed::colorOffEdge() const
{
    Q_D(const StatusLed);
    return d->m_colorsOff[1];
}

//---------------------------------------------------------------------------
QColor StatusLed::colorOffCenter() const
{
    Q_D(const StatusLed);
    return d->m_colorsOff[0];
}


//---------------------------------------------------------------------------
void StatusLed::setColorOnEdge(const QColor &color)
{
    Q_D(StatusLed);
    d->m_colorsOn[1] = color;
    d->m_gradientOn.setColorAt(1.0, color);
    update();
}

//---------------------------------------------------------------------------
void StatusLed::setColorOnCenter(const QColor &color)
{
    Q_D(StatusLed);
    d->m_colorsOn[0] = color;
    d->m_gradientOn.setColorAt(0.0, color);
    update();
}

//---------------------------------------------------------------------------
void StatusLed::setColorOffEdge(const QColor &color)
{
    Q_D(StatusLed);
    d->m_colorsOff[1] = color;
    d->m_gradientOff.setColorAt(1.0, color);
    update();
}

//---------------------------------------------------------------------------
void StatusLed::setColorOffCenter(const QColor &color)
{
    Q_D(StatusLed);
    d->m_colorsOff[0] = color;
    d->m_gradientOff.setColorAt(0.0, color);
    update();
}

//---------------------------------------------------------------------------
void StatusLed::setChecked(bool checked)
{
    Q_D(StatusLed);
    d->m_status = checked;
    update();
}

//---------------------------------------------------------------------------
void StatusLed::paintEvent(QPaintEvent * /*event*/)
{
    Q_D(StatusLed);

    QPainter p(this);



    p.setRenderHint(QPainter::Antialiasing);

    int radius = qMin(rect().width(), rect().height()) / 2 - 2;

    if (isEnabled())
    {
        QRadialGradient *g = d->m_status ? &(d->m_gradientOn) : &(d->m_gradientOff);
        g->setCenter(rect().center());
        g->setFocalPoint(rect().center() - QPoint(radius / 2, radius / 2));
        g->setRadius(radius);

        p.setBrush(*g);

        if (d->m_status)
        {
            p.setPen(QPen(d->m_colorsOn->darker(200)));
        }
        else
        {
            p.setPen(QPen(d->m_colorsOff->darker(200)));
        }
    }
    else
    {
        QRadialGradient gradient;
        gradient.setColorAt(0.0, Qt::white);
        gradient.setColorAt(1.0, Qt::gray);
        gradient.setCenter(rect().center());
        gradient.setFocalPoint(rect().center() - QPoint(radius / 2, radius / 2));
        gradient.setRadius(radius);

        p.setBrush(gradient);

        p.setPen(QPen(QColor(Qt::gray).darker(200)));
    }

    p.drawEllipse(rect().center(), radius, radius);
}
