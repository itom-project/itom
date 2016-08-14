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

#ifndef STATUSLED_H
#define STATUSLED_H

#include <QWidget>

#include "commonWidgets.h"

/**
    \brief Round LED-style widget with gradient fill.

        The gradient consists of two colors: led color (main color of the widget)
        and highlight color at the top-left corner (typically, white).
*/
class ITOMWIDGETS_EXPORT StatusLed : public QWidget
{
    Q_OBJECT
public:
    explicit StatusLed(QWidget *parent = 0);

    virtual QSize minimumSizeHint() const
    { return QSize(12,12); }

    virtual int heightForWidth(int w) const
    { return w; }

public Q_SLOTS:
    /**
     * @brief setColor Funtion sets color of LED to \a ledColor. Highlight color is set to Qt::white.
     * @param ledColor Color to set (Qt::gray is the default value).
     */
    void setColor(const QColor &ledColor);
    /**
     * @brief setColors Funtion sets color of LED to \a ledColor and its highlight color to \a blickColor.
     * @param ledColor (Qt::gray is the default value).
     * @param highlightColor (Qt::white is the default value).
     */
    void setColors(const QColor &ledColor, const QColor &highlightColor);

protected:
    virtual void paintEvent(QPaintEvent *event);

    QRadialGradient m_gradient;
};

#endif // STATUSLED_H
