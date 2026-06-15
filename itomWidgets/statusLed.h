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

#ifndef STATUSLED_H
#define STATUSLED_H

#include <QWidget>
#include <qpen.h>

#include "commonWidgets.h"

class StatusLedPrivate; // forward declare

/**
    \brief Round LED-style widget with gradient fill.

        The gradient consists of two colors: led color (main color of the widget)
        and highlight color at the top-left corner (typically, white).
*/
class ITOMWIDGETS_EXPORT StatusLed : public QWidget
{
    Q_OBJECT

    Q_PROPERTY(QColor colorOnEdge READ colorOnEdge WRITE setColorOnEdge);
    Q_PROPERTY(QColor colorOnCenter READ colorOnCenter WRITE setColorOnCenter);
    Q_PROPERTY(QColor colorOffEdge READ colorOffEdge WRITE setColorOffEdge);
    Q_PROPERTY(QColor colorOffCenter READ colorOffCenter WRITE setColorOffCenter);
    Q_PROPERTY(bool checked READ checked WRITE setChecked);

public:

    explicit StatusLed(QWidget *parent = 0);
    virtual ~StatusLed();

    virtual QSize sizeHint() const
    {
        return QSize(32,32);
    }

    virtual QSize minimumSizeHint() const
    {
        return QSize(16,16);
    }

    virtual int heightForWidth(int w) const
    {
        return w;
    }

    bool checked() const;
    QColor colorOnEdge() const;
    QColor colorOnCenter() const;
    QColor colorOffEdge() const;
    QColor colorOffCenter() const;

public Q_SLOTS:
    void setColorOnEdge(const QColor &color);
    void setColorOnCenter(const QColor &color);
    void setColorOffEdge(const QColor &color);
    void setColorOffCenter(const QColor &color);
    void setChecked(bool checked);

protected:
    virtual void paintEvent(QPaintEvent *event);

    QScopedPointer<StatusLedPrivate> d_ptr; // QScopedPointer to forward declared class

private:
    Q_DECLARE_PRIVATE(StatusLed);
    Q_DISABLE_COPY(StatusLed);
};

#endif // STATUSLED_H
