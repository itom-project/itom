/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut für Technische Optik (ITO),
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


#pragma once

#include <QApplication>
#include <QMouseEvent>
#include <QPoint>
#include <QtTest/QTest>
#include <QWidget>

namespace itomWidgetsTest {

//! shows the widget with a fixed size and waits until it has been exposed.
/*
 * Many widgets only compute their geometry (or hide/show their children)
 * once they have been created and shown. Returns false if the widget
 * could not be exposed within the timeout.
 */
inline bool showAndWait(QWidget& widget, const QSize& size = QSize(300, 60))
{
    widget.resize(size);
    widget.show();
    return QTest::qWaitForWindowExposed(&widget);
}

//! sends a mouse event with an explicit button state to the widget.
/*
 * QTest::mouseMove does not reliably carry the pressed button state across
 * all supported Qt versions (5.11 - 6.x). Drag operations therefore use
 * explicit QMouseEvent objects. This constructor signature is available
 * in Qt5 and Qt6.
 */
inline void sendMouseEvent(
    QWidget* widget,
    QEvent::Type type,
    const QPoint& pos,
    Qt::MouseButton button,
    Qt::MouseButtons buttons)
{
    QMouseEvent event(
        type, QPointF(pos), QPointF(widget->mapToGlobal(pos)), button, buttons, Qt::NoModifier);
    QApplication::sendEvent(widget, &event);
}

//! drags with the left mouse button from 'from' to 'to' in a number of steps.
inline void dragMouse(QWidget* widget, const QPoint& from, const QPoint& to, int steps = 10)
{
    sendMouseEvent(widget, QEvent::MouseButtonPress, from, Qt::LeftButton, Qt::LeftButton);

    for (int i = 1; i <= steps; ++i)
    {
        const QPoint p = from + (to - from) * i / steps;
        sendMouseEvent(widget, QEvent::MouseMove, p, Qt::NoButton, Qt::LeftButton);
    }

    sendMouseEvent(widget, QEvent::MouseButtonRelease, to, Qt::LeftButton, Qt::NoButton);
}

} // namespace itomWidgetsTest
