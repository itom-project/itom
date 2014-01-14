/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
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

#ifndef MATPLOTLIBWIDGET_H
#define MATPLOTLIBWIDGET_H

#include <qwidget.h>
#include <qstring.h>
#include <qevent.h>
#include <qpainter.h>
#include <qpoint.h>
#include <qtimer.h>
#include <qcoreapplication.h>
#include <qapplication.h>
#include <qgraphicsview.h>
#include <qgraphicsscene.h>
#include <qgraphicsitem.h>
#include <qqueue.h>

class MatplotlibWidget : public QGraphicsView
{
    Q_OBJECT
public:
    MatplotlibWidget(QWidget * parent = 0);
    ~MatplotlibWidget() {};

protected:
    virtual void keyPressEvent ( QKeyEvent * event );
    virtual void keyReleaseEvent ( QKeyEvent * event );
    virtual void leaveEvent ( QEvent * event );
    virtual void enterEvent ( QEvent * event );
    virtual void wheelEvent( QWheelEvent * event );
    virtual void mouseDoubleClickEvent ( QMouseEvent * event );
    virtual void mouseMoveEvent ( QMouseEvent * event );
    virtual void mousePressEvent ( QMouseEvent * event );
    virtual void mouseReleaseEvent ( QMouseEvent * event );
    virtual void resizeEvent ( QResizeEvent * event );
    //virtual void paintEvent ( QPaintEvent * event );

    virtual void showEvent ( QShowEvent * event );


private:

    struct PendingEvent
    { 
    public:
        PendingEvent() : m_type(0), m_x(0), m_y(0), m_button(0), m_h(0), m_w(0), m_valid(false) {};
        PendingEvent(int x, int y, int button) : m_type(typeMouseMove), m_x(x), m_y(y), m_button(button), m_h(0), m_w(0), m_valid(true) {};
        PendingEvent(int h, int w) : m_type(typeResize), m_x(0), m_y(0), m_button(0), m_h(h), m_w(w), m_valid(true)  {};
        PendingEvent(const PendingEvent &cpy) : m_type(cpy.m_type), m_x(cpy.m_x), m_y(cpy.m_y), m_button(cpy.m_button), m_h(cpy.m_h), m_w(cpy.m_w), m_valid(cpy.m_valid) {};
        bool isValid() { return m_valid; };
        void clear() { m_valid = false; };
        enum tPendingEventType { typeResize, typeMouseMove };

        tPendingEventType m_type;
        int m_x;
        int m_y;
        int m_button;
        int m_h;
        int m_w;
        bool m_valid;
    };

    void handleMouseEvent( int type, QMouseEvent *event);

    QPixmap m_pixmap;
    QRect m_pixmapRect;

    QTimer m_timer;
    bool m_internalResize; //resize has been done, but resizeEvent should not request a python-side refresh of the image (if true)

    PendingEvent m_pendingEvent;

    QGraphicsScene *m_scene;
    QGraphicsRectItem *m_rectItem;
    QGraphicsPixmapItem *m_pixmapItem;

signals:
    void eventLeaveEnter(bool enter);
    void eventMouse(int type, int x, int y, int button);
    void eventWheel(int x, int y, int delta, int orientation);
    void eventKey(int type, int keyId, QString keyString, bool autoRepeat);
    void eventResize(int w, int h);
    void eventPaintRequest();
    void eventIdle();

public slots:
    void externalResize(int width, int height);
    void paintResult(QByteArray imageString, int x, int y, int w, int h, bool blit );
    void paintRect(bool drawRect, int x = 0, int y = 0, int w = 0, int h = 0);
    void paintTimeout();
    void stopTimer()
    {
        m_timer.stop();
    }

    void setCursors(int cursorId)
    {
        QApplication::restoreOverrideCursor();
        switch(cursorId)
        {
        case Qt::ArrowCursor:
            QApplication::setOverrideCursor( QCursor(Qt::ArrowCursor) );
            break;
        case Qt::CrossCursor:
            QApplication::setOverrideCursor( QCursor(Qt::CrossCursor) );
            break;
        case Qt::SizeAllCursor:
            QApplication::setOverrideCursor( QCursor(Qt::SizeAllCursor) );
            break;
        case Qt::PointingHandCursor:
            QApplication::setOverrideCursor( QCursor(Qt::PointingHandCursor) );
            break;
        }
    };


};

#endif
