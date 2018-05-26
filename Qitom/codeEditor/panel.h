#ifndef PANEL_H
#define PANEL_H

/*
This module contains the Panel API.
*/



#include <qwidget.h>
#include <qevent.h>
#include <qbrush.h>
#include <qpen.h>
#include "mode.h"

namespace ito {

class CodeEditor;

/*
Base class for editor panels.

A panel is a mode and a QWidget.

.. note:: Use enabled to disable panel actions and setVisible to change the
    visibility of the panel.
*/

class Panel : public QWidget, public Mode
{
    Q_OBJECT

public:
    enum Position
    {
        Top = 0,
        Left = 1,
        Right = 2,
        Bottom = 3,
        Floating = 4
    };

    typedef QSharedPointer<Panel> Ptr;

    Panel(const QString &name, bool dynamic, const QString &description = "", QWidget *parent = NULL);
    virtual ~Panel();

    void setVisible(bool visible);

    bool scrollable() const;
    void setScrollable(bool value);

    int orderInZone() const;
    void setOrderInZone(int orderInZone);

    Position position() const;
    void setPosition(Position pos);

    QBrush backgroundBrush() const { return m_backgroundBrush; }
    QPen foregroundPen() const { return m_foregroundPen; }

    virtual void onInstall(CodeEditor *editor);

protected:
    virtual void paintEvent(QPaintEvent *e);
    

private:
    bool m_dynamic;
    int m_orderInZone;
    bool m_scrollable;
    QBrush m_backgroundBrush;
    QPen m_foregroundPen;

    //!< position in the editor (top, left, right, bottom)
    Position m_position;

    Q_DISABLE_COPY(Panel)
};

} //end namespace ito

#endif