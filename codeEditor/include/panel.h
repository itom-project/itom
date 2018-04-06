#ifndef PANEL_H
#define PANEL_H

/*
This module contains the Panel API.
*/

class CodeEditor;

#include <qwidget.h>
#include <qevent.h>
#include <qbrush.h>
#include <qpen.h>
#include "mode.h"

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
        Bottom = 3
    };

    Panel(const QString &name, bool dynamic, const QString &description = "", QWidget *parent = NULL);
    virtual ~Panel();

    void setVisible(bool visible);

    bool scrollable() const;
    void setScrollable(bool value);

    void setOrderInZone(int orderInZone);

    Position position() const;
    void setPosition(Position pos);

    virtual void onInstall(CodeEditor *editor);

protected:
    virtual void paintEvent(QPaintEvent *e);
    

private:
    CodeEditor* m_pEditor;
    bool m_dynamic;
    int m_orderInZone;
    bool m_scrollable;
    QBrush m_backgroundBrush;
    QPen m_foregroundPen;

    //!< position in the editor (top, left, right, bottom)
    int m_position;
};


#endif