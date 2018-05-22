#ifndef LINENUMBER_H
#define LINENUMBER_H

/*
This module contains the line number panel
*/

#include "panel.h"
#include "utils/utils.h"

#include <qevent.h>
#include <qsize.h>
#include <qcolor.h>

/*
Displays the document line numbers.
*/
class LineNumberPanel : public Panel
{
    Q_OBJECT
public:
    LineNumberPanel(const QString &description = "", QWidget *parent = NULL);
    virtual ~LineNumberPanel();

    virtual QSize sizeHint() const;
    int lineNumberAreaWidth() const;
    void cancelSelection();

protected:
    virtual void paintEvent(QPaintEvent *e);
    virtual void mousePressEvent(QMouseEvent *e);
    virtual void mouseMoveEvent(QMouseEvent *e);
    virtual void mouseReleaseEvent(QMouseEvent *e);
    virtual void wheelEvent(QWheelEvent *e);

private:
    bool m_selecting;
    int m_startLine;
    int m_selStart;
    QColor m_lineColorU;
    QColor m_lineColorS;
};

#endif
