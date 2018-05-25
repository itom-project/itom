#ifndef CARETLINHIGHLIGHT_H
#define CARETLINHIGHLIGHT_H

/*
This module contains the care line highlighter mode
*/

#include "../textDecoration.h"
#include "../mode.h"

#include <qcolor.h>

/*
Highlights the caret line
*/
class CaretLineHighlighterMode : public QObject, public Mode
{
    Q_OBJECT
public:
    CaretLineHighlighterMode(const QString &description = "", QObject *parent = NULL);
    virtual ~CaretLineHighlighterMode();

    QColor background() const;
    void setBackground(const QColor &color);

    virtual void onInstall(CodeEditor *editor);
    virtual void onStateChanged(bool state);

public slots:
    void refresh();

protected:
    void clearDeco();

    QColor m_color;
    int m_pos;
    TextDecoration::Ptr m_decoration;
};

#endif
