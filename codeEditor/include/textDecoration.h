#ifndef TEXTDECORATIONS_H
#define TEXTDECORATIONS_H

/*
This module contains the text decoration API.
*/

class CodeEditor;

#include <qwidget.h>
#include <qtextedit.h>
#include <qtextcursor.h>
#include <qsharedpointer.h>

/*
Holds the signals for a TextDecoration (since we cannot make it a
QObject, we need to store its signals in an external QObject).
*/
class TextDecorationsSignals : public QObject
{
    Q_OBJECT
public:
    TextDecorationsSignals(QObject *parent = NULL) : QObject(parent) {}
    virtual ~TextDecorationsSignals() {}

signals:
    //Signal emitted when a TextDecoration has been clicked.
    void clicked();
};
/*
Helper class to quickly create a text decoration. The text decoration is an
utility class that adds a few utility methods to QTextEdit.ExtraSelection.

In addition to the helper methods, a tooltip can be added to a decoration.
(useful for errors markers and so on...)

Text decoration expose a **clicked** signal stored in a separate QObject:
    :attr:`pyqode.core.api.TextDecoration.Signals`

.. code-block:: python
    deco = TextDecoration()
    deco.signals.clicked.connect(a_slot)
    def a_slot(decoration):
        print(decoration)
*/
struct TextDecoration : public QTextEdit::ExtraSelection
{

public:
    TextDecoration(QTextCursor cursor, int startPos = -1, int endPos = -1, \
        int startLine = -1, int endLine =-1, int drawOrder = 0, const QString &tooltip = "", \
        bool fullWidth = false);
    virtual ~TextDecoration();

    bool operator==(const TextDecoration &other);

    int drawOrder() const { return m_drawOrder; }

    void setAsBold();
    void setForeground(const QColor &color);
    void setBackground(const QBrush &brush);
    void setOutline(const QColor &color);
    void selectLine();
    void setFullWidth(bool flag = true, bool clear = true);

    void setAsUnderlined(const QColor &color = QColor("blue"));
    void setAsSpellCheck(const QColor &color = QColor("blue"));
    void setAsError(const QColor &color = QColor("red"));
    void setAsWarning(const QColor &color = QColor("orange"));

protected:
    bool containsCursor(const QTextCursor &cursor);

    QSharedPointer<TextDecorationsSignals> m_signals;
    bool m_drawOrder;
    QString m_tooltip;

private:
    
};


#endif