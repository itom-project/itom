#ifndef TEXTDECORATIONS_H
#define TEXTDECORATIONS_H

/*
This module contains the text decoration API.
*/

#include <qwidget.h>
#include <qtextedit.h>
#include <qtextcursor.h>
#include <qsharedpointer.h>
#include <qmap.h>
#include <qvariant.h>
#include <QTextBlock>

class QTextDecoration;

namespace ito {

class TextDecorationsSignals;
class CodeEditor;

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
    typedef QSharedPointer<TextDecoration> Ptr;

    TextDecoration();
    TextDecoration(const QTextCursor &cursor, int startPos = -1, int endPos = -1, \
        int startLine = -1, int endLine =-1, int drawOrder = 0, const QString &tooltip = "", \
        bool fullWidth = false);
    TextDecoration(QTextDocument *document, int startPos = -1, int endPos = -1, \
        int startLine = -1, int endLine =-1, int drawOrder = 0, const QString &tooltip = "", \
        bool fullWidth = false);
    virtual ~TextDecoration();

    bool operator==(const TextDecoration &other) const;

    int drawOrder() const { return m_drawOrder; }
    void setDrawOrder(int order) { m_drawOrder = order; }

    QVariantMap &properties() { return m_properties; }

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

    bool containsCursor(const QTextCursor &cursor) const;

    void emitClicked(TextDecoration::Ptr selection) const;

    QString tooltip() const { return m_tooltip; }
    void setTooltip(const QString &tooltip) { m_tooltip = tooltip; }

    bool isValid() const { return m_drawOrder >= 0; }

    QTextBlock block() const { return m_block; }
    void setBlock(const QTextBlock &block) { m_block = block; }

    QMetaObject::Connection connect(const char* signal, QObject *receiver, const char *slot);

protected:
    QSharedPointer<TextDecorationsSignals> m_signals;
    int m_drawOrder;
    QString m_tooltip;
    QVariantMap m_properties;
    QTextBlock m_block;

private:
    
};

} //end namespace ito

Q_DECLARE_METATYPE(ito::TextDecoration)
Q_DECLARE_METATYPE(ito::TextDecoration::Ptr)

namespace ito {


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
    void clicked(TextDecoration::Ptr selection);
};

} //end namespace ito


#endif