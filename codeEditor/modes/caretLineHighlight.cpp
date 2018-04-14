#include "caretLineHighlight.h"

#include "codeEditor.h"
#include "managers/textDecorationsManager.h"
#include "utils/utils.h"

#include <qbrush.h>


CaretLineHighlighterMode::CaretLineHighlighterMode(const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    Mode("CaretLineHighlighterMode", description),
    QObject(parent),
    m_decoration(NULL),
    m_pos(-1),
    m_color(QColor())
{
}

//----------------------------------------------------------
/*
*/
CaretLineHighlighterMode::~CaretLineHighlighterMode()
{
}

//----------------------------------------------------------
/*
Background color of the caret line. Default is to use a color slightly
darker/lighter than the background color. You can override the
automatic color by setting up this property
*/
QColor CaretLineHighlighterMode::background() const
{
    if (m_color.isValid() || !editor())
    {
        return m_color;
    }
    else
    {
        return Utils::driftColor(editor()->background(), 110);
    }
}

//----------------------------------------------------------
/*
*/
void CaretLineHighlighterMode::setBackground(const QColor &color)
{
    m_color = color;
    refresh();
}

//----------------------------------------------------------
/*
*/

void CaretLineHighlighterMode::onInstall(CodeEditor *editor)
{
    Mode::onInstall(editor);
    refresh();
}

//----------------------------------------------------------
/*
*/
void CaretLineHighlighterMode::onStateChanged(bool state)
{
    if (state)
    {
        connect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(refresh()));
        connect(editor(), SIGNAL(newTextSet()), this, SLOT(refresh()));
        refresh();
    }
    else
    {
        disconnect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(refresh()));
        disconnect(editor(), SIGNAL(newTextSet()), this, SLOT(refresh()));
        clearDeco();
    }
            
}

//----------------------------------------------------------
/*
Updates the current line decoration
*/
void CaretLineHighlighterMode::refresh()
{
    if (enabled())
    {
        QBrush brush;

        clearDeco();
        if (m_color.isValid())
        {
            brush = QBrush(m_color);
        }
        else
        {
            brush = Utils::driftColor(editor()->background(), 110);
        }

        m_decoration = TextDecoration::Ptr(new TextDecoration(editor()->textCursor()));
        m_decoration->setBackground(brush);
        m_decoration->setFullWidth();
        editor()->decorations()->append(m_decoration);
    }
}

//----------------------------------------------------------
/*
Clear line decoration
*/
void CaretLineHighlighterMode::clearDeco()
{
    if (m_decoration.isNull() == false)
    {
        editor()->decorations()->remove(m_decoration);
    }

    m_decoration.clear();
}