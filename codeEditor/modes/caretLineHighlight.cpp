#include "caretLineHighlight.h"

#include "codeEditor.h"
#include "managers/textDecorationsManager.h"
#include "utils/utils.h"

#include <qbrush.h>


CaretLineHighlighterMode::CaretLineHighlighterMode(QObject *parent /*= NULL*/) :
    Mode("CaretLineHighlighterMode"),
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
    if (m_color.isValid() || !m_editor)
    {
        return m_color;
    }
    else
    {
        return Utils::driftColor(m_editor->background(), 110);
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
        connect(m_editor, SIGNAL(cursorPositionChanged()), this, SLOT(refresh()));
        connect(m_editor, SIGNAL(newTextSet()), this, SLOT(refresh()));
        refresh();
    }
    else
    {
        disconnect(m_editor, SIGNAL(cursorPositionChanged()), this, SLOT(refresh()));
        disconnect(m_editor, SIGNAL(newTextSet()), this, SLOT(refresh()));
        clearDeco();
    }
            
}

//----------------------------------------------------------
/*
Updates the current line decoration
*/
void CaretLineHighlighterMode::refresh()
{
    if (m_enabled)
    {
        QBrush brush;

        clearDeco();
        if (m_color.isValid())
        {
            brush = QBrush(m_color);
        }
        else
        {
            brush = Utils::driftColor(m_editor->background(), 110);
        }

        m_decoration = TextDecoration::Ptr(new TextDecoration(m_editor->textCursor()));
        m_decoration->setBackground(brush);
        m_decoration->setFullWidth();
        m_editor->decorations()->append(m_decoration);
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
        m_editor->decorations()->remove(m_decoration);
    }

    m_decoration.clear();
}