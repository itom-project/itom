#include "autoindent.h"

#include "codeEditor.h"


AutoIndentMode::AutoIndentMode(const QString &name, const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    Mode(name, description),
    QObject(parent)
{
}

//----------------------------------------------------------
/*
*/
AutoIndentMode::~AutoIndentMode()
{
}


//----------------------------------------------------------
/*
*/
void AutoIndentMode::onStateChanged(bool state)
{
    if (state)
    {
        connect(editor(), SIGNAL(keyPressed(QKeyEvent*)), this, SLOT(onKeyPressed(QKeyEvent*)));
    }
    else
    {
        disconnect(editor(), SIGNAL(keyPressed(QKeyEvent*)), this, SLOT(onKeyPressed(QKeyEvent*)));
    }       
}

//----------------------------------------------------------
/*
Return the indentation text (a series of spaces or tabs)

:param cursor: QTextCursor

:returns: Tuple (text before new line, text after new line)
*/
QPair<QString, QString> AutoIndentMode::getIndent(const QTextCursor &cursor) const
{
    QString indent = QString(editor()->lineIndent(-1), QChar(' '));
    return QPair<QString,QString>("", indent);
}

//----------------------------------------------------------
/*
Auto indent if the released key is the return key.
:param event: the key event
*/
void AutoIndentMode::onKeyPressed(QKeyEvent *e)
{
    if (!e->isAccepted())
    {
        if ((e->key() == Qt::Key_Return) || (e->key() == Qt::Key_Enter))
        {
            QTextCursor cursor = editor()->textCursor();
            QPair<QString,QString> pre_post = getIndent(cursor);
            cursor.beginEditBlock();
            cursor.insertText(QString("%1\n%2").arg(pre_post.first, pre_post.second));

            //eats possible whitespaces
            cursor.movePosition(QTextCursor::WordRight, QTextCursor::KeepAnchor);
            QString txt = cursor.selectedText();
            if (txt.startsWith(" "))
            {
                QString new_txt = txt.replace(" ", "");
                if (txt.size() > new_txt.size())
                {
                    cursor.insertText(new_txt);
                }
            }
            cursor.endEditBlock();
            e->accept();
        }
    }
}