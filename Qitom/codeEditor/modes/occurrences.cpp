/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

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

    Further hints:
    ------------------------

    This file belongs to the code editor of itom. The code editor is
    in major parts a fork / rewritten version of the python-based source
    code editor PyQode from Colin Duquesnoy and others
    (see https://github.com/pyQode). PyQode itself is licensed under
    the MIT License (MIT).

    Some parts of the code editor of itom are also inspired by the
    source code editor of the Spyder IDE (https://github.com/spyder-ide),
    also licensed under the MIT License and developed by the Spyder Project
    Contributors.

*********************************************************************** */

#include "occurrences.h"

#include "../codeEditor.h"
#include "../managers/textDecorationsManager.h"
#include "../utils/utils.h"

#include <qbrush.h>
#include <qregularexpression.h>
#include <QtConcurrent/QtConcurrentRun>

namespace ito {

OccurrencesHighlighterMode::OccurrencesHighlighterMode(const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    Mode("OccurrencesHighlighterMode", description),
    QObject(parent),
    m_background(QColor("#CCFFCC")),
    m_foreground(QColor("magenta")),
    m_underlined(false),
    m_caseSensitive(false),
    m_wholeWord(true),
    m_selectOnDoubleClick(false)
{
    // Timer used to run the search request with a specific delay
    m_pTimer = new DelayJobRunnerNoArgs<OccurrencesHighlighterMode, void(OccurrencesHighlighterMode::*)()>(1000);
    connect(&m_asyncFindAllWatcher, SIGNAL(finished()), this, SLOT(asyncFindAllWatcherFinished()));
}

//----------------------------------------------------------
/*
*/
OccurrencesHighlighterMode::~OccurrencesHighlighterMode()
{
    delete m_pTimer;
    m_pTimer = NULL;
}

//----------------------------------------------------------
/*
Background or underline color (if underlined is True).
*/
QColor OccurrencesHighlighterMode::background() const
{
    return m_background;
}

//----------------------------------------------------------
/*
*/
void OccurrencesHighlighterMode::setBackground(const QColor &color)
{
    m_background = color;
}

//----------------------------------------------------------
/*
Foreground color of occurences, not used if underlined is True.
*/
QColor OccurrencesHighlighterMode::foreground() const
{
    return m_foreground;
}

//----------------------------------------------------------
/*
*/
void OccurrencesHighlighterMode::setForeground(const QColor &color)
{
    m_foreground = color;
}

//----------------------------------------------------------
/*
Delay before searching for occurrences. The timer is rearmed as soon
as the cursor position changed.
*/
int OccurrencesHighlighterMode::delay() const
{
    return m_pTimer->delay();
}

void OccurrencesHighlighterMode::setDelay(int delay)
{
    m_pTimer->setDelay(delay);
}

//----------------------------------------------------------
/*
True to use to underlined occurrences instead of
changing the background. Default is True.

If this mode is ON, the foreground color is ignored, the
background color is then used as the underline color.
*/
bool OccurrencesHighlighterMode::underlined() const
{
    return m_underlined;
}

void OccurrencesHighlighterMode::setUnderlined(bool value)
{
    m_underlined = value;
}

//----------------------------------------------------------
/*
*/
bool OccurrencesHighlighterMode::wholeWord() const
{
    return m_wholeWord;
}

void OccurrencesHighlighterMode::setWholeWord(bool value)
{
    m_wholeWord = value;
}

//----------------------------------------------------------
/*
*/
bool OccurrencesHighlighterMode::caseSensitive() const
{
    return m_caseSensitive;
}

void OccurrencesHighlighterMode::setCaseSensitive(bool value)
{
    m_caseSensitive = value;
}

//----------------------------------------------------------
/*
*/
bool OccurrencesHighlighterMode::selectOnDoubleClick() const
{
    return m_selectOnDoubleClick;
}

void OccurrencesHighlighterMode::setSelectOnDoubleClick(bool value)
{
    if (m_selectOnDoubleClick != value)
    {
        m_selectOnDoubleClick = value;
    }
}

//----------------------------------------------------------
/*
*/
void OccurrencesHighlighterMode::onStateChanged(bool state)
{
    if (state)
    {
        connect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(requestHighlightPosChanged()));
        connect(editor(), SIGNAL(mouseDoubleClicked(QMouseEvent*)), this, SLOT(requestHighlightDoubleClick()));
    }
    else
    {
        disconnect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(requestHighlightPosChanged()));
        disconnect(editor(), SIGNAL(mouseDoubleClicked(QMouseEvent*)), this, SLOT(requestHighlightDoubleClick()));
        m_pTimer->cancelRequests();
    }
}

//----------------------------------------------------------
void OccurrencesHighlighterMode::requestHighlightPosChanged()
{
    if (!m_selectOnDoubleClick)
    {
        requestHighlight();
    }
    else if (m_decorations.size() > 0)
    {
        m_pTimer->cancelRequests();
        clearDecorations();
        m_sub = "";
    }
}

//----------------------------------------------------------
void OccurrencesHighlighterMode::requestHighlightDoubleClick()
{
    if (m_selectOnDoubleClick)
    {
        m_pTimer->cancelRequests();
        requestHighlight();
    }
}

//----------------------------------------------------------
/*
Updates the current line decoration
*/
void OccurrencesHighlighterMode::requestHighlight()
{
    if (editor())
    {
        QString sub = editor()->wordUnderCursor(true).selectedText();
        if (sub != m_sub)
        {
            clearDecorations();
            if (sub.size() > 1)
            {
                DELAY_JOB_RUNNER_NOARGS(m_pTimer, OccurrencesHighlighterMode, void(OccurrencesHighlighterMode::*)())->requestJob( \
                this, &OccurrencesHighlighterMode::sendRequest);
            }
            else
            {
                m_sub = "";
            }
        }
    }
}

//----------------------------------------------------------
void OccurrencesHighlighterMode::sendRequest()
{
    if (!editor())
    {
        return;
    }

    QTextCursor cursor = editor()->textCursor();
    m_sub = editor()->wordUnderCursor(true).selectedText();
    if (!cursor.hasSelection() || cursor.selectedText() == m_sub)
    {
        if (m_sub != "")
        {
            if (1)
            {
                //concurrent
                if (!m_asyncFindAllWatcher.isRunning())
                {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
                    m_asyncFindAllWatcher.setFuture(
                        QtConcurrent::run(
                            &OccurrencesHighlighterMode::findAll,
                            this,
                            editor()->toPlainText(),
                            m_sub,
                            m_wholeWord,
                            m_caseSensitive));
#else
                    m_asyncFindAllWatcher.setFuture(
                        QtConcurrent::run(
                            this,
                            &OccurrencesHighlighterMode::findAll,
                            editor()->toPlainText(),
                            m_sub,
                            m_wholeWord,
                            m_caseSensitive));
#endif
                }
                else
                {
                    m_sub == "";
                    requestHighlight();
                }
            }
            else
            {
                //or: direct
                QList<QPair<int,int> > matches = findAll(editor()->toPlainText(), m_sub, m_wholeWord, m_caseSensitive);
                onResultsAvailable(matches);
            }
        }
    }
}

//----------------------------------------------------------
void OccurrencesHighlighterMode::asyncFindAllWatcherFinished()
{
    onResultsAvailable(m_asyncFindAllWatcher.future().result());
}

//----------------------------------------------------------
void OccurrencesHighlighterMode::onResultsAvailable(QList<QPair<int,int> > results)
{
    if (results.size() > 500)
    {
        /*limit number of results (on very big file where a lots of
        # occurrences can be found, this would totally freeze the editor
        # during a few seconds, with a limit of 500 we can make sure
        # the editor will always remain responsive).*/
        results = results.mid(0, 500);
    }

    m_sub == ""; //todo: is it good to delete the recently detected word?

    int current = editor()->textCursor().position();
    if (results.size() > 1)
    {
        int start, end;
        TextDecoration::Ptr deco;
        for (int i = 0; i < results.size(); ++i)
        {
            start = results[i].first;
            end = results[i].second;

            if ((start <= current) && (current <= end))
            {
                continue;
            }

            deco = TextDecoration::Ptr(
                new TextDecoration(
                    editor()->textCursor(), start, end,-1, -1, 200));

            if (m_underlined)
            {
                deco->setAsUnderlined(m_background);
            }
            else
            {
                deco->setBackground(QBrush(m_background));

                if (m_foreground.isValid())
                {
                    deco->setForeground(m_foreground);
                }
            }

            editor()->decorations()->append(deco);
            m_decorations.append(deco);
        }
    }


}

//----------------------------------------------------------
/*
Clear line decoration
*/
//-------------------------------------------------------------
void OccurrencesHighlighterMode::clearDecorations()
{
    foreach(TextDecoration::Ptr deco, m_decorations)
    {
        editor()->decorations()->remove(deco);
    }
    m_decorations.clear();
}

//----------------------------------------------------------
/*
Generator that finds all occurrences of ``sub`` in  ``string``

:param string: string to parse
:param sub: string to search
:param regex: True to search using regex
:param case_sensitive: True to match case, False to ignore case
:param whole_word: True to returns only whole words
:return:
*/
//-------------------------------------------------------------
QList<QPair<int,int> > OccurrencesHighlighterMode::findAll(const QString &text, const QString &sub, bool wholeWord, bool caseSensitive)
{
    QList<QPair<int, int> > results;

    if (sub != "")
    {
        QRegularExpression rx;

        if (!caseSensitive)
        {
            rx.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        }

        if (wholeWord)
        {
            rx.setPattern(QString("\\b%1\\b").arg(sub));
            //offset = -1;
        }
        else
        {
            rx.setPattern(sub);
        }

        int pos = 0;
        int length = sub.size();
        QRegularExpressionMatch match;

        while ((match = rx.match(text, pos)).hasMatch())
        {
            pos = match.capturedStart();
            results.append(QPair<int,int>(pos, pos + length));
            //qDebug() << rx.pattern() << rx.matchedLength();
            pos = match.capturedEnd();
        }
    }

    return results;
}

} //end namespace ito
