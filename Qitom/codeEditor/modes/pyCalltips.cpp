/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#include "pyCalltips.h"

#include "../codeEditor.h"
#include "../utils/utils.h"
#include "../managers/panelsManager.h"
#include "AppManagement.h"
#include "../../widgets/scriptEditorWidget.h"

#include "python/pythonEngine.h"
#include "languageServer/languageServerManager.h"

#include <qdir.h>
#include <qtooltip.h>

namespace ito {

PyCalltipsMode::PyCalltipsMode(const QString &name, const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    Mode(name, description),
    QObject(parent),
    m_pPythonEngine(NULL),
    m_requestCount(0),
    m_pConnectedBackend(NULL),
    m_pendingCalltipId(-1),
    m_hasDeferredRequest(false),
    m_deferredLine(-1),
    m_deferredCol(-1)
{
    m_pPythonEngine = AppManagement::getPythonEngine();

    m_disablingKeys << Qt::Key_ParenRight << \
            Qt::Key_Return << \
            Qt::Key_Left << \
            Qt::Key_Right << \
            Qt::Key_Up << \
            Qt::Key_Down << \
            Qt::Key_End << \
            Qt::Key_Home << \
            Qt::Key_PageDown << \
            Qt::Key_PageUp << \
            Qt::Key_Backspace << \
            Qt::Key_Delete;
}

//----------------------------------------------------------
/*
*/
PyCalltipsMode::~PyCalltipsMode()
{
}


//----------------------------------------------------------
/*
*/
void PyCalltipsMode::onStateChanged(bool state)
{
    if (m_pPythonEngine)
    {
        if (state)
        {
            connect(editor(), SIGNAL(keyReleased(QKeyEvent*)), this, SLOT(onKeyReleased(QKeyEvent*)));
        }
        else
        {
            disconnect(editor(), SIGNAL(keyReleased(QKeyEvent*)), this, SLOT(onKeyReleased(QKeyEvent*)));
        }
    }
}

//----------------------------------------------------------
/*
Auto indent if the released key is the return key.
:param event: the key event
*/
void PyCalltipsMode::onKeyReleased(QKeyEvent *e)
{
    if (e->key() == Qt::Key_ParenLeft || \
            e->key() == Qt::Key_Comma)
    {
        QTextCursor tc = editor()->textCursor();
        int line = tc.blockNumber();
        int col = tc.columnNumber();

        QString encoding = "utf8";

        QString source = editor()->codeText(line, col); // line and col might be changed if code is a virtual code (e.g. for command line, containing all its history)
        // jedi has a bug if the statement has a closing parenthesis
        // remove it!
        QStringList lines = Utils::splitlines(source);
        QString l;

        if (line >= 0 && line < lines.size())
        {
            l = Utils::rstrip(lines[line]);
        }
        else
        {
            // at the beginning of the last line (empty)
            return;
        }

        if (l.endsWith(")"))
        {
            lines[line] = l.left(l.size() - 1);
            if (col > lines[line].size())
            {
                col = lines[line].size();
            }
        }

        source = lines.join("\n");
        requestCalltip(source, line, col, encoding);
    }
    else if (m_disablingKeys.contains(e->key()))
    {
        //QToolTip::hideText();
        ToolTip::hideText();
        //m_toolTipWidget->hide();
    }
}

//--------------------------------------------------------------------------------
void PyCalltipsMode::requestCalltip(const QString &source, int line, int col, const QString &encoding)
{
    if (m_requestCount != 0)
    {
        return;
    }

    ILanguageServerBackend* backend = LanguageServerManager::getInstance()->activeBackend();

    if (!backend)
    {
        // The backend is created lazily and it might be initialized asynchronously
        // (the zuban language server for instance is an external process). In this
        // case activeBackend() returns nullptr for the moment and the manager emits
        // backendChanged() as soon as the backend is ready. Therefore remember this
        // request and re-issue it at that time instead of disabling the calltips.
        deferCalltipRequest(source, line, col, encoding);
        return;
    }

    ScriptEditorWidget *sew = qobject_cast<ScriptEditorWidget*>(editor());
    QString filename;

    if (sew)
    {
        filename = sew->getFilename();
    }

    if (filename == "")
    {
        filename = QDir::cleanPath(QDir::current().absoluteFilePath("__temporaryfile__.py"));
    }

    connectToBackend(backend);

    ito::JediCalltipRequest request;
    request.m_col = col;
    request.m_line = line;
    request.m_path = filename;
    request.m_source = source;

    int requestId = backend->requestCalltip(request);

    if (requestId >= 0)
    {
        m_pendingCalltipId = requestId;
        m_requestCount += 1;
    }
}

//--------------------------------------------------------------------------------
void PyCalltipsMode::deferCalltipRequest(const QString &source, int line, int col, const QString &encoding)
{
    // only the most recent request is kept, since an outdated calltip must not be shown.
    m_deferredSource = source;
    m_deferredEncoding = encoding;
    m_deferredLine = line;
    m_deferredCol = col;
    m_hasDeferredRequest = true;

    if (!m_backendChangedConnection)
    {
        m_backendChangedConnection = connect(
            LanguageServerManager::getInstance(), &LanguageServerManager::backendChanged,
            this, &PyCalltipsMode::onBackendChanged);
    }
}

//--------------------------------------------------------------------------------
void PyCalltipsMode::onBackendChanged()
{
    if (!m_hasDeferredRequest)
    {
        return;
    }

    const QString source = m_deferredSource;
    const QString encoding = m_deferredEncoding;
    const int line = m_deferredLine;
    const int col = m_deferredCol;

    m_hasDeferredRequest = false;
    m_deferredSource.clear();
    m_deferredEncoding.clear();
    m_deferredLine = -1;
    m_deferredCol = -1;

    // the initialization of the backend may have taken a while, hence only show a
    // calltip, if the user is still typing in this editor.
    if (!editor() || !editor()->hasFocus())
    {
        return;
    }

    requestCalltip(source, line, col, encoding);
}

//--------------------------------------------------------------------------------
void PyCalltipsMode::connectToBackend(ILanguageServerBackend* backend)
{
    if (backend == m_pConnectedBackend)
    {
        return;
    }

    if (m_pConnectedBackend)
    {
        disconnect(
            m_pConnectedBackend, &ILanguageServerBackend::calltipReady,
            this, &PyCalltipsMode::onCalltipReady);
    }

    connect(
        backend, &ILanguageServerBackend::calltipReady,
        this, &PyCalltipsMode::onCalltipReady);

    m_pConnectedBackend = backend;
}

//--------------------------------------------------------------------------------
void PyCalltipsMode::onCalltipReady(int requestId, QVector<ito::JediCalltip> calltips)
{
    // every mode is connected to the same backend, therefore only handle
    // the result, that belongs to the request of this instance.
    if (requestId < 0 || requestId != m_pendingCalltipId)
    {
        return;
    }

    m_pendingCalltipId = -1;

    onJediCalltipResultAvailable(calltips);
}

//--------------------------------------------------------------------------------
bool PyCalltipsMode::isLastChardEndOfWord() const
{
    QTextCursor tc = editor()->wordUnderCursor(false);
    tc.setPosition(tc.position());
    tc.movePosition(QTextCursor::StartOfLine, QTextCursor::KeepAnchor);
    QString l = tc.selectedText();

    if (l.size() > 0)
    {
        QChar lastChar = l[l.size() - 1];
        QString seps = editor()->wordSeparators();
        QString symbols = ", (";
        return (seps.contains(lastChar)) && !(symbols.contains(lastChar));
    }
    else
    {
        return false;
    }
}

//--------------------------------------------------------------------------------
QString parseCalltip(const ito::JediCalltip &tip, bool compactLayout)
{
    if (tip.m_calltipParams.size() == 0)
    {
        return QString("<p><nobr>%1()</nobr></p>").arg(tip.m_calltipMethodName);
    }
    else
    {
        int paramLength = 0;
        const int maxLineLength = 88;

        foreach(const QString &p, tip.m_calltipParams)
        {
            paramLength += 2 + p.size();
        }

        if (paramLength + tip.m_calltipMethodName.size() + 2 < (2 * maxLineLength))
        {
            return QString("<p><nobr>%1(%2)</nobr></p>")
                .arg(tip.m_calltipMethodName)
                .arg(tip.m_calltipParams.join(", "));
        }
        else if (compactLayout)
        {
            QString currentLine;
            QString params;

            foreach (const QString& p, tip.m_calltipParams)
            {
                if (currentLine == "")
                {
                    // at least one argument in the line
                    currentLine = p;
                }
                else if (currentLine.size() + p.size() > maxLineLength)
                {
                    if (params != "")
                    {
                        params += ", ";
                    }

                    params += "<br>&nbsp;&nbsp;&nbsp;&nbsp;";
                    params += currentLine;

                    currentLine = p;
                }
                else
                {
                    currentLine = currentLine + ", " + p;
                }
            }

            if (currentLine != "")
            {
                if (params != "")
                {
                    params += ", ";
                }

                params += "<br>&nbsp;&nbsp;&nbsp;&nbsp;";
                params += currentLine;
            }

            return QString("<p><nobr>%1(%2)</nobr></p>")
                .arg(tip.m_calltipMethodName)
                .arg(params);
        }
        else
        {
            return QString("<p><nobr>%1(<br>&nbsp;&nbsp;&nbsp;&nbsp;%2)</nobr></p>")
                .arg(tip.m_calltipMethodName)
                .arg(tip.m_calltipParams.join(",<br>&nbsp;&nbsp;&nbsp;&nbsp;"));
        }
    }
}

//--------------------------------------------------------------------------------
void PyCalltipsMode::onJediCalltipResultAvailable(QVector<ito::JediCalltip> calltips)
{
     m_requestCount--;

    if (isLastChardEndOfWord() || calltips.size() == 0)
    {
        return;
    }

    JediCalltip first_calltip = calltips[0];

    QString text;

    if (calltips.size() > 0)
    {
        // estimate the number of lines if every argument will be
        // in a new line
        int noLines = calltips.size();

        foreach (const JediCalltip& tip, calltips)
        {
            noLines += tip.m_calltipParams.size();
        }

        foreach(const JediCalltip &tip, calltips)
        {
            // newline not necessary, since each calltip is in a <p>...</p> block
            text.append(parseCalltip(tip, noLines > 20));
        }
    }
    else
    {
        text = parseCalltip(first_calltip, false);
    }

    // set tool tip position at the start of the bracket

#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))
    int char_width = editor()->fontMetrics().horizontalAdvance('A');
#else
    int char_width = editor()->fontMetrics().width('A');
#endif
    int w_offset = (first_calltip.m_column - first_calltip.m_bracketStartCol) * char_width;
    QRect cursorRect = editor()->cursorRect();
    QPoint position(
        cursorRect.x() - w_offset + editor()->panels()->marginSize(ito::Panel::Left),
        cursorRect.y() + char_width + //cursorRect.height() +
        editor()->panels()->marginSize(ito::Panel::Top));
    position = editor()->mapToGlobal(position);

    //position = QPoint(0, 0);

    // show tooltip
    ToolTip::showText(position, text, editor(), QRect(), true);
}

} //end namespace ito
