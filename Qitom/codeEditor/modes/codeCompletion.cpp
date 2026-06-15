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

#include "codeCompletion.h"

#include "../codeEditor.h"
#include "../utils/utils.h"
#include "../managers/panelsManager.h"
#include "AppManagement.h"

#include "../../python/pythonEngine.h"
#include "../../widgets/scriptEditorWidget.h"
#include "../../helper/compatHelper.h"

#include <qtooltip.h>
#include <qabstractitemview.h>
#include <qstandarditemmodel.h>
#include <qscrollbar.h>
#include <qdir.h>


namespace ito {

//--------------------------------------------------------------------
/*
*/
SubsequenceSortFilterProxyModel::SubsequenceSortFilterProxyModel(Qt::CaseSensitivity caseSensitivity, QObject *parent /*= NULL*/) :
    QSortFilterProxyModel(parent),
    m_caseSensitivity(caseSensitivity)
{
}

//--------------------------------------------------------------------
/*
*/
void SubsequenceSortFilterProxyModel::setPrefix(const QString &prefix)
{
    m_filterPatterns.clear();
    m_filterPatternsCaseSensitive.clear();
    m_sortPatterns.clear();

    QString ptrn;

    for (int i = prefix.size(); i >= 1; --i)
    {
        ptrn = CompatHelper::regExpAnchoredPattern(QString(".*%1.*%2").arg(prefix.left(i), prefix.mid(i)));
        QRegularExpression regExp(ptrn);
        m_filterPatternsCaseSensitive.append(regExp);

        if (!m_caseSensitivity)
        {
            regExp.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        }

        m_filterPatterns.append(regExp);

        ptrn = QString("%1.*%1").arg(prefix.left(i), prefix.mid(i));
        regExp.setPattern(ptrn);

        if (m_caseSensitivity)
        {
            regExp.setPatternOptions(QRegularExpression::PatternOptions());
        }
        else
        {
            regExp.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        }

        m_sortPatterns.append(regExp);
    }
    m_prefix = prefix;
}

//--------------------------------------------------------------------
/*
*/
bool SubsequenceSortFilterProxyModel::filterAcceptsRow(int source_row, const QModelIndex &source_parent) const
{
    QString completion = sourceModel()->data(sourceModel()->index(source_row, 0)).toString();
    if (completion.size() < m_prefix.size())
    {
        return false;
    }

    QString prefix;
    int rank;

    if (m_prefix.size() == 1)
    {
        prefix = m_prefix;
        if (m_caseSensitivity == Qt::CaseInsensitive)
        {
            completion = completion.toLower();
            prefix = prefix.toLower();
        }
        rank = completion.indexOf(prefix);
        if (rank >= 0)
        {
            sourceModel()->setData(sourceModel()->index(source_row, 0), rank, Qt::UserRole);
            return completion.contains(prefix);
        }
        else
        {
            return false;
        }
    }

    for (int idx = 0; idx < m_filterPatterns.size(); ++idx)
    {
        if (m_filterPatterns[idx].match(completion).hasMatch())
        {
            // exact match due to pattern of regular expression
            // compute rank, the lowest rank the closer it is from the
            // completion
            int start = completion.lastIndexOf(m_sortPatterns[idx]);
            if (start == -1)
            {
                start = INT_MAX;
            }
            rank = start + idx * 10;
            if (m_filterPatternsCaseSensitive[idx].match(completion).hasMatch())
            {
                // exact match due to pattern of regular expression
                // favorise completions where case is matched
                rank -= 10;
            }
            sourceModel()->setData(sourceModel()->index(source_row, 0), rank, Qt::UserRole);
            return true;
        }
    }

    return m_prefix.size() == 0;
}



//--------------------------------------------------------------------
/*
*/
SubsequenceCompleter::SubsequenceCompleter(QObject *parent /*= NULL*/) :
    QCompleter(parent),
    m_pFilterProxyModel(NULL)
{
    m_localCompletionPrefix = "";
    m_pSourceModel = nullptr;
    m_pFilterProxyModel = new SubsequenceSortFilterProxyModel(caseSensitivity(), this);
    m_pFilterProxyModel->setSortRole(Qt::UserRole);
    m_forceNextUpdate = true;
}

//--------------------------------------------------------------------
/*
*/
void SubsequenceCompleter::setModel(QAbstractItemModel *model)
{
    m_pSourceModel = model;
    DELETE_AND_SET_NULL(m_pFilterProxyModel);
    m_pFilterProxyModel = new SubsequenceSortFilterProxyModel(caseSensitivity(), this);
    m_pFilterProxyModel->setSortRole(Qt::UserRole);
    m_pFilterProxyModel->setPrefix(m_localCompletionPrefix);
    m_pFilterProxyModel->setSourceModel(m_pSourceModel);
    QCompleter::setModel(m_pFilterProxyModel);
    m_pFilterProxyModel->invalidate();
    m_pFilterProxyModel->sort(0);
    m_forceNextUpdate = true;
}

//--------------------------------------------------------------------
/*
*/
void SubsequenceCompleter::updateModel()
{
    if (completionCount() || (m_localCompletionPrefix.size() <= 1) || m_forceNextUpdate)
    {
        m_pFilterProxyModel->setPrefix(m_localCompletionPrefix);
        m_pFilterProxyModel->invalidate(); // force sorting/filtering
    }
    if (completionCount() > 1)
    {
        m_pFilterProxyModel->sort(0);
    }
    m_forceNextUpdate = false;
}

//--------------------------------------------------------------------
/*
*/
QStringList SubsequenceCompleter::splitPath(const QString &path) const
{
    SubsequenceCompleter *c = const_cast<SubsequenceCompleter*>(this);
    c->m_localCompletionPrefix = path;
    c->updateModel();
    return QStringList() << "";
}




//-------------------------------------------------------------------
CodeCompletionMode::CodeCompletionMode(const QString &name, const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    Mode(name, description),
    QObject(parent),
    m_currentCompletion(""),
    m_triggerKey(Qt::Key_Space),
    m_triggerLen(1),
    m_triggerSymbols(QStringList() << "."),
    m_caseSensitive(false),
    m_pCompleter(NULL),
    m_filterMode(FilterFuzzy),
    m_lastCursorLine(-1),
    m_lastCursorColumn(-1),
    m_showTooltips(false),
    m_requestId(0),
    m_lastRequestId(0),
    m_tooltipsMaxLength(300),
    m_selectWithReturn(true)
{
    m_pPythonEngine = AppManagement::getPythonEngine();
}

//-------------------------------------------------------------------
/*virtual*/ CodeCompletionMode::~CodeCompletionMode()
{
    if (m_pCompleter)
    {
        m_pCompleter->deleteLater();
        m_pCompleter = nullptr;
    }
}

//-------------------------------------------------------------------
void CodeCompletionMode::createCompleter()
{
    if (m_filterMode != FilterFuzzy)
    {
        m_pCompleter = new QCompleter(QStringList() << "", editor());

        if (m_filterMode == FilterContains)
        {
            m_pCompleter->setFilterMode(Qt::MatchContains);
        }
    }
    else
    {
        m_pCompleter = new SubsequenceCompleter(editor());
    }

    m_pCompleter->setCompletionMode(QCompleter::PopupCompletion);

    if (m_caseSensitive)
    {
        m_pCompleter->setCaseSensitivity(Qt::CaseSensitive);
    }
    else
    {
        m_pCompleter->setCaseSensitivity(Qt::CaseInsensitive);
    }

    connect(m_pCompleter, SIGNAL(activated(QString)), this, SLOT(insertCompletion(QString)));
    connect(m_pCompleter, SIGNAL(highlighted(QString)), this, SLOT(onSelectedCompletionChanged(QString)));
    connect(m_pCompleter, SIGNAL(highlighted(QString)), this, SLOT(displayCompletionTooltip(QString)));
}

//-------------------------------------------------------------------
void CodeCompletionMode::onStateChanged(bool state)
{
    if (state)
    {
        connect(editor(), SIGNAL(focusedIn(QFocusEvent*)), this, SLOT(onFocusIn(QFocusEvent*)));
        connect(editor(), SIGNAL(keyPressed(QKeyEvent*)), this, SLOT(onKeyPressed(QKeyEvent*)));
        connect(editor(), SIGNAL(postKeyPressed(QKeyEvent*)), this, SLOT(onKeyReleased(QKeyEvent*)));
    }
    else
    {
        disconnect(editor(), SIGNAL(focusedIn(QFocusEvent*)), this, SLOT(onFocusIn(QFocusEvent*)));
        disconnect(editor(), SIGNAL(keyPressed(QKeyEvent*)), this, SLOT(onKeyPressed(QKeyEvent*)));
        disconnect(editor(), SIGNAL(postKeyPressed(QKeyEvent*)), this, SLOT(onKeyReleased(QKeyEvent*)));
    }
}

//-------------------------------------------------------------------
void CodeCompletionMode::onInstall(CodeEditor *editor)
{
    createCompleter();
    m_pCompleter->setModel(new QStandardItemModel(this));
    Mode::onInstall(editor);
}

//-------------------------------------------------------------------
void CodeCompletionMode::onUninstall()
{
    Mode::onUninstall();
    m_pCompleter->popup()->hide();
    DELETE_AND_SET_NULL(m_pCompleter);
}


//-------------------------------------------------------------------
/*
*/
void CodeCompletionMode::handleCompleterEvents(QKeyEvent *e)
{
    bool nav_key = isNavigationKey(e);
    bool ctrl = int(e->modifiers() & Qt::ControlModifier) == Qt::ControlModifier;
    // complete
    if (e->key() == Qt::Key_Enter || \
        (m_selectWithReturn && (e->key() == Qt::Key_Return)) || \
        e->key() == Qt::Key_Tab)
    {
        insertCompletion(m_currentCompletion);
        hidePopup();
        e->accept();
    }
    // hide
    else if (e->key() == Qt::Key_Escape || \
        e->key() == Qt::Key_Backtab || \
        (!m_selectWithReturn && (e->key() == Qt::Key_Return)) || \
        (nav_key && ctrl))
    {
        resetSyncDataAndHidePopup();
        e->accept();
    }
    // move into list
    else if (e->key() == Qt::Key_Home)
    {
        showPopup(0);
        e->accept();
    }
    else if (e->key() == Qt::Key_End)
    {
        showPopup(m_pCompleter->completionCount() - 1);
        e->accept();
    }
}

//-------------------------------------------------------------------
/*
*/
void CodeCompletionMode::onKeyPressed(QKeyEvent *e)
{
    //debug('key pressed: %s' % e->text())
    bool is_shortcut = isShortcut(e);
    // handle completer popup events ourselves
    if (m_pCompleter->popup()->isVisible())
    {
        if (is_shortcut)
        {
            e->accept();
        }
        else
        {
            handleCompleterEvents(e);
        }
    }
    else if (is_shortcut)
    {
        resetSyncDataAndHidePopup();
        requestCompletion();
        e->accept();
    }
}

//-------------------------------------------------------------------
/*
*/
void CodeCompletionMode::onKeyReleased(QKeyEvent *e)
{
        if (isShortcut(e) || e->isAccepted())
        {
            return;
        }

        //debug('key released:%s' % e->text())
        QTextCursor cursor = editor()->wordUnderCursor(true);
        QString word = cursor.selectedText();
        QTextCursor current_cursor = editor()->textCursor();

        //debug('word: %s' % word)
        if (e->text() != "" && !editor()->isCommentOrString(current_cursor) && !editor()->isNumber(current_cursor))
        {
            if (e->key() == Qt::Key_Escape)
            {
                hidePopup();
                return;
            }
            if (isNavigationKey(e) && \
                    (!isPopupVisible() || word == ""))
            {
                resetSyncDataAndHidePopup();
                return;
            }
            if (e->key() == Qt::Key_Return)
            {
                return;
            }
            if (m_triggerSymbols.contains(e->text()))
            {
                // symbol trigger, force request
                resetSyncDataAndHidePopup();
                requestCompletion();
            }
            else if (
                (word.size() >= m_triggerLen || m_pCompleter->popup()->isVisible())
                && !editor()->wordSeparators().contains(e->text()))
            {
                // Length trigger
                if (e->modifiers() == Qt::NoModifier || e->modifiers() == Qt::ShiftModifier)
                {
                    requestCompletion();
                }
                else
                {
                    hidePopup();
                }
            }
            else
            {
                resetSyncDataAndHidePopup();
            }
        }
        else
        {
            if (isNavigationKey(e))
            {
                if (isPopupVisible() && word != "")
                {
                    showPopup();
                    return;
                }
                else
                {
                    resetSyncDataAndHidePopup();
                }
            }
        }
}

//-------------------------------------------------------------------
/*
Resets completer's widget

:param event: QFocusEvents
*/
void CodeCompletionMode::onFocusIn(QFocusEvent *e)
{
    m_pCompleter->setWidget(editor());
}


//-------------------------------------------------------------------
/*
The completion filter mode
*/
CodeCompletionMode::FilterMode CodeCompletionMode::filterMode() const
{
    return m_filterMode;
}

void CodeCompletionMode::setFilterMode(FilterMode mode)
{
    m_filterMode = mode;
}


//-------------------------------------------------------------------
/*
The key that triggers code completion (Default is **Space**:
        Ctrl + Space).
*/
Qt::Key CodeCompletionMode::triggerKey() const
{
    return m_triggerKey;
}

void CodeCompletionMode::setTriggerKey(Qt::Key key)
{
    m_triggerKey = key;
}


//-------------------------------------------------------------------
/*
The trigger length defines the word length required to run code
        completion.
*/
int CodeCompletionMode::triggerLength() const
{
    return m_triggerLen;
}

void CodeCompletionMode::setTriggerLength(int length)
{
    m_triggerLen = length;
}


//-------------------------------------------------------------------
/*
Defines the list of symbols that immediately trigger a code completion
requiest. BY default, this list contains the dot character.

For C++, we would add the '->' operator to that list.
*/
QStringList CodeCompletionMode::triggerSymbols() const
{
    return m_triggerSymbols;
}

void CodeCompletionMode::setTriggerSymbols(const QStringList &symbols)
{
    m_triggerSymbols = symbols;
}


//-------------------------------------------------------------------
/*
True to performs case sensitive completion matching.
*/
bool CodeCompletionMode::caseSensitive() const
{
    return m_caseSensitive;
}

void CodeCompletionMode::setCaseSensitive(bool cs)
{
    m_caseSensitive = cs;
}


//-------------------------------------------------------------------
/*
Returns the current completion prefix
*/
QString CodeCompletionMode::completionPrefix() const
{
    return m_completionPrefix;
}


//-------------------------------------------------------------------
/*
True to show tooltips next to the current completion.
*/
bool CodeCompletionMode::showTooltips() const
{
    return m_showTooltips;
}

void CodeCompletionMode::setShowTooltips(bool show)
{
    m_showTooltips = show;
}

//-------------------------------------------------------------------
/*
True to show tooltips next to the current completion.
*/
int CodeCompletionMode::tooltipsMaxLength() const
{
    return m_tooltipsMaxLength;
}

void CodeCompletionMode::setTooltipsMaxLength(int length)
{
    m_tooltipsMaxLength = length;
}



//-------------------------------------------------------------------
void CodeCompletionMode::onSelectedCompletionChanged(const QString &completion)
{
    m_currentCompletion = completion;
}

//-------------------------------------------------------------------
void CodeCompletionMode::insertCompletion(const QString &completion)
{
    QTextCursor cursor = editor()->wordUnderCursor(false);
    cursor.insertText(completion);
    editor()->setTextCursor(cursor);
}

//-------------------------------------------------------------------
void CodeCompletionMode::onJediCompletionResultAvailable(int line, int col, int requestId, QVector<ito::JediCompletion> completions)
{
    m_lastRequestId = requestId;

    if (line == m_lastCursorLine && \
            col == m_lastCursorColumn)
    {
        if (editor())
        {
            showCompletions(completions);
        }
    }
    else
    {
        //qDebug() << "outdated request" << requestId << ": dropping.";
    }
}


//-------------------------------------------------------------------
bool CodeCompletionMode::isPopupVisible() const
{
    return m_pCompleter->popup()->isVisible();
}

//-------------------------------------------------------------------
void CodeCompletionMode::resetSyncDataAndHidePopup()
{
    //debug('reset sync data and hide popup')
    m_lastCursorLine = -1;
    m_lastCursorColumn = -1;
    hidePopup();
}

//-------------------------------------------------------------------
bool CodeCompletionMode::requestCompletion()
{
    int line = editor()->currentLineNumber();
    int col = editor()->currentColumnNumber() - m_completionPrefix.size();
    bool sameContext = (line == m_lastCursorLine && col == m_lastCursorColumn);

    if (sameContext)
    {
        if (m_requestId - 1 == m_lastRequestId)
        {
            // context has not changed and the correct results can be
            // directly shown
            //debug('request completion ignored, context has not '
            //                'changed')
            showPopup();
        }
        else
        {
            // same context but result not yet available
        }

        return true;
    }
    else
    {
        PythonEngine *pyEng = (PythonEngine*)m_pPythonEngine;
        if (pyEng)
        {
            QString filename;

            ScriptEditorWidget *sew = qobject_cast<ScriptEditorWidget*>(editor());

            if (sew)
            {
                filename = sew->getFilename();
            }

            if (filename == "")
            {
                filename = QDir::cleanPath(QDir::current().absoluteFilePath("__temporaryfile__.py"));
            }

            if (pyEng->tryToLoadJediIfNotYetDone())
            {
                // line and col might be changed if code is a virtual code (e.g. for command line, containing all its history)
                QString code = editor()->codeText(line, col);

                ito::JediCompletionRequest request;
                request.m_source = code;
                request.m_line = line;
                request.m_col = col;
                request.m_path = filename;
                request.m_prefix = m_completionPrefix;
                request.m_requestId = m_requestId;
                request.m_callbackFctName = "onJediCompletionResultAvailable";
                request.m_sender = this;

                pyEng->enqueueJediCompletionRequest(request);

                m_lastCursorColumn = col;
                m_lastCursorLine = line;
                m_requestId += 1;

                if (m_requestId == INT_MAX)
                {
                    m_requestId = 0;
                }
            }
            else
            {
                onStateChanged(false);
            }
        }

        return true;
    }
}

//--------------------------------------------------------------------
/*
Checks if the event's key and modifiers make the completion shortcut
(Ctrl+Space)

:param event: QKeyEvent

:return: bool
*/
bool CodeCompletionMode::isShortcut(QKeyEvent *e) const
{
#ifdef __APPLE__
    Qt::KeyboardModifier modifier = Qt::MetaModifier;
#else
    Qt::KeyboardModifier modifier = Qt::ControlModifier;
#endif
    bool valid_modifier = int(e->modifiers() & modifier) == modifier;
    bool valid_key = (e->key() == m_triggerKey);
    return valid_key && valid_modifier;
}

//--------------------------------------------------------------------
bool CodeCompletionMode::selectWithReturn() const
{
    return m_selectWithReturn;
}

//--------------------------------------------------------------------
void CodeCompletionMode::setSelectWithReturn(bool select)
{
    m_selectWithReturn = select;
}

//--------------------------------------------------------------------
/*
Hides the completer popup
*/
void CodeCompletionMode::hidePopup()
{
    m_lastCursorColumn = -1;
    m_lastCursorLine = -1;

    if (m_pCompleter->popup() && \
            m_pCompleter->popup()->isVisible())
    {
        m_pCompleter->popup()->hide();

        ToolTip::hideText();
    }
}

//--------------------------------------------------------------------
/*
*/
QRect CodeCompletionMode::getPopupRect() const
{
    QRect cursor_rec = editor()->cursorRect();

#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))
    int char_width = editor()->fontMetrics().horizontalAdvance('A');
#else
    int char_width = editor()->fontMetrics().width('A');
#endif

    int prefix_len = (m_completionPrefix.size() * char_width);
    cursor_rec.translate(
        editor()->panels()->marginSize() - prefix_len,
        editor()->panels()->marginSize(Panel::Top) + 5);
    int width = m_pCompleter->popup()->verticalScrollBar()->sizeHint().width();
    cursor_rec.setWidth(
        m_pCompleter->popup()->sizeHintForColumn(0) + width);
    return cursor_rec;
}

//--------------------------------------------------------------------
/*
Shows the popup at the specified index.
    :param index: index
    :return:
*/
void CodeCompletionMode::showPopup(int index /*= 0*/)
{
    QString fullPrefix = editor()->wordUnderCursor(false).selectedText();

    if (m_caseSensitive)
    {
        m_pCompleter->setCaseSensitivity(Qt::CaseSensitive);
    }
    else
    {
        m_pCompleter->setCaseSensitivity(Qt::CaseInsensitive);
    }

    // set prefix
    m_pCompleter->setCompletionPrefix(m_completionPrefix);
    int cnt = m_pCompleter->completionCount();
    QString selected = m_pCompleter->currentCompletion();

    if ((fullPrefix == selected) && (cnt == 1))
    {
        //debug('user already typed the only completion that we have')
        hidePopup();
    }
    else
    {
        // show the completion list
        if (editor()->isVisible())
        {
            if (m_pCompleter->widget() != editor())
            {
                m_pCompleter->setWidget(editor());
            }

            m_pCompleter->complete(getPopupRect());
            m_pCompleter->popup()->setCurrentIndex(m_pCompleter->completionModel()->index(index, 0));
            //debug(
            //    "popup shown: %r" % m_pCompleter->popup().isVisible())
        }
        //else:
        //    debug('cannot show popup, editor is not visible')
    }
}

//--------------------------------------------------------------------
/*
*/
void CodeCompletionMode::showCompletions(const QVector<JediCompletion> &completions)
{
    updateModel(completions);
    showPopup();
}

//--------------------------------------------------------------------
QPair<QStringList, QString> CodeCompletionMode::parseTooltipDocstring(const QString &docstring) const
{
    QStringList lines = Utils::strip(docstring).split("\n");
    QStringList signatures;
    int idx = 0;

    for (; idx < lines.size(); ++idx)
    {
        if (lines[idx] == "" || (lines[idx][0] == ' ' && lines[idx].trimmed() == ""))
        {
            // empty line or line with spaces only. skip. the real docstring comes now.
            break;
        }

        signatures << lines[idx];
    }

    QString docstr = lines.mid(idx + 1).join("\n");

    if (docstr.size() > m_tooltipsMaxLength)
    {
        int idx = docstr.lastIndexOf(' ', m_tooltipsMaxLength);

        if (idx > 0)
        {
            docstr = docstr.left(idx);
        }
        else
        {
            docstr = docstr.left(m_tooltipsMaxLength);
        }

        docstr += tr("...");
    }

    return qMakePair<QStringList, QString>(std::move(signatures), std::move(docstr));
}

//--------------------------------------------------------------------
/*
Creates a QStandardModel that holds the suggestion from the completion
models for the QCompleter

:param completionPrefix:
*/
QStandardItemModel* CodeCompletionMode::updateModel(const QVector<JediCompletion> &completions)
{
    // build the completion model
    QStandardItemModel* cc_model = new QStandardItemModel(this);
    m_tooltips.clear();
    QString name;
    QStandardItem *item;
    QIcon icon;

    foreach (const JediCompletion &completion, completions)
    {
        name = completion.m_name;
        item = new QStandardItem();
        item->setData(name, Qt::DisplayRole);

        QList<QPair<QStringList, QString>> tooltips;
        QPair<QStringList, QString> tooltip;

        if (completion.m_tooltips.size() > 0)
        {
            foreach(const QString &tt, completion.m_tooltips)
            {
                if (tt != "")
                {
                    tooltip = parseTooltipDocstring(tt);

                    if (tooltips.size() > 0 && tooltips.last().second == tooltip.second)
                    {
                        //same docstring -> add signatures to previous
                        tooltips[tooltips.size() - 1].first.append(tooltip.first);
                    }
                    else
                    {
                        tooltips.append(tooltip);
                    }
                }
            }
        }

        if (tooltips.size() == 0 && completion.m_description != "")
        {
            tooltips.append(parseTooltipDocstring(completion.m_description));
        }

        m_tooltips[name] = tooltips;

        if (completion.m_icon != "")
        {
            icon = QIcon(completion.m_icon);
            //if isinstance(icon, list):
            //    icon = QIcon.fromTheme(icon[0], QIcon(icon[1]))
            //else:
            //    icon = QIcon(icon)
            item->setData(QIcon(icon), Qt::DecorationRole);
        }
        cc_model->appendRow(item);
    }

    if (!m_pCompleter)
    {
        createCompleter();
    }

    m_pCompleter->setModel(cc_model);

    return cc_model;
}


//--------------------------------------------------------------------
/*
*/
void CodeCompletionMode::displayCompletionTooltip(const QString &completion) const
{
    if (!m_showTooltips)
    {
        return;
    }

    if (!m_tooltips.contains(completion))
    {
        ToolTip::hideText();
        return;
    }

    auto tooltips = m_tooltips[completion];

    /* tasks: convert tooltip to html, check for the first
    section with the definitions and wrap after maximum line length.
    Make a <hr> after the first section
    */
    QStringList styledTooltips;

    foreach(const auto &tip, tooltips)
    {
        // the signature is represented as <code> monospace section.
        // this requires much more space than ordinary letters.
        // Therefore reduce the maximum line length to 88/2.
        styledTooltips << Utils::parseStyledTooltipsFromSignature(
            tip.first,
            tip.second,
            44,
            m_tooltipsMaxLength
        );
    }

    if (styledTooltips.size() == 0)
    {
        ToolTip::hideText();
        return;
    }

    QPoint pos = m_pCompleter->popup()->pos();
    pos.setX(pos.x() + m_pCompleter->popup()->size().width());
    pos.ry() -= 15;
    QPoint altTopRightPos = m_pCompleter->popup()->pos();
    altTopRightPos.ry() += 40;
    ToolTip::showText(pos, styledTooltips.join("<hr>"), editor(), altTopRightPos);
}

//--------------------------------------------------------------------
/*
*/
/*static*/ bool CodeCompletionMode::isNavigationKey(QKeyEvent *e)
{
    return (e->key() == Qt::Key_Backspace || \
                e->key() == Qt::Key_Back || \
                e->key() == Qt::Key_Delete || \
                e->key() == Qt::Key_End || \
                e->key() == Qt::Key_Home || \
                e->key() == Qt::Key_Left || \
                e->key() == Qt::Key_Right || \
                e->key() == Qt::Key_Up || \
                e->key() == Qt::Key_Down || \
                e->key() == Qt::Key_Space);
}


} //end namespace ito
