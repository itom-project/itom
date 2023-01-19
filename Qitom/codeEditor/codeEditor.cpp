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

#include "codeEditor.h"

#include <qapplication.h>
#include <qdebug.h>
#include <qevent.h>
#include <qmenu.h>
#include <qmimedata.h>
#include <qpainter.h>
#include <qtextdocument.h>
#include <qtooltip.h>

#include "delayJobRunner.h"
#include "managers/modesManager.h"
#include "managers/panelsManager.h"
#include "managers/textDecorationsManager.h"
#include "panels/foldingPanel.h"
#include "utils/utils.h"

#include <iostream>


namespace ito {

CodeEditor::CodeEditor(QWidget* parent /*= NULL*/, bool createDefaultActions /*= true*/) :
    QPlainTextEdit(parent), m_showCtxMenu(true), m_defaultFontSize(10),
    m_useSpacesInsteadOfTabs(true), m_showWhitespaces(false), m_tabLength(0), m_zoomLevel(0),
    m_fontSize(10), m_fontFamily("Verdana"), m_selectLineOnCopyEmpty(true),
    m_wordSeparators("~!@#$%^&*()+{}|:\"'<>?,./;[]\\\n\t=- "), m_pPanels(NULL),
    m_pDecorations(NULL), m_pModes(NULL), m_lastMousePos(QPoint(0, 0)), m_prevTooltipBlockNbr(-1),
    m_pTooltipsRunner(NULL), m_edgeMode(EdgeNone), m_edgeColumn(88), m_edgeColor(Qt::darkGray),
    m_showIndentationGuides(true), m_indentationGuidesColor(Qt::darkGray), m_redoAvailable(false),
    m_undoAvailable(false), m_pContextMenu(NULL), m_minLineJumpsForGoBackNavigationReport(11)
{
    installEventFilter(this);
    connect(document(), SIGNAL(modificationChanged(bool)), this, SLOT(emitDirtyChanged(bool)));

    // connect slots
    connect(this, SIGNAL(blockCountChanged(int)), this, SLOT(update()));
    connect(this, SIGNAL(cursorPositionChanged()), this, SLOT(update()));
    connect(this, SIGNAL(selectionChanged()), this, SLOT(update()));
    connect(this, SIGNAL(undoAvailable(bool)), this, SLOT(setUndoAvailable(bool)));
    connect(this, SIGNAL(redoAvailable(bool)), this, SLOT(setRedoAvailable(bool)));

    setMouseTracking(true);
    setCenterOnScroll(true);
    setLineWrapMode(QPlainTextEdit::NoWrap);
    setCursorWidth(2);
    setTabLength(4);

    m_pPanels = new PanelsManager(this);
    m_pDecorations = new TextDecorationsManager(this);
    m_pModes = new ModesManager(this);

    m_pTooltipsRunner = new DelayJobRunner<CodeEditor, void (CodeEditor::*)(QList<QVariant>)>(700);

    m_pContextMenu = new QMenu(this);

    initStyle();
    resetStylesheet();
}

//-----------------------------------------------------------
CodeEditor::~CodeEditor()
{
    foreach (auto item, m_textBlockUserDataList)
    {
        item->removeCodeEditorRef();
    }

    delete m_pPanels;
    m_pPanels = NULL;

    delete m_pDecorations;
    m_pDecorations = NULL;

    delete m_pModes;
    m_pModes = NULL;

    if (m_pTooltipsRunner)
    {
        m_pTooltipsRunner->cancelRequests();
        delete m_pTooltipsRunner;
        m_pTooltipsRunner = NULL;
    }

    delete m_pContextMenu;
    m_pContextMenu = NULL;
}

//-----------------------------------------------------------
/*
Returns a reference to the :class:`pyqode.core.managers.PanelsManager`
used to manage the collection of installed panels
*/
PanelsManager* CodeEditor::panels() const
{
    return m_pPanels;
}

//-----------------------------------------------------------
/*
Returns a reference to the
:class:`pyqode.core.managers.TextDecorationManager` used to manage the
list of :class:`pyqode.core.api.TextDecoration`
*/
TextDecorationsManager* CodeEditor::decorations() const
{
    return m_pDecorations;
}

//-----------------------------------------------------------
/*
Returns a reference to the :class:`pyqode.core.managers.ModesManager`
        used to manage the collection of installed modes.
*/
ModesManager* CodeEditor::modes() const
{
    return m_pModes;
}

//-----------------------------------------------------------
/*
Returns a reference to the syntax highlighter mode currently used to
        highlight the editor content.

        :return: :class:`pyqode.core.api.SyntaxHighlighter`
*/
SyntaxHighlighterBase* CodeEditor::syntaxHighlighter() const
{
    if (!m_pModes)
        return NULL;

    SyntaxHighlighterBase* out = NULL;
    ModesManager::const_iterator it = m_pModes->constBegin();
    while (it != m_pModes->constEnd())
    {
        out = dynamic_cast<SyntaxHighlighterBase*>(it.value().data());
        if (out)
        {
            break;
        }

        it++;
    }

    return out;
}

//-----------------------------------------------------------
bool CodeEditor::useSpacesInsteadOfTabs() const
{
    return m_useSpacesInsteadOfTabs;
}

void CodeEditor::setUseSpacesInsteadOfTabs(bool value)
{
    m_useSpacesInsteadOfTabs = value;
    updateTabStopAndIndentationWidth();
}

//-----------------------------------------------------------
/*
The editor background color (QColor)
*/
QColor CodeEditor::background() const
{
    return m_background;
}

void CodeEditor::setBackground(const QColor& value)
{
    if (m_background != value)
    {
        m_background = value;
        resetStylesheet();
    }
}

//-----------------------------------------------------------
/*
The editor foreground color (QColor)
*/
QColor CodeEditor::foreground() const
{
    return m_foreground;
}

void CodeEditor::setForeground(const QColor& value)
{
    if (m_foreground != value)
    {
        m_foreground = value;
        resetStylesheet();
    }
}

//-----------------------------------------------------------
/*
The editor selection's foreground color.
*/
QColor CodeEditor::selectionForeground() const
{
    return m_selForeground;
}

void CodeEditor::setSelectionForeground(const QColor& value)
{
    if (m_selForeground != value)
    {
        m_selForeground = value;
        resetStylesheet();
    }
}

//-----------------------------------------------------------
/*
The editor selection's background color.
*/
QColor CodeEditor::selectionBackground() const
{
    return m_selBackground;
}

void CodeEditor::setSelectionBackground(const QColor& value)
{
    if (m_selBackground != value)
    {
        m_selBackground = value;
        resetStylesheet();
    }
}

//-----------------------------------------------------------
bool CodeEditor::selectLineOnCopyEmpty() const
{
    return m_selectLineOnCopyEmpty;
}

void CodeEditor::setSelectLineOnCopyEmpty(bool value)
{
    m_selectLineOnCopyEmpty = value;
}

//-----------------------------------------------------------
CodeEditor::EdgeMode CodeEditor::edgeMode() const
{
    return m_edgeMode;
}

void CodeEditor::setEdgeMode(CodeEditor::EdgeMode mode)
{
    if (m_edgeMode != mode)
    {
        m_edgeMode = mode;
        update();
    }
}

//-----------------------------------------------------------
int CodeEditor::edgeColumn() const
{
    return m_edgeColumn;
}

void CodeEditor::setEdgeColumn(int column)
{
    if (m_edgeColumn != column)
    {
        m_edgeColumn = column;
        update();
    }
}

//-----------------------------------------------------
QColor CodeEditor::edgeColor() const
{
    return m_edgeColor;
}

void CodeEditor::setEdgeColor(const QColor& color)
{
    if (m_edgeColor != color)
    {
        m_edgeColor = color;
        update();
    }
}

//-----------------------------------------------------
bool CodeEditor::showIndentationGuides() const
{
    return m_showIndentationGuides;
}

void CodeEditor::setShowIndentationGuides(bool value)
{
    if (m_showIndentationGuides != value)
    {
        m_showIndentationGuides = value;
        updateTabStopAndIndentationWidth();
        update();
    }
}

//-----------------------------------------------------
QColor CodeEditor::indentationGuidesColor() const
{
    return m_indentationGuidesColor;
}

void CodeEditor::setIndentationGuidesColor(const QColor& color)
{
    if (m_indentationGuidesColor != color)
    {
        m_indentationGuidesColor = color;
        update();
    }
}

//-----------------------------------------------------------
bool CodeEditor::showContextMenu() const
{
    return m_showCtxMenu;
}

void CodeEditor::setShowContextMenu(bool value)
{
    m_showCtxMenu = value;
}

//-----------------------------------------------------------
bool CodeEditor::isModified() const
{
    return document()->isModified();
}

void CodeEditor::setModified(bool modified)
{
    document()->setModified(modified);
}

//-----------------------------------------------------------
bool CodeEditor::showWhitespaces() const
{
    return m_showWhitespaces;
}

void CodeEditor::setShowWhitespaces(bool value)
{
    if (m_showWhitespaces != value)
    {
        m_showWhitespaces = value;
        setWhitespacesFlags(value);
        updateTabStopAndIndentationWidth();
        rehighlight();
    }
}

//-----------------------------------------------------------
QString CodeEditor::fontName() const
{
    return m_fontFamily;
}

void CodeEditor::setFontName(const QString& value)
{
    if (value == "")
    {
        m_fontFamily = "Verdana";
    }
    else
    {
        m_fontFamily = value;
    }
}

//-----------------------------------------------------------
int CodeEditor::fontSize() const
{
    return m_fontSize;
}

void CodeEditor::setFontSize(int fontSize)
{
    m_fontSize = fontSize;
}

//-----------------------------------------------------------
int CodeEditor::zoomLevel() const
{
    return m_zoomLevel;
}

void CodeEditor::setZoomLevel(int value)
{
    m_zoomLevel = value;
}

//-----------------------------------------------------------
int CodeEditor::tabLength() const
{
    return m_tabLength;
}

void CodeEditor::setTabLength(int value)
{
    if (m_tabLength != value)
    {
        m_tabLength = qBound(2, value, 2000);
        updateTabStopAndIndentationWidth();
    }
}

//-----------------------------------------------------------
QColor CodeEditor::whitespacesForeground() const
{
    if (syntaxHighlighter())
    {
        return syntaxHighlighter()
            ->editorStyle()
            ->format(StyleItem::KeyWhitespace)
            .foreground()
            .color();
    }
    else
    {
        return m_whitespacesForeground;
    }
}

void CodeEditor::setWhitespacesForeground(const QColor& value)
{
    if (syntaxHighlighter())
    {
        syntaxHighlighter()->editorStyle()->rformat(StyleItem::KeyWhitespace).setForeground(value);
    }

    m_whitespacesForeground = value;

    // updateTabStopAndIndentationWidth();
}

//-----------------------------------------------------------
QList<VisibleBlock> CodeEditor::visibleBlocks() const
{
    return m_visibleBlocks;
}

//-----------------------------------------------------------
void CodeEditor::emitDirtyChanged(bool state)
{
    emit dirtyChanged(state);
}

//-----------------------------------------------------------
void CodeEditor::cursorPosition(int& line, int& column) const
{
    line = textCursor().blockNumber();
    column = textCursor().columnNumber();
}

//-----------------------------------------------------------
bool CodeEditor::dirty() const
{
    return document()->isModified();
}

//-----------------------------------------------------------
void CodeEditor::setMouseCursor(const QCursor& cursor)
{
    viewport()->setCursor(cursor);
}

//-----------------------------------------------------------
void CodeEditor::initSettings()
{
    // init settings
    m_showWhitespaces = false;
    m_tabLength = 4;
    m_useSpacesInsteadOfTabs = true;
    setWhitespacesFlags(m_showWhitespaces);
    updateTabStopAndIndentationWidth();
}

//-----------------------------------------------------------
void CodeEditor::updateTabStopAndIndentationWidth()
{
    QFontMetrics fm = fontMetrics();

    if (syntaxHighlighter())
    {
        QFont f = syntaxHighlighter()->editorStyle()->rformat(StyleItem::KeyWhitespace).font();
        fm = QFontMetrics(f);
    }

#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))
    setTabStopDistance(tabLength() * fm.horizontalAdvance(" "));
#else
    setTabStopWidth(tabLength() * fm.width(" "));
#endif

    if (useSpacesInsteadOfTabs())
    {
        QString tab_text = QString(tabLength(), ' ');
#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))
        m_indentationBarWidth = fm.horizontalAdvance(tab_text);
#else
        m_indentationBarWidth = fm.width(tab_text);
#endif
    }
    else
    {
#if (QT_VERSION >= QT_VERSION_CHECK(5, 10, 0))
        m_indentationBarWidth = tabStopDistance();
#else
        m_indentationBarWidth = tabStopWidth();
#endif
    }
}

//-----------------------------------------------------------
void CodeEditor::initStyle()
{
    // Inits style options
    m_background = QColor("white");
    m_foreground = QColor("black");
    m_whitespacesForeground = QColor("light gray");
    QApplication* app = qobject_cast<QApplication*>(QApplication::instance());
    m_selBackground = app->palette().highlight().color();
    m_selForeground = app->palette().highlightedText().color();
    m_fontSize = 10;
    setFontName("");
}

//-----------------------------------------------------------
void CodeEditor::setWhitespacesFlags(bool show)
{
    // Sets show white spaces flag
    QTextDocument* doc = document();
    QTextOption options = doc->defaultTextOption();
    if (show)
    {
        options.setFlags(options.flags() | QTextOption::ShowTabsAndSpaces);
    }
    else
    {
        options.setFlags(options.flags() & ~QTextOption::ShowTabsAndSpaces);
    }

    doc->setDefaultTextOption(options);

    updateTabStopAndIndentationWidth();
}

//-----------------------------------------------------------
void CodeEditor::setViewportMargins(int left, int top, int right, int bottom)
{
    QPlainTextEdit::setViewportMargins(left, top, right, bottom);
}

//-----------------------------------------------------------
void CodeEditor::resizeEvent(QResizeEvent* e)
{
    /*
    Overrides resize event to resize the editor's panels.
    :param e: resize event
    */
    QPlainTextEdit::resizeEvent(e);
    m_pPanels->resize();
}

//-----------------------------------------------------------
void CodeEditor::closeEvent(QCloseEvent* e)
{
    close();
    QPlainTextEdit::closeEvent(e);
}

//-----------------------------------------------------------
/*
Overrides paint event to update the list of visible blocks and emit
the painted e->
:param e: paint event
*/
void CodeEditor::paintEvent(QPaintEvent* e)
{
    updateVisibleBlocks(); //_update_visible_blocks

    QTextCursor tc = textCursor();
    tc.movePosition(QTextCursor::Start);
    int xoffset = cursorRect(tc).x(); // left offset of first character

    switch (m_edgeMode)
    {
    case EdgeNone:
        break;
    case EdgeLine: {
        QPainter painter(viewport());
        painter.setPen(m_edgeColor);

#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))
        int x = fontMetrics().horizontalAdvance(QString(m_edgeColumn, '9'));
#else
        int x = fontMetrics().width(QString(m_edgeColumn, '9'));
#endif
        x += xoffset;
        painter.drawLine(x, 0, x, size().height());
    }
    break;
    case EdgeBackground: {
        QPainter painter(viewport());
        painter.setBrush(m_edgeColor);
        painter.setPen(Qt::NoPen);

#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))
        int x = fontMetrics().horizontalAdvance(QString(m_edgeColumn, '9'));
#else
        int x = fontMetrics().width(QString(m_edgeColumn, '9'));
#endif
        x += xoffset;
        painter.drawRect(x, 0, size().width() - x, size().height());
    }
    break;
    }

    QPlainTextEdit::paintEvent(e);

    if (m_showIndentationGuides)
    {
        QPainter painter(viewport());

        QColor color = m_indentationGuidesColor;
        color.setAlphaF(.5);
        painter.setPen(color);
        int bottom;
        int indentation;

        foreach (const VisibleBlock& block, visibleBlocks())
        {
            bottom = block.topPosition + blockBoundingRect(block.textBlock).height() - 1;
            indentation = Utils::TextBlockHelper::getFoldLvl(block.textBlock);

            for (int i = 1; i < indentation; ++i)
            {
                painter.drawLine(
                    xoffset + m_indentationBarWidth * i,
                    block.topPosition,
                    xoffset + m_indentationBarWidth * i,
                    bottom);
            }
        }
    }

    emit painted(e);
}

//-----------------------------------------------------------
void CodeEditor::keyPressEvent(QKeyEvent* e)
{
    /*
    Overrides the keyPressEvent to emit the key_pressed signal.
    Also takes care of indenting and handling smarter home key.
    :param e: QKeyEvent
    */
    if (isReadOnly())
    {
        return;
    }
    bool initial_state = e->isAccepted();
    e->ignore();
    emit keyPressed(e);
    bool state = e->isAccepted();
    bool forward = false;

    if (!e->isAccepted())
    {
        forward = keyPressInternalEvent(e);
    }

    if (!e->isAccepted() && forward)
    {
        forward = false;

        if (e->key() == Qt::Key_Tab && e->modifiers() == Qt::NoModifier)
        {
            indent();
            e->accept();
        }
        else if (e->key() == Qt::Key_Backtab && e->modifiers() == Qt::NoModifier)
        {
            unindent();
            e->accept();
        }
        else if (e->key() == Qt::Key_Backtab && e->modifiers() == Qt::ShiftModifier)
        {
            unindent();
            e->accept();
        }
        else if (e->key() == Qt::Key_Home && (int(e->modifiers()) & Qt::ControlModifier) == 0)
        {
            doHomeKey(e, int(e->modifiers()) & Qt::ShiftModifier);
        }
        else if (
            (e->modifiers() & Qt::ShiftModifier) &&
            ((e->key() == Qt::Key_Enter) || (e->key() == Qt::Key_Return)))
        {
            // deny soft line break. not desired in editor.
            e->accept(); // do not further process this key
        }
        else if (
            (e->modifiers() & Qt::AltModifier) && (e->modifiers() & Qt::ControlModifier) &&
            (e->key() == Qt::Key_D))
        {
            // ignore this key, even if the generate docstring action is currently disabled.
            e->accept();
        }

        if (!e->isAccepted())
        {
            e->setAccepted(initial_state);
            QPlainTextEdit::keyPressEvent(e);
        }
    }

    if (forward)
    {
        e->setAccepted(initial_state);
        QPlainTextEdit::keyPressEvent(e);
    }

    bool new_state = e->isAccepted();
    e->setAccepted(state);
    emit postKeyPressed(e);
    e->setAccepted(new_state);
}

//-----------------------------------------------------------
void CodeEditor::keyReleaseEvent(QKeyEvent* e)
{
    /*
    Overrides keyReleaseEvent to emit the key_released signal.
    :param e: QKeyEvent
    */
    if (isReadOnly())
    {
        return;
    }
    bool initial_state = e->isAccepted();
    e->ignore();
    emit keyReleased(e);
    if (!e->isAccepted())
    {
        e->setAccepted(initial_state);
        QPlainTextEdit::keyReleaseEvent(e);
    }
}

//-----------------------------------------------------------
void CodeEditor::mouseDoubleClickEvent(QMouseEvent* e)
{
    bool initial_state = e->isAccepted();
    e->ignore();
    emit mouseDoubleClicked(e);
    if (!e->isAccepted())
    {
        e->setAccepted(initial_state);
        QPlainTextEdit::mouseDoubleClickEvent(e);
    }
}

//-----------------------------------------------------------
/*
Overrides focusInEvent to emits the focused_in signal
:param event: QFocusEvent
*/
void CodeEditor::focusInEvent(QFocusEvent* e)
{
    emit focusedIn(e);
    QPlainTextEdit::focusInEvent(e);
}

//-----------------------------------------------------------
/* Saves content if save_on_focus_out is True.
 */
void CodeEditor::focusOutEvent(QFocusEvent* e)
{
    QPlainTextEdit::focusOutEvent(e);
}

//-----------------------------------------------------------
/*
    Overrides mousePressEvent to emits mousePressed signal

    :param event: QMouseEvent
*/
void CodeEditor::mousePressEvent(QMouseEvent* e)
{
    bool initialState = e->isAccepted();
    e->ignore();
    emit mousePressed(e);

    if (e->button() == Qt::LeftButton)
    {
        QTextCursor cursor = cursorForPosition(e->pos());

        TextDecorationsManager::const_iterator it = m_pDecorations->constBegin();
        while (it != m_pDecorations->constEnd())
        {
            if (it->isNull() == false)
            {
                if (it->data()->cursor.blockNumber() == cursor.blockNumber())
                {
                    if (it->data()->containsCursor(cursor))
                    {
                        it->data()->emitClicked(*it);
                    }
                }
            }

            ++it;
        }
    }

    if (!e->isAccepted())
    {
        e->setAccepted(initialState);
        QPlainTextEdit::mousePressEvent(e);
    }
}

//-----------------------------------------------------------
/*
Emits mouse_released signal.
:param event: QMouseEvent
*/
void CodeEditor::mouseReleaseEvent(QMouseEvent* e)
{
    bool initialState = e->isAccepted();
    e->ignore();
    emit mouseReleased(e);
    if (!e->isAccepted())
    {
        e->setAccepted(initialState);
        QPlainTextEdit::mouseReleaseEvent(e);
    }
}

//-----------------------------------------------------------
void CodeEditor::callWheelEvent(QWheelEvent* e)
{
    wheelEvent(e);
}

//-----------------------------------------------------------
void CodeEditor::wheelEvent(QWheelEvent* e)
{
    /*
    Emits the mouse_wheel_activated signal.
    :param event: QMouseEvent
    */
    bool initialState = e->isAccepted();
    e->ignore();
    emit mouseWheelActivated(e);
    if (!e->isAccepted())
    {
        e->setAccepted(initialState);
        QPlainTextEdit::wheelEvent(e);
    }
}

//-----------------------------------------------------------
/*
    Overrides mouseMovedEvent to display any decoration tooltip and emits
    the mouse_moved e->
    :param event: QMouseEvent
*/
void CodeEditor::mouseMoveEvent(QMouseEvent* e)
{
    QTextCursor cursor = this->cursorForPosition(e->pos());
    m_lastMousePos = e->pos();
    bool blockFound = false;

    TextDecorationsManager::const_iterator it = m_pDecorations->constBegin();
    TextDecoration::Ptr itPtr;

    while (it != m_pDecorations->constEnd())
    {
        itPtr = *it;
        if (itPtr->containsCursor(cursor) && (itPtr->tooltip() != ""))
        {
            if (m_prevTooltipBlockNbr != cursor.blockNumber() || !QToolTip::isVisible())
            {
                QPoint position = e->pos();
                // add left margin
                position.setX(position.x() + m_pPanels->marginSize(Panel::Left));
                // add top margin
                position.setY(position.y() + m_pPanels->marginSize(Panel::Top));

                QList<QVariant> args;
                args << mapToGlobal(position);
                args << itPtr->tooltip().left(1024);
                args << QVariant::fromValue(*it);

                if (m_pTooltipsRunner)
                {
                    DELAY_JOB_RUNNER(
                        m_pTooltipsRunner, CodeEditor, void (CodeEditor::*)(QList<QVariant>))
                        ->requestJob(this, &CodeEditor::showTooltipDelayJobRunner, args);
                }

                m_prevTooltipBlockNbr = cursor.blockNumber();
            }
            blockFound = true;
            break;
        }

        it++;
    }

    if (!blockFound && m_prevTooltipBlockNbr != -1)
    {
        QToolTip::hideText();
        m_prevTooltipBlockNbr = -1;

        if (m_pTooltipsRunner)
        {
            m_pTooltipsRunner->cancelRequests();
        }
    }

    emit mouseMoved(e);
    QPlainTextEdit::mouseMoveEvent(e);
}

//----------------------------------------------------------
/*
    Show a tool tip at the specified position

    :param pos: Tooltip position
    :param tooltip: Tooltip text
    :param _sender_deco: TextDecoration which is the sender of the show
        tooltip request. (for internal use only).
*/
void CodeEditor::showTooltip(
    const QPoint& pos, const QString& tooltip, const TextDecoration::Ptr& senderDeco)
{
    if (!m_pDecorations->contains(senderDeco))
    {
        return;
    }

    QToolTip::showText(pos, tooltip.left(1024), this);
}

void CodeEditor::showTooltip(const QPoint& pos, const QString& tooltip)
{
    QToolTip::showText(pos, tooltip.left(1024), this);
}

//-----------------------------------------------------------
void CodeEditor::showEvent(QShowEvent* e)
{
    /* Overrides showEvent to update the viewport margins */
    QPlainTextEdit::showEvent(e);
    m_pPanels->refresh();
}


//-----------------------------------------------------------
/*
Indents the text cursor or the selection.

Emits the :attr:`pyqode.core.api.CodeEdit.indent_requested`
signal, the :class:`pyqode.core.modes.IndenterMode` will
perform the actual indentation.
*/
void CodeEditor::indent()
{
    emit indentRequested();
}

//-----------------------------------------------------------
/*
Un-indents the text cursor or the selection.

Emits the :attr:`pyqode.core.api.CodeEdit.unindent_requested`
signal, the :class:`pyqode.core.modes.IndenterMode` will
perform the actual un-indentation.
*/
void CodeEditor::unindent()
{
    emit unindentRequested();
}


//-----------------------------------------------------------
/*
Performs home key action
*/
void CodeEditor::doHomeKey(QEvent* event /*= NULL*/, bool select /* = false*/)
{
    // get nb char to first significative char
    QTextCursor cursor = textCursor();
    int delta = (cursor.positionInBlock() - lineIndent());

    QTextCursor::MoveMode move = QTextCursor::MoveAnchor;
    if (select)
    {
        move = QTextCursor::KeepAnchor;
    }

    if (delta > 0)
    {
        cursor.movePosition(QTextCursor::Left, move, delta);
    }
    else
    {
        cursor.movePosition(QTextCursor::StartOfBlock, move);
    }

    setTextCursor(cursor);
    reportGoBackNavigationCursorMovement(CursorPosition(cursor, -1), "homeKey");

    if (event)
    {
        event->accept();
    }
}

//-----------------------------------------------------------
void CodeEditor::reportGoBackNavigationCursorMovement(
    const CursorPosition& cursor, const QString& origin) const
{
    // do nothing
}


//-----------------------------------------------------------
/*
Updates the list of visible blocks
*/
void CodeEditor::updateVisibleBlocks()
{
    m_visibleBlocks.clear();
    QTextBlock block = firstVisibleBlock();
    int block_nbr = block.blockNumber();
    int top = int(blockBoundingGeometry(block).translated(contentOffset()).top());
    int bottom = top + int(blockBoundingRect(block).height());
    int ebottom_top = 0;
    int ebottom_bottom = height();
    while (block.isValid())
    {
        bool visible = ((top >= ebottom_top) && (bottom <= ebottom_bottom));
        if (!visible)
        {
            break;
        }
        if (block.isVisible())
        {
            VisibleBlock vb;
            vb.lineNumber = block_nbr;
            vb.textBlock = block;
            vb.topPosition = top;
            m_visibleBlocks.append(vb);
        }
        block = block.next();
        top = bottom;
        bottom = top + int(blockBoundingRect(block).height());
        block_nbr = block.blockNumber();
    }
}

//-----------------------------------------------------------
/*
return the line number of the first visible line (or -1 if no line available)
*/
int CodeEditor::firstVisibleLine() const
{
    QTextBlock block = firstVisibleBlock();
    if (block.isValid())
    {
        return block.blockNumber();
    }
    else
    {
        return -1;
    }
}

//-----------------------------------------------------------
/*
Set the number of the first visible line to line.
*/
void CodeEditor::setFirstVisibleLine(int line)
{
    moveCursor(QTextCursor::End);
    QTextCursor cursor(document()->findBlockByNumber(line));
    setTextCursor(cursor);
}

//-----------------------------------------------------------
/*
Moves the text cursor to the specified position..

:param line: Number of the line to go to (0 based)
:param column: Optional column number. Default is 0 (start of line).
:param move: True to move the cursor. False will return the cursor
                without setting it on the editor.
:return: The new text cursor
:rtype: QtGui.QTextCursor
*/
QTextCursor CodeEditor::gotoLine(int line, int column, bool move /*= true*/)
{
    QTextCursor text_cursor = moveCursorTo(line);

    if (column >= 0)
    {
        text_cursor.movePosition(QTextCursor::Right, QTextCursor::MoveAnchor, column);
    }

    if (move)
    {
        setTextCursor(text_cursor);
        unfoldCursorPosition();
        ensureCursorVisible();
    }

    reportGoBackNavigationCursorMovement(CursorPosition(text_cursor), "gotoLine");

    return text_cursor;
}

//-----------------------------------------------------------
void CodeEditor::reportPositionAsGoBackNavigationItem(
    const QTextCursor& cursor, const QString& reason) const
{
    reportGoBackNavigationCursorMovement(CursorPosition(cursor), reason);
}

//-----------------------------------------------------------
/*
Returns the indent level of the specified line

    :param line_nbr: Number of the line to get indentation (1 base).
        Pass None to use the current line number. Note that you can also
        pass a QTextBlock instance instead of an int.
    :return: Number of spaces that makes the indentation level of the
                current line
*/
int CodeEditor::lineIndent(int lineNumber /*= -1*/) const
{
    if (lineNumber == -1)
    {
        lineNumber = currentLineNumber();
    }

    QString line = lineText(lineNumber);

    int lindent = 0;
    while (lindent < line.size() && line[lindent].isSpace())
    {
        lindent++;
    }

    return lindent;
}

//-------------------------------------------------------------
int CodeEditor::lineIndent(const QTextBlock* lineNbr) const
{
    if (lineNbr == NULL)
    {
        return lineIndent(-1);
    }
    else
    {
        return lineIndent(lineNbr->blockNumber());
    }
}

//-------------------------------------------------------------
/*virtual*/ bool CodeEditor::eventFilter(QObject* obj, QEvent* e)
{
    if ((obj == this) && (e->type() == QEvent::KeyPress))
    {
        QKeyEvent* ke = dynamic_cast<QKeyEvent*>(e);
        if (ke->matches(QKeySequence::Cut))
        {
            cut();
            return true;
        }
        else if (ke->matches(QKeySequence::Copy))
        {
            copy();
            return true;
        }
    }
    return false;
}

//-------------------------------------------------------------
/*
Cuts the selected text or the whole line if no text was selected,
the current line is not empty and selectLineOnCopy() is true.
(the latter feature can be turned off by
setting :attr:`select_line_on_copy_empty` to False).
*/
void CodeEditor::cut()
{
    QTextCursor tc = textCursor();
    tc.beginEditBlock();
    bool from_selection = false;

    if (!tc.hasSelection() && selectLineOnCopyEmpty() && currentLineText() != "")
    {
        tc.movePosition(QTextCursor::StartOfLine);
        tc.movePosition(QTextCursor::Left);
        tc.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor);
        tc.movePosition(QTextCursor::EndOfLine, QTextCursor::KeepAnchor);
        from_selection = true;
    }
    else
    {
        from_selection = true;
    }

    tc.endEditBlock();
    setTextCursor(tc);

    if (from_selection)
    {
        QPlainTextEdit::cut();
    }
}

//-------------------------------------------------------------
/*
Copy the selected text to the clipboard. If no text was selected, the
entire line is copied (this feature can be turned off by
setting :attr:`select_line_on_copy_empty` to False).
*/
void CodeEditor::copy()
{
    if (selectLineOnCopyEmpty() && !textCursor().hasSelection())
    {
        selectWholeLine();
    }

    QPlainTextEdit::copy();
}

//-----------------------------------------------------------
/*
Gets the text of the specified line

:param line_idx: The line number of the text to get
:return: Entire line's text
:rtype: str
*/
QString CodeEditor::lineText(int lineIdx) const
{
    if (lineIdx < 0)
    {
        return "";
    }

    const QTextBlock& block = document()->findBlockByNumber(lineIdx);
    return block.text();
}

/*
Returns the text cursor's line number.

:return: Line number
*/
int CodeEditor::currentLineNumber() const
{
    return textCursor().blockNumber();
}

//------------------------------------------------------------
/*
Returns the text cursor's column number.

:return: Column number
*/
int CodeEditor::currentColumnNumber() const
{
    return textCursor().columnNumber();
}

//------------------------------------------------------------
/*
Gets the previous line text (relative to the current cursor pos).
:return: previous line text (str)
*/
QString CodeEditor::previousLineText() const
{
    if (currentLineNumber() > 0)
    {
        return lineText(currentLineNumber() - 1);
    }
    return "";
}

//------------------------------------------------------------
/*
QScintilla uses the combination of a line number and a character
index from the start of that line to specify the position of a
character within the text. The underlying Scintilla instead uses
a byte index from the start of the text. This will return the byte
index corresponding to the line line number and index character index.
*/
int CodeEditor::positionFromLineIndex(int line, int column) const
{
    QTextBlock block = document()->findBlockByNumber(line);
    if (block.isValid())
    {
        if (block.length() >= column)
        {
            return block.position() + column;
        }
    }
    return -1;
}

//------------------------------------------------------------
/*
Returns the text of the current line.

:return: Text of the current line
*/
QString CodeEditor::currentLineText() const
{
    return lineText(currentLineNumber());
}

//------------------------------------------------------------
/*
Extends setPlainText to force the user to setup an encoding and a mime type.

Emits the new_text_set signal.

:param txt: The new text to set.
:param mime_type: Associated mimetype. Setting the mime will update the
                    pygments lexer.
:param encoding: text encoding
*/
void CodeEditor::setPlainText(
    const QString& text, const QString& mimeType /*= ""*/, const QString& encoding /*= ""*/)
{
    QPlainTextEdit::setPlainText(text);
    emit newTextSet();
    setRedoAvailable(false);
    setUndoAvailable(false);
}

//------------------------------------------------------------
/*
 */
QString CodeEditor::selectedText() const
{
    return textCursor().selectedText().replace(QChar(0x2029), '\n');
}

//------------------------------------------------------------
void CodeEditor::removeSelectedText()
{
    textCursor().removeSelectedText();
}

//------------------------------------------------------------
void CodeEditor::append(const QString& text)
{
    QTextCursor cursor = textCursor();
    cursor.movePosition(QTextCursor::End);
    cursor.insertText(text);
}

//------------------------------------------------------------
void CodeEditor::insertAt(const QString& text, int line, int index)
{
    setCursorPosition(line, index, true);
    insertPlainText(text);
}

//------------------------------------------------------------
/*
Replace the current selection, set by a previous call to
findFirst(), findFirstInSelection() or findNext(), with replaceStr.
*/
void CodeEditor::replace(const QString& text)
{
    QTextCursor cursor = textCursor();

    int start = cursor.selectionStart();
    int end = cursor.selectionEnd();

    if (!cursor.hasSelection())
    {
        return;
    }

    cursor.removeSelectedText();
    cursor.setPosition(start);
    cursor.insertText(text);
    cursor.setPosition(cursor.position() + text.size(), QTextCursor::MoveAnchor);
}

//--------------------------------------------------------------
/*
 */
bool CodeEditor::findFirst(
    const QString& expr,
    bool re,
    bool cs,
    bool wo,
    bool wrap,
    bool forward /*= true*/,
    int line /*= -1*/,
    int index /*= -1*/,
    bool show /*= true*/)
{
    QTextCursor current_cursor = textCursor();

    if (line >= 0 && index >= 0)
    {
        current_cursor = setCursorPosition(line, index, false);
    }

    QTextCursor cursor;

    QTextDocument::FindFlags flags;
    if (!forward)
    {
        flags |= QTextDocument::FindBackward;
    }
    if (wo)
    {
        flags |= QTextDocument::FindWholeWords;
    }
    if (cs)
    {
        flags |= QTextDocument::FindCaseSensitively;
    }

    if (re)
    {
        QRegularExpression regExp(expr);

        if (!cs)
        {
            regExp.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        }

        cursor = document()->find(regExp, current_cursor, flags);

        if (cursor.isNull() && wrap)
        {
            if (forward)
            {
                current_cursor.setPosition(0);
                cursor = document()->find(regExp, current_cursor, flags);
            }
            else
            {
                QTextBlock block = document()->lastBlock();
                current_cursor.setPosition(block.position() + block.length());
                cursor = document()->find(regExp, current_cursor, flags);
            }
        }
    }
    else
    {
        cursor = document()->find(expr, current_cursor, flags);

        if (cursor.isNull() && wrap)
        {
            if (forward)
            {
                current_cursor.setPosition(0);
                cursor = document()->find(expr, current_cursor, flags);
            }
            else
            {
                current_cursor.movePosition(QTextCursor::End);
                cursor = document()->find(expr, current_cursor, flags);
            }
        }
    }

    m_lastFindOptions.valid = true;
    m_lastFindOptions.cs = cs;
    m_lastFindOptions.expr = expr;
    m_lastFindOptions.forward = forward;
    m_lastFindOptions.re = re;
    m_lastFindOptions.show = show;
    m_lastFindOptions.wo = wo;
    m_lastFindOptions.wrap = wrap;

    if (cursor.isNull() == false)
    {
        setTextCursor(cursor);

        if (show)
        {
            unfoldCursorPosition();
            ensureCursorVisible();

            reportGoBackNavigationCursorMovement(CursorPosition(cursor), "findFirst");
        }

        return true;
    }

    return false;
}

//--------------------------------------------------------------
/*
 */
bool CodeEditor::findNext()
{
    const FindOptions& f = m_lastFindOptions;

    if (f.valid)
    {
        return findFirst(f.expr, f.re, f.cs, f.wo, f.wrap, f.forward, -1, -1, f.show);
    }

    return false;
}

//--------------------------------------------------------------
/*
Resets stylesheet
*/
void CodeEditor::resetStylesheet()
{
    setFont(QFont(m_fontFamily, m_fontSize + m_zoomLevel));

    bool flg_stylesheet = property("flg_stylesheet").isValid();
    if (qApp->styleSheet() != "" || flg_stylesheet)
    {
        setProperty("flg_stylesheet", true);
        /*On Window, if the application once had a stylesheet, we must
        keep on using a stylesheet otherwise strange colors appear
        see https://github.com/OpenCobolIDE/OpenCobolIDE/issues/65
        Also happen on plasma 5*/
        QByteArray ds = qgetenv("DESKTOP_SESSION");
        if (ds.isEmpty() == false && ds == "plasma")
        {
            setStyleSheet(QString("QPlainTextEdit \
            { \
                background-color: %1; \
                color: %2; \
            }")
                              .arg(m_background.name(), m_foreground.name()));
        }
        else
        {
#if WIN32
            setStyleSheet(QString("QPlainTextEdit \
            { \
                background-color: %1; \
                color: %2; \
            }")
                              .arg(m_background.name(), m_foreground.name()));
#else
            /*on linux/osx we just have to set an empty stylesheet to
            cancel any previous stylesheet and still keep a correct
            style for scrollbars*/
            setStyleSheet("");
#endif
        }
    }
    else
    {
        QPalette p = palette();
        p.setColor(QPalette::Base, m_background);
        p.setColor(QPalette::Text, m_foreground);
        p.setColor(QPalette::Highlight, m_selBackground);
        p.setColor(QPalette::HighlightedText, m_selForeground);
        setPalette(p);
    }
    repaint();
}

//------------------------------------------------------------
/*
Checks if a block/cursor is a string or a comment.

:param cursor_or_block: QTextCursor or QTextBlock
:param formats: the list of color scheme formats to consider. By
    default, it will consider the following keys: 'comment', 'string',
    'docstring'.
*/
bool CodeEditor::isCommentOrString(
    const QTextCursor& cursor,
    const QList<StyleItem::StyleType>& formats /*= QList<StyleItem::StyleType>()*/)
{
    return isCommentOrString(cursor.block(), formats);
}

//------------------------------------------------------------
/*
Checks if a block/cursor is a string or a comment.

:param cursor_or_block: QTextCursor or QTextBlock
:param formats: the list of color scheme formats to consider. By
    default, it will consider the following keys: 'comment', 'string',
    'docstring'.
*/
bool CodeEditor::isCommentOrString(
    const QTextBlock& block,
    const QList<StyleItem::StyleType>& formats /*= QList<StyleItem::StyleType>()*/)
{
    QList<StyleItem::StyleType> formats_ = formats;
    if (formats_.size() == 0)
    {
        formats_ << StyleItem::KeyComment << StyleItem::KeyString << StyleItem::KeyDocstring;
    }

    int pos = block.text().size() - 1;
    const QTextLayout* layout = block.layout();
    bool is_user_obj;

    if (layout)
    {
        auto additional_formats = layout->formats();
        const SyntaxHighlighterBase* sh = syntaxHighlighter();

        if (sh)
        {
            QSharedPointer<CodeEditorStyle> ref_formats = sh->editorStyle();

            foreach (const QTextLayout::FormatRange& r, additional_formats)
            {
                if ((r.start <= pos) && (pos < (r.start + r.length)))
                {
                    is_user_obj = (r.format.objectType() == StyleItem::GroupCommentOrString);

                    foreach (StyleItem::StyleType fmtType, formats_)
                    {
                        if (is_user_obj && (ref_formats->format(fmtType) == r.format))
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }

    return false;
}


//------------------------------------------------------------
/*
Checks if a block/cursor is a number (int, float, complex...).

:param cursor_or_block: QTextCursor or QTextBlock
:param formats: the list of color scheme formats to consider. By
    default, it will consider the following keys: 'comment', 'string',
    'docstring'.
*/
bool CodeEditor::isNumber(const QTextCursor& cursor) const
{
    return isNumber(cursor.block());
}

//------------------------------------------------------------
/*
Checks if a block/cursor is a number (int, float, complex...).

:param cursor_or_block: QTextCursor or QTextBlock
*/
bool CodeEditor::isNumber(const QTextBlock& block) const
{
    int pos = block.text().size() - 1;
    const QTextLayout* layout = block.layout();
    bool is_user_obj;

    if (layout)
    {
        auto additional_formats = layout->formats();
        const SyntaxHighlighterBase* sh = syntaxHighlighter();

        if (sh)
        {
            QSharedPointer<CodeEditorStyle> ref_formats = sh->editorStyle();

            foreach (const QTextLayout::FormatRange& r, additional_formats)
            {
                is_user_obj = (r.format.objectType() == StyleItem::GroupNumber);

                if ((r.start <= pos) && (pos < (r.start + r.length)))
                {
                    if (is_user_obj && (ref_formats->format(StyleItem::KeyNumber) == r.format))
                    {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

//------------------------------------------------------------
/*
Gets the word under cursor using the separators defined by
:attr:`pyqode.core.api.CodeEdit.word_separators`.

.. note: Instead of returning the word string, this function returns
    a QTextCursor, that way you may get more information than just the
    string. To get the word, just call ``selectedText`` on the returned
    value.

:param select_whole_word: If set to true the whole word is selected,
    else the selection stops at the cursor position.
:param text_cursor: Optional custom text cursor (e.g. from a
    QTextDocument clone)
:returns: The QTextCursor that contains the selected word.
*/
QTextCursor CodeEditor::wordUnderCursor(bool selectWholeWord) const
{
    return wordUnderCursor(textCursor(), selectWholeWord);
}

//----------------------------------------------------------------------------------------------------------------------------------
QTextCursor CodeEditor::wordUnderCursor(const QTextCursor& cursor, bool selectWholeWord) const
{
    QTextCursor text_cursor = cursor;
    int endPos, startPos;
    endPos = startPos = text_cursor.position();
    QString selectedText;
    QChar firstChar;
    // select char by char until we are at the original cursor position.
    while (!text_cursor.atStart())
    {
        text_cursor.movePosition(QTextCursor::Left, QTextCursor::KeepAnchor, 1);
        selectedText = text_cursor.selectedText();
        if (selectedText.size() > 0)
        {
            firstChar = selectedText[0];
            if (m_wordSeparators.contains(firstChar) &&
                    (selectedText != "n" && selectedText != "t") ||
                firstChar.isSpace())
            {
                break; // start boundary found
            }
        }
        startPos = text_cursor.position();
        text_cursor.setPosition(startPos);
    }

    if (selectWholeWord)
    {
        // select the rest of the word
        text_cursor.setPosition(endPos);
        while (!text_cursor.atEnd())
        {
            text_cursor.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor, 1);
            selectedText = text_cursor.selectedText();
            if (selectedText.size() > 0)
            {
                firstChar = selectedText[0];
                if (m_wordSeparators.contains(selectedText) &&
                        (selectedText != "n" && selectedText != "t") ||
                    firstChar.isSpace())
                {
                    break; // end boundary found
                }
                endPos = text_cursor.position();
                text_cursor.setPosition(endPos);
            }
        }
    }
    // now that we habe the boundaries, we can select the text
    text_cursor.setPosition(startPos);
    text_cursor.setPosition(endPos, QTextCursor::KeepAnchor);

    return text_cursor;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*
Selects the word under the **mouse** cursor.

:return: A QTextCursor with the word under mouse cursor selected.
*/
QTextCursor CodeEditor::wordUnderMouseCursor() const
{
    QTextCursor text_cursor = cursorForPosition(m_lastMousePos);

    // check if text_cursor is not behind end of line
    QTextCursor end_of_line_cursor(text_cursor);
    end_of_line_cursor.movePosition(QTextCursor::EndOfLine, QTextCursor::MoveAnchor);

    if (text_cursor.position() < end_of_line_cursor.position())
    {
        text_cursor.select(QTextCursor::WordUnderCursor);
        if (text_cursor.selectedText().isEmpty())
        {
            return QTextCursor();
        }
        else
        {
            return text_cursor;
        }
    }
    else
    {
        return QTextCursor();
    }
}


//----------------------------------------------------------------------------------------------------------------------------------
QString CodeEditor::wordAtPosition(int line, int index, bool selectWholeWord) const
{
    if (line < 0 || line >= lineCount())
    {
        return "";
    }

    if (index < 0 || index >= lineLength(line))
    {
        return "";
    }

    QTextCursor text_cursor = textCursor();
    text_cursor.movePosition(QTextCursor::Start);
    text_cursor.movePosition(QTextCursor::Down, QTextCursor::MoveAnchor, line); // go down y-times
    text_cursor.movePosition(QTextCursor::Right, QTextCursor::MoveAnchor, index); // go right
                                                                                  // x-times
    return wordUnderCursor(text_cursor, selectWholeWord).selectedText();
}

//------------------------------------------------------------
/*
Returns the line number from the y_pos

:param y_pos: Y pos in the editor
:return: Line number (0 based), -1 if out of range
*/
int CodeEditor::lineNbrFromPosition(int yPos) const
{
    int height = fontMetrics().height();
    foreach (const VisibleBlock& vb, visibleBlocks())
    {
        if ((vb.topPosition <= yPos) && (yPos <= (vb.topPosition + height)))
        {
            return vb.lineNumber;
        }
    }
    return -1;
}

//------------------------------------------------------------
void CodeEditor::lineIndexFromPosition(const QPoint& pos, int* line, int* column) const
{
    QTextCursor cursor = cursorForPosition(pos);

    if (line)
    {
        *line = cursor.blockNumber();
    }

    if (column)
    {
        *column = cursor.columnNumber();
    }
}

//------------------------------------------------------------
void CodeEditor::lineIndexFromPosition(int pos, int* line, int* column) const
{
    QTextBlock block = document()->findBlock(pos);
    if (block.isValid())
    {
        if (line)
        {
            *line = block.blockNumber();
        }
        if (column)
        {
            *column = pos - block.position();
        }
    }
    else
    {
        if (line)
        {
            *line = -1;
        }

        if (column)
        {
            *column = -1;
        }
    }
}

//------------------------------------------------------------
void CodeEditor::getCursorPosition(int* line, int* column) const
{
    QTextCursor cursor = textCursor();
    if (line)
    {
        *line = cursor.blockNumber();
    }
    if (column)
    {
        *column = cursor.positionInBlock();
    }
}

//------------------------------------------------------------
QTextCursor CodeEditor::setCursorPosition(int line, int column, bool applySelection /*= true*/)
{
    QTextCursor cursor = textCursor();
    QTextBlock block = document()->findBlockByNumber(line);
    cursor.setPosition(block.position() + column);
    if (applySelection)
    {
        setTextCursor(cursor);
    }
    return cursor;
}

//------------------------------------------------------------
void CodeEditor::unfoldCursorPosition()
{
    QTextBlock block = textCursor().block();
    // unfold parent fold trigger if the block is collapsed

    Panel::Ptr panel = panels()->get("FoldingPanel");
    if (panel)
    {
        QSharedPointer<FoldingPanel> fp = panel.dynamicCast<FoldingPanel>();
        if (fp)
        {
            if (!block.isVisible())
            {
                block = FoldScope::findParentScope(block);

                while (block.isValid())
                {
                    // qDebug() << block.blockNumber() <<
                    // Utils::TextBlockHelper::isFoldTrigger(block) <<
                    // Utils::TextBlockHelper::isCollapsed(block);
                    if (Utils::TextBlockHelper::isCollapsed(block))
                    {
                        fp->toggleFoldTrigger(block);
                    }

                    block = block.previous();
                    block = FoldScope::findParentScope(block);
                }
            }
        }
    }
}

//------------------------------------------------------------
void CodeEditor::ensureLineVisible(int line)
{
    setCursorPosition(line, 0);
    unfoldCursorPosition();
    ensureCursorVisible();
}

//------------------------------------------------------------
/*
Returns a pointer to the TextBlockUserData, assigned to line 'lineIndex'.
If no userData is currently assigned to this line, a new TextBlockUserData
structure is allocated, attached to the line and returned.

Returns NULL if the line does not exist
*/
TextBlockUserData* CodeEditor::getTextBlockUserData(int lineIndex, bool createIfNotExist /*= true*/)
{
    QTextBlock block = document()->findBlockByNumber(lineIndex);

    if (block.isValid())
    {
        // set docstring dynamic attribute, used by the fold detector.
        TextBlockUserData* userData = dynamic_cast<TextBlockUserData*>(block.userData());
        if (userData == NULL && createIfNotExist)
        {
            userData = new TextBlockUserData(this);
            userData->m_currentLineIdx = lineIndex;
            block.setUserData(userData);
        }
        return userData;
    }
    else
    {
        return NULL;
    }
}

//------------------------------------------------------------
const TextBlockUserData* CodeEditor::getConstTextBlockUserData(int lineIndex) const
{
    QTextBlock block = document()->findBlockByNumber(lineIndex);

    if (block.isValid())
    {
        return dynamic_cast<TextBlockUserData*>(block.userData());
    }
    else
    {
        return NULL;
    }
}

//------------------------------------------------------------
/*
Returns a pointer to the TextBlockUserData, assigned to line 'lineNbr'.
If no userData is currently assigned to this line, a new TextBlockUserData
structure is allocated, attached to the line and returned.

Returns NULL if the line does not exist
*/
TextBlockUserData* CodeEditor::getTextBlockUserData(
    QTextBlock& block, bool createIfNotExist /*= true*/)
{
    if (block.isValid())
    {
        // set docstring dynamic attribute, used by the fold detector.
        TextBlockUserData* userData = dynamic_cast<TextBlockUserData*>(block.userData());
        if (userData == NULL && createIfNotExist)
        {
            userData = new TextBlockUserData(this);
            userData->m_currentLineIdx = block.blockNumber();
            block.setUserData(userData);
        }
        return userData;
    }
    else
    {
        return NULL;
    }
}

//------------------------------------------------------------
/*
Returns true if at least one bookmark is set, else false
*/
bool CodeEditor::bookmarksAvailable() const
{
    foreach (TextBlockUserData* tbud, textBlockUserDataList())
    {
        if (tbud->m_bookmark)
        {
            return true;
        }
    }
    return false;
}

//------------------------------------------------------------
/*
Returns true if at least one bookmark is set, else false
*/
bool CodeEditor::breakpointsAvailable() const
{
    foreach (TextBlockUserData* tbud, textBlockUserDataList())
    {
        if (tbud->m_breakpointType != TextBlockUserData::TypeNoBp)
        {
            return true;
        }
    }
    return false;
}

//------------------------------------------------------------
/*

*/
/*virtual*/ bool CodeEditor::removeTextBlockUserData(TextBlockUserData* userData)
{
    return m_textBlockUserDataList.remove(userData);
}

//------------------------------------------------------------
/*
Returns the line count
*/
int CodeEditor::lineCount() const
{
    return document()->blockCount();
}

//------------------------------------------------------------
/*
Returns the length of line \a line in characters.
*/
int CodeEditor::lineLength(int line) const
{
    if (line < 0 || line >= lineCount())
    {
        return -1;
    }

    QTextBlock block = document()->findBlockByNumber(line);
    if (block.isValid())
    {
        return block.text().length(); // length();
    }

    return -1;
}

//------------------------------------------------------------
/*
Selects entire lines between start and end line numbers.

This functions apply the selection and returns the text cursor that
contains the selection.

Optionally it is possible to prevent the selection from being applied
on the code editor widget by setting ``apply_selection`` to False.

:param start: Start line number (0 based)
:param end: End line number (0 based). Use -1 to select up to the
    end of the document
:param apply_selection: True to apply the selection before returning
    the QTextCursor.
:returns: A QTextCursor that holds the requested selection
*/
QTextCursor CodeEditor::selectLines(
    int start /*= 0*/, int end /*= -1*/, bool applySelection /*= true*/)
{
    if (end == -1)
    {
        end = lineCount() - 1;
    }
    if (start < 0)
    {
        start = 0;
    }

    QTextCursor text_cursor = moveCursorTo(start); // reports goback navigation if necessary

    if (end > start) // Going down
    {
        text_cursor.movePosition(QTextCursor::Down, QTextCursor::KeepAnchor, end - start);
        text_cursor.movePosition(QTextCursor::EndOfLine, QTextCursor::KeepAnchor);
    }
    else if (end < start) // going up
    {
        // don't miss end of line !
        text_cursor.movePosition(QTextCursor::EndOfLine, QTextCursor::MoveAnchor);
        text_cursor.movePosition(QTextCursor::Up, QTextCursor::KeepAnchor, start - end);
        text_cursor.movePosition(QTextCursor::StartOfLine, QTextCursor::KeepAnchor);
    }
    else
    {
        text_cursor.movePosition(QTextCursor::EndOfLine, QTextCursor::KeepAnchor);
    }

    if (applySelection)
    {
        setTextCursor(text_cursor);
    }

    return text_cursor;
}

//------------------------------------------------------------
/*
Selects an entire line.

:param line: Line to select. If -1, the current line will be selected
:param apply_selection: True to apply selection on the text editor
    widget, False to just return the text cursor without setting it
    on the editor.
:return: QTextCursor
*/
QTextCursor CodeEditor::selectWholeLine(int line /*= -1*/, bool applySelection /*= true*/)
{
    if (line == -1)
    {
        line = currentLineNumber();
    }
    return selectLines(line, line, applySelection);
}

//------------------------------------------------------------
/*
Returns the selected lines boundaries (start line, end line)

:return: tuple(int, int)
*/
QPair<int, int> CodeEditor::selectionRange() const
{
    int start = document()->findBlock(textCursor().selectionStart()).blockNumber();
    int end = document()->findBlock(textCursor().selectionEnd()).blockNumber();
    QTextCursor text_cursor = textCursor();
    text_cursor.setPosition(textCursor().selectionEnd());
    if ((text_cursor.columnNumber() == 0) && (start != end))
    {
        end -= 1;
    }
    return QPair<int, int>(start, end);
}

//------------------------------------------------------------
/*
If there is a selection, *lineFrom is set to the line number in which the selection
begins and *lineTo is set to the line number in which the selection ends.
(They could be the same.) *indexFrom is set to the index at which the selection
begins within *lineFrom, and *indexTo is set to the index at which the selection ends within
*lineTo. If there is no selection, *lineFrom, *indexFrom, *lineTo and *indexTo are all set to
-1.
*/
void CodeEditor::getSelection(int* lineFrom, int* indexFrom, int* lineTo, int* indexTo)
{
    const QTextCursor& cursor = textCursor();
    if (!cursor.hasSelection())
    {
        if (lineFrom)
            *lineFrom = -1;
        if (lineTo)
            *lineTo = -1;
        if (indexFrom)
            *indexFrom = -1;
        if (indexTo)
            *indexTo = -1;
    }
    else
    {
        int start = cursor.selectionStart();
        int end = cursor.selectionEnd();
        QTextBlock block = document()->findBlock(start);

        if (lineFrom)
            *lineFrom = block.blockNumber();
        if (indexFrom)
            *indexFrom = start - block.position();

        block = document()->findBlock(end);
        if (lineTo)
            *lineTo = block.blockNumber();
        if (indexTo)
            *indexTo = end - block.position();
    }
}

//------------------------------------------------------------
void CodeEditor::setSelection(int lineFrom, int indexFrom, int lineTo, int indexTo)
{
    QTextCursor cursor = textCursor();
    int currentLine = cursor.blockNumber();

    QTextBlock firstBlock = document()->findBlockByNumber(lineFrom);
    QTextBlock lastBlock = document()->findBlockByNumber(lineTo);

    if (firstBlock.isValid() && lastBlock.isValid())
    {
        cursor.setPosition(firstBlock.position() + indexFrom, QTextCursor::MoveAnchor);
        cursor.setPosition(lastBlock.position() + indexTo, QTextCursor::KeepAnchor);
        setTextCursor(cursor);

        if ((firstBlock.blockNumber() - currentLine >= m_minLineJumpsForGoBackNavigationReport) ||
            (currentLine - lastBlock.blockNumber() >= m_minLineJumpsForGoBackNavigationReport))
        {
            reportGoBackNavigationCursorMovement(CursorPosition(cursor), "setSelection");
        }
    }
}

//------------------------------------------------------------
bool CodeEditor::hasSelectedText() const
{
    const QTextCursor& cursor = textCursor();
    return cursor.hasSelection();
}

//------------------------------------------------------------
/*
Computes line position on Y-Axis (at the center of the line) from line
number.

:param line_number: The line number for which we want to know the
                    position in pixels.
:return: The center position of the line.
*/
int CodeEditor::linePosFromNumber(int lineNumber) const
{
    QTextBlock block = document()->findBlockByNumber(lineNumber);
    if (block.isValid())
    {
        return (int)(blockBoundingGeometry(block).translated(contentOffset()).top());
    }

    if (lineNumber <= 0)
    {
        return 0;
    }
    else
    {
        return (int)(blockBoundingGeometry(block.previous()).translated(contentOffset()).bottom());
    }
}

//------------------------------------------------------------
QTextCursor CodeEditor::moveCursorTo(int line) const
{
    QTextCursor cursor = textCursor();
    int currentLine = cursor.blockNumber();
    QTextBlock block = document()->findBlockByNumber(line);
    cursor.setPosition(block.position());

    return cursor;
}

//------------------------------------------------------------
/*
Marks the whole document as dirty to force a full refresh. **SLOW**
*/
void CodeEditor::markWholeDocDirty()
{
    QTextCursor text_cursor = textCursor();
    text_cursor.select(QTextCursor::Document);
    document()->markContentsDirty(text_cursor.selectionStart(), text_cursor.selectionEnd());
}

//------------------------------------------------------------
/*
Calls ``rehighlight`` on the installed syntax highlighter mode.
*/
void CodeEditor::rehighlight()
{
    if (syntaxHighlighter())
    {
        syntaxHighlighter()->rehighlight();
    }
}

//------------------------------------------------------------
/*
Calls ``rehighlightBlock`` on the installed syntax highlighter mode.
*/
void CodeEditor::rehighlightBlock(int lineFromIdx, int lineToIdx /*=-1*/)
{
    if (syntaxHighlighter())
    {
        QTextBlock begin = document()->findBlockByNumber(lineFromIdx);
        QTextBlock end =
            lineToIdx == -1 ? begin : document()->findBlockByNumber(qMax(lineFromIdx, lineToIdx));
        end = end.next();

        while (begin != end)
        {
            syntaxHighlighter()->rehighlightBlock(begin);
            begin = begin.next();
        }
    }
}

//------------------------------------------------------------
/*
Show a tool tip at the specified position

:param pos: QPoint Tooltip position
:param tooltip: Tooltip text

:param _sender_deco: TextDecoration which is the sender of the show
    tooltip request. (for internal use only).
*/
void CodeEditor::showTooltipDelayJobRunner(QList<QVariant> args)
{
    QPoint pos = args[0].toPoint();
    QString tooltip = args[1].toString();
    TextDecoration::Ptr senderDeco = args[2].value<TextDecoration::Ptr>();

    if (senderDeco && !this->decorations()->contains(senderDeco))
    {
        return;
    }

    QToolTip::showText(pos, tooltip.left(1024));
}

//------------------------------------------------------------
void CodeEditor::setUndoAvailable(bool available)
{
    m_undoAvailable = available;

    emit updateActions();
}

//------------------------------------------------------------
void CodeEditor::setRedoAvailable(bool available)
{
    m_redoAvailable = available;

    emit updateActions();
}

//------------------------------------------------------------
void CodeEditor::contextMenuEvent(QContextMenuEvent* e)
{
    if (m_showCtxMenu)
    {
        e->accept();
        int line = -1;
        int index = -1;
        lineIndexFromPosition(e->pos(), &line, &index);

        int selLineFrom, selLineTo;
        int selIndexFrom, selIndexTo;
        getSelection(&selLineFrom, &selIndexFrom, &selLineTo, &selIndexTo);

        if (selLineFrom ==
            -1) // nothing selected yet -> set the cursor to the current mouse position
        {
            setCursorPosition(line, index);
        }
        else if (
            line < selLineFrom || (line == selLineFrom && index < selIndexFrom) ||
            line > selLineTo ||
            (line == selLineTo &&
             index > selIndexTo)) // the right mouse click happened out of the current
                                  // selection, move the cursor to the clicked position
        {
            setCursorPosition(line, index);
        }

        contextMenuAboutToShow(line);
        m_pContextMenu->exec(e->globalPos());
    }
}

//------------------------------------------------------------
void CodeEditor::contextMenuAboutToShow(int contextMenuLine)
{
}


} // end namespace ito
