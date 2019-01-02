#include "codeEditor.h"

#include <qapplication.h>
#include <qevent.h>
#include <qtooltip.h>
#include <qtextdocument.h>
#include <qdebug.h>
#include <qpainter.h>

#include "managers/panelsManager.h"
#include "managers/textDecorationsManager.h"
#include "managers/modesManager.h"
#include "delayJobRunner.h"

CodeEditor::CodeEditor(QWidget *parent /*= NULL*/, bool createDefaultActions /*= true*/)
    : QPlainTextEdit(parent),
    m_showCtxMenu(true),
    m_defaultFontSize(10),
    m_useSpacesInsteadOfTabs(true),
    m_showWhitespaces(false),
    m_tabLength(4),
    m_zoomLevel(0),
    m_fontSize(10),
    m_fontFamily("Arial"),
    m_selectLineOnCopyEmpty(true),
    m_wordSeparators("~!@#$%^&*()+{}|:\"'<>?,./;[]\\\n\t=- "),
    m_dirty(false),
    m_cleaning(false),
    m_pPanels(NULL),
    m_pDecorations(NULL),
    m_pModes(NULL),
    m_saveOnFocusOut(false),
    m_lastMousePos(QPoint(0,0)),
    m_prevTooltipBlockNbr(-1),
    m_pTooltipsRunner(NULL),
    m_edgeLineShow(false),
    m_edgeLineColumn(79),
    m_edgeLineColor(Qt::darkGray)
{
    installEventFilter(this);
    connect(document(), SIGNAL(modificationChanged(bool)), this, SLOT(emitDirtyChanged(bool)));

    // connect slots
    connect(this, SIGNAL(textChanged()), this, SLOT(onTextChanged()));
    connect(this, SIGNAL(blockCountChanged()), this, SLOT(update()));
    connect(this, SIGNAL(cursorPositionChanged()), this, SLOT(update()));
    connect(this, SIGNAL(selectionChanged()), this, SLOT(update()));

    setMouseTracking(true);
    setCenterOnScroll(true);
    setLineWrapMode(QPlainTextEdit::NoWrap);
    setCursorWidth(2);

    m_pPanels = new PanelsManager(this);
    m_pDecorations = new TextDecorationsManager(this);
    m_pModes = new ModesManager(this);

    m_pTooltipsRunner = new DelayJobRunner<CodeEditor, void(CodeEditor::*)(QList<QVariant>)>(700);

    initStyle();
}

//-----------------------------------------------------------
CodeEditor::~CodeEditor()
{
    delete m_pPanels;
    m_pPanels = NULL;

    delete m_pDecorations;
    m_pDecorations = NULL;

    delete m_pModes;
    m_pModes = NULL;

    m_pTooltipsRunner->deleteLater();
    m_pTooltipsRunner = NULL;
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
}

//-----------------------------------------------------------
/*
The editor background color (QColor)
*/
QColor CodeEditor::background() const
{
    return m_background;
}

void CodeEditor::setBackground(const QColor &value)
{
    m_background = value;
    resetStylesheet();
}

//-----------------------------------------------------------
/*
The editor foreground color (QColor)
*/
QColor CodeEditor::foreground() const
{
    return m_foreground;
}

void CodeEditor::setForeground(const QColor &value)
{
    m_foreground = value;
    resetStylesheet();
}

//-----------------------------------------------------------
/*
The editor selection's foreground color.
*/
QColor CodeEditor::selectionForeground() const
{
    return m_selForeground;
}

void CodeEditor::setSelectionForeground(const QColor &value)
{
    m_selForeground = value;
    resetStylesheet();
}

//-----------------------------------------------------------
/*
The editor selection's background color.
*/
QColor CodeEditor::selectionBackground() const
{
    return m_selBackground;
}

void CodeEditor::setSelectionBackground(const QColor &value)
{
    m_selBackground = value;
    resetStylesheet();
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
bool CodeEditor::edgeLineVisible() const
{
    return m_edgeLineShow;
}

void CodeEditor::setEdgeLineVisible(bool value)
{
    if (m_edgeLineShow != value)
    {
        m_edgeLineShow = value;
        update();
    }
}

//-----------------------------------------------------------
int CodeEditor::edgeLineColumn() const
{
    return m_edgeLineColumn;
}

void CodeEditor::setEdgeLineColumn(int column)
{
    if (m_edgeLineColumn != column)
    {
        m_edgeLineColumn = column;
        update();
    }
}

//-----------------------------------------------------
QColor CodeEditor::edgeLineColor() const
{
    return m_edgeLineColor;
}

void CodeEditor::setEdgeLineColor(const QColor &color)
{
    if (m_edgeLineColor != color)
    {
        m_edgeLineColor = color;
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
    m_fontFamily = value;
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
    m_tabLength = value;
}

//-----------------------------------------------------------
QColor CodeEditor::whitespacesForeground() const
{
    return m_whitespacesForeground;
}

void CodeEditor::setWhitespacesForeground(const QColor &value)
{
    m_whitespacesForeground = value;
}

//-----------------------------------------------------------
/*
Automatically saves editor content on focus out.
Default is False.
*/
bool CodeEditor::saveOnFocusOut() const
{
    return m_saveOnFocusOut;
}

void CodeEditor::setSaveOnFocusOut(bool value)
{
    m_saveOnFocusOut = value;
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
void CodeEditor::onTextChanged()
{
    // Adjust dirty flag depending on editor's content
    if (!m_cleaning)
    {
        int line, column;
        cursorPosition(line, column);
        m_modifiedLines << line;
    }
}

//-----------------------------------------------------------
void CodeEditor::cursorPosition(int &line, int &column) const
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
void CodeEditor::setMouseCursor(const QCursor &cursor)
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
    setTabStopWidth(m_tabLength * fontMetrics().width(" "));
    setWhitespacesFlags(m_showWhitespaces);
}

//-----------------------------------------------------------
void CodeEditor::initStyle()
{
    //Inits style options
    m_background = QColor("white");
    m_foreground = QColor("black");
    m_whitespacesForeground = QColor("light gray");
    QApplication *app = qobject_cast<QApplication*>(QApplication::instance());
    m_selBackground = app->palette().highlight().color();
    m_selForeground = app->palette().highlightedText().color();
    m_fontSize = 10;
    setFontName("");
}

//-----------------------------------------------------------
void CodeEditor::initActions(bool createStandardActions)
{
    //todo
}

//-----------------------------------------------------------
void CodeEditor::setWhitespacesFlags(bool show)
{
    //Sets show white spaces flag
    QTextDocument *doc = document();
    QTextOption options = doc->defaultTextOption();
    if (show)
    {
        options.setFlags(options.flags() |
                            QTextOption::ShowTabsAndSpaces);
    }
    else
    {
        options.setFlags(
            options.flags() & ~QTextOption::ShowTabsAndSpaces);
    }
    doc->setDefaultTextOption(options);
}

//-----------------------------------------------------------
void CodeEditor::setViewportMargins(int left, int top, int right, int bottom)
{
    QPlainTextEdit::setViewportMargins(left, top, right, bottom);
}

//-----------------------------------------------------------
void CodeEditor::resizeEvent(QResizeEvent *e)
{
    /*
    Overrides resize event to resize the editor's panels.
    :param e: resize event
    */
    QPlainTextEdit::resizeEvent(e);
    m_pPanels->resize();
}

//-----------------------------------------------------------
void CodeEditor::closeEvent(QCloseEvent *e)
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
void CodeEditor::paintEvent(QPaintEvent *e)
{
    updateVisibleBlocks(); //_update_visible_blocks
    QPlainTextEdit::paintEvent(e);

    if (m_edgeLineShow)
    {
        QPainter painter(viewport());
        QColor color(m_edgeLineColor);
        color.setAlphaF(.5);
        painter.setPen(color);

        int x = fontMetrics().width(QString(m_edgeLineColumn, '9'));
        painter.drawLine(x, 0, x, size().height());
    }

    emit painted(e);
}

//-----------------------------------------------------------
void CodeEditor::keyPressEvent(QKeyEvent *e)
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
    if (!e->isAccepted())
    {
        if (e->key() == Qt::Key_Tab && e->modifiers() == \
            Qt::NoModifier)
        {
            indent();
            e->accept();
        }
        else if (e->key() == Qt::Key_Backtab && \
            e->modifiers() == Qt::NoModifier)
        {
            unindent();
            e->accept();
        }
        else if (e->key() == Qt::Key_Backtab && \
            e->modifiers() == Qt::ShiftModifier)
        {
            unindent();
            e->accept();
        }
        else if (e->key() == Qt::Key_Home && \
                (int(e->modifiers()) & Qt::ControlModifier) == 0)
        {
            doHomeKey(e, int(e->modifiers()) & Qt::ShiftModifier);
        }
        if (!e->isAccepted())
        {
            e->setAccepted(initial_state);
            QPlainTextEdit::keyPressEvent(e);
        }
    }
    bool new_state = e->isAccepted();
    e->setAccepted(state);
    emit postKeyPressed(e);
    e->setAccepted(new_state);
}

//-----------------------------------------------------------
void CodeEditor::keyReleaseEvent(QKeyEvent *e)
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
void CodeEditor::mouseDoubleClickEvent(QMouseEvent *e)
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
void CodeEditor::focusInEvent(QFocusEvent *e)
{   
    emit focusedIn(e);
    QPlainTextEdit::focusInEvent(e);
}

//-----------------------------------------------------------
/* Saves content if save_on_focus_out is True.
*/
void CodeEditor::focusOutEvent(QFocusEvent *e)
{
    if (m_saveOnFocusOut && m_dirty ) //todo:  && this->file.path)
    {
        //TODO
        //file.save();
    }
    QPlainTextEdit::focusOutEvent(e);
}

//-----------------------------------------------------------
/*
    Overrides mousePressEvent to emits mousePressed signal

    :param event: QMouseEvent
*/
void CodeEditor::mousePressEvent(QMouseEvent *e)
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
void CodeEditor::callWheelEvent(QWheelEvent *e)
{
    wheelEvent(e);
}

//-----------------------------------------------------------
void CodeEditor::wheelEvent(QWheelEvent *e)
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
    
    QTextCursor cursor  = this->cursorForPosition(e->pos());
    m_lastMousePos = e->pos();
    bool blockFound = false;

    TextDecorationsManager::const_iterator it = m_pDecorations->constBegin();
    TextDecoration::Ptr itPtr;

    while (it != m_pDecorations->constEnd())
    {
        itPtr = *it;
        if (itPtr->containsCursor(cursor) && (itPtr->tooltip() != ""))
        {
            if (m_prevTooltipBlockNbr != cursor.blockNumber() || \
                    !QToolTip::isVisible())
            {
                QPoint position = e->pos();
                //add left margin
                position.setX(position.x() + m_pPanels->marginSize(Panel::Left));
                //add top margin
                position.setY(position.y() + m_pPanels->marginSize(Panel::Top));

                QList<QVariant> args;
                args << mapToGlobal(position);
                args << itPtr->tooltip().left(1024);
                args << QVariant::fromValue(*it);
                DELAY_JOB_RUNNER(m_pTooltipsRunner, CodeEditor, void(CodeEditor::*)(QList<QVariant>))->requestJob( \
                    this, &CodeEditor::showTooltipDelayJobRunner, args);
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
        m_pTooltipsRunner->cancelRequests();
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
void CodeEditor::showTooltip(const QPoint &pos, const QString &tooltip, const TextDecoration::Ptr &senderDeco)
{
    if (!m_pDecorations->contains(senderDeco))
    {
        return;
    }

    QToolTip::showText(pos, tooltip.left(1024), this);
}

void CodeEditor::showTooltip(const QPoint &pos, const QString &tooltip)
{
    QToolTip::showText(pos, tooltip.left(1024), this);
}

//-----------------------------------------------------------
void CodeEditor::showEvent(QShowEvent *e)
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
void CodeEditor::doHomeKey(QEvent *event /*= NULL*/, bool select /* = false*/)
{
    //get nb char to first significative char
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
    if (event)
    {
        event->accept();
    }
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
    int top = int(blockBoundingGeometry(block).translated(
        contentOffset()).top());
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
int CodeEditor::lineIndent(const QTextBlock *lineNbr) const
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
/*virtual*/ bool CodeEditor::eventFilter(QObject *obj, QEvent *e)
{
    if ((obj == this) && (e->type() == QEvent::KeyPress))
    {
        QKeyEvent *ke = dynamic_cast<QKeyEvent*>(e);
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
Cuts the selected text or the whole line if no text was selected.
*/
void CodeEditor::cut()
{
    QTextCursor tc = textCursor();
    tc.beginEditBlock();
    bool no_selection = false;
    if (currentLineText() != "")
    {
        tc.deleteChar();
    }
    else
    {
        if (!textCursor().hasSelection())
        {
            no_selection = true;
            selectWholeLine();
        }
        QPlainTextEdit::cut();
        if (no_selection)
        {
            tc.deleteChar();
        }
    }
    tc.endEditBlock();
    setTextCursor(tc);
}

//-------------------------------------------------------------
/*
Copy the selected text to the clipboard. If no text was selected, the
entire line is copied (this feature can be turned off by
setting :attr:`select_line_on_copy_empty` to False.
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

:param line_nbr: The line number of the text to get
:return: Entire line's text
:rtype: str
*/
QString CodeEditor::lineText(int lineNbr) const
{
    const QTextBlock &block = document()->findBlockByNumber(lineNbr);
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
Returns the text of the current line.

:return: Text of the current line
*/
QString CodeEditor::currentLineText() const
{
    return lineText(currentLineNumber());
}

//------------------------------------------------------------
/*
Extends setPlainText to force the user to setup an encoding and amime type.

Emits the new_text_set signal.

:param txt: The new text to set.
:param mime_type: Associated mimetype. Setting the mime will update the
                    pygments lexer.
:param encoding: text encoding
*/
void CodeEditor::setPlainText(const QString &text, const QString &mimeType /*= ""*/, const QString &encoding /*= ""*/)
{
    //TODO: mimeType, encoding
    m_modifiedLines.clear();

    QPlainTextEdit::setPlainText(text);
    emit newTextSet();
    emit redoAvailable(false);
    emit undoAvailable(false);
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
                color: %1; \
            }").arg(m_background.name(), m_foreground.name()));
        }
        else
        {
#if WIN32
            setStyleSheet(QString("QPlainTextEdit \
            { \
                background-color: %1; \
                color: %1; \
            }").arg(m_background.name(), m_foreground.name()));
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
        p.setColor(QPalette::Highlight,
                    m_selBackground);
        p.setColor(QPalette::HighlightedText,
                    m_selForeground);
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
bool CodeEditor::isCommentOrString(const QTextCursor &cursor, const QList<ColorScheme::Keys> &formats /*= QList<ColorScheme::Keys>()*/)
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
bool CodeEditor::isCommentOrString(const QTextBlock &block, const QList<ColorScheme::Keys> &formats /*= QList<ColorScheme::Keys>()*/)
{
    QList<ColorScheme::Keys> formats_ = formats;
    if (formats_.size() == 0)
    {
        formats_ << ColorScheme::KeyComment << ColorScheme::KeyString << ColorScheme::KeyDocstring;
    }

    QTextLayout *layout = NULL;

    int pos = block.text().size() - 1;
    layout = block.layout();
    bool is_user_obj;

    if (layout)
    {
        QList<QTextLayout::FormatRange> additional_formats = layout->additionalFormats();
        SyntaxHighlighterBase *sh = syntaxHighlighter();
        if (sh)
        {
            const ColorScheme &ref_formats = sh->colorScheme();
            foreach (const QTextLayout::FormatRange &r, additional_formats)
            {
                if ((r.start <= pos) && (pos < (r.start + r.length)))
                {
                    foreach (int fmtType, formats_)
                    {
                        is_user_obj = (r.format.objectType() == r.format.UserObject);
                        if ((ref_formats[fmtType] == r.format) && is_user_obj)
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
QTextCursor CodeEditor::wordUnderCursor(bool selectWholeWord)
{
    QTextCursor text_cursor = textCursor();
    int endPos, startPos;
    endPos = startPos = text_cursor.position();
    QString selectedText;
    QChar firstChar;
    //select char by char until we are at the original cursor position.
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
                break;  //start boundary found
            }
        }
        startPos = text_cursor.position();
        text_cursor.setPosition(startPos);
    }

    if (selectWholeWord)
    {
        //select the resot of the word
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
                    break;  //end boundary found
                }
                endPos = text_cursor.position();
                text_cursor.setPosition(endPos);
            }
        }
    }
    //now that we habe the boundaries, we can select the text
    text_cursor.setPosition(startPos);
    text_cursor.setPosition(endPos, QTextCursor::KeepAnchor);
    return text_cursor;
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
    foreach(const VisibleBlock &vb, visibleBlocks())
    {
        if ((vb.topPosition <= yPos) && \
            (yPos <= (vb.topPosition + height)))
        {
            return vb.lineNumber;
        }
    }
    return -1;
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
QTextCursor CodeEditor::selectLines(int start /*= 0*/, int end /*= -1*/, bool applySelection /*= true*/)
{
    if (end == -1)
    {
        end = lineCount() - 1;
    }
    if (start < 0)
    {
        start = 0;
    }
    QTextCursor text_cursor = moveCursorTo(start);
    if (end > start)  //Going down
    {
        text_cursor.movePosition(QTextCursor::Down,
                                    QTextCursor::KeepAnchor, end - start);
        text_cursor.movePosition(QTextCursor::EndOfLine,
                                    QTextCursor::KeepAnchor);
    }
    else if (end < start)  //going up
    {
        // don't miss end of line !
        text_cursor.movePosition(QTextCursor::EndOfLine,
                                    QTextCursor::MoveAnchor);
        text_cursor.movePosition(QTextCursor::Up,
                                    QTextCursor::KeepAnchor, start - end);
        text_cursor.movePosition(QTextCursor::StartOfLine,
                                    QTextCursor::KeepAnchor);
    }
    else
    {
        text_cursor.movePosition(QTextCursor::EndOfLine,
                                    QTextCursor::KeepAnchor);
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
QPair<int,int> CodeEditor::selectionRange() const
{        
    int start = document()->findBlock(
        textCursor().selectionStart()).blockNumber();
    int end = document()->findBlock(
        textCursor().selectionEnd()).blockNumber();
    QTextCursor text_cursor = textCursor();
    text_cursor.setPosition(textCursor().selectionEnd());
    if ((text_cursor.columnNumber() == 0) && (start != end))
    {
        end -= 1;
    }
    return QPair<int,int>(start, end);
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
QTextCursor CodeEditor::moveCursorTo(int line)
{
    QTextCursor cursor = textCursor();
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
    document()->markContentsDirty(text_cursor.selectionStart(),
                                                text_cursor.selectionEnd());
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