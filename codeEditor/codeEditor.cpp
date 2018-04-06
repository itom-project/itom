#include "codeEditor.h"

#include "qapplication.h"
#include "qevent.h"

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
    m_cleaning(false)
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
}

//-----------------------------------------------------------
CodeEditor::~CodeEditor()
{
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
bool CodeEditor::selectLineOnCopyEmpty() const
{
    return m_selectLineOnCopyEmpty;
}

void CodeEditor::setSelectLineOnCopyEmpty(bool value)
{
    m_selectLineOnCopyEmpty = value;
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
    m_showWhitespaces = value;
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
    panels.resize()
}

//-----------------------------------------------------------
void CodeEditor::closeEvent(QCloseEvent *e)
{
    close();
    QPlainTextEdit::closeEvent(e);
}

//-----------------------------------------------------------
void CodeEditor::paintEvent(QPaintEvent *e)
{
    /*
    Overrides paint event to update the list of visible blocks and emit
    the painted e->
    :param e: paint event
    */
    updateVisibleBlocks(e); //_update_visible_blocks
    QPlainTextEdit::paintEvent(e);
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
            this->indent();
            e->accept();
        }
        else if (e->key() == Qt::Key_Backtab && \
                e->modifiers() == Qt::NoModifier)
        {
            this->unindent();
            e->accept();
        }
        else if (e->key() == Qt::Key_Home && \
                int(e->modifiers()) & Qt::ControlModifier == 0)
        {
            this->_do_home_key(
                event, int(e->modifiers()) & Qt::ShiftModifier)
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
void CodeEditor::focusInEvent(self, event):
    """
    Overrides focusInEvent to emits the focused_in signal
    :param event: QFocusEvent
    """
    this->focused_in.emit(event)
    QPlainTextEdit::focusInEvent(event)

//-----------------------------------------------------------
void CodeEditor::focusOutEvent(self, event):
    # Saves content if save_on_focus_out is True.
    if this->_save_on_focus_out and this->dirty and this->file.path:
        this->file.save()
    QPlainTextEdit::focusOutEvent(event)

//-----------------------------------------------------------
void CodeEditor::mousePressEvent(self, event):
    """
    Overrides mousePressEvent to emits mouse_pressed signal
    :param event: QMouseEvent
    """
    initial_state = e->isAccepted()
    e->ignore()
    this->mouse_pressed.emit(event)
    if e->button() == Qt::LeftButton:
        cursor = this->cursorForPosition(e->pos())
        for sel in this->decorations:
            if sel.cursor.blockNumber() == cursor.blockNumber():
                if sel.contains_cursor(cursor):
                    sel.signals.clicked.emit(sel)
    if not e->isAccepted():
        e->setAccepted(initial_state)
        QPlainTextEdit::mousePressEvent(event)

//-----------------------------------------------------------
void CodeEditor::mouseReleaseEvent(self, event):
    """
    Emits mouse_released signal.
    :param event: QMouseEvent
    """
    initial_state = e->isAccepted()
    e->ignore()
    this->mouse_released.emit(event)
    if not e->isAccepted():
        e->setAccepted(initial_state)
        QPlainTextEdit::mouseReleaseEvent(event)

//-----------------------------------------------------------
void CodeEditor::wheelEvent(QWheelEvent *e)
{
    /*
    Emits the mouse_wheel_activated signal.
    :param event: QMouseEvent
    */
    bool initial_state = e->isAccepted();
    e->ignore();
    this->mouse_wheel_activated.emit(event)
    if not e->isAccepted():
        e->setAccepted(initial_state)
        QPlainTextEdit::wheelEvent(event)
}

//-----------------------------------------------------------
void CodeEditor::mouseMoveEvent(self, event)
{
    """
    Overrides mouseMovedEvent to display any decoration tooltip and emits
    the mouse_moved e->
    :param event: QMouseEvent
    """
    cursor = this->cursorForPosition(e->pos())
    this->_last_mouse_pos = e->pos()
    block_found = False
    for sel in this->decorations:
        if sel.contains_cursor(cursor) and sel.tooltip:
            if (this->_prev_tooltip_block_nbr != cursor.blockNumber() or
                    not QtWidgets.QToolTip.isVisible()):
                pos = e->pos()
                # add left margin
                pos.setX(pos.x() + this->panels.margin_size())
                # add top margin
                pos.setY(pos.y() + this->panels.margin_size(0))
                this->_tooltips_runner.request_job(
                    this->show_tooltip,
                    this->mapToGlobal(pos), sel.tooltip[0: 1024], sel)
                this->_prev_tooltip_block_nbr = cursor.blockNumber()
            block_found = True
            break
    if not block_found and this->_prev_tooltip_block_nbr != -1:
        QtWidgets.QToolTip.hideText()
        this->_prev_tooltip_block_nbr = -1
        this->_tooltips_runner.cancel_requests()
    this->mouse_moved.emit(event)
    QPlainTextEdit::mouseMoveEvent(event)
}

//-----------------------------------------------------------
void CodeEditor::showEvent(QShowEvent *e)
{
    /* Overrides showEvent to update the viewport margins */
    QPlainTextEdit::showEvent(e);
    this->panels.refresh();
}