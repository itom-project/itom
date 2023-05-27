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

#include "foldingPanel.h"

#include "../codeEditor.h"
#include "../foldDetector/foldDetector.h"
#include <qpainter.h>
#include "../delayJobRunner.h"
#include <qfontmetrics.h>
#include <qtextdocument.h>
#include <qapplication.h>
#include <QStyleOptionViewItem>
#include <qdebug.h>

#include "../managers/textDecorationsManager.h"

namespace ito {

//----------------------------------------------------------
/*
*/
FoldingPanel::FoldingPanel(bool highlightCaretScope /*= false*/, const QString &description /*= ""*/, QWidget *parent /*= NULL*/) :
    Panel("FoldingPanel", false, description, parent),
    m_highlightCaretScope(highlightCaretScope),
    m_highlightCaret(false),
    m_native(true),
    m_customColor(QColor("green")),
    m_blockNbr(-1),
    m_indicSize(16),
    m_pHighlightRunner(NULL),
    m_mouseOverLine(-1)
{
    m_customIndicators << ":/pyqode-icons/rc/arrow_right_off.png" << \
            ":/pyqode-icons/rc/arrow_right_on.png" << \
            ":/pyqode-icons/rc/arrow_down_off.png" << \
            ":/pyqode-icons/rc/arrow_down_on.png";

    //: the list of deco used to highlight the current fold region (
    //: surrounding regions are darker)
    m_scopeDecos = QList<TextDecoration::Ptr>();
    //: the list of folded blocs decorations
    m_blockDecos = QList<TextDecoration::Ptr>();
    setMouseTracking(true);
    setScrollable(true);
        /*
        self._current_scope = None
        self._prev_cursor = None
        self.context_menu = None
        self.action_collapse = None
        self.action_expand = None
        self.action_collapse_all = None
        self.action_expand_all = None
        self._original_background = None*/


    m_pHighlightRunner = new DelayJobRunnerArgTextBlock<FoldingPanel, void(FoldingPanel::*)(QTextBlock)>(250);
}

//----------------------------------------------------------
/*
*/
FoldingPanel::~FoldingPanel()
{
    delete m_pHighlightRunner;
    m_pHighlightRunner = NULL;
}

//--------------------------------------------------------------------
/*
Defines whether the panel will use native indicator icons and color or
use custom one.

If you want to use custom indicator icons and color, you must first
set this flag to False.
*/
bool FoldingPanel::nativeLook() const
{
    return m_native;
}

void FoldingPanel::setNativeLook(bool native)
{
    m_native = native;
}

//--------------------------------------------------------------------
/*
Gets/sets the custom icon for the fold indicators.

The list of indicators is interpreted as follow::

    (COLLAPSED_OFF, COLLAPSED_ON, EXPANDED_OFF, EXPANDED_ON)

To use this property you must first set `native_look` to False.

:returns: tuple(str, str, str, str)
*/
QStringList FoldingPanel::customIndicatorIcons() const
{
    return m_customIndicators;
}

void FoldingPanel::setCustomIndicatorIcons(const QStringList &icons)
{
    m_customIndicators = icons;
}


//--------------------------------------------------------------------
/*
Custom base color for the fold region background

:return: QColor
*/
QColor FoldingPanel::customFoldRegionBackground() const
{
    return m_customColor;
}

void FoldingPanel::setCustomFoldRegionBackground(const QColor &color)
{
    m_customColor = color;
}


//--------------------------------------------------------------------
/*
*/
bool FoldingPanel::highlightCaretScope() const
{
    return m_highlightCaretScope;
}

void FoldingPanel::setHighlightCaretScope(bool value)
{
    if (value != m_highlightCaret)
    {
        m_highlightCaret = value;
        if (editor())
        {
            if (value)
            {
                m_blockNbr = -1;
                connect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(highlightCaretScopeSlot()));
            }
            else
            {
                m_blockNbr = -1;
                connect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(highlightCaretScopeSlot()));

            }
        }
    }
}

//-----------------------------------------------------------------------
/*
Highlight the scope surrounding the current caret position.

This get called only if :attr:`
pyqode.core.panels.FoldingPanel.highlight_care_scope` is True.
*/
void FoldingPanel::highlightCaretScopeSlot()
{
    QTextCursor cursor = editor()->textCursor();
    int block_nbr = cursor.blockNumber();
    bool valid;

    if (m_blockNbr != block_nbr)
    {
        QTextBlock block = FoldScope::findParentScope( \
            editor()->textCursor().block());

        FoldScope s(block, valid);

        if (valid)
        {
            m_mouseOverLine = block.blockNumber();
            if (Utils::TextBlockHelper::isFoldTrigger(block))
            {
                highlightSurroundingScopes(block);
            }
        }
        else
        {
            clearScopeDecos();
        }
    }
    m_blockNbr = block_nbr;
}


//------------------------------------------------------------
/*
Returns the widget size hint (based on the editor font size)
*/
QSize FoldingPanel::sizeHint() const
{
    QFontMetricsF fm(editor()->font());
    QSize size_hint(fm.height(), fm.height());
    if (size_hint.width() > 16)
    {
        size_hint.setWidth(16);
    }
    return size_hint;
}

//------------------------------------------------------------
/*
Add the folding menu to the editor, on install.

:param editor: editor instance on which the mode has been installed to.
*/
void FoldingPanel::onInstall(CodeEditor *editor)
{
    Panel::onInstall(editor);
    //TODO
    /*self.context_menu = QtWidgets.QMenu(_('Folding'), self.editor)
    action = self.action_collapse = QtWidgets.QAction(
        _('Collapse'), self.context_menu)
    action.setShortcut('Shift+-')
    action.triggered.connect(self._on_action_toggle)
    self.context_menu.addAction(action)
    action = self.action_expand = QtWidgets.QAction(_('Expand'),
                                                    self.context_menu)
    action.setShortcut('Shift++')
    action.triggered.connect(self._on_action_toggle)
    self.context_menu.addAction(action)
    self.context_menu.addSeparator()
    action = self.action_collapse_all = QtWidgets.QAction(
        _('Collapse all'), self.context_menu)
    action.setShortcut('Ctrl+Shift+-')
    action.triggered.connect(self._on_action_collapse_all_triggered)
    self.context_menu.addAction(action)
    action = self.action_expand_all = QtWidgets.QAction(
        _('Expand all'), self.context_menu)
    action.setShortcut('Ctrl+Shift++')
    action.triggered.connect(self._on_action_expand_all_triggered)
    self.context_menu.addAction(action)
    editor()->add_menu(self.context_menu)*/
}

//----------------------------------------------------------
/*
On state changed we (dis)connect to the cursorPositionChanged signal
*/
void FoldingPanel::onStateChanged(bool state)
{
    if (state)
    {
        connect(editor(), SIGNAL(keyPressed(QKeyEvent*)), this, SLOT(onKeyPressed(QKeyEvent*)));
        if (m_highlightCaret)
        {
            connect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(highlightCaretScopeSlot()));
            m_blockNbr = -1;
        }
        connect(editor(), SIGNAL(newTextSet()), this, SLOT(clearBlockDeco()));
    }
    else
    {
        disconnect(editor(), SIGNAL(keyPressed(QKeyEvent*)), this, SLOT(onKeyPressed(QKeyEvent*)));
        if (m_highlightCaret)
        {
            disconnect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(highlightCaretScopeSlot()));
            m_blockNbr = -1;
        }
        disconnect(editor(), SIGNAL(newTextSet()), this, SLOT(clearBlockDeco()));
    }
}


//----------------------------------------------------------
/*
Starts selecting
*/
void FoldingPanel::paintEvent(QPaintEvent *e)
{
    // Paints the fold indicators and the possible fold region background
    // on the folding panel.
    Panel::paintEvent(e);
    QPainter painter(this);

    QTextBlock block;

    // Draw background over the selected non collapsed fold region
    if (m_mouseOverLine != -1)
    {
        block = editor()->document()->findBlockByNumber(m_mouseOverLine);
        drawFoldRegionBackground(block, painter);
    }

    int top_position, line_number;
    bool collapsed;
    bool mouseOver;

    // Draw fold triggers
    foreach (const VisibleBlock &b, editor()->visibleBlocks())
    {
        top_position = b.topPosition;
        line_number = b.lineNumber;
        block = b.textBlock;

        if (Utils::TextBlockHelper::isFoldTrigger(block))
        {
            collapsed = Utils::TextBlockHelper::isCollapsed(block);
            mouseOver = (m_mouseOverLine == line_number);
            drawFoldIndicator(top_position, mouseOver, collapsed, &painter);

            if (collapsed)
            {
                bool found = false;

                // check if the block already has a decoration, it might
                // have been folded by the parent editor/document in the
                // case of cloned editor
                foreach(const TextDecoration::Ptr &deco, m_blockDecos)
                {
                    if (deco->block() == block)
                    {
                        found = true;
                        // no need to add a deco, just go to the next block
                        break;
                    }
                }

                if (!found)//TODO: was ist das fuer eine Struktur
                {
                    bool valid;  //valid should always be true, since check for fold trigger was already done above
                    addFoldDecoration(block, FoldScope(block, valid));
                }
            }
            else
            {
                foreach(const TextDecoration::Ptr &deco, m_blockDecos)
                {
                    // check if the block decoration has been removed, it
                    // might have been unfolded by the parent
                    // editor/document in the case of cloned editor
                    if (deco->block() == block)
                    {
                        // remove it and
                        m_blockDecos.removeOne(deco);
                        editor()->decorations()->remove(deco);
                        break;
                    }
                }
            }
        }
    }
}

//----------------------------------------------------------
/*
Draw the fold region when the mouse is over and non collapsed
indicator.

:param top: Top position
:param block: Current block.
:param painter: QPainter
*/
void FoldingPanel::drawFoldRegionBackground(const QTextBlock &block, QPainter &painter) const
{
    bool valid;
    FoldScope r(block, valid);
    if (!valid)
    {
        return;
    }
    QPair<int,int> start_end = r.getRange(true);
    int top = 0;
    if (start_end.first > 0)
    {
        top = editor()->linePosFromNumber(start_end.first);
    }
    int bottom = editor()->linePosFromNumber(start_end.second + 1);
    int h = bottom - top;
    if (h == 0)
    {
        h = sizeHint().height();
    }
    int w = sizeHint().width();

    drawRect(QRectF(0, top, w, h), painter);
}

//----------------------------------------------------------
/*
Draw the background rectangle using the current style primitive color
or foldIndicatorBackground if nativeFoldingIndicator is true.

:param rect: The fold zone rect to draw

:param painter: The widget's painter.
*/
void FoldingPanel::drawRect(const QRectF &rect, QPainter &painter) const
{
    QColor c = m_customColor;
    if (m_native)
    {
        c = getSystemBckColor();
    }
    QLinearGradient grad(rect.topLeft(), rect.topRight());
    QColor outline;
#ifdef __APPLE__
    grad.setColorAt(0, c.lighter(100));
    grad.setColorAt(1, c.lighter(110));
    outline = c.darker(110);
#else
    grad.setColorAt(0, c.lighter(110));
    grad.setColorAt(1, c.lighter(130));
    outline = c.darker(100);
#endif

    painter.fillRect(rect, grad);
    painter.setPen(QPen(outline));
    painter.drawLine(rect.topLeft() +
                        QPointF(1, 0),
                        rect.topRight() -
                        QPointF(1, 0));
    painter.drawLine(rect.bottomLeft() +
                        QPointF(1, 0),
                        rect.bottomRight() -
                        QPointF(1, 0));
    painter.drawLine(rect.topRight() +
                        QPointF(0, 1),
                        rect.bottomRight() -
                        QPointF(0, 1));
    painter.drawLine(rect.topLeft() +
                        QPointF(0, 1),
                        rect.bottomLeft() -
                        QPointF(0, 1));
}

QColor mergedColor(const QColor &colorA, const QColor &colorB, float factor)
{
    float maxFactor = 100.0;
    QColor tmp = colorA;
    tmp.setRed((tmp.red() * factor) / maxFactor +
                (colorB.red() * (maxFactor - factor)) / maxFactor);
    tmp.setGreen((tmp.green() * factor) / maxFactor +
                    (colorB.green() * (maxFactor - factor)) / maxFactor);
    tmp.setBlue((tmp.blue() * factor) / maxFactor +
                (colorB.blue() * (maxFactor - factor)) / maxFactor);
    return tmp;
}

//----------------------------------------------------------
/*
Gets a system color for drawing the fold scope background.
*/
/*static*/ QColor FoldingPanel::getSystemBckColor()
{
    QPalette pal = qApp->palette();
    QColor b = pal.window().color();
    QColor h = pal.highlight().color();
    return mergedColor(b, h, 50);
}

//----------------------------------------------------------
/*
Draw the fold indicator/trigger (arrow).

:param top: Top position
:param mouse_over: Whether the mouse is over the indicator
:param collapsed: Whether the trigger is collapsed or not.
:param painter: QPainter
*/
void FoldingPanel::drawFoldIndicator(int top, bool mouseOver, bool collapsed, QPainter *painter) const
{
    QRect rect(0, top, sizeHint().width(), sizeHint().height());
    if (m_native)
    {
        QStyleOptionViewItem opt;
        opt.rect = rect;
        opt.state = (QStyle::State_Active | \
                        QStyle::State_Item | \
                        QStyle::State_Children);
        if (!collapsed)
        {
            opt.state |= QStyle::State_Open;
        }
        if (mouseOver)
        {
            opt.state |= (QStyle::State_MouseOver | \
                            QStyle::State_Enabled | \
                            QStyle::State_Selected);
            opt.palette.setBrush(QPalette::Window, \
                                    palette().highlight());
        }
        opt.rect.translate(-2, 0);
        style()->drawPrimitive(QStyle::PE_IndicatorBranch, \
                                    &opt, painter, this);
    }
    else
    {
        int index = 0;
        if (!collapsed)
        {
            index = 2;
        }
        if (mouseOver)
        {
            index += 1;
        }

        QIcon(m_customIndicators[index]).paint(painter, rect);
    }
}

//----------------------------------------------------------
/*
Add fold decorations (boxes arround a folded block in the editor
widget).
*/
void FoldingPanel::addFoldDecoration(const QTextBlock &block, const FoldScope &region)
{
    TextDecoration::Ptr deco(new TextDecoration(QTextCursor(block)));
    deco->connect(SIGNAL(clicked(TextDecoration::Ptr)), this, SLOT(onFoldDecoClicked(TextDecoration::Ptr)));
    deco->setTooltip(region.text(/*maxLines*/ 25));
    deco->setDrawOrder(1);
    deco->setBlock(block);
    deco->selectLine();
    deco->setOutline(Utils::driftColor(getScopeHighlightColor(), 110));
    deco->setBackground(getScopeHighlightColor());
    deco->setForeground(QColor("#808080"));
    m_blockDecos.append(deco);
    editor()->decorations()->append(deco);
}

//----------------------------------------------------------
/*
Adds a scope decoration that enclose the current scope
:param start: Start of the current scope
:param end: End of the current scope
:param parent_start: Start of the parent scope
:param parent_end: End of the parent scope
:param base_color: base color for scope decoration
:param factor: color factor to apply on the base color (to make it
    darker).
*/
void FoldingPanel::addScopeDeco(int start, int end, int parentStart, int parentEnd, const QColor &baseColor, int factor)
{
    QColor color = Utils::driftColor(baseColor, factor);
    // upper part
    if (start > 0)
    {
        TextDecoration::Ptr d(new TextDecoration(editor()->document(), -1, -1,
                            parentStart, start));
        d->setFullWidth(true, false);
        d->setDrawOrder(2);
        d->setBackground(color);
        editor()->decorations()->append(d);
        m_scopeDecos.append(d);
    }

    // lower part
    int blockCount = editor()->document()->blockCount();

    if (end < blockCount)
    {
        TextDecoration::Ptr d(new TextDecoration(editor()->document(), -1, -1,
                            end, parentEnd + 1));
        d->setFullWidth(true, false);
        d->setDrawOrder(2);
        d->setBackground(color);
        editor()->decorations()->append(d);
        m_scopeDecos.append(d);
    }
}

//----------------------------------------------------------------
/*
Gets the base scope highlight color (derivated from the editor
background)
*/
QColor FoldingPanel::getScopeHighlightColor() const
{
    QColor color = editor()->background();
    if (color.lightness() < 128)
    {
        color = Utils::driftColor(color, 130);
    }
    else
    {
        color = Utils::driftColor(color, 105);
    }
    return color;
}


//----------------------------------------------------------
/*
Unfold a folded block that has just been clicked by the user
*/
void FoldingPanel::onFoldDecoClicked(TextDecoration::Ptr deco)
{
    toggleFoldTrigger(deco->block());
}

//----------------------------------------------------------
/*
Toggle a fold trigger block (expand or collapse it).

:param block: The QTextBlock to expand/collapse
*/
void FoldingPanel::toggleFoldTrigger(const QTextBlock &block, bool refreshEditor /*= true*/)
{
    if (!Utils::TextBlockHelper::isFoldTrigger(block))
    {
        return;
    }

    bool valid;
    FoldScope region(block, valid); //valid is true always, since block is a fold trigger (see check above)

    if (region.collapsed())
    {
        region.unfold(false);
        if (m_mouseOverLine >= 0)
        {
            QPair<int,int> start_end = region.getRange();
            addScopeDecorations(region.trigger(), start_end.first, start_end.second);
        }
    }
    else
    {
        region.fold();
        clearScopeDecos();
    }

    if (refreshEditor)
    {
        refreshEditorAndScrollbars();
    }
    emit triggerStateChanged(region.trigger(), region.collapsed());
}

//----------------------------------------------------------
/*
Refrehes editor content and scollbars.

We generate a fake resize event to refresh scroll bar.

We have the same problem as described here:
http://www.qtcentre.org/threads/44803 and we apply the same solution
(don't worry, there is no visual effect, the editor does not grow up
at all, even with a value = 500)
*/
void FoldingPanel::refreshEditorAndScrollbars()
{

    //editor()->markWholeDocDirty();
    editor()->repaint();
    QSize s = editor()->size();
    s.setWidth(s.width() + 1);
    QResizeEvent evt(editor()->size(), s);
    editor()->callResizeEvent(&evt);
}

//----------------------------------------------------------
/*
Show a scope decoration on the editor widget

:param start: Start line
:param end: End line
*/
void FoldingPanel::addScopeDecorations(const QTextBlock &block, int start, int end)
{
    bool valid;
    FoldScope blockScope(block, valid);
    if (!valid)
    {
        qDebug() << "FoldingPanel::addScopeDecorations: block is no fold trigger (this should not happen!)";
        return;
    }
    QSharedPointer<FoldScope> parent = blockScope.parent();

    if (Utils::TextBlockHelper::isFoldTrigger(block))
    {
        QColor base_color = getScopeHighlightColor();
        int factor_step = 5;
        int factor = 100;

        if (base_color.lightness() < 128)
        {
            factor_step = 10;
            factor = 70;
        }

        QPair<int,int> parent_start_end;

        while (!parent.isNull())
        {
            // highlight parent scope
            parent_start_end = parent->getRange();
            addScopeDeco(start, end + 1, parent_start_end.first, parent_start_end.second,
                base_color, factor);
            // next parent scope
            start = parent_start_end.first;
            end = parent_start_end.second;
            parent = parent->parent();
            factor += factor_step;
        }
        // global scope
        parent_start_end.first = 0;
        parent_start_end.second = editor()->document()->blockCount();
        addScopeDeco(start, end + 1, parent_start_end.first, parent_start_end.second, base_color,
            factor + factor_step);
    }
    else
    {
        clearScopeDecos();
    }
}

//----------------------------------------------------------
/*
Clear scope decorations (on the editor)
*/
void FoldingPanel::clearScopeDecos()
{
    foreach (TextDecoration::Ptr deco, m_scopeDecos)
    {
        editor()->decorations()->remove(deco);
    }

    m_scopeDecos.clear();
}

//----------------------------------------------------------
/*
Clear the folded block decorations.
*/
void FoldingPanel::clearBlockDeco()
{
    foreach(TextDecoration::Ptr deco, this->m_blockDecos)
    {
        editor()->decorations()->remove(deco);
    }

    m_blockDecos.clear();
}

//----------------------------------------------------------
/*
Collapses all triggers and makes all blocks with fold level > 0
invisible.
*/
void FoldingPanel::collapseAll()
{
    clearBlockDeco();
    QTextBlock block = editor()->document()->firstBlock();
    const QTextBlock &last = editor()->document()->lastBlock();
    int lvl;
    bool trigger;
    while (block.isValid())
    {
        lvl = Utils::TextBlockHelper::getFoldLvl(block);
        trigger = Utils::TextBlockHelper::isFoldTrigger(block);
        if (trigger)
        {
            if (lvl == 0)
            {
                FoldingPanel::showPreviousBlankLines(block);
            }
            Utils::TextBlockHelper::setCollapsed(block, true);
        }
        block.setVisible(lvl == 0);
        if ((block == last) && (Utils::strip(block.text()) == ""))
        {
            block.setVisible(true);
            FoldingPanel::showPreviousBlankLines(block);
        }
        block = block.next();
    }

    refreshEditorAndScrollbars();
    QTextCursor tc = editor()->textCursor();
    tc.movePosition(QTextCursor::Start);
    editor()->setTextCursor(tc);
    emit collapseAllTriggered();
}

//----------------------------------------------------------
/*
Expands all fold triggers.
*/
void FoldingPanel::expandAll()
{
    QTextBlock block = editor()->document()->firstBlock();
    while (block.isValid())
    {
        Utils::TextBlockHelper::setCollapsed(block, false);
        block.setVisible(true);
        block = block.next();
    }

    clearBlockDeco();
    refreshEditorAndScrollbars();
    emit expandAllTriggered();
}

//----------------------------------------------------------
/*
toggles all folds
*/
void FoldingPanel::toggleFold(bool topLevelOnly)
{
    QTextBlock block = editor()->document()->firstBlock();
    bool trigger;
    int lvl;

    if (topLevelOnly)
    {
        while (block.isValid())
        {
            lvl = Utils::TextBlockHelper::getFoldLvl(block);
            trigger = Utils::TextBlockHelper::isFoldTrigger(block);
            if (lvl == 0 && trigger)
            {
                toggleFoldTrigger(block, true);
            }
            block = block.next();
        }

        clearBlockDeco();
        refreshEditorAndScrollbars();
    }
    else
    {
        //at first, toggle all the top level entries and refresh editor, afterwards silently toggle all deeper levels
        QTextBlock block = editor()->document()->firstBlock();
        bool trigger;
        while (block.isValid())
        {
            lvl = Utils::TextBlockHelper::getFoldLvl(block);
            trigger = Utils::TextBlockHelper::isFoldTrigger(block);
            if (lvl == 0 && trigger)
            {
                toggleFoldTrigger(block, false);
            }
            block = block.next();
        }

        clearBlockDeco();
        refreshEditorAndScrollbars();

        //2nd run
        block = editor()->document()->firstBlock();
        while (block.isValid())
        {
            lvl = Utils::TextBlockHelper::getFoldLvl(block);
            trigger = Utils::TextBlockHelper::isFoldTrigger(block);
            if (lvl > 0 && trigger)
            {
                toggleFoldTrigger(block, false);
            }
            block = block.next();
        }
    }
}

//----------------------------------------------------------
/*
Show the block previous blank lines
*/
/*static*/ void FoldingPanel::showPreviousBlankLines(const QTextBlock &block)
{
    // set previous blank lines visibles
    QTextBlock pblock = block.previous();
    while ((Utils::strip(pblock.text()) == "") && \
            (pblock.blockNumber() >= 0))
    {
        pblock.setVisible(true);
        pblock = pblock.previous();
    }
}

//----------------------------------------------------------
/*
Find parent scope, if the block is not a fold trigger.
*/
/*static*/ QTextBlock FoldingPanel::findParentScope(const QTextBlock &block)
{
    QTextBlock block2 = block;
    QTextBlock original = block;
    int ref_lvl;

    if (!Utils::TextBlockHelper::isFoldTrigger(block))
    {
        // search level of next non blank line
        while ((Utils::strip(block2.text()) == "") && block2.isValid())
        {
            block2 = block2.next();
        }
        ref_lvl = Utils::TextBlockHelper::getFoldLvl(block2) - 1;
        block2 = original;
        while (block2.blockNumber() && \
                (!Utils::TextBlockHelper::isFoldTrigger(block2) || \
                (Utils::TextBlockHelper::getFoldLvl(block2) > ref_lvl)))
        {
            block2 = block2.previous();
        }
    }

    return block2;
}

//----------------------------------------------------------
/*
Folds/unfolds the pressed indicator if any.
*/
void FoldingPanel::mousePressEvent(QMouseEvent *e)
{
    if (m_mouseOverLine >= 0)
    {
        QTextBlock block = editor()->document()->findBlockByNumber(m_mouseOverLine);
        toggleFoldTrigger(block);
    }
}

//----------------------------------------------------------
/*
Highlights the scopes surrounding the current fold scope.

    :param block: Block that starts the current fold scope.
*/
void FoldingPanel::highlightSurroundingScopes(QTextBlock block)
{
    bool valid;
    FoldScope scope(block, valid);
    if (!valid)
    {
        return;
    }

    if (!m_currentScope.isValid() ||
            (m_currentScope.getRange() != scope.getRange()))
    {
        m_currentScope = scope;
        clearScopeDecos();
        // highlight surrounding parent scopes with a darker color
        //start, end = scope.getRange()
        if (!Utils::TextBlockHelper::isCollapsed(block))
        {
            QPair<int,int> start_end = scope.getRange();
            addScopeDecorations(block, start_end.first, start_end.second);
        }
    }
}

//----------------------------------------------------------
/*
Detect mouser over indicator and highlight the current scope in the
editor (up and down decoration arround the foldable text when the mouse
is over an indicator).

:param event: event
*/
void FoldingPanel::mouseMoveEvent(QMouseEvent *e)
{
    Panel::mouseMoveEvent(e);
    int line = editor()->lineNbrFromPosition(e->pos().y());
    if (line >= 0)
    {
        QTextBlock block = FoldScope::findParentScope( \
            editor()->document()->findBlockByNumber(line));
        if (Utils::TextBlockHelper::isFoldTrigger(block))
        {
            if (m_mouseOverLine == -1)
            {
                // mouse enter fold scope
                QApplication::setOverrideCursor( \
                    QCursor(Qt::PointingHandCursor));
            }
            if ((m_mouseOverLine != block.blockNumber()) && \
                    (m_mouseOverLine >= 0))
            {
                // fold scope changed, a previous block was highlighter so
                // we quickly update our highlighting
                m_mouseOverLine = block.blockNumber();
                highlightSurroundingScopes(block);
            }
            else
            {
                // same fold scope, request highlight
                m_mouseOverLine = block.blockNumber();
                DELAY_JOB_RUNNER_ARGTEXTBLOCK(m_pHighlightRunner, FoldingPanel, void(FoldingPanel::*)(QTextBlock))->requestJob(this, &FoldingPanel::highlightSurroundingScopes, block);
            }
            //self._highight_block = block
        }
        else
        {
            // no fold scope to highlight, cancel any pending requests
            DELAY_JOB_RUNNER_ARGTEXTBLOCK(m_pHighlightRunner, FoldingPanel, void(FoldingPanel::*)(QTextBlock))->cancelRequests();
            m_mouseOverLine = -1;
            QApplication::restoreOverrideCursor();
        }
        repaint();
    }
}

//----------------------------------------------------------
/*
Removes scope decorations and background from the editor and the panel
if highlight_caret_scope, else simply update the scope decorations to
match the caret scope.
*/
void FoldingPanel::leaveEvent(QEvent *e)
{
    Panel::leaveEvent(e);
    QApplication::restoreOverrideCursor();
    DELAY_JOB_RUNNER_ARGTEXTBLOCK(m_pHighlightRunner, FoldingPanel, void(FoldingPanel::*)(QTextBlock))->cancelRequests();

    if (!m_highlightCaretScope)
    {
        clearScopeDecos();
        m_mouseOverLine = -1;
        m_currentScope = FoldScope();
    }
    else
    {
        m_blockNbr = -1;
        highlightCaretScopeSlot();
    }
    editor()->repaint();
}

//----------------------------------------------------------
/*
Override key press to select the current scope if the user wants
to deleted a folded scope (without selecting it).
*/
void FoldingPanel::onKeyPressed(QKeyEvent *e)
{
    bool delete_request = (e->key() == Qt::Key_Backspace) || (e->key() == Qt::Key_Delete);

    if ((e->text() != "") || delete_request)
    {
        QTextCursor cursor = editor()->textCursor();
        QList<int> positions_to_check;
        QTextCursor tc;
        QTextBlock block;
        int start, end;
        FoldScope scope;
        QPair<int,int> start_end;
        bool valid;

        if (cursor.hasSelection())
        {
            // change selection to encompass the whole scope.
            positions_to_check << cursor.selectionStart() << cursor.selectionEnd();
        }
        else
        {
            positions_to_check << cursor.position();
        }

        foreach (int pos, positions_to_check)
        {
            block = editor()->document()->findBlock(pos);

            if ((Utils::TextBlockHelper::isFoldTrigger(block)) && (Utils::TextBlockHelper::isCollapsed(block)))
            {
                toggleFoldTrigger(block);
                if (delete_request && cursor.hasSelection())
                {
                    scope = FoldScope(FoldingPanel::findParentScope(block), valid);
                    if (!valid)
                    {
                        continue; //this should never happen
                    }

                    start_end = scope.getRange();
                    tc = editor()->selectLines(start_end.first, start_end.second);
                    if (tc.selectionStart() > cursor.selectionStart())
                    {
                        start = cursor.selectionStart();
                    }
                    else
                    {
                        start = tc.selectionStart();
                    }
                    if (tc.selectionEnd() < cursor.selectionEnd())
                    {
                        end = cursor.selectionEnd();
                    }
                    else
                    {
                        end = tc.selectionEnd();
                    }
                    tc.setPosition(start);
                    tc.setPosition(end, QTextCursor::KeepAnchor);
                    editor()->setTextCursor(tc);
                }
            }
        }
    }
}


//----------------------------------------------------------
/*
Refresh decorations colors. This function is called by the syntax
highlighter when the style changed so that we may update our
decorations colors according to the new style.
*/
void FoldingPanel::refreshDecorations(bool force /*= false*/)
{
    QTextCursor cursor = editor()->textCursor();
    if (m_prevCursor.isNull() || force || \
            m_prevCursor.blockNumber() != cursor.blockNumber())
    {
        foreach (const TextDecoration::Ptr &deco, m_blockDecos)
        {
            editor()->decorations()->remove(deco);
        }

        foreach (TextDecoration::Ptr deco, m_blockDecos)
        {
            deco->setOutline(Utils::driftColor(getScopeHighlightColor(), 110));
            deco->setBackground(getScopeHighlightColor());
            editor()->decorations()->append(deco);
        }
    }
    m_prevCursor = cursor;
}

} //end namespace ito
