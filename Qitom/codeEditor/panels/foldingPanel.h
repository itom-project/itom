#ifndef FOLDINGPANEL_H
#define FOLDINGPANEL_H

/*
This module contains the marker panel
*/

#include "../panel.h"
#include "../utils/utils.h"

#include <qevent.h>
#include <qsize.h>
#include <qcolor.h>
#include "../foldDetector/foldDetector.h"
#include "../textDecoration.h"

class DelayJobRunnerBase;

/*
Displays the document outline and lets the user collapse/expand blocks.

The data represented by the panel come from the text block user state and
is set by the SyntaxHighlighter mode.

The panel does not expose any function that you can use directly. To
interact with the fold tree, you need to modify text block fold level or
trigger state using :class:`pyqode.core.api.utils.TextBlockHelper` or
:mod:`pyqode.core.api.folding`
*/
class FoldingPanel : public Panel
{
    Q_OBJECT
public:
    FoldingPanel(bool highlightCaretScope = false, const QString &description = "", QWidget *parent = NULL);
    virtual ~FoldingPanel();

    bool nativeLook() const;
    void setNativeLook(bool native);

    QStringList customIndicatorIcons() const;
    void setCustomIndicatorIcons(const QStringList &icons);

    QColor customFoldRegionBackground() const;
    void setCustomFoldRegionBackground(const QColor &color);

    bool highlightCaretScope() const;
    void setHighlightCaretScope(bool value);

    virtual void onInstall(CodeEditor *editor);
    virtual void onStateChanged(bool state);
    virtual QSize sizeHint() const;

    void collapseAll();
    void expandAll();

    void toggleFoldTrigger(const QTextBlock &block);

signals:
    void triggerStateChanged(QTextBlock, bool);
    void collapseAllTriggered();
    void expandAllTriggered();

protected:
    virtual void paintEvent(QPaintEvent *e);
    virtual void mouseMoveEvent(QMouseEvent *e);
    virtual void mousePressEvent(QMouseEvent *e);
    virtual void leaveEvent(QEvent *e);

    void drawFoldRegionBackground(const QTextBlock &block, QPainter &painter) const;
    void drawRect(const QRectF &rect, QPainter &painter) const;
    void drawFoldIndicator(int top, bool mouseOver, bool collapsed, QPainter *painter) const;
    void addFoldDecoration(const QTextBlock &block, const FoldScope &region);
    void addScopeDecorations(const QTextBlock &block, int start, int end);
    QColor getScopeHighlightColor() const;
    void clearScopeDecos();
    
    void addScopeDeco(int start, int end, int parentStart, int parentEnd, const QColor &baseColor, int factor);
    void refreshEditorAndScrollbars();

    static QColor getSystemBckColor();
    static void showPreviousBlankLines(const QTextBlock &block);
    static QTextBlock findParentScope(const QTextBlock &block);

    void highlightSurroundingScopes(QTextBlock block);

private slots:
    void clearBlockDeco();
    void highlightCaretScopeSlot();
    void onFoldDecoClicked(TextDecoration::Ptr deco);
    void onKeyPressed(QKeyEvent *e);

private:
    bool m_native;
    QStringList m_customIndicators;
    FoldScope m_currentScope;
    QColor m_customColor;
    bool m_highlightCaret;
    bool m_highlightCaretScope;
    int m_blockNbr;
    int m_indicSize;
    QList<TextDecoration::Ptr> m_scopeDecos;
    QList<TextDecoration::Ptr> m_blockDecos;
    int m_mouseOverLine; //-1 -> invalid
    DelayJobRunnerBase *m_pHighlightRunner;
};

#endif
