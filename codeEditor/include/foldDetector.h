#ifndef FOLDDETECTOR_H
#define FOLDDETECTOR_H

#include <qtextedit.h>
#include <qstring.h>
#include <qsharedpointer.h>
#include <QTextBlock>

/*
This module contains the code folding API.
*/

class CodeEditor;
class FoldDetectorPrivate;


/*
Base class for fold detectors.

A fold detector takes care of detecting the text blocks fold levels that
are used by the FoldingPanel to render the document outline.

To use a FoldDetector, simply set it on a syntax_highlighter::

    editor.syntax_highlighter.fold_detector = my_fold_detector
*/
class FoldDetector : public QObject
{
    Q_OBJECT
public:
    FoldDetector(QObject *parent = NULL);

    virtual ~FoldDetector();

    void processBlock(QTextBlock &currentBlock, QTextBlock &previousBlock, const QString &text);

    CodeEditor* editor() const;
    void setEditor(CodeEditor *editor);
    
    /*
    Detects the block fold level.

    The default implementation is based on the block **indentation**.

    .. note:: Blocks fold level must be contiguous, there cannot be
        a difference greater than 1 between two successive block fold
        levels.

    :param prev_block: first previous **non-blank** block or None if this
        is the first line of the document
    :param block: The block to process.
    :return: Fold level
    */
    virtual int detectFoldLevel(const QTextBlock &previousBlock, const QTextBlock &block) = 0;
private:
    FoldDetectorPrivate *d_ptr;
    Q_DECLARE_PRIVATE(FoldDetector);
};


/*
Utility class for manipulating fold-able code scope (fold/unfold,
get range, child and parent scopes and so on).

A scope is built from a fold trigger (QTextBlock).
*/
class FoldScope
{
public:
    FoldScope(const QTextBlock &block);
    virtual ~FoldScope();

    int triggerLevel() const;
    int scopeLevel() const;
    bool collapsed() const;
    QPair<int, int> getRange(bool ignoreBlankLines = true) const;
    void fold();
    void unfold();
    QString text(int maxLines) const;
    QSharedPointer<FoldScope> parent() const;
    QList<FoldScope> childRegions() const;
    QList<QTextBlock> blocks(bool ignoreBlankLines = true) const;

    static QTextBlock findParentScope(QTextBlock block);

private:
    QTextBlock m_trigger;
};


#endif