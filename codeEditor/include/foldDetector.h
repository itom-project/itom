#ifndef FOLDDETECTOR_H
#define FOLDDETECTOR_H

#include <qtextedit.h>
#include <qstring.h>

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


#endif