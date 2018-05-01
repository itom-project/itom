#ifndef INDENTFOLDDETECTOR_H
#define INDENTFOLDDETECTOR_H

#include "foldDetector.h"

/*
This module contains the code folding API.
*/


/*
Simple fold detector based on the line indentation level
*/
class IndentFoldDetector : public FoldDetector
{
    Q_OBJECT
public:
    IndentFoldDetector(QObject *parent = NULL);

    virtual ~IndentFoldDetector();

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
};


#endif