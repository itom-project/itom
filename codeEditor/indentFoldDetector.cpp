#include "indentFoldDetector.h"

#include "codeEditor.h"
#include <qpointer.h>
#include "utils/utils.h"



//--------------------------------------------------
IndentFoldDetector::IndentFoldDetector(QObject *parent /*= NULL*/) :
    FoldDetector(parent)
{
}

//--------------------------------------------------
IndentFoldDetector::~IndentFoldDetector()
{
}

//--------------------------------------------------
/*
Detects fold level by looking at the block indentation.

:param prev_block: previous text block
:param block: current block to highlight
*/
int IndentFoldDetector::detectFoldLevel(const QTextBlock &previousBlock, const QTextBlock &block)
{
    QString text = block.text();
    // round down to previous indentation guide to ensure contiguous block
    // fold level evolution.
    return (text.size() - Utils::lstrip(text).size()); // self.editor.tab_length
}
