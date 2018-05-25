#include "charBasedFoldDetector.h"

#include "../codeEditor.h"
#include <qpointer.h>
#include "../utils/utils.h"

class CharBasedFoldDetectorPrivate
{
public:
    CharBasedFoldDetectorPrivate()
    {
        /*
        #: Reference to the parent editor, automatically set by the syntax
        #: highlighter before process any block.

        Fold level limit, any level greater or equal is skipped.
        #: Default is sys.maxsize (i.e. all levels are accepted)*/
    }

    QString m_openChars;
    QString m_closeChars;
};

//--------------------------------------------------
CharBasedFoldDetector::CharBasedFoldDetector(QChar openChars /*= '{'*/, QChar closeChars /*= '}'*/, QObject *parent /*= NULL*/) :
    FoldDetector(parent),
    d_ptr(new CharBasedFoldDetectorPrivate)
{
    d_ptr->m_openChars = openChars;
    d_ptr->m_closeChars = closeChars;
}

//--------------------------------------------------
CharBasedFoldDetector::~CharBasedFoldDetector()
{
    delete d_ptr;
    d_ptr = NULL;
}

//--------------------------------------------------
/*
Detects fold level by looking at the block indentation.

:param prev_block: previous text block
:param block: current block to highlight
*/
int CharBasedFoldDetector::detectFoldLevel(const QTextBlock &previousBlock, const QTextBlock &block)
{
    Q_D(CharBasedFoldDetector);
    

    QString prev_text;

    if (previousBlock.isValid())
    {
        prev_text = Utils::strip(previousBlock.text());
    }
    else
    {
        prev_text = "";
    }
    QString text = Utils::strip(block.text());
    if (d->m_openChars.contains(text))
    {
        return Utils::TextBlockHelper::getFoldLvl(previousBlock) + 1;
    }
    if (prev_text.endsWith(d->m_openChars) && !d->m_openChars.contains(prev_text))
    {
        return Utils::TextBlockHelper::getFoldLvl(previousBlock) + 1;
    }
    if (prev_text.contains(d->m_openChars))
    {
        return Utils::TextBlockHelper::getFoldLvl(previousBlock) - 1;
    }
    return Utils::TextBlockHelper::getFoldLvl(previousBlock);
}
