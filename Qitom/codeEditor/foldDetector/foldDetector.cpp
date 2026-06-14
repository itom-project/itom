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

#include "foldDetector.h"

#include "../codeEditor.h"
#include <qpointer.h>
#include "../utils/utils.h"
#include <limits>

#include <qdebug.h>

namespace ito {

class FoldDetectorPrivate
{
public:
    FoldDetectorPrivate() :
        m_limit(0x3FF) //see capacity of fold level in TextBlockHelper::setFoldLvl
    {
        /*
        //: Reference to the parent editor, automatically set by the syntax
        //: highlighter before process any block.

        Fold level limit, any level greater or equal is skipped.
        //: Default is sys.maxsize (i.e. all levels are accepted)*/
    }

    QPointer<CodeEditor> m_editor;
    int m_limit;
};

//--------------------------------------------------
FoldDetector::FoldDetector(QObject *parent /*= NULL*/) :
    QObject(parent),
    d_ptr(new FoldDetectorPrivate)
{
}

//--------------------------------------------------
FoldDetector::~FoldDetector()
{
    delete d_ptr;
    d_ptr = NULL;
}

//--------------------------------------------------
/*
Processes a block and setup its folding info.

This method call ``detect_fold_level`` and handles most of the tricky
corner cases so that all you have to do is focus on getting the proper
fold level foreach meaningful block, skipping the blank ones.

:param current_block: current block to process
:param previous_block: previous block
:param text: current block text
*/
void FoldDetector::processBlock(QTextBlock &currentBlock, QTextBlock &previousBlock, const QString &text)
{
    Q_D(FoldDetector);

    int prev_fold_level = Utils::TextBlockHelper::getFoldLvl(previousBlock);
    int fold_level;

    if (text.trimmed() == "")
    {
        // blank line always have the same level as the previous line
        fold_level = prev_fold_level;
    }
    else
    {
        fold_level = detectFoldLevel(previousBlock, currentBlock);

        if (fold_level > d->m_limit)
        {
            fold_level = d->m_limit;
        }
    }

    prev_fold_level = Utils::TextBlockHelper::getFoldLvl(previousBlock);

    if (fold_level > prev_fold_level)
    {
        // apply on previous blank lines
        QTextBlock block = currentBlock.previous();

        while (block.isValid() && block.text().trimmed() == "")
        {
            Utils::TextBlockHelper::setFoldLvl(block, fold_level);
            block = block.previous();
        }

        Utils::TextBlockHelper::setFoldTrigger(block, true);
    }

    // update block fold level
    if (text.trimmed() != "")
    {
        Utils::TextBlockHelper::setFoldTrigger(previousBlock, fold_level > prev_fold_level);
    }

    Utils::TextBlockHelper::setFoldLvl(currentBlock, fold_level);

    // user pressed enter at the beginning of a fold trigger line
    // the previous blank line will keep the trigger state and the new line
    // (which actually contains the trigger) must use the prev state (
    // and prev state must then be reset).
    QTextBlock prev = currentBlock.previous();  // real prev block (may be blank)

    if (prev.isValid()
        && prev.text().trimmed() == ""
        && Utils::TextBlockHelper::isFoldTrigger(prev))
    {
        // prev line has the correct trigger fold state
        Utils::TextBlockHelper::setCollapsed(currentBlock, Utils::TextBlockHelper::isCollapsed(prev));
        // make empty line not a trigger
        Utils::TextBlockHelper::setFoldTrigger(prev, false);
        Utils::TextBlockHelper::setCollapsed(prev, false);
    }
}

//--------------------------------------------------
CodeEditor* FoldDetector::editor() const
{
    Q_D(const FoldDetector);
    return d->m_editor;
}

//--------------------------------------------------
void FoldDetector::setEditor(CodeEditor *editor)
{
    Q_D(FoldDetector);
    d->m_editor = QPointer<CodeEditor>(editor);
}

//////////////////////////////////////////////////////////////////////////////////////

//------------------------------------------------
/*
Create a fold-able region from a fold trigger block.

:param block: The block **must** be a fold trigger.
:type block: QTextBlock

:param valid: false if the block is not a fold trigger
:type valid: bool
*/
FoldScope::FoldScope(const QTextBlock &block, bool &valid)
{
    if (!Utils::TextBlockHelper::isFoldTrigger(block))
    {
        valid = false; //not a fold trigger
        return;
    }

    m_trigger = block;
    valid = true;
}

//------------------------------------------------
FoldScope::FoldScope()
{
}

//------------------------------------------------
FoldScope::~FoldScope()
{
}

//------------------------------------------------
/*
Returns the fold level of the block trigger
:return:
*/
int FoldScope::triggerLevel() const
{
    return Utils::TextBlockHelper::getFoldLvl(m_trigger);
}

//------------------------------------------------
/*
Returns the fold level of the first block of the foldable scope (
just after the trigger)

:return:
*/
int FoldScope::scopeLevel() const
{
    return Utils::TextBlockHelper::getFoldLvl(m_trigger.next());
}

//------------------------------------------------
/*
*/
bool FoldScope::isValid() const
{
    return m_trigger.isValid();
}

//------------------------------------------------
/*
Returns True if the block is collasped, False if it is expanded.
*/
bool FoldScope::collapsed() const
{
    return Utils::TextBlockHelper::isCollapsed(m_trigger);
}



//------------------------------------------------
/*
Gets the fold region range (start and end line).

.. note:: Start line do no encompass the trigger line.

:param ignore_blank_lines: True to ignore blank lines at the end of the
    scope (the method will rewind to find that last meaningful block
    that is part of the fold scope).
:returns: tuple(int, int)
*/
QPair<int,int> FoldScope::getRange(bool ignoreBlankLines /*= true*/) const
{
    int ref_lvl = triggerLevel();
    int first_line = m_trigger.blockNumber();
    QTextBlock block = m_trigger.next();
    int last_line = block.blockNumber();
    int lvl = scopeLevel();
    if (ref_lvl == lvl)  // for zone set programmatically such as imports
    {
                        // in pyqode.python
        ref_lvl -= 1;
    }
    while (block.isValid() &&
            (Utils::TextBlockHelper::getFoldLvl(block) > ref_lvl))
    {
        last_line = block.blockNumber();
        block = block.next();
    }

    if (ignoreBlankLines && last_line)
    {
        block = block.document()->findBlockByNumber(last_line);
        while (block.blockNumber() && Utils::strip(block.text()) == "")
        {
            block = block.previous();
            last_line = block.blockNumber();
        }
    }
    return QPair<int,int>(first_line, last_line);
}

//------------------------------------------------
/*
Folds the region.
*/
void FoldScope::fold()
{
    QPair<int,int> start_end = getRange();
    Utils::TextBlockHelper::setCollapsed(m_trigger, true);
    QTextBlock block = m_trigger.next();
    while ((block.blockNumber() <= start_end.second) && block.isValid())
    {
        block.setVisible(false);
        block = block.next();
    }
}

//------------------------------------------------
/*
Unfolds the region.
*/
void FoldScope::unfold(bool unfoldChildBlocks /*= true*/)
{
    // set all direct child blocks which are not triggers to be visible
    m_trigger.setVisible(true);
    Utils::TextBlockHelper::setCollapsed(m_trigger, false);
    QList<QTextBlock> subblocksOfTrigger = blocks(false);

    if (unfoldChildBlocks)
    {
        foreach (QTextBlock block, subblocksOfTrigger)
        {
            block.setVisible(true);
            if (Utils::TextBlockHelper::isFoldTrigger(block))
            {
                Utils::TextBlockHelper::setCollapsed(block, false);
            }
        }
    }
    else if (subblocksOfTrigger.size() > 0)
    {
        int hideUntilThisBlockNumber = -1; //if >= 0, don't show blocks until this number (inclusive)
        bool valid;

        foreach(QTextBlock block, subblocksOfTrigger)
        {
            if ((hideUntilThisBlockNumber == -1) ||
                (block.blockNumber() > hideUntilThisBlockNumber))
            {
                block.setVisible(true);
                hideUntilThisBlockNumber = -1;

                if (Utils::TextBlockHelper::isFoldTrigger(block) &&
                    Utils::TextBlockHelper::isCollapsed(block))
                {
                    FoldScope scope(block, valid);
                    if (valid)
                    {
                        QPair<int, int> scopeRange = scope.getRange(true);
                        hideUntilThisBlockNumber = scopeRange.second;
                    }
                }
            }
        }
    }
}


//------------------------------------------------
/*
This generator generates the list of blocks directly under the fold
region. This list does not contain blocks from child regions.

:param ignore_blank_lines: True to ignore last blank lines.
*/
QList<QTextBlock> FoldScope::blocks(bool ignoreBlankLines /*= true*/) const
{
    QList<QTextBlock> retlist;
    QPair<int,int> start_end = getRange(ignoreBlankLines);
    QTextBlock block = m_trigger.next();

    while ((block.blockNumber() <= start_end.second) && block.isValid())
    {
        retlist << block;
        block = block.next();
    }

    return retlist;
}

//------------------------------------------------
/*
This generator generates the list of direct child regions.
*/
QList<FoldScope> FoldScope::childRegions() const
{
    QPair<int,int> start_end = getRange();
    QTextBlock block = m_trigger.next();
    int ref_lvl = scopeLevel();
    bool trigger;
    int lvl;
    QList<FoldScope> retlist;
    bool valid;

    while ((block.blockNumber() <= start_end.second) && block.isValid())
    {
        lvl = Utils::TextBlockHelper::getFoldLvl(block);
        trigger = Utils::TextBlockHelper::isFoldTrigger(block);
        if ((lvl == ref_lvl) && trigger)
        {
            retlist << FoldScope(block, valid); //valid has to be true, since check for fold trigger was already done above
        }
        block = block.next();
    }

    return retlist;
}

//------------------------------------------------
/*
Return the parent scope.

:return: FoldScope or None
*/
QSharedPointer<FoldScope> FoldScope::parent() const
{
    if ((Utils::TextBlockHelper::getFoldLvl(m_trigger) > 0) && \
            m_trigger.blockNumber())
    {
        QTextBlock block = m_trigger.previous();
        int ref_lvl = triggerLevel() - 1;
        while (block.blockNumber() &&
                (!Utils::TextBlockHelper::isFoldTrigger(block) || \
                    Utils::TextBlockHelper::getFoldLvl(block) > ref_lvl))
        {
            block = block.previous();
        }

        if (Utils::TextBlockHelper::isFoldTrigger(block))
        {
            bool valid;
            return QSharedPointer<FoldScope>(new FoldScope(block, valid));  //valid has to be true, since check for fold trigger was already done above
        }
        else
        {
            return QSharedPointer<FoldScope>();
        }
    }
    return QSharedPointer<FoldScope>();
}

//------------------------------------------------
/*
Get the scope text, with a possible maximum number of lines.

:param max_lines: limit the number of lines returned to a maximum.
:return: str
*/
QString FoldScope::text(int maxLines) const
{
    QStringList ret_val;
    QTextBlock block = m_trigger.next();
    QPair<int, int> start_end = getRange();
    while (block.isValid() && (block.blockNumber() <= start_end.second) && \
            (ret_val.size() < maxLines))
    {
        ret_val.append(block.text());
        block = block.next();
    }
    return ret_val.join("\n");
}

//------------------------------------------------
/*
Find parent scope, if the block is not a fold trigger.

:param block: block from which the research will start
*/
/*static*/ QTextBlock FoldScope::findParentScope(QTextBlock block)
{
    // if we moved up for more than n lines, just give up otherwise this
    // would take too much time.
    int limit = 5000;
    int counter = 0;
    int ref_lvl;

    QTextBlock original = block;
    if (block.isValid() && !Utils::TextBlockHelper::isFoldTrigger(block))
    {
        // search level of next non blank line
        while ((Utils::strip(block.text()) == "") && block.isValid())
        {
            block = block.next();
        }

        ref_lvl = Utils::TextBlockHelper::getFoldLvl(block) - 1;
        block = original;

        while (block.blockNumber() && (counter < limit) && \
                (!Utils::TextBlockHelper::isFoldTrigger(block) || \
                (Utils::TextBlockHelper::getFoldLvl(block) > ref_lvl)))
        {
            counter += 1;
            block = block.previous();
        }
    }

    if (counter < limit)
    {
        return block;
    }

    return QTextBlock();
}

//------------------------------------------------
QTextBlock FoldScope::trigger() const
{
    return m_trigger;
}

} //end namespace ito
