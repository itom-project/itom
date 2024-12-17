/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2024, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#include "codeCellHighlight.h"

#include "../codeEditor.h"
#include "../managers/textDecorationsManager.h"
#include "../utils/utils.h"
#include "widgets/scriptEditorWidget.h"

#include <qbrush.h>

namespace ito {

//------------------------------------------------------------------------------
CodeCellHighlighterMode::CodeCellHighlighterMode(const QString& description /*= ""*/, QObject* parent /*= NULL*/) :
    Mode("CodeCellHighlighterMode", description),
    QObject(parent),
    m_headlineBgColor(QColor(191, 242, 31)),
    m_activeCellBgColor(QColor(242, 242, 210)),
    m_activeCodeCellDecorator(nullptr),
    m_rootOutline(nullptr)
{
}

//------------------------------------------------------------------------------
/*
*/
CodeCellHighlighterMode::~CodeCellHighlighterMode()
{
}

//------------------------------------------------------------------------------
/*
*/

void CodeCellHighlighterMode::onInstall(CodeEditor* editor)
{
    Mode::onInstall(editor);
}

//------------------------------------------------------------------------------
/*
*/
void CodeCellHighlighterMode::onStateChanged(bool state)
{
    if (state)
    {
        connect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(updateActiveCodeCell()));
    }
    else
    {
        disconnect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(updateActiveCodeCell()));
        clearAllDecorators(true, true);
    }
}

//----------------------------------------------------------------------------------------
void CodeCellHighlighterMode::updateActiveCodeCell()
{
    int currentLine = editor()->currentLineNumber();
    bool withinCodeCell = false;
    int codeCellStartIndex = -1;
    int codeCellEndIndex = -1;

    if (m_rootOutline)
    {
        foreach(const auto & childItem, m_rootOutline->m_childs)
        {
            if (childItem->m_type == OutlineItem::typeCodeCell)
            {
                if (childItem->m_startLineIdx <= currentLine && childItem->m_endLineIdx >= currentLine)
                {
                    withinCodeCell = true;
                    codeCellStartIndex = childItem->m_startLineIdx;
                    codeCellEndIndex = childItem->m_endLineIdx;
                    break;
                }
            }
        }
    }

    if (!withinCodeCell)
    {
        clearAllDecorators(true, false);
    }
    else
    {
        int numLines = codeCellEndIndex - codeCellStartIndex;

        if (m_activeCodeCellDecorator &&
            m_activeCodeCellDecorator->block().firstLineNumber() == codeCellStartIndex + 1 &&
            m_activeCodeCellDecorator->block().lineCount() == numLines)
        {
            // no changes
            return;
        }
        else
        {
            clearAllDecorators(true, false);

            if (numLines > 0)
            {
                m_activeCodeCellDecorator = TextDecoration::Ptr(
                    new TextDecoration(editor()->document(), -1, -1, codeCellStartIndex + 1, codeCellEndIndex + 1, 1));
                m_activeCodeCellDecorator->setBackground(QBrush(m_activeCellBgColor));
                m_activeCodeCellDecorator->setFullWidth(true, false);
                editor()->decorations()->append(m_activeCodeCellDecorator);
            }
        }
    }
}

//--------------------------------------------------------------
/*
*/
void CodeCellHighlighterMode::clearAllDecorators(bool removeActiveCodeCell, bool removeCodeCellHeadings)
{
    if (m_activeCodeCellDecorator && removeActiveCodeCell)
    {
        editor()->decorations()->remove(m_activeCodeCellDecorator);
        m_activeCodeCellDecorator = nullptr;
    }

    if (removeCodeCellHeadings)
    {
        // remove non-confirmed indices
        for (int i = m_codeCellHeadlineDecorators.size() - 1; i >= 0; --i)
        {
            editor()->decorations()->remove(m_codeCellHeadlineDecorators[i]);
        }

        m_codeCellHeadlineDecorators.clear();
    }
}

//------------------------------------------------------------------------------
void CodeCellHighlighterMode::outlineModelChanged(ScriptEditorWidget* /*sew*/, QSharedPointer<OutlineItem> rootItem)
{
    // code cell headlines
    QList<int> confirmedIndices;
    int startLineIdx;
    bool found;

    foreach(const auto & childItem, rootItem->m_childs)
    {
        if (childItem->m_type == OutlineItem::typeCodeCell)
        {
            startLineIdx = childItem->m_startLineIdx;

            // check if the item already exists in m_codeCellHeadlineDecorators
            found = false;

            for (int i = 0; i < m_codeCellHeadlineDecorators.size(); ++i)
            {
                if (m_codeCellHeadlineDecorators[i]->block().blockNumber() == startLineIdx)
                {
                    confirmedIndices << i;
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                auto newDecorator = TextDecoration::Ptr(new TextDecoration(editor()->document(), -1, -1, startLineIdx, startLineIdx, 1));
                newDecorator->setBackground(QBrush(m_headlineBgColor));
                newDecorator->setFullWidth();
                editor()->decorations()->append(newDecorator);
                m_codeCellHeadlineDecorators << newDecorator;
                confirmedIndices << m_codeCellHeadlineDecorators.size() - 1;
            }

        }
    }

    // remove non-confirmed indices
    for (int i = m_codeCellHeadlineDecorators.size() - 1; i >= 0; --i)
    {
        if (!confirmedIndices.contains(i))
        {
            editor()->decorations()->remove(m_codeCellHeadlineDecorators[i]);
            m_codeCellHeadlineDecorators.takeAt(i);
        }
    }

    m_rootOutline = rootItem;

    updateActiveCodeCell();
}


} //end namespace ito
