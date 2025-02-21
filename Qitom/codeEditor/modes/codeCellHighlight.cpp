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
#include <qtextcursor.h>
#include <qdebug.h>

namespace ito {

//------------------------------------------------------------------------------
CodeCellHighlighterMode::CodeCellHighlighterMode(const QString& description /*= ""*/, QObject* parent /*= NULL*/) :
    Mode("CodeCellHighlighterMode", description),
    QObject(parent),
    m_headlineBgColor(QColor(240,240,240)),
    m_activeCodeCellBgColor(QColor(242, 242, 210)),
    m_rootOutline(nullptr),
    m_activeCodeCellLineRange(qMakePair<int, int>(-1, -1))
{
}

//------------------------------------------------------------------------------
/*
*/
CodeCellHighlighterMode::~CodeCellHighlighterMode()
{
}

//------------------------------------------------------------------------------
void CodeCellHighlighterMode::setHeadlineBgColor(const QColor& color)
{
    if (m_headlineBgColor != color)
    {
        m_headlineBgColor = color;
    }

    outlineModelChanged(nullptr, m_rootOutline);
}

//------------------------------------------------------------------------------
void CodeCellHighlighterMode::setActiveCodeCellBgColor(const QColor& color)
{
    if (m_activeCodeCellBgColor != color)
    {
        m_activeCodeCellBgColor = color;
    }

    outlineModelChanged(nullptr, m_rootOutline);
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
        m_activeCodeCellLineRange = qMakePair<int, int>(-1, -1);
    }
    else
    {
        m_activeCodeCellLineRange = QPair<int, int>(codeCellStartIndex + 1, codeCellEndIndex);
    }

    editor()->viewport()->update();
}

//------------------------------------------------------------------------------
void CodeCellHighlighterMode::outlineModelChanged(ScriptEditorWidget* /*sew*/, QSharedPointer<OutlineItem> rootItem)
{
    // code cell headlines
    QList<int> confirmedIndices;
    int startLineIdx;
    bool found;

    if (!rootItem)
    {
        return;
    }

    m_rootOutline = rootItem;

    updateActiveCodeCell();
}


} //end namespace ito
