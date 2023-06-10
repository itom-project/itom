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

#include "indentFoldDetector.h"

#include "../codeEditor.h"
#include "../utils/utils.h"
#include "../syntaxHighlighter/pythonSyntaxHighlighter.h"
#include <qpointer.h>


namespace ito {

//--------------------------------------------------
IndentFoldDetector::IndentFoldDetector(QObject *parent /*= NULL*/) :
    FoldDetector(parent)
{
    m_reContinuationLine = QRegularExpression(
        "(\\sand|\\sor|\\+|\\-|\\*|\\^|>>|<<|\\*|\\*{2}|\\||//|/|,|\\\\)$"
    );

    // it is possible, that a signature ends
    // with
    // ): or ) -> type: # comment
    // in the last line
    m_lastSignatureLine = QRegularExpression(
        "^\\s*\\)\\s*(->\\s*.+)?:\\s*(#.*)?$"
    );

    /*bool a = m_lastSignatureLine.match("):").hasMatch();
    a = m_lastSignatureLine.match("   ) :    ").hasMatch();
    a = m_lastSignatureLine.match("  ) -> :  ").hasMatch();
    a = m_lastSignatureLine.match(") -> 'class' :").hasMatch();
    a = m_lastSignatureLine.match(")").hasMatch();
    a = m_lastSignatureLine.match("# ): sdfwer").hasMatch();
    a = m_lastSignatureLine.match("):#sfdwer").hasMatch();
    a = m_lastSignatureLine.match("   ) :    # :)'").hasMatch();
    a = m_lastSignatureLine.match("  ) -> :  #wersdf").hasMatch();
    a = m_lastSignatureLine.match(") -> 'class' :'lwers").hasMatch();*/
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
    int min_lvl = 0;
    int level;

    if (previousBlock.isValid())
    {
        int prev_lvl = Utils::TextBlockHelper::getFoldLvl(previousBlock);
        QString prev_text = previousBlock.text();
        int prev_state = Utils::TextBlockHelper::getState(previousBlock);
        int prev_prev_state = PythonSyntaxHighlighter::Normal;
        QTextBlock prevPrevBlock = previousBlock.previous();

        if (prevPrevBlock.isValid())
        {
            prev_prev_state = Utils::TextBlockHelper::getState(prevPrevBlock);
        }

        if (prev_state == PythonSyntaxHighlighter::InsideDq3String ||
            prev_state == PythonSyntaxHighlighter::InsideSq3String)
        {
            // it is assumed, that the last line of a multiline string
            // does not have the InsideXq3String state set!
            min_lvl = prev_lvl;

            if (prev_prev_state != prev_state)
            {
                //this is the 2nd line of a multi line string. Indent it.
                min_lvl++;
            }
        }
        else if (!Utils::lstrip(prev_text).startsWith("#"))
        {
            // ignore commented lines(could have arbitary indentation)
            // Verify if the previous line ends with a continuation line
            // with a regex.
            // The 2nd case is for this (e.g. produced by black):
            /* def test(
                    a: int,
                    b: int
               ):
                    pass
            */
            if (m_reContinuationLine.match(prev_text).hasMatch())
            {
                min_lvl = prev_lvl;
            }
            else if (m_lastSignatureLine.match(text).hasMatch())
            {
                min_lvl = prev_lvl;
            }
        }
    }

    // round down to previous indentation guide to ensure contiguous block
    // fold level evolution.
    if (editor()->useSpacesInsteadOfTabs())
    {
        level = (text.size() - Utils::lstrip(text).size()) / editor()->tabLength();
    }
    else
    {
        level = (text.size() - Utils::lstrip(text).size());
    }

    return qMax(min_lvl, level);
}

} //end namespace ito
