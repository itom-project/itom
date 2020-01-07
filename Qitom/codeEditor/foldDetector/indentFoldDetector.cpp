/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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
#include <qpointer.h>
#include "../utils/utils.h"

namespace ito {

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
    if (editor()->useSpacesInsteadOfTabs())
    {
        return (text.size() - Utils::lstrip(text).size()) / editor()->tabLength();
    }
    else
    {
        return (text.size() - Utils::lstrip(text).size());
    }
}

} //end namespace ito
