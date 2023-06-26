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

#include "charBasedFoldDetector.h"

#include "../codeEditor.h"
#include <qpointer.h>
#include "../utils/utils.h"

namespace ito {

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

} //end namespace ito
