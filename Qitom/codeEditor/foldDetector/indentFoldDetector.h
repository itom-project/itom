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

#ifndef INDENTFOLDDETECTOR_H
#define INDENTFOLDDETECTOR_H

#include "foldDetector.h"

#include <qregularexpression.h>

namespace ito {

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
    virtual int detectFoldLevel(const QTextBlock &previousBlock, const QTextBlock &block);
private:

    QRegularExpression m_reContinuationLine;
    QRegularExpression m_lastSignatureLine;
};

} //end namespace ito


#endif
