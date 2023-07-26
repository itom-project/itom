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

#ifndef FOLDDETECTOR_H
#define FOLDDETECTOR_H

#include <qtextedit.h>
#include <qstring.h>
#include <qsharedpointer.h>
#include <QTextBlock>

namespace ito {

/*
This module contains the code folding API.
*/

class CodeEditor;
class FoldDetectorPrivate;


/*
Base class for fold detectors.

A fold detector takes care of detecting the text blocks fold levels that
are used by the FoldingPanel to render the document outline.

To use a FoldDetector, simply set it on a syntax_highlighter::

    editor.syntax_highlighter.fold_detector = my_fold_detector
*/
class FoldDetector : public QObject
{
    Q_OBJECT
public:
    FoldDetector(QObject *parent = NULL);

    virtual ~FoldDetector();

    void processBlock(QTextBlock &currentBlock, QTextBlock &previousBlock, const QString &text);

    CodeEditor* editor() const;
    void setEditor(CodeEditor *editor);

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
    FoldDetectorPrivate *d_ptr;
    Q_DECLARE_PRIVATE(FoldDetector);
};


/*
Utility class for manipulating fold-able code scope (fold/unfold,
get range, child and parent scopes and so on).

A scope is built from a fold trigger (QTextBlock).
*/
class FoldScope
{
public:
    FoldScope();
    FoldScope(const QTextBlock &block, bool &valid);
    virtual ~FoldScope();

    int triggerLevel() const;
    int scopeLevel() const;
    bool collapsed() const;
    bool isValid() const;

    //returns index of first and last line of entire fold range
    QPair<int, int> getRange(bool ignoreBlankLines = true) const;
    void fold();
    void unfold(bool unfoldChildBlocks = true);
    QString text(int maxLines) const;
    QSharedPointer<FoldScope> parent() const;
    QTextBlock trigger() const;
    QList<FoldScope> childRegions() const;
    QList<QTextBlock> blocks(bool ignoreBlankLines = true) const;

    static QTextBlock findParentScope(QTextBlock block);

private:
    QTextBlock m_trigger;
};

} //end namespace ito

#endif
