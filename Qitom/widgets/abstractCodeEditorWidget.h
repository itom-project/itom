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
*********************************************************************** */

#ifndef ABSTRACTCODEEDITORWIDGET_H
#define ABSTRACTCODEEDITORWIDGET_H

#include "../common/sharedStructures.h"
#include "../codeEditor/codeEditor.h"

#include <qevent.h>
#include "../codeEditor/syntaxHighlighter/syntaxHighlighterBase.h"
#include "../codeEditor/modes/symbolMatcherMode.h"
#include "../codeEditor/modes/caretLineHighlight.h"
#include "../codeEditor/modes/pyCalltips.h"
#include "../codeEditor/modes/codeCompletion.h"
#include "../codeEditor/modes/pyCalltips.h"
#include "../codeEditor/modes/pyAutoIndent.h"

#include <qstringlist.h>
#include <qrect.h>
#include <qpixmap.h>
#include <qsharedpointer.h>

namespace ito {

class AbstractCodeEditorWidget : public CodeEditor
{
    Q_OBJECT

public:
    AbstractCodeEditorWidget(QWidget* parent = NULL);
    ~AbstractCodeEditorWidget();

    QString getWordAtPosition(const int &line, const int &index) const;

protected:

    enum tUserSelectionState { selNo, selRange };

    void init();

    virtual void loadSettings(); //overwrite this method if you want to load further settings

    QString formatCodeBeforeInsertion(const QString &text, int &lineCount, bool trimText = false, const QString &newIndent = "") const;
    QString formatCodeForClipboard(const QString &code, const QString &prependedTextInFirstLine) const;

    QPixmap loadMarker(const QString &name, int sizeAt96dpi) const;

    tUserSelectionState m_userSelectionState;

    QSharedPointer<SyntaxHighlighterBase> m_pythonSyntaxHighlighter;
    QSharedPointer<CodeEditorStyle> m_editorStyle;
    QSharedPointer<SymbolMatcherMode> m_symbolMatcher;
    QSharedPointer<CaretLineHighlighterMode> m_caretLineHighlighter;
    QSharedPointer<PyCalltipsMode> m_calltipsMode;
    QSharedPointer<CodeCompletionMode> m_codeCompletionMode;
    QSharedPointer<PyAutoIndentMode> m_pyAutoIndentMode;

    virtual int startLineOffset(int lineIdx) const { return 0; }

private:
    int getSpaceTabCount(const QString &text) const;

public slots:
    void reloadSettings() { loadSettings(); };

    virtual void copy();

    virtual void paste();

    virtual void cut();

signals:
    void userSelectionChanged(int lineFrom, int indexFrom, int lineTo, int indexTo);

};

} //end namespace ito

#endif
