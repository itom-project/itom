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

#ifndef TEXTBLOCKUSERDATA_H
#define TEXTBLOCKUSERDATA_H

#include <qstring.h>

#include <QTextBlockUserData>
#include <qpointer.h>
#include "codeCheckerItem.h"

namespace ito {

class CodeEditor;

/*
Custom text block user data, mainly used to store checker messages and
    markers.
*/
class TextBlockUserData : public QTextBlockUserData
{
public:

    enum BreakpointType
    {
        TypeNoBp = 0,
        TypeBp = 0x0001,
        TypeBpEdit = 0x0002,
        TypeFlagDisabled = 0x0004,
        TypeBpDisabled = TypeBp | TypeFlagDisabled,
        TypeBpEditDisabled = TypeBpEdit | TypeFlagDisabled
    };

    enum StyleType
    {
        StylePython,
        StyleOutput,
        StyleError
    };

    TextBlockUserData(CodeEditor *editor);

    virtual ~TextBlockUserData();

    void removeCodeEditorRef();

    //List of checker messages associated with the block.
    QList<CodeCheckerItem> m_checkerMessages;

    //List of markers draw by a marker panel.
    QStringList m_markers;

    BreakpointType m_breakpointType;

    bool m_bookmark;

    QSharedPointer<TextBlockUserData> m_syntaxStack; //e.g. for python syntax highlighter

    int m_currentLineIdx;

    StyleType m_syntaxStyle;

private:
    QPointer<CodeEditor> m_codeEditor;
};

} //end namespace ito

#endif
