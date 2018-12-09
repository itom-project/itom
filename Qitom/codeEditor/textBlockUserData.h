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

#ifndef TEXTBLOCKUSERDATA_H
#define TEXTBLOCKUSERDATA_H

#include <qstring.h>

#include <QTextBlockUserData>
#include <qpointer.h>

namespace ito {

class CodeEditor;

//------------------------------------------------------
/*
Holds data for a message displayed by the
:class:`pyqode.core.modes.CheckerMode`.
*/
class CheckerMessage
{
public:
    enum CheckerStatus
    {
        StatusInfo = 1,
        StatusWarning = 2,
        StatusError = 3
    };

    //---------------------------------------------------------
    /*
    :param description: The message description (used as a tooltip)
    :param status: The status associated with the message.
    :param line: The message line number
    :param col: The message start column (at the moment the message ends at
                the end of the line).
    :param icon: Unused, we keep it for backward compatiblity.
    :param color: Text decoration color
    :param path: file path. Optional
    */
    explicit CheckerMessage(const QString &description, CheckerStatus status, \
            int col = -1, const QColor color = QColor(), const QString path = QString()) :
        m_description(description), //The description of the message, used as a tooltip.
        m_status(status), //The status associated with the message
        m_col(col),    //: The start column (used for the text decoration). If the col is None,
                       //: the whole line is highlighted.
        m_color(color), //The color used for the text decoration. If None, the default color
                       //: is used (:const:`pyqode.core.CheckerMessage.COLORS`)
        m_path(path)
    {
        if (m_color.isValid() == false)
        {
            m_color = statusToColor(m_status);
        }
    }

    //------------------------------------------------------------
    /*
    Converts a message status to a string.

    :param status: Status to convert (p yqode.core.modes.CheckerMessages)
    :return: The status string.
    :rtype: str
    */
    static QString statusToString(CheckerStatus status)
    {
        switch (status)
        {
        case StatusInfo:
            return QObject::tr("Info");
        case StatusWarning:
            return QObject::tr("Warning");
        case StatusError:
            return QObject::tr("Error");
        }
    }

    //------------------------------------------------------------
    /*
    Converts a message status to a color.

    :param status: Status to convert (p yqode.core.modes.CheckerMessages)
    :return: The status color.
    :rtype: QColor
    */
    static QColor statusToColor(CheckerStatus status)
    {
        switch (status)
        {
        case StatusInfo:
            return QColor("#4040DD");
        case StatusWarning:
            return QColor("#DDDD40");
        case StatusError:
        default:
            return QColor("#DD4040");
        }
    }

    //----------------------------------------------------
    /*
    Returns the message status as a string.

    :return: The status string.
    */
    QString statusString() const
    {
        return statusToString(m_status);
    }

public:
    CheckerStatus m_status;
    QString m_description;
    QString m_path;
    int m_col;
    QColor m_color;
};


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

    TextBlockUserData(CodeEditor *editor);

    virtual ~TextBlockUserData();

    void removeCodeEditorRef();

    //List of checker messages associated with the block.
    QList<CheckerMessage> m_checkerMessages;

    //List of markers draw by a marker panel.
    QStringList m_markers;

    BreakpointType m_breakpointType;

    bool m_bookmark;

    QSharedPointer<TextBlockUserData> m_syntaxStack; //e.g. for python syntax highlighter

    int m_currentLineIdx;

    bool m_noSyntaxHighlighting;

private:
    QPointer<CodeEditor> m_codeEditor;
};

} //end namespace ito

#endif