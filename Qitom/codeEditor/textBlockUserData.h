#ifndef TEXTBLOCKUSERDATA_H
#define TEXTBLOCKUSERDATA_H

#include <qstring.h>

#include <QTextBlockUserData>

namespace ito {

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
    explicit CheckerMessage(const QString &description, CheckerStatus status, int line, \
            int col = -1, const QColor color = QColor(), const QString path = QString()) :
        m_description(description), //The description of the message, used as a tooltip.
        m_status(status), //The status associated with the message
        m_line(line),    //: The line of the message
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
    int m_line;
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
        TypeBpDisabled = TypeBp | 0x0004,
        TypeBpEditDisabled = TypeBpEdit | 0x0004
    };

    TextBlockUserData() :
       QTextBlockUserData(),
        m_importStmt(false),
        m_breakpointType(TypeNoBp),
        m_bookmark(false)
    {
    }

    //List of checker messages associated with the block.
    QList<CheckerMessage> m_checkerMessages;

    //List of markers draw by a marker panel.
    QStringList m_markers;

    BreakpointType m_breakpointType;

    bool m_bookmark;

    QSharedPointer<TextBlockUserData> m_syntaxStack; //e.g. for python syntax highlighter

    bool m_docstring; //special item for python-related code editor

    bool m_importStmt;
};

} //end namespace ito

#endif