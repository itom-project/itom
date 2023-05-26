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

#ifndef CODECHECKERITEM_H
#define CODECHECKERITEM_H

#include <qstring.h>
#include <qcolor.h>
#include <qobject.h>

namespace ito {

    class CodeEditor;

    //------------------------------------------------------
    /*
    Holds data for a message displayed by the
    :class:`pyqode.core.modes.CheckerMode`.

    One entry as a result of the code checker (linter, e.g. pyflakes, flake8...).
    */
    class CodeCheckerItem
    {

    public:
        enum CheckerType
        {
            Info =    0x001, //must be a bitmask
            Warning = 0x002,
            Error =   0x004,
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
        explicit CodeCheckerItem(CheckerType messageType,
                                const QString &description,
                                const QString &messageCode = "",
                                int lineNumber = -1,
                                int col = -1,
                                const QString canonicalPath = QString()) :
            m_description(description), //The description of the message, used as a tooltip.
            m_code(messageCode),
            m_type(messageType), //The status associated with the message
            m_col(col),    //: The start column (used for the text decoration). If the col is None,
                           //: the whole line is highlighted.
            m_lineNumber(lineNumber),
            m_filePath(canonicalPath)
        {
            if (m_color.isValid() == false)
            {
                m_color = statusToColor(m_type);
            }
        }

        //-----------------------------------------------------------
        CodeCheckerItem(const CodeCheckerItem &other) :
            m_description(other.m_description), //The description of the message, used as a tooltip.
            m_code(other.m_code),
            m_type(other.m_type), //The status associated with the message
            m_col(other.m_col),    //: The start column (used for the text decoration). If the col is None,
                           //: the whole line is highlighted.
            m_lineNumber(other.m_lineNumber),
            m_filePath(other.m_filePath),
            m_color(other.m_color)
        {

        }

        //------------------------------------------------------------
        CheckerType type() const
        {
            return m_type;
        }

        //------------------------------------------------------------
        QString description() const
        {
            return m_description;
        }

        //------------------------------------------------------------
        int lineNumber() const
        {
            return m_lineNumber;
        }

        //------------------------------------------------------------
        /*
        Converts a message status to a string.

        :param status: Status to convert (p yqode.core.modes.CheckerMessages)
        :return: The status string.
        :rtype: str
        */
        static QString statusToString(CheckerType status)
        {
            switch (status)
            {
            case Info:
                return QObject::tr("Info");
            case Warning:
                return QObject::tr("Warning");
            case Error:
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
        static QColor statusToColor(CheckerType status)
        {
            switch (status)
            {
            case Info:
                return QColor("#4040DD");
            case Warning:
                return QColor("#DDDD40");
            case Error:
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
            return statusToString(m_type);
        }

        //----------------------------------------------------
        /* return a string representation of this checker item. */
        QString checkerItemText(bool addShortType = false, int wordWrapLength = -1) const
        {
            QString prefix;
            QString text;

            if (addShortType)
            {
                switch (m_type)
                {
                case Info:
                    prefix = QObject::tr("[I] ");
                    break;
                case Warning:
                    prefix = QObject::tr("[W] ");
                    break;
                case Error:
                    prefix = QObject::tr("[E] ");
                    break;
                }

            }

            if (m_code.isEmpty())
            {
                if (m_col != -1)
                {
                    text = QObject::tr("%1 (Column %2)").arg(prefix + m_description).arg(m_col + 1);
                }
                else
                {
                    text = m_description;
                }
            }
            else
            {
                if (m_col != -1)
                {
                    text = QObject::tr("%1: %2 (Column %3)")
                               .arg(prefix + m_code)
                               .arg(m_description)
                               .arg(m_col + 1);
                }
                else
                {
                    text = QString("%1: %2").arg(prefix + m_code).arg(m_description);
                }
            }

            if (wordWrapLength > 0 && text.size() > wordWrapLength)
            {
                // text too long. Wrap it in multiple lines, but inherit
                // every next line by 8 spaces.
                QStringList words = text.split(" ");
                int len = 0;
                QStringList finalParts;
                QStringList parts;

                foreach (const QString& word, text.split(" "))
                {
                    parts << word;
                    len += (word.size() + 1);

                    if (len >= wordWrapLength)
                    {
                        finalParts.append(parts.join(" "));
                        len = 0;
                        parts.clear();
                    }
                }

                if (len > 0)
                {
                    finalParts.append(parts.join(" "));
                }

                text = finalParts.join("\n        ");
            }

            return text;
        }

    private:
        CheckerType m_type;
        QString m_code; //code of the message
        QString m_description;
        QString m_filePath;
        int m_lineNumber;
        int m_col;
        QColor m_color;
    };

} //end namespace ito

#endif
