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

#ifndef PYTHONJEDI_H
#define PYTHONJEDI_H

#include <QMetaType>
#include <qstring.h>
#include <qpointer.h>
#include <qobject.h>

namespace ito
{
    //--------------------------------------------------------------------------------------
    struct JediCalltipRequest
    {
        QString m_source;
        int     m_line;
        int     m_col;
        QString m_path;
        QByteArray m_callbackFctName;
        QPointer<QObject> m_sender;
    };

    //--------------------------------------------------------------------------------------
    struct JediCalltip
    {
        JediCalltip() : m_column(-1), m_bracketStartCol(-1), m_bracketStartLine(-1) {};
        JediCalltip(const QString &methodName, const QStringList &params, int column, int bracketStartLine, int bracketStartCol) :
            m_calltipMethodName(methodName),
            m_calltipParams(params),
            m_column(column),
            m_bracketStartCol(bracketStartCol),
            m_bracketStartLine(bracketStartLine)
        {}

        QString m_calltipMethodName;
        QStringList m_calltipParams;
        int m_column;
        int m_bracketStartCol;
        int m_bracketStartLine;
    };

    //--------------------------------------------------------------------------------------
    struct JediCompletionRequest
    {
        QString m_source;
        int     m_line;
        int     m_col;
        QString m_path;
        QString m_prefix;
        int m_requestId;
        QByteArray m_callbackFctName;
        QPointer<QObject> m_sender;
    };

    //--------------------------------------------------------------------------------------
    struct JediCompletion
    {
        JediCompletion() {};
        JediCompletion(const QString &name, const QStringList &tooltips, const QString &icon = QString(), const QString &description = QString()) :
            m_name(name),
            m_tooltips(tooltips),
            m_icon(icon),
            m_description(description)
        {}

        QString m_name;
        QStringList m_tooltips; //!< can be multiple tooltips for overloaded methods
        QString m_icon;
        QString m_description;
    };

    //--------------------------------------------------------------------------------------
    struct JediAssignmentRequest
    {
        QString m_source;
        int m_line;
        int m_col;
        QString m_path;
        int m_mode;
        QByteArray m_callbackFctName;
        QPointer<QObject> m_sender;
    };

    //--------------------------------------------------------------------------------------
    struct JediAssignment
    {
        JediAssignment() {};
        JediAssignment(const QString &path, int line, int column, const QString &fullName) :
            m_path(path),
            m_line(line),
            m_column(column),
            m_fullName(fullName)
        {}

        QString m_path; // File path of the module where the assignment can be found
        int m_line; //line number
        int m_column; //column number
        QString m_fullName; //assignement full name
    };

    //--------------------------------------------------------------------------------------
    struct JediGetHelpRequest
    {
        QString m_source;
        int m_line;
        int m_col;
        QString m_path;
        QByteArray m_callbackFctName;
        QPointer<QObject> m_sender;
    };

    //--------------------------------------------------------------------------------------
    struct JediGetHelp
    {
        JediGetHelp() {};
        JediGetHelp(const QString &description, const QStringList &tooltips) :
            m_description(description),
            m_tooltips(tooltips)
        {}

        QString m_description;
        QStringList m_tooltips; //!< can be multiple tooltips for overloaded methods
    };

} //end namespace ito

Q_DECLARE_METATYPE(ito::JediCalltip)
Q_DECLARE_METATYPE(ito::JediCompletion)
Q_DECLARE_METATYPE(ito::JediAssignment)
Q_DECLARE_METATYPE(ito::JediGetHelp)

#endif
