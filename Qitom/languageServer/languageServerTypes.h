/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#ifndef LANGUAGESERVERTYPES_H
#define LANGUAGESERVERTYPES_H

#include <QMetaType>
#include <qobject.h>
#include <qpointer.h>
#include <qstring.h>
#include <qvector.h>
#include <qstringlist.h>

namespace ito {
//--------------------------------------------------------------------------------------
struct CalltipRequest
{
    QString m_source;
    int m_line;
    int m_col;
    QString m_path;
    QByteArray m_callbackFctName;
    QPointer<QObject> m_sender;
};

//--------------------------------------------------------------------------------------
struct CalltipItem
{
    CalltipItem() : m_column(-1), m_bracketStartCol(-1), m_bracketStartLine(-1){};
    CalltipItem(
        const QString& methodName,
        const QStringList& params,
        int column,
        int bracketStartLine,
        int bracketStartCol) :
        m_calltipMethodName(methodName),
        m_calltipParams(params), m_column(column), m_bracketStartCol(bracketStartCol),
        m_bracketStartLine(bracketStartLine)
    {
    }

    QString m_calltipMethodName;
    QStringList m_calltipParams;
    int m_column;
    int m_bracketStartCol;
    int m_bracketStartLine;
};

//--------------------------------------------------------------------------------------
struct CompletionRequest
{
    QString m_source;
    int m_line;
    int m_col;
    QString m_path;
    QString m_prefix;
    int m_requestId;
    QByteArray m_callbackFctName;
    QPointer<QObject> m_sender;
};

//--------------------------------------------------------------------------------------
struct CompletionItem
{
    CompletionItem(){};
    CompletionItem(
        const QString& name,
        const QStringList& tooltips,
        const QString& icon = QString(),
        const QString& description = QString()) :
        m_name(name),
        m_tooltips(tooltips), m_icon(icon), m_description(description)
    {
    }

    QString m_name;
    QStringList m_tooltips; //!< can be multiple tooltips for overloaded methods
    QString m_icon;
    QString m_description;
};

//--------------------------------------------------------------------------------------
struct AssignmentRequest
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
struct AssignmentItem
{
    AssignmentItem(){};
    AssignmentItem(const QString& path, int line, int column, const QString& fullName) :
        m_path(path), m_line(line), m_column(column), m_fullName(fullName)
    {
    }

    QString m_path; // File path of the module where the assignment can be found
    int m_line; // line number
    int m_column; // column number
    QString m_fullName; // assignment full name
};

//--------------------------------------------------------------------------------------
struct GetHelpRequest
{
    QString m_source;
    int m_line;
    int m_col;
    QString m_path;
    QByteArray m_callbackFctName;
    QPointer<QObject> m_sender;
};

//--------------------------------------------------------------------------------------
struct HelpItem
{
    HelpItem(){};
    HelpItem(const QString& description, const QStringList& tooltips) :
        m_description(description), m_tooltips(tooltips)
    {
    }

    QString m_description;
    QStringList m_tooltips; //!< can be multiple tooltips for overloaded methods
    bool m_markdown = false; //!< true, if m_description contains markdown syntax (e.g. from a language server)
};

//--------------------------------------------------------------------------------------
struct RenameRequest
{
    QString m_code;
    int m_line;
    int m_col;
    QString m_filepath; //!< full filepath, or a non-existing py-file in the current directory in case of an unsafed file
    bool m_fileModified;
    bool m_untitledFile; //!< true if file has not been safed yet
    QString m_untitledName; //!< only != null, if this file is unsafed
    QByteArray m_callbackFctName;
    QPointer<QObject> m_sender;
};

//-------------------------------------------------------------------------------------
struct RenameTextItem
{
    int lineNumber;
    int startColumnIndex;
    int oldWordSize;
};

//--------------------------------------------------------------------------------------
struct RenameFileChanges
{
    RenameFileChanges(){};

    QString m_filePath; //!< either a true filepath or a non-existing dummy file in the current dir, if the file is untitled.
    bool m_untitledFile;
    QString m_untitledFilename; //!< null if a real file
    bool m_mainFile;

    //!< defines if this file is with the project of the initial file of the rename operation.
    /* The project is defined by jedi.api.project.get_default_project()
    */
    bool m_fileInProject;

    QVector<RenameTextItem> m_items;
};

} // end namespace ito

Q_DECLARE_METATYPE(ito::CalltipItem)
Q_DECLARE_METATYPE(ito::CompletionItem)
Q_DECLARE_METATYPE(ito::AssignmentItem)
Q_DECLARE_METATYPE(ito::HelpItem)
Q_DECLARE_METATYPE(ito::RenameFileChanges)
Q_DECLARE_METATYPE(ito::RenameTextItem)

#endif
