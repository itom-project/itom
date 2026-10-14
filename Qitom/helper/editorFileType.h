/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut für Technische Optik (ITO),
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

#ifndef EDITORFILETYPE_H
#define EDITORFILETYPE_H

#include <qicon.h>
#include <qstring.h>
#include <qstringlist.h>

namespace ito {

//! Central registry that maps filenames to the type of document that is opened in the
//! script editor. Python files are fully supported (run, debug, code check, ...). All other
//! types are text-like files, that are only edited with a type specific syntax highlighting.
class EditorFileType
{
public:
    enum Type
    {
        Python,
        Json,
        Yaml,
        Xml,
        Ini,
        Markdown,
        CMake,
        Batch,
        PlainText, //!< any other text-like file
        Unknown //!< not supported by the editor
    };

    //! type of the file, derived from the suffix (and if unknown, from the mime type).
    static Type fromFilename(const QString& filename);

    //! true, if this file can be opened in the script editor
    static bool isEditable(const QString& filename);

    //! returns the line comment token or an empty string, if no line comments exist.
    static QString commentToken(Type type);

    //! returns the name of a type to be shown in dialogs
    static QString displayName(Type type);

    //! icon for the tab of the script editor
    static QIcon icon(Type type);

    //! suffixes (lower case, without dot) of all text-like files besides python files,
    //! as configured in the settings (CodeEditor/textFileExtensions).
    static QStringList textSuffixes();
    static QStringList defaultTextSuffixes();

    //! file filter for the open dialog of the editor, e.g. "Python (*.py);;Text files (...)"
    static QString openFileFilter();

    //! file filter for the save as dialog. The filter of the given type is the first one.
    static QString saveFileFilter(Type type);

    //! name filters (e.g. "*.py") with all supported files, used for QDir::entryInfoList.
    static QStringList nameFilters();

    //! forces a reload of the suffix list from the settings at the next access
    static void invalidateCache();

private:
    static Type fromSuffix(const QString& suffix);
    static QStringList m_cachedSuffixes;
    static bool m_cacheValid;
};

} // end namespace ito

#endif
