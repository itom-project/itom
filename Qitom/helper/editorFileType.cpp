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

#include "editorFileType.h"

#include "../AppManagement.h"

#include <qcoreapplication.h>
#include <qfileinfo.h>
#include <qmimedatabase.h>
#include <qmimetype.h>
#include <qsettings.h>

namespace ito {

QStringList EditorFileType::m_cachedSuffixes;
bool EditorFileType::m_cacheValid = false;

//-------------------------------------------------------------------------------------
/*static*/ QStringList EditorFileType::defaultTextSuffixes()
{
    return QStringList() << "txt"
                         << "json"
                         << "yaml"
                         << "yml"
                         << "xml"
                         << "ini"
                         << "cfg"
                         << "md"
                         << "csv"
                         << "toml"
                         << "cmake"
                         << "bat"
                         << "cmd"
                         << "log"
                         << "html"
                         << "htm"
                         << "css"
                         << "qss"
                         << "rst"
                         << "sh";
}

//-------------------------------------------------------------------------------------
/*static*/ QStringList EditorFileType::textSuffixes()
{
    if (!m_cacheValid)
    {
        QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
        settings.beginGroup("CodeEditor");
        QStringList list =
            settings.value("textFileExtensions", defaultTextSuffixes()).toStringList();
        settings.endGroup();

        m_cachedSuffixes.clear();

        foreach (QString s, list)
        {
            s = s.trimmed().toLower();

            while (s.startsWith("*") || s.startsWith("."))
            {
                s.remove(0, 1);
            }

            if (!s.isEmpty() && s != "py" && !m_cachedSuffixes.contains(s))
            {
                m_cachedSuffixes << s;
            }
        }

        m_cacheValid = true;
    }

    return m_cachedSuffixes;
}

//-------------------------------------------------------------------------------------
/*static*/ void EditorFileType::invalidateCache()
{
    m_cacheValid = false;
}

//-------------------------------------------------------------------------------------
/*static*/ EditorFileType::Type EditorFileType::fromSuffix(const QString& suffix)
{
    if (suffix == "py")
    {
        return Python;
    }
    else if (suffix == "json")
    {
        return Json;
    }
    else if (suffix == "yaml" || suffix == "yml")
    {
        return Yaml;
    }
    else if (
        suffix == "xml" || suffix == "html" || suffix == "htm" || suffix == "ui" ||
        suffix == "qrc")
    {
        return Xml;
    }
    else if (suffix == "ini" || suffix == "cfg" || suffix == "toml")
    {
        return Ini;
    }
    else if (suffix == "md" || suffix == "rst")
    {
        return Markdown;
    }
    else if (suffix == "csv")
    {
        return Csv;
    }
    else if (suffix == "txt")
    {
        return PlainText;
    }
    else if (suffix == "cmake")
    {
        return CMake;
    }
    else if (suffix == "bat" || suffix == "cmd")
    {
        return Batch;
    }

    return Unknown;
}

//-------------------------------------------------------------------------------------
/*static*/ EditorFileType::Type EditorFileType::fromFilename(const QString& filename)
{
    QFileInfo fi(filename);
    QString suffix = fi.suffix().toLower();
    Type t = fromSuffix(suffix);

    if (t == Python)
    {
        return Python;
    }

    if (fi.fileName().compare("CMakeLists.txt", Qt::CaseInsensitive) == 0)
    {
        return CMake;
    }

    if (textSuffixes().contains(suffix))
    {
        return (t == Unknown) ? Generic : t;
    }

    // not registered by suffix: check the mime type. Every text file is accepted.
    if (fi.exists() && fi.isFile())
    {
        QMimeDatabase db;
        QMimeType mime = db.mimeTypeForFile(fi, QMimeDatabase::MatchContent);

        if (mime.isValid() && mime.inherits("text/plain"))
        {
            return (t == Unknown) ? Generic : t;
        }
    }

    return Unknown;
}

//-------------------------------------------------------------------------------------
/*static*/ bool EditorFileType::isEditable(const QString& filename)
{
    return fromFilename(filename) != Unknown;
}

//-------------------------------------------------------------------------------------
/*static*/ QString EditorFileType::commentToken(Type type)
{
    switch (type)
    {
    case Python:
    case Yaml:
    case CMake:
        return "#";
    case Ini:
        return ";";
    case Batch:
        return "REM ";
    default:
        return QString();
    }
}

//-------------------------------------------------------------------------------------
/*static*/ QString EditorFileType::displayName(Type type)
{
    switch (type)
    {
    case Python:
        return QCoreApplication::translate("EditorFileType", "Python");
    case Json:
        return QCoreApplication::translate("EditorFileType", "JSON");
    case Yaml:
        return QCoreApplication::translate("EditorFileType", "YAML");
    case Xml:
        return QCoreApplication::translate("EditorFileType", "XML / HTML");
    case Ini:
        return QCoreApplication::translate("EditorFileType", "INI / Config");
    case Markdown:
        return QCoreApplication::translate("EditorFileType", "Markdown / reStructuredText");
    case Csv:
        return QCoreApplication::translate("EditorFileType", "CSV");
    case CMake:
        return QCoreApplication::translate("EditorFileType", "CMake");
    case Batch:
        return QCoreApplication::translate("EditorFileType", "Batch");
    default:
        return QCoreApplication::translate("EditorFileType", "Text");
    }
}

//-------------------------------------------------------------------------------------
/*static*/ QIcon EditorFileType::icon(Type type)
{
    QString path = ":/files/icons/fileGeneric.svg";

    switch (type)
    {
    case Python:
        path = ":/files/icons/filePython.png";
        break;
    case Json:
        path = ":/files/icons/fileJson.svg";
        break;
    case Yaml:
        path = ":/files/icons/fileYaml.svg";
        break;
    case Xml:
        path = ":/files/icons/fileXml.svg";
        break;
    case Ini:
        path = ":/files/icons/fileIni.svg";
        break;
    case Markdown:
        path = ":/files/icons/fileMd.svg";
        break;
    case Csv:
        path = ":/files/icons/fileCsv.svg";
        break;
    case PlainText:
        path = ":/files/icons/fileTxt.svg";
        break;
    case Generic:
    case CMake:
    case Batch:
    case Unknown:
    default:
        break;
    }

    QIcon icon(path);

    if (!icon.isNull())
    {
        return icon;
    }

    icon = QIcon(":/files/icons/fileGeneric.svg");

    if (!icon.isNull())
    {
        return icon;
    }

    return QIcon(":/files/icons/filePython.png");
}

//-------------------------------------------------------------------------------------
/*static*/ QStringList EditorFileType::nameFilters()
{
    QStringList list;
    list << "*.py";

    foreach (const QString& s, textSuffixes())
    {
        list << "*." + s;
    }

    return list;
}

//-------------------------------------------------------------------------------------
/*static*/ QString EditorFileType::openFileFilter()
{
    QStringList textPatterns;

    foreach (const QString& s, textSuffixes())
    {
        textPatterns << "*." + s;
    }

    // the first filter is selected by default in the file dialog
    QStringList filters;
    filters << QCoreApplication::translate("EditorFileType", "All supported files (%1)")
                   .arg((QStringList("*.py") + textPatterns).join(" "));
    filters << QCoreApplication::translate("EditorFileType", "Python (*.py)");
    filters << QCoreApplication::translate("EditorFileType", "Text files (%1)")
                   .arg(textPatterns.join(" "));
    filters << QCoreApplication::translate("EditorFileType", "All files (*)");
    return filters.join(";;");
}

//-------------------------------------------------------------------------------------
/*static*/ QString EditorFileType::saveFileFilter(Type type)
{
    QString python = QCoreApplication::translate("EditorFileType", "Python (*.py)");
    QStringList textPatterns;

    foreach (const QString& s, textSuffixes())
    {
        textPatterns << "*." + s;
    }

    QString text = QCoreApplication::translate("EditorFileType", "Text files (%1)")
                       .arg(textPatterns.join(" "));
    QString all = QCoreApplication::translate("EditorFileType", "All files (*)");

    if (type == Python || type == Unknown)
    {
        return QStringList({python, text, all}).join(";;");
    }

    return QStringList({text, python, all}).join(";;");
}

} // end namespace ito
