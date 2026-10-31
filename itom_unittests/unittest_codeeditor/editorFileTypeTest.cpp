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

#include "codeEditorTestHelper.h"
#include "gtest/gtest.h"

#include <qdir.h>
#include <qfile.h>
#include <qsettings.h>
#include <qtemporarydir.h>

using namespace ito;
using namespace ito::test;

//! Tests for the registry, that maps filenames to editor file types.
class EditorFileTypeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // remove any setting of a previous test and reload the defaults
        QFile::remove(settingsFilePath());
        EditorFileType::invalidateCache();
    }

    void TearDown() override
    {
        QFile::remove(settingsFilePath());
        EditorFileType::invalidateCache();
    }

    //! writes the list of text file extensions to the settings and invalidates the cache
    void setTextFileExtensions(const QStringList& list)
    {
        QSettings settings(settingsFilePath(), QSettings::IniFormat);
        settings.beginGroup("CodeEditor");
        settings.setValue("textFileExtensions", list);
        settings.endGroup();
        settings.sync();
        EditorFileType::invalidateCache();
    }

    //! creates a file with the given content in the temporary directory
    QString createFile(const QString& name, const QByteArray& content)
    {
        QString path = m_dir.path() + QDir::separator() + name;
        QFile file(path);
        EXPECT_TRUE(file.open(QIODevice::WriteOnly));
        file.write(content);
        file.close();
        return path;
    }

    QTemporaryDir m_dir;
};

//-------------------------------------------------------------------------------------
TEST_F(EditorFileTypeTest, defaultSuffixesAreRegistered)
{
    QStringList suffixes = EditorFileType::textSuffixes();

    EXPECT_EQ(suffixes, EditorFileType::defaultTextSuffixes());
    EXPECT_FALSE(suffixes.contains("py")); // python is not a "text" file
    EXPECT_TRUE(suffixes.contains("txt"));
    EXPECT_TRUE(suffixes.contains("json"));
    EXPECT_TRUE(suffixes.contains("yaml"));
    EXPECT_TRUE(suffixes.contains("md"));
}

//-------------------------------------------------------------------------------------
TEST_F(EditorFileTypeTest, typeFromSuffixOfNonExistingFiles)
{
    // the type is derived from the suffix only, the file does not need to exist
    EXPECT_EQ(EditorFileType::fromFilename("a.py"), EditorFileType::Python);
    EXPECT_EQ(EditorFileType::fromFilename("/some/dir/a.PY"), EditorFileType::Python);
    EXPECT_EQ(EditorFileType::fromFilename("a.json"), EditorFileType::Json);
    EXPECT_EQ(EditorFileType::fromFilename("a.yaml"), EditorFileType::Yaml);
    EXPECT_EQ(EditorFileType::fromFilename("a.yml"), EditorFileType::Yaml);
    EXPECT_EQ(EditorFileType::fromFilename("a.xml"), EditorFileType::Xml);
    EXPECT_EQ(EditorFileType::fromFilename("a.html"), EditorFileType::Xml);
    EXPECT_EQ(EditorFileType::fromFilename("a.htm"), EditorFileType::Xml);
    EXPECT_EQ(EditorFileType::fromFilename("a.ini"), EditorFileType::Ini);
    EXPECT_EQ(EditorFileType::fromFilename("a.cfg"), EditorFileType::Ini);
    EXPECT_EQ(EditorFileType::fromFilename("a.toml"), EditorFileType::Ini);
    EXPECT_EQ(EditorFileType::fromFilename("a.md"), EditorFileType::Markdown);
    EXPECT_EQ(EditorFileType::fromFilename("a.rst"), EditorFileType::Markdown);
    EXPECT_EQ(EditorFileType::fromFilename("a.cmake"), EditorFileType::CMake);
    EXPECT_EQ(EditorFileType::fromFilename("a.bat"), EditorFileType::Batch);
    EXPECT_EQ(EditorFileType::fromFilename("a.cmd"), EditorFileType::Batch);

    // registered suffixes without a specific highlighter
    EXPECT_EQ(EditorFileType::fromFilename("a.txt"), EditorFileType::PlainText);
    EXPECT_EQ(EditorFileType::fromFilename("a.csv"), EditorFileType::Csv);
    EXPECT_EQ(EditorFileType::fromFilename("a.log"), EditorFileType::Generic);
    EXPECT_EQ(EditorFileType::fromFilename("a.sh"), EditorFileType::Generic);
    EXPECT_EQ(EditorFileType::fromFilename("a.css"), EditorFileType::Generic);

    // the suffix is compared case insensitive
    EXPECT_EQ(EditorFileType::fromFilename("A.JSON"), EditorFileType::Json);
    EXPECT_EQ(EditorFileType::fromFilename("A.Txt"), EditorFileType::PlainText);

    // not registered and not existing -> unknown
    EXPECT_EQ(EditorFileType::fromFilename("a.exe"), EditorFileType::Unknown);
    EXPECT_EQ(EditorFileType::fromFilename("a.foo"), EditorFileType::Unknown);
    EXPECT_EQ(EditorFileType::fromFilename("noSuffix"), EditorFileType::Unknown);
    EXPECT_EQ(EditorFileType::fromFilename(""), EditorFileType::Unknown);
}

//-------------------------------------------------------------------------------------
TEST_F(EditorFileTypeTest, cmakeListsIsCMake)
{
    EXPECT_EQ(EditorFileType::fromFilename("CMakeLists.txt"), EditorFileType::CMake);
    EXPECT_EQ(EditorFileType::fromFilename("/x/y/cmakelists.txt"), EditorFileType::CMake);
    EXPECT_EQ(EditorFileType::fromFilename("notes.txt"), EditorFileType::PlainText);
}

//-------------------------------------------------------------------------------------
TEST_F(EditorFileTypeTest, existingFilesWithUnregisteredSuffixUseMimeType)
{
    // a text file with an unknown suffix is editable as generic text file
    QString textFile = createFile("data.foo", "hello world\nsecond line\n");
    EXPECT_EQ(EditorFileType::fromFilename(textFile), EditorFileType::Generic);
    EXPECT_TRUE(EditorFileType::isEditable(textFile));

    // a text file without suffix
    QString noSuffix = createFile("README", "readme text\n");
    EXPECT_EQ(EditorFileType::fromFilename(noSuffix), EditorFileType::Generic);

    // ui and qrc are not registered by default, but if they exist and are text,
    // the xml highlighter is used
    QString uiFile = createFile("dialog.ui", "<ui version=\"4.0\"><class>Dialog</class></ui>\n");
    EXPECT_EQ(EditorFileType::fromFilename(uiFile), EditorFileType::Xml);

    // a binary file is never editable
    QByteArray binary(512, '\0');

    for (int i = 0; i < binary.size(); ++i)
    {
        binary[i] = (char)(i * 7 + 3);
    }

    QString binFile = createFile("data.foo2", binary);
    EXPECT_EQ(EditorFileType::fromFilename(binFile), EditorFileType::Unknown);
    EXPECT_FALSE(EditorFileType::isEditable(binFile));

    // a directory is not editable
    EXPECT_EQ(EditorFileType::fromFilename(m_dir.path()), EditorFileType::Unknown);
}

//-------------------------------------------------------------------------------------
TEST_F(EditorFileTypeTest, suffixesFromSettings)
{
    // the settings may contain patterns with "*." or "." prefixes, upper case letters
    // and duplicates. Python is never part of the text suffixes.
    setTextFileExtensions(QStringList() << "*.abc" << ".DEF" << "py" << "abc" << " ghi " << "");

    EXPECT_EQ(EditorFileType::textSuffixes(), QStringList() << "abc" << "def" << "ghi");

    EXPECT_EQ(EditorFileType::fromFilename("a.abc"), EditorFileType::Generic);
    EXPECT_EQ(EditorFileType::fromFilename("a.def"), EditorFileType::Generic);
    EXPECT_EQ(EditorFileType::fromFilename("a.py"), EditorFileType::Python);

    // txt is no longer registered: non existing txt file is unknown now...
    EXPECT_EQ(EditorFileType::fromFilename("a.txt"), EditorFileType::Unknown);

    // ...but an existing one is still detected by its mime type
    QString txtFile = createFile("a.txt", "some text\n");
    EXPECT_EQ(EditorFileType::fromFilename(txtFile), EditorFileType::PlainText);

    // the specific highlighter is kept for known suffixes, that are registered in the settings
    setTextFileExtensions(QStringList() << "json");
    EXPECT_EQ(EditorFileType::fromFilename("a.json"), EditorFileType::Json);
    EXPECT_EQ(EditorFileType::fromFilename("a.yaml"), EditorFileType::Unknown);

    EXPECT_EQ(EditorFileType::nameFilters(), QStringList() << "*.py" << "*.json");
}

//-------------------------------------------------------------------------------------
TEST_F(EditorFileTypeTest, settingsCacheIsOnlyReloadedAfterInvalidation)
{
    EXPECT_TRUE(EditorFileType::textSuffixes().contains("txt"));

    QSettings settings(settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");
    settings.setValue("textFileExtensions", QStringList() << "xyz");
    settings.endGroup();
    settings.sync();

    // still cached
    EXPECT_TRUE(EditorFileType::textSuffixes().contains("txt"));

    EditorFileType::invalidateCache();
    EXPECT_EQ(EditorFileType::textSuffixes(), QStringList() << "xyz");
}

//-------------------------------------------------------------------------------------
TEST_F(EditorFileTypeTest, commentTokens)
{
    EXPECT_EQ(EditorFileType::commentToken(EditorFileType::Python), "#");
    EXPECT_EQ(EditorFileType::commentToken(EditorFileType::Yaml), "#");
    EXPECT_EQ(EditorFileType::commentToken(EditorFileType::CMake), "#");
    EXPECT_EQ(EditorFileType::commentToken(EditorFileType::Ini), ";");
    EXPECT_EQ(EditorFileType::commentToken(EditorFileType::Batch), "REM ");

    // no line comments
    EXPECT_TRUE(EditorFileType::commentToken(EditorFileType::Json).isEmpty());
    EXPECT_TRUE(EditorFileType::commentToken(EditorFileType::Xml).isEmpty());
    EXPECT_TRUE(EditorFileType::commentToken(EditorFileType::Markdown).isEmpty());
    EXPECT_TRUE(EditorFileType::commentToken(EditorFileType::Csv).isEmpty());
    EXPECT_TRUE(EditorFileType::commentToken(EditorFileType::PlainText).isEmpty());
    EXPECT_TRUE(EditorFileType::commentToken(EditorFileType::Generic).isEmpty());
    EXPECT_TRUE(EditorFileType::commentToken(EditorFileType::Unknown).isEmpty());
}

//-------------------------------------------------------------------------------------
TEST_F(EditorFileTypeTest, displayNames)
{
    const EditorFileType::Type types[] = {
        EditorFileType::Python,
        EditorFileType::Json,
        EditorFileType::Yaml,
        EditorFileType::Xml,
        EditorFileType::Ini,
        EditorFileType::Markdown,
        EditorFileType::Csv,
        EditorFileType::CMake,
        EditorFileType::Batch,
        EditorFileType::PlainText,
        EditorFileType::Generic,
        EditorFileType::Unknown};

    QSet<QString> names;

    for (EditorFileType::Type t : types)
    {
        EXPECT_FALSE(EditorFileType::displayName(t).isEmpty());
        names.insert(EditorFileType::displayName(t));
    }

    // all types besides PlainText, Generic and Unknown have an own name
    EXPECT_EQ(names.size(), 10);
    EXPECT_EQ(
        EditorFileType::displayName(EditorFileType::PlainText),
        EditorFileType::displayName(EditorFileType::Unknown));
    EXPECT_EQ(
        EditorFileType::displayName(EditorFileType::Generic),
        EditorFileType::displayName(EditorFileType::Unknown));
}

//-------------------------------------------------------------------------------------
TEST_F(EditorFileTypeTest, fileDialogFilters)
{
    QStringList nameFilters = EditorFileType::nameFilters();
    EXPECT_EQ(nameFilters.first(), "*.py");
    EXPECT_EQ(nameFilters.size(), EditorFileType::defaultTextSuffixes().size() + 1);
    EXPECT_TRUE(nameFilters.contains("*.txt"));

    QString open = EditorFileType::openFileFilter();
    QStringList openFilters = open.split(";;");
    EXPECT_EQ(openFilters.size(), 4);
    EXPECT_TRUE(openFilters[0].contains("*.py"));
    EXPECT_TRUE(openFilters[0].contains("*.json"));
    EXPECT_TRUE(openFilters[1].contains("*.py"));
    EXPECT_FALSE(openFilters[1].contains("*.json"));
    EXPECT_TRUE(openFilters[2].contains("*.json"));
    EXPECT_FALSE(openFilters[2].contains("*.py"));
    EXPECT_TRUE(openFilters[3].contains("*"));

    // save as: the filter of the current type comes first
    QStringList savePy = EditorFileType::saveFileFilter(EditorFileType::Python).split(";;");
    EXPECT_EQ(savePy.size(), 3);
    EXPECT_TRUE(savePy[0].contains("*.py"));
    EXPECT_TRUE(savePy[1].contains("*.txt"));

    QStringList saveJson = EditorFileType::saveFileFilter(EditorFileType::Json).split(";;");
    EXPECT_EQ(saveJson.size(), 3);
    EXPECT_TRUE(saveJson[0].contains("*.json"));
    EXPECT_TRUE(saveJson[1].contains("*.py"));

    // an unknown type is handled like python (new, unsaved document)
    EXPECT_EQ(
        EditorFileType::saveFileFilter(EditorFileType::Unknown),
        EditorFileType::saveFileFilter(EditorFileType::Python));
}
