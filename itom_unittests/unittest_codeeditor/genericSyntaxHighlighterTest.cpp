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

#include "codeEditor/foldDetector/indentFoldDetector.h"
#include "codeEditor/utils/utils.h"

using namespace ito;
using namespace ito::test;

/* The expected highlighting is written as a "style string" with one letter per character:

    ' ' default      C comment     N number      S string      K keyword / key
    L class/section  F function    O operator    D decorator   B builtin (attribute)
    T tag            P punctuation X constant    W operator word

   Trailing spaces of the expected string can be omitted.
*/

//-------------------------------------------------------------------------------------
// gtest takes its arguments by reference, which needs an out-of-class definition of the
// static constant. A local copy avoids this.
static const int kFirstMultiLineState = GenericSyntaxHighlighter::FirstMultiLineState;

//-------------------------------------------------------------------------------------
//! compares the highlighting of one line with the expected style string (the expected
//! string is padded with spaces to the length of the line, trailing spaces can be omitted)
static ::testing::AssertionResult styleMatches(
    const HighlightedEditor& hed, int line, const char* expected)
{
    QString actual = hed.styleString(line);
    QString exp = QString::fromLatin1(expected);

    while (exp.size() < actual.size())
    {
        exp += ' ';
    }

    if (actual == exp)
    {
        return ::testing::AssertionSuccess();
    }

    return ::testing::AssertionFailure()
        << "line " << line << ": '"
        << hed.editor()->document()->findBlockByNumber(line).text().toStdString() << "'\n"
        << "  expected: '" << exp.toStdString() << "'\n"
        << "  actual:   '" << actual.toStdString() << "'";
}

#define EXPECT_STYLE(hed, line, expected) EXPECT_TRUE(styleMatches(hed, line, expected))

//-------------------------------------------------------------------------------------
TEST(GenericSyntaxHighlighterTest, plainTextHasNoHighlighting)
{
    HighlightedEditor e(EditorFileType::PlainText);
    e.setText("plain text 42 \"string\" # no comment\n[section] key = value\n");

    ASSERT_EQ(e.lineCount(), 3);
    EXPECT_STYLE(e, 0, "");
    EXPECT_STYLE(e, 1, "");
    EXPECT_EQ(e.blockState(0), 0);
    EXPECT_EQ(e.blockState(1), 0);
}

//-------------------------------------------------------------------------------------
TEST(GenericSyntaxHighlighterTest, json)
{
    HighlightedEditor e(EditorFileType::Json);
    e.setText(
        "{\"a\": 1, \"b\": [true, null], \"c\": \"text\",\n"
        "  // comment\n"
        "  \"d\": -2.5e3 /* multi\n"
        " line */ , \"e\": \"x\"}\n");

    ASSERT_EQ(e.lineCount(), 5);
    EXPECT_STYLE(e, 0, "PKKKP NP KKKP PXXXXP XXXXPP KKKP SSSSSSP");
    EXPECT_STYLE(e, 1, "  CCCCCCCCCC");
    EXPECT_STYLE(e, 2, "  KKKP NNNNNN CCCCCCCC");
    EXPECT_STYLE(e, 3, "CCCCCCCC P KKKP SSSP");

    // the block comment is open at the end of line 2 and closed in line 3
    EXPECT_EQ(e.blockState(1), 0);
    EXPECT_GE(e.blockState(2), kFirstMultiLineState);
    EXPECT_EQ(e.blockState(3), 0);

    // keys and string values are distinguished
    EXPECT_EQ(e.styleAt(0, 1), StyleItem::KeyKeyword);
    EXPECT_EQ(e.styleAt(0, 33), StyleItem::KeyString);
}

//-------------------------------------------------------------------------------------
TEST(GenericSyntaxHighlighterTest, jsonEscapedQuotesInStrings)
{
    HighlightedEditor e(EditorFileType::Json);
    e.setText("{\"k\\\"ey\": \"va\\\"lue\"}");

    EXPECT_STYLE(e, 0, "PKKKKKKKP SSSSSSSSSP");
}

//-------------------------------------------------------------------------------------
TEST(GenericSyntaxHighlighterTest, yaml)
{
    HighlightedEditor e(EditorFileType::Yaml);
    e.setText(
        "# comment\n"
        "key: value\n"
        "list:\n"
        "  - 1\n"
        "  - two\n"
        "  - \"q\" # c\n"
        "---\n"
        "anchor: &a !!str yes\n"
        "ref: *a\n"
        "url: http://x#notacomment\n");

    ASSERT_EQ(e.lineCount(), 11);
    EXPECT_STYLE(e, 0, "CCCCCCCCC");
    EXPECT_STYLE(e, 1, "KKK");
    EXPECT_STYLE(e, 2, "KKKK");
    EXPECT_STYLE(e, 3, "PPP N");
    EXPECT_STYLE(e, 4, "PPP");
    EXPECT_STYLE(e, 5, "PPP SSS CCC");
    EXPECT_STYLE(e, 6, "TTT");
    EXPECT_STYLE(e, 7, "KKKKKK  DD DDDDD XXX");
    EXPECT_STYLE(e, 8, "KKK  DD");
    // a '#' without leading whitespace does not start a comment
    EXPECT_STYLE(e, 9, "KKK");
}

//-------------------------------------------------------------------------------------
TEST(GenericSyntaxHighlighterTest, xml)
{
    HighlightedEditor e(EditorFileType::Xml);
    e.setText(
        "<?xml version=\"1.0\"?>\n"
        "<!-- c -->\n"
        "<a b=\"c\">text &amp;</a>\n"
        "<!-- multi\n"
        "line --><x/>\n"
        "<![CDATA[raw <>]]>\n");

    ASSERT_EQ(e.lineCount(), 7);
    EXPECT_STYLE(e, 0, "DDDDDDDDDDDDDDDDDDDDD");
    EXPECT_STYLE(e, 1, "CCCCCCCCCC");
    EXPECT_STYLE(e, 2, "TT B SSST     XXXXXTTTT");
    EXPECT_STYLE(e, 3, "CCCCCCCCCC");
    EXPECT_STYLE(e, 4, "CCCCCCCCTTTT");
    EXPECT_STYLE(e, 5, "SSSSSSSSSSSSSSSSSS");

    EXPECT_GE(e.blockState(3), kFirstMultiLineState);
    EXPECT_EQ(e.blockState(4), 0);
}

//-------------------------------------------------------------------------------------
TEST(GenericSyntaxHighlighterTest, ini)
{
    HighlightedEditor e(EditorFileType::Ini);
    e.setText(
        "; comment\n"
        "# c2\n"
        "[section]\n"
        "key = value ; c\n"
        "num = 42\n"
        "flag = true\n"
        "str = \"s\"\n"
        "path = c:\\dir#x\n");

    ASSERT_EQ(e.lineCount(), 9);
    EXPECT_STYLE(e, 0, "CCCCCCCCC");
    EXPECT_STYLE(e, 1, "CCCC");
    EXPECT_STYLE(e, 2, "LLLLLLLLL");
    EXPECT_STYLE(e, 3, "KKK        CCCC");
    EXPECT_STYLE(e, 4, "KKK   NN");
    EXPECT_STYLE(e, 5, "KKKK   XXXX");
    EXPECT_STYLE(e, 6, "KKK   SSS");
    // '#' inside a value (without leading whitespace) is no comment
    EXPECT_STYLE(e, 7, "KKKK");
}

//-------------------------------------------------------------------------------------
TEST(GenericSyntaxHighlighterTest, markdown)
{
    HighlightedEditor e(EditorFileType::Markdown);
    e.setText(
        "# Headline\n"
        "\n"
        "* item **bold** `code`\n"
        "\n"
        "```\n"
        "code # not a headline\n"
        "```\n"
        "[l](http://x)\n"
        "> quote\n"
        "====\n"
        ".. note::\n"
        "1. numbered\n");

    ASSERT_EQ(e.lineCount(), 13);
    EXPECT_STYLE(e, 0, "LLLLLLLLLL");
    EXPECT_STYLE(e, 1, "");
    EXPECT_STYLE(e, 2, "O      KKKKKKKK SSSSSS");
    EXPECT_STYLE(e, 4, "SSS");
    EXPECT_STYLE(e, 5, "SSSSSSSSSSSSSSSSSSSSS");
    EXPECT_STYLE(e, 6, "SSS");
    EXPECT_STYLE(e, 7, "FFFFFFFFFFFFF");
    EXPECT_STYLE(e, 8, "CCCCCCC");
    EXPECT_STYLE(e, 9, "LLLL");
    EXPECT_STYLE(e, 10, "DDDDDDDDD");
    EXPECT_STYLE(e, 11, "OO");

    // fenced code block keeps its state until the closing fence
    EXPECT_GE(e.blockState(4), kFirstMultiLineState);
    EXPECT_GE(e.blockState(5), kFirstMultiLineState);
    EXPECT_EQ(e.blockState(6), 0);
}

//-------------------------------------------------------------------------------------
TEST(GenericSyntaxHighlighterTest, cmake)
{
    HighlightedEditor e(EditorFileType::CMake);
    e.setText(
        "# comment\n"
        "if(WIN32)\n"
        "  set(A ${B} \"s\" 3)\n"
        "endif()\n"
        "#[[ block\n"
        "comment ]] message(x)\n"
        "if(NOT a STREQUAL b)\n");

    ASSERT_EQ(e.lineCount(), 8);
    EXPECT_STYLE(e, 0, "CCCCCCCCC");
    EXPECT_STYLE(e, 1, "KK");
    EXPECT_STYLE(e, 2, "  FFF   XXXX SSS N");
    EXPECT_STYLE(e, 3, "KKKKK");
    EXPECT_STYLE(e, 4, "CCCCCCCCC");
    EXPECT_STYLE(e, 5, "CCCCCCCCCC FFFFFFF");
    EXPECT_STYLE(e, 6, "KK WWW   WWWWWWWW");

    EXPECT_GE(e.blockState(4), kFirstMultiLineState);
    EXPECT_EQ(e.blockState(5), 0);
}

//-------------------------------------------------------------------------------------
TEST(GenericSyntaxHighlighterTest, batch)
{
    HighlightedEditor e(EditorFileType::Batch);
    e.setText(
        "REM comment\n"
        "rem lower case\n"
        "@echo %PATH%\n"
        ":label\n"
        "if not exist \"f\" goto label\n"
        ":: c\n"
        "set x=%~dp0 %1 %%i %~nx1\n");

    ASSERT_EQ(e.lineCount(), 8);
    EXPECT_STYLE(e, 0, "CCCCCCCCCCC");
    EXPECT_STYLE(e, 1, "CCCCCCCCCCCCCC");
    EXPECT_STYLE(e, 2, " KKKK XXXXXX");
    EXPECT_STYLE(e, 3, "LLLLLL");
    EXPECT_STYLE(e, 4, "KK KKK KKKKK SSS KKKK");
    EXPECT_STYLE(e, 5, "CCCC");
    EXPECT_STYLE(e, 6, "KKK   XXXXX XX XXX XXXXX");
}

//-------------------------------------------------------------------------------------
TEST(GenericSyntaxHighlighterTest, multiLineStateIsResetAfterEdit)
{
    HighlightedEditor e(EditorFileType::Json);
    e.setText("/* a\nb\nc */ 1\n2");

    EXPECT_STYLE(e, 0, "CCCC");
    EXPECT_STYLE(e, 1, "C");
    EXPECT_STYLE(e, 2, "CCCC N");
    EXPECT_STYLE(e, 3, "N");

    // closing the comment in the first line must re-highlight the following lines
    e.setText("/* a */\nb\nc */ 1\n2");

    EXPECT_STYLE(e, 0, "CCCCCCC");
    EXPECT_STYLE(e, 1, "");
    EXPECT_STYLE(e, 2, "     N");
    EXPECT_EQ(e.blockState(0), 0);
    EXPECT_EQ(e.blockState(1), 0);
}

//-------------------------------------------------------------------------------------
TEST(GenericSyntaxHighlighterTest, veryLongLinesAreNotHighlighted)
{
    // e.g. minified json: lines longer than the internal limit are skipped
    QString longLine = "{";

    for (int i = 0; i < 2000; ++i)
    {
        longLine += "\"k\": 1, "; // 8 chars -> 16000 chars
    }

    longLine += "\"end\": 2}";

    HighlightedEditor e(EditorFileType::Json);
    e.setText(longLine + "\n{\"a\": 1}");

    EXPECT_STYLE(e, 0, "");
    EXPECT_EQ(e.blockState(0), 0);
    EXPECT_STYLE(e, 1, "PKKKP NP");
}

//-------------------------------------------------------------------------------------
TEST(GenericSyntaxHighlighterTest, fileTypeAndFoldDetector)
{
    HighlightedEditor e(EditorFileType::Yaml);
    EXPECT_EQ(e.highlighter()->fileType(), EditorFileType::Yaml);

    // folding by indentation for structured formats, none for the others
    EXPECT_FALSE(GenericSyntaxHighlighter::createFoldDetector(EditorFileType::Yaml).isNull());
    EXPECT_FALSE(GenericSyntaxHighlighter::createFoldDetector(EditorFileType::Json).isNull());
    EXPECT_FALSE(GenericSyntaxHighlighter::createFoldDetector(EditorFileType::Xml).isNull());
    EXPECT_TRUE(GenericSyntaxHighlighter::createFoldDetector(EditorFileType::Ini).isNull());
    EXPECT_TRUE(GenericSyntaxHighlighter::createFoldDetector(EditorFileType::Markdown).isNull());
    EXPECT_TRUE(GenericSyntaxHighlighter::createFoldDetector(EditorFileType::CMake).isNull());
    EXPECT_TRUE(GenericSyntaxHighlighter::createFoldDetector(EditorFileType::Batch).isNull());
    EXPECT_TRUE(GenericSyntaxHighlighter::createFoldDetector(EditorFileType::PlainText).isNull());
}

//-------------------------------------------------------------------------------------
TEST(GenericSyntaxHighlighterTest, yamlFoldLevels)
{
    // The indent fold detector derives the fold level from the indentation divided by
    // the tab length of the editor (default: 4). Therefore the folding of yaml files with
    // 2 spaces indentation is only detected at every second level (see second part).
    HighlightedEditor e(EditorFileType::Yaml);
    e.setText("root:\n    child: 1\n        grand: 2\nother: 3\n");

    QTextBlock b0 = e.editor()->document()->findBlockByNumber(0);
    QTextBlock b1 = e.editor()->document()->findBlockByNumber(1);
    QTextBlock b2 = e.editor()->document()->findBlockByNumber(2);
    QTextBlock b3 = e.editor()->document()->findBlockByNumber(3);

    EXPECT_LT(Utils::TextBlockHelper::getFoldLvl(b0), Utils::TextBlockHelper::getFoldLvl(b1));
    EXPECT_LT(Utils::TextBlockHelper::getFoldLvl(b1), Utils::TextBlockHelper::getFoldLvl(b2));
    EXPECT_EQ(Utils::TextBlockHelper::getFoldLvl(b0), Utils::TextBlockHelper::getFoldLvl(b3));
    EXPECT_TRUE(Utils::TextBlockHelper::isFoldTrigger(b0));
    EXPECT_TRUE(Utils::TextBlockHelper::isFoldTrigger(b1));
    EXPECT_FALSE(Utils::TextBlockHelper::isFoldTrigger(b3));

    // characterization of the current behavior with 2 spaces indentation
    e.setText("root:\n  child: 1\n    grand: 2\n");
    b0 = e.editor()->document()->findBlockByNumber(0);
    b1 = e.editor()->document()->findBlockByNumber(1);
    b2 = e.editor()->document()->findBlockByNumber(2);
    EXPECT_EQ(Utils::TextBlockHelper::getFoldLvl(b0), Utils::TextBlockHelper::getFoldLvl(b1));
    EXPECT_LT(Utils::TextBlockHelper::getFoldLvl(b1), Utils::TextBlockHelper::getFoldLvl(b2));
}
