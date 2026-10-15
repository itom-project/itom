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

    Further hints:
    ------------------------

    This file belongs to the code editor of itom. The code editor is
    in major parts a fork / rewritten version of the python-based source
    code editor PyQode from Colin Duquesnoy and others
    (see https://github.com/pyQode). PyQode itself is licensed under
    the MIT License (MIT).

*********************************************************************** */

#include "genericSyntaxHighlighter.h"

#include "../foldDetector/indentFoldDetector.h"
#include "../textBlockUserData.h"
#include "../utils/utils.h"

#include <climits>
#include <qvector.h>

namespace ito {

//! lines longer than this are not highlighted (e.g. minified json) to keep the editor responsive
static const int MaxHighlightedLineLength = 10000;

//-------------------------------------------------------------------------------------
GenericSyntaxHighlighter::GenericSyntaxHighlighter(
    EditorFileType::Type type,
    QTextDocument* parent,
    QSharedPointer<CodeEditorStyle> editorStyle /*= QSharedPointer<CodeEditorStyle>()*/) :
    SyntaxHighlighterBase("GenericSyntaxHighlighter", parent, "", editorStyle),
    m_type(type)
{
    setupRules();
}

//-------------------------------------------------------------------------------------
GenericSyntaxHighlighter::~GenericSyntaxHighlighter()
{
}

//-------------------------------------------------------------------------------------
/*static*/ QSharedPointer<FoldDetector> GenericSyntaxHighlighter::createFoldDetector(
    EditorFileType::Type type)
{
    switch (type)
    {
    case EditorFileType::Yaml:
    case EditorFileType::Json:
    case EditorFileType::Xml:
        return QSharedPointer<FoldDetector>(new IndentFoldDetector());
    default:
        return QSharedPointer<FoldDetector>();
    }
}

//-------------------------------------------------------------------------------------
void GenericSyntaxHighlighter::addKeywords(
    const QStringList& words, StyleItem::StyleType style, bool caseInsensitive /*= false*/)
{
    Rule r(QString("\\b(?:%1)\\b").arg(words.join("|")), style);

    if (caseInsensitive)
    {
        r.regExp.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
    }

    m_rules.append(r);
}

//-------------------------------------------------------------------------------------
// Rules are evaluated like the python highlighter: the match that starts first wins. If
// two rules start at the same column, the rule that has been added first wins. Therefore,
// more specific rules must be added before more general rules.
void GenericSyntaxHighlighter::setupRules()
{
    const QString dqString = "\"(?:[^\"\\\\]|\\\\.)*\"";
    const QString sqString = "'(?:[^'\\\\]|\\\\.)*'";
    const QString number = "(?<![\\w.])[-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?(?![\\w.])";

    switch (m_type)
    {
    case EditorFileType::Json:
        addRule(dqString + "(?=\\s*:)", StyleItem::KeyKeyword); // keys
        addRule(dqString, StyleItem::KeyString);
        addRule(number, StyleItem::KeyNumber);
        addRule("\\b(?:true|false|null)\\b", StyleItem::KeyConstant);
        addRule("[{}\\[\\],:]", StyleItem::KeyPunctuation);
        // json with comments (jsonc)
        addRule("//.*", StyleItem::KeyComment);
        m_multiLines.append(MultiLine("/\\*", "\\*/", StyleItem::KeyComment));
        break;
    case EditorFileType::Yaml:
        addRule("(?<!\\S)#.*", StyleItem::KeyComment);
        addRule("^(?:---|\\.\\.\\.)(?=\\s|$)", StyleItem::KeyTag);
        addRule(
            "^\\s*(?:-\\s+)*((?:\"[^\"]*\"|'[^']*'|[^\\s#:\\-'\"\\[\\]{},&*!|>%@`][^:#]*?|-[^\\s:#][^:#]*?))\\s*:(?=\\s|$)",
            StyleItem::KeyKeyword,
            1);
        addRule("^\\s*-(?=\\s|$)", StyleItem::KeyPunctuation);
        addRule(dqString, StyleItem::KeyString);
        addRule("'(?:[^']|'')*'", StyleItem::KeyString);
        addRule("[&*][\\w-]+", StyleItem::KeyDecorator); // anchors and aliases
        addRule("![\\w!<>:/.-]*", StyleItem::KeyDecorator); // tags
        addRule(number, StyleItem::KeyNumber);
        addRule("(?<![\\w-])(?:true|false|null|yes|no|on|off|~)(?![\\w-])", StyleItem::KeyConstant);
        break;
    case EditorFileType::Xml:
        m_multiLines.append(MultiLine("<!--", "-->", StyleItem::KeyComment));
        m_multiLines.append(MultiLine("<!\\[CDATA\\[", "\\]\\]>", StyleItem::KeyString));
        addRule("<\\?.*?\\?>", StyleItem::KeyDecorator);
        addRule("<!DOCTYPE[^>]*>", StyleItem::KeyDecorator);
        addRule("</?[\\w:.-]+|/?>", StyleItem::KeyTag);
        addRule("=\\s*(\"[^\"]*\"|'[^']*')", StyleItem::KeyString, 1);
        addRule("(?<=\\s)[\\w:.-]+(?=\\s*=)", StyleItem::KeyBuiltin);
        addRule("&#?\\w+;", StyleItem::KeyConstant);
        break;
    case EditorFileType::Ini:
        addRule("^\\s*[;#].*", StyleItem::KeyComment);
        addRule("^\\s*\\[[^\\]]*\\]", StyleItem::KeyClass);
        addRule("^\\s*([^=:;#\\[\\s][^=:]*?)\\s*(?==|:)", StyleItem::KeyKeyword, 1);
        addRule("\\s[;#].*", StyleItem::KeyComment);
        addRule(dqString, StyleItem::KeyString);
        addRule(sqString, StyleItem::KeyString);
        addRule(number, StyleItem::KeyNumber);
        addRule("\\b(?:true|false|yes|no|on|off)\\b", StyleItem::KeyConstant);
        break;
    case EditorFileType::Markdown:
        m_multiLines.append(MultiLine("^\\s*(?:```|~~~)", "^\\s*(?:```|~~~)", StyleItem::KeyString));
        addRule("^#{1,6}\\s.*", StyleItem::KeyClass); // markdown headlines
        addRule("^(?:={3,}|-{3,}|~{3,}|\\^{3,}|\"{3,}|\\+{3,})\\s*$", StyleItem::KeyClass); // rst
        addRule("^\\.\\.\\s.*", StyleItem::KeyDecorator); // rst directives and comments
        addRule("^\\s*>.*", StyleItem::KeyComment);
        addRule("`[^`]+`", StyleItem::KeyString);
        addRule("\\*\\*[^*]+\\*\\*|__[^_]+__", StyleItem::KeyKeyword);
        addRule("!?\\[[^\\]]*\\]\\([^)]*\\)", StyleItem::KeyFunction);
        addRule("^\\s*(?:[-*+]|\\d+[.)])(?=\\s)", StyleItem::KeyOperator);
        addRule("<https?://[^>]+>|https?://[^\\s)]+", StyleItem::KeyFunction);
        break;
    case EditorFileType::CMake:
        m_multiLines.append(MultiLine("#\\[=*\\[", "\\]=*\\]", StyleItem::KeyComment));
        addRule("#.*", StyleItem::KeyComment);
        addRule(dqString, StyleItem::KeyString);
        addRule("\\$(?:ENV|CACHE)?\\{[^}]*\\}", StyleItem::KeyConstant);
        addKeywords(
            QStringList() << "if"
                          << "else"
                          << "elseif"
                          << "endif"
                          << "foreach"
                          << "endforeach"
                          << "while"
                          << "endwhile"
                          << "function"
                          << "endfunction"
                          << "macro"
                          << "endmacro"
                          << "block"
                          << "endblock"
                          << "return"
                          << "break"
                          << "continue",
            StyleItem::KeyKeyword,
            true);
        addRule("\\b[A-Za-z_]\\w*(?=\\s*\\()", StyleItem::KeyFunction);
        addRule(
            "\\b(?:AND|OR|NOT|STREQUAL|EQUAL|MATCHES|DEFINED|EXISTS|IN_LIST|VERSION_LESS|"
            "VERSION_GREATER|VERSION_EQUAL|ON|OFF|TRUE|FALSE)\\b",
            StyleItem::KeyOperatorWord);
        addRule(number, StyleItem::KeyNumber);
        break;
    case EditorFileType::Batch:
        addRule("^\\s*@?(?:rem\\b|::).*", StyleItem::KeyComment);
        m_rules.last().regExp.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
        addRule("^\\s*:[\\w.-]+", StyleItem::KeyClass); // labels
        addRule(dqString, StyleItem::KeyString);
        addRule("%[\\w~:*.]+%|%%?~?[\\w*]|![\\w:~]+!", StyleItem::KeyConstant);
        addKeywords(
            QStringList() << "if"
                          << "else"
                          << "for"
                          << "in"
                          << "do"
                          << "goto"
                          << "call"
                          << "set"
                          << "echo"
                          << "exit"
                          << "not"
                          << "exist"
                          << "defined"
                          << "errorlevel"
                          << "setlocal"
                          << "endlocal"
                          << "pause"
                          << "shift"
                          << "start"
                          << "cd"
                          << "copy"
                          << "del"
                          << "mkdir"
                          << "rmdir"
                          << "move"
                          << "type",
            StyleItem::KeyKeyword,
            true);
        addRule(number, StyleItem::KeyNumber);
        break;
    default:
        // plain text: no rules. Only the whitespace highlighting of the base class is applied.
        break;
    }
}

//-------------------------------------------------------------------------------------
void GenericSyntaxHighlighter::highlight_block(const QString& text, QTextBlock& block)
{
    QTextBlock prevBlock = block.previous();
    int prevState = prevBlock.isValid() ? Utils::TextBlockHelper::getState(prevBlock) : 0;
    int state = 0;
    int pos = 0;
    const int len = text.size();

    setFormat(0, len, getFormatFromStyle(StyleItem::KeyDefault));

    if ((m_rules.isEmpty() && m_multiLines.isEmpty()) || len > MaxHighlightedLineLength)
    {
        Utils::TextBlockHelper::setState(block, 0);
        return;
    }

    // continue a multi line construct of the previous block
    if (prevState >= FirstMultiLineState &&
        prevState - FirstMultiLineState < m_multiLines.size())
    {
        const MultiLine& ml = m_multiLines.at(prevState - FirstMultiLineState);
        QRegularExpressionMatch endMatch = ml.endExp.match(text, 0);

        if (!endMatch.hasMatch())
        {
            setFormat(0, len, getFormatFromStyle(ml.style));
            Utils::TextBlockHelper::setState(block, prevState);
            return;
        }

        pos = endMatch.capturedEnd();
        setFormat(0, pos, getFormatFromStyle(ml.style));
    }

    const int nrRules = m_rules.size();
    const int nrMulti = m_multiLines.size();
    QVector<QRegularExpressionMatch> ruleMatches(nrRules);
    QVector<int> ruleStarts(nrRules, -2); // -2: not yet evaluated, -1: no (further) match
    QVector<QRegularExpressionMatch> multiMatches(nrMulti);
    QVector<int> multiStarts(nrMulti, -2);

    while (pos < len)
    {
        int bestStart = INT_MAX;
        int bestRule = -1;
        bool bestIsMulti = false;

        for (int i = 0; i < nrRules; ++i)
        {
            if (ruleStarts[i] == -2 || (ruleStarts[i] >= 0 && ruleStarts[i] < pos))
            {
                ruleMatches[i] = m_rules[i].regExp.match(text, pos);
                int s = ruleMatches[i].hasMatch() ? ruleMatches[i].capturedStart(m_rules[i].group)
                                                  : -1;
                // groups that did not participate or empty matches are ignored
                if (s >= 0 && ruleMatches[i].capturedLength(m_rules[i].group) <= 0)
                {
                    s = -1;
                }
                ruleStarts[i] = s;
            }

            if (ruleStarts[i] >= 0 && ruleStarts[i] < bestStart)
            {
                bestStart = ruleStarts[i];
                bestRule = i;
                bestIsMulti = false;
            }
        }

        for (int i = 0; i < nrMulti; ++i)
        {
            if (multiStarts[i] == -2 || (multiStarts[i] >= 0 && multiStarts[i] < pos))
            {
                multiMatches[i] = m_multiLines[i].beginExp.match(text, pos);
                multiStarts[i] = multiMatches[i].hasMatch() ? multiMatches[i].capturedStart() : -1;
            }

            // multi line constructs win against single line rules at the same column
            if (multiStarts[i] >= 0 && multiStarts[i] <= bestStart)
            {
                bestStart = multiStarts[i];
                bestRule = i;
                bestIsMulti = true;
            }
        }

        if (bestRule < 0)
        {
            break;
        }

        if (bestIsMulti)
        {
            const MultiLine& ml = m_multiLines.at(bestRule);
            const QRegularExpressionMatch& beginMatch = multiMatches.at(bestRule);
            int beginEnd = beginMatch.capturedEnd();
            QRegularExpressionMatch endMatch = ml.endExp.match(text, beginEnd);

            if (endMatch.hasMatch())
            {
                int end = endMatch.capturedEnd();
                setFormat(bestStart, end - bestStart, getFormatFromStyle(ml.style));
                pos = end;
            }
            else
            {
                setFormat(bestStart, len - bestStart, getFormatFromStyle(ml.style));
                state = FirstMultiLineState + bestRule;
                break;
            }
        }
        else
        {
            const Rule& rule = m_rules.at(bestRule);
            const QRegularExpressionMatch& m = ruleMatches.at(bestRule);
            int length = m.capturedLength(rule.group);
            setFormat(bestStart, length, getFormatFromStyle(rule.style));
            pos = bestStart + length;
        }
    }

    Utils::TextBlockHelper::setState(block, state);
}

//-------------------------------------------------------------------------------------
void GenericSyntaxHighlighter::default_highlight_block(
    const QString& text, const TextBlockUserData* textBlockUserData)
{
    bool outputNotError = textBlockUserData->m_syntaxStyle == TextBlockUserData::StyleOutput;
    setFormat(
        0,
        text.size(),
        getFormatFromStyle(outputNotError ? StyleItem::KeyStreamOutput : StyleItem::KeyStreamError));
}

//-------------------------------------------------------------------------------------
QTextCharFormat GenericSyntaxHighlighter::getFormatFromStyle(StyleItem::StyleType token) const
{
    return m_editorStyle->formatWithFontSizeOffset(token);
}

//-------------------------------------------------------------------------------------
/*virtual*/ void GenericSyntaxHighlighter::rehighlight()
{
}

} // end namespace ito
