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

#include "pythonSyntaxHighlighter.h"

// for Python version check
#include "patchlevel.h"

#include "../codeEditor.h"
#include <qapplication.h>
#include <qtextdocument.h>

#include "../managers/modesManager.h"
#include "../managers/panelsManager.h"
#include "../modes/caretLineHighlight.h"
#include "../utils/utils.h"

#include <qdebug.h>
#include <iostream>

namespace ito {

/*static*/ QList<PythonSyntaxHighlighter::NamedRegExp> PythonSyntaxHighlighter::regExpProg = PythonSyntaxHighlighter::makePythonPatterns();
/*static*/ QRegularExpression PythonSyntaxHighlighter::regExpIdProg = QRegularExpression("\\s+(\\w+)");
/*static*/ QRegularExpression PythonSyntaxHighlighter::regExpAsProg = QRegularExpression(".*?\\b(as)\\b");
/*static*/ QRegularExpression PythonSyntaxHighlighter::regExpOeComment = QRegularExpression("^(// ?--[-]+|##[#]+ )[ -]*[^- ]+");


//-------------------------------------------------------------------
PythonSyntaxHighlighter::PythonSyntaxHighlighter(QTextDocument *parent, const QString &description /*= ""*/, QSharedPointer<CodeEditorStyle> editorStyle /*= QSharedPointer<CodeEditorStyle>()*/ ) :
    SyntaxHighlighterBase("PythonSyntaxHighlighter", parent, description, editorStyle)
{

}

//-------------------------------------------------------------------
PythonSyntaxHighlighter::~PythonSyntaxHighlighter()
{
}

//-------------------------------------------------------------------
const QTextCharFormat PythonSyntaxHighlighter::getTextCharFormat(const QString &colorName, const QString &style)
{
    QTextCharFormat charFormat;
    QColor color(colorName);
    charFormat.setForeground(color);
    if (style.contains("bold", Qt::CaseInsensitive))
    {
        charFormat.setFontWeight(QFont::Bold);
    }
    if (style.contains("italic", Qt::CaseInsensitive))
    {
        charFormat.setFontItalic(true);
    }
    return charFormat;
}


QList<QPair<QRegularExpressionMatch, QStringList> >::const_iterator
hasNextMatch(const QList<QPair<QRegularExpressionMatch, QStringList> > &matches, QString &captureGroup)
{
    QList<QPair<QRegularExpressionMatch, QStringList> >::const_iterator it_dest = matches.constEnd();
    QList<QPair<QRegularExpressionMatch, QStringList> >::const_iterator it = matches.constBegin();
    bool hasMatch = false;
    int pos = INT_MAX;

    while (it != matches.constEnd())
    {
        if (it->first.hasMatch())
        {
            hasMatch = true;
            foreach (const QString &cg, it->second)
            {
                if (it->first.capturedStart(cg) < pos)
                {
                    pos = it->first.capturedStart(cg);
                    captureGroup = cg;
                    it_dest = it;
                }
            }
        }
        it++;
    }

    return it_dest;
}


void PythonSyntaxHighlighter::default_highlight_block(const QString &text, bool outputNotError)
{
    if (outputNotError)
    {
        setFormat(0, text.size(), getFormatFromStyle(StyleItem::KeyStreamOutput));
    }
    else
    {
        setFormat(0, text.size(), getFormatFromStyle(StyleItem::KeyStreamError));
    }
}
//-------------------------------------------------------------------
/*
Highlights the block using a pygments lexer.

:param text: text of the block to highlight
:param block: block to highlight
*/
void PythonSyntaxHighlighter::highlight_block(const QString &text, QTextBlock &block)
{
    QTextBlock prev_block = block.previous();
    int prev_state = Utils::TextBlockHelper::getState(prev_block);
    int offset;
    QString text2 = text;
    if (text2.count(QChar('\\')) > 100)
    {
        text2 = text2.left(250); //if text is too long, the regular expressions can cause a stack overflow! (especially the string expressions)
    }

    if (prev_state == InsideDq3String)
    {
        offset = -4;
        text2 = "\"\"\" " + text;
    }
    else if (prev_state == InsideSq3String)
    {
        offset = -4;
        text2 = "''' " + text;
    }
    else if (prev_state == InsideDqString)
    {
        offset = -2;
        text2 = "\" " + text;
    }
    else if (prev_state == InsideSqString)
    {
        offset = -2;
        text2 = "' " + text;
    }
    else
    {
        offset = 0;
    }

    setFormat(0, text2.size(), getFormatFromStyle(StyleItem::KeyDefault));

    State state = Normal;


    QString key;
    int pos = 0;
    int length;
    int start;
    int end;
    QString value;

    QList<QPair<QRegularExpressionMatch, QStringList> > matches;
    QList<QPair<QRegularExpressionMatch, QStringList> >::const_iterator match_iter;

    matches.reserve(regExpProg.size());

    try
    {
        for (int i = 0; i < regExpProg.size(); ++i)
        {
            matches.append(QPair<QRegularExpressionMatch, QStringList>(regExpProg[i].regExp.match(text2, 0), regExpProg[i].groupNames));
        }
    }
    catch (...)
    {
        std::cerr << "Stack overflow in regular expression (syntax highlighter).\n" << std::endl;
        return;
    }

    while ((match_iter = hasNextMatch(matches, key)) != matches.constEnd())
    {
        {
            start = match_iter->first.capturedStart(key);
            value = match_iter->first.captured(key);
            end = match_iter->first.capturedEnd(key);
            pos = std::max(0, end);
            start = std::max(0, start + offset);
            end = std::max(0, end + offset);
            length = match_iter->first.capturedLength(key);

            //qDebug() << key << start << end << value.toHtmlEscaped() << QString::number(value.right(1)[0].cell(),16) << QString::number(value.right(1)[0].row(),16);

            if (key == "uf_sq3string")
            {
                setFormat(start, length,
                    getFormatFromStyle(StyleItem::KeyDocstring));
                state = InsideSq3String;
            }
            else if (key == "uf_dq3string")
            {
                setFormat(start, length,
                    getFormatFromStyle(StyleItem::KeyDocstring));
                state = InsideDq3String;
            }
            else if (key == "uf_sqstring")
            {
                setFormat(start, length,
                    getFormatFromStyle(StyleItem::KeyString));
                state = InsideSqString;
            }
            else if (key == "uf_dqstring")
            {
                setFormat(start, length,
                    getFormatFromStyle(StyleItem::KeyString));
                state = InsideDqString;
            }
            else if (key == "builtin_fct")
            {
                //trick to highlight __init__, __add__ and so on with
                //builtin color
                setFormat(start, length,
                    getFormatFromStyle(StyleItem::KeyConstant));
            }
            else
            {
                if ((value.contains("\"\"\"") || value.contains("'''")) && key != "comment")
                {
                    // highlight docstring with a different color
                    setFormat(start, length,
                        getFormatFromStyle(StyleItem::KeyDocstring));
                }
                else if (key == "decorator")
                {
                    // highlight decorators
                    setFormat(start, length,
                        getFormatFromStyle(StyleItem::KeyDecorator));
                }
                else if (value == "self" || value == "cls")
                {
                    // highlight self attribute
                    setFormat(start, length,
                        getFormatFromStyle(StyleItem::KeySelf));
                }
                else if (key == "number")
                {
                    setFormat(start, length,
                        getFormatFromStyle(StyleItem::KeyNumber));
                }
                else if (key == "keyword")
                {
                    setFormat(start, length,
                        getFormatFromStyle(StyleItem::KeyKeyword));
                }
                else if (key == "comment")
                {
                    setFormat(start, length,
                        getFormatFromStyle(StyleItem::KeyComment));
                }
                else if (key == "operator_word")
                {
                    setFormat(start, length,
                        getFormatFromStyle(StyleItem::KeyOperatorWord));
                }
                else if (key == "string")
                {
                    setFormat(start, length,
                        getFormatFromStyle(StyleItem::KeyString));
                }
                else if (key == "namespace")
                {
                    setFormat(start, length,
                        getFormatFromStyle(StyleItem::KeyNamespace));
                }
                else if (key == "builtin")
                {
                    setFormat(start, length,
                        getFormatFromStyle(StyleItem::KeyBuiltin));
                }
                else
                {
                    // highlight all other tokens
                    setFormat(start, length,
                        getFormatFromStyle(StyleItem::KeyTag /*key*/)); //TODO
                }

                if (key == "keyword")
                {
                    if (value == "def" || value == "class")
                    {
                        auto match = PythonSyntaxHighlighter::regExpIdProg.match(text2, end);

                        if (match.hasMatch())
                        {
                            int start1 = match.capturedStart();
                            int end1 = match.capturedEnd();
                            QTextCharFormat fmt = getFormatFromStyle(value == "class" ? StyleItem::KeyClass : StyleItem::KeyFunction);
                            setFormat(start1, end1 - start1, fmt);
                        }
                    }
                }
                else if (key == "namespace")
                {
                    // color all the "as" words on same line, except
                    // if in a comment; cheap approximation to the
                    // truth
                    int endpos;
                    QString text2stripped;

                    if (text2.contains('#'))
                    {
                        endpos = text2.indexOf('#');
                        text2stripped = text2.left(endpos);
                    }
                    else
                    {
                        endpos = text2.size();
                        text2stripped = text2;
                    }

                    QRegularExpressionMatch match;

                    while (true)
                    {
                        match = PythonSyntaxHighlighter::regExpAsProg.match(text2stripped, end);

                        if (!match.hasMatch())
                        {
                            break;
                        }

                        start = match.capturedStart(1);
                        end = match.capturedEnd(1);
                        QTextCharFormat fmt = getFormatFromStyle(StyleItem::KeyNamespace);
                        setFormat(start, end - start, fmt);
                    }
                }
            }
        }

        for (int i = 0; i < regExpProg.size(); ++i)
        {
            matches[i].first = regExpProg[i].regExp.match(text2, pos);
        }
    }

    Utils::TextBlockHelper::setState(block, state);
}



//-------------------------------------------------------------------
/*
Returns a QTextCharFormat for token
*/
QTextCharFormat PythonSyntaxHighlighter::getFormatFromStyle(StyleItem::StyleType token) const
{
    return m_editorStyle->format(token);
}

//--------------------------------------------------------------------
/*virtual*/ void PythonSyntaxHighlighter::rehighlight()
{
}


//----------------------------------------------------------------
QString any(const QString &name, const QStringList &alternates)
{
    //Return a named group pattern matching list of alternates.
    return QString("(?<%1>%2)").arg(name).arg(alternates.join("|"));
}


//-------------------------------------------------------------------------------------
/*static*/ QList<PythonSyntaxHighlighter::NamedRegExp> PythonSyntaxHighlighter::makePythonPatterns(
    const QStringList &additionalKeywords,
    const QStringList &additionalBuiltins)
{
    QList<NamedRegExp> regExpressions;

    QStringList kwlist = QStringList() << "self"
                                       << "False"
                                       << "None"
                                       << "True"
                                       << "assert"
                                       << "break"
                                       << "class"
                                       << "continue"
                                       << "def"
                                       << "del"
                                       << "elif"
                                       << "else"
                                       << "except"
                                       << "finally"
                                       << "for"
                                       << "global"
                                       << "if"
                                       << "lambda"
                                       << "nonlocal"
                                       << "pass"
                                       << "raise"
                                       << "return"
                                       << "try"
                                       << "while"
                                       << "with"
                                       << "yield"
                                       << "async"
                                       << "await";

#if (PY_VERSION_HEX >= 0x030A0000)
    kwlist << "match"
           << "case"; // match/case is new in Python 3.10
#endif

    QStringList kwNamespaceList = QStringList() << "from" << "import" << "as";
    QStringList wordopList = QStringList() << "and" << "or" << "not" << "in" << "is";

    //Strongly inspired from idlelib.ColorDelegator.make_pat

    // a keyword is one of the given lists, bounded by a word boundary on both sides
    QString kw = "\\b" + any("keyword", kwlist + additionalKeywords) + "\\b";

    // the same holds for namespace keywords
    QString kw_namespace = "\\b" + any("namespace", kwNamespaceList) + "\\b";

    // ... and for text operators
    QString word_operators = "\\b" + any("operator_word", wordopList) + "\\b";

    //TODO: obtain the following list by the following python script:
    /*
    import builtins
    text = ["\"%s\"" % str(name) for name in dir(builtins) if not name.startswith('_')]
    print("QStringList() << " + " << ".join(text) + ";")
    */
    //The following builtins are based on Python 3.7
    QStringList builtinlist = QStringList() << "ArithmeticError" << "AssertionError" << "AttributeError" << \
        "BaseException" << "BlockingIOError" << "BrokenPipeError" << "BufferError" << "BytesWarning" <<
        "ChildProcessError" << "ConnectionAbortedError" << "ConnectionError" << "ConnectionRefusedError" << \
        "ConnectionResetError" << "DeprecationWarning" << "EOFError" << "Ellipsis" << "EnvironmentError" << \
        "Exception" << "False" << "FileExistsError" << "FileNotFoundError" << "FloatingPointError" << "FutureWarning" << \
        "GeneratorExit" << "IOError" << "ImportError" << "ImportWarning" << "IndentationError" << "IndexError" << \
        "InterruptedError" << "IsADirectoryError" << "KeyError" << "KeyboardInterrupt" << "LookupError" << "MemoryError" << \
        "ModuleNotFoundError" << "NameError" << "None" << "NotADirectoryError" << "NotImplemented" << "NotImplementedError" << "OSError" << "OverflowError" << \
        "PendingDeprecationWarning" << "PermissionError" << "ProcessLookupError" << "RecursionError" << "ReferenceError" << "ResourceWarning" << "RuntimeError" << \
        "RuntimeWarning" << "StopAsyncIteration" << "StopIteration" << "SyntaxError" << "SyntaxWarning" << "SystemError" << "SystemExit" << "TabError" << \
        "TimeoutError" << "True" << "TypeError" << "UnboundLocalError" << "UnicodeDecodeError" << "UnicodeEncodeError" << "UnicodeError" << \
        "UnicodeTranslateError" << "UnicodeWarning" << "UserWarning" << "ValueError" << "Warning" << "WindowsError" << "ZeroDivisionError" << \
        "abs" << "all" << "any" << "ascii" << "bin" << "bool" << "breakpoint" << "bytearray" << "bytes" << "callable" << "chr" << "classmethod" << \
        "compile" << "complex" << "copyright" << "credits" << "delattr" << "dict" << "dir" << "divmod" << "enumerate" << "eval" << \
        "exec" << "exit" << "filter" << "float" << "format" << "frozenset" << "getattr" << "globals" << "hasattr" << "hash" << "help" << \
        "hex" << "id" << "input" << "int" << "isinstance" << "issubclass" << "iter" << "len" << "license" << "list" << "locals" << "map" << \
        "max" << "memoryview" << "min" << "next" << "object" << "oct" << "open" << "ord" << "pow" << "print" << "property" << "quit" << \
        "range" << "repr" << "reversed" << "round" << "set" << "setattr" << "slice" << "sorted" << "staticmethod" << "str" << "sum" << \
        "super" << "tuple" << "type" << "vars" << "zip";

    builtinlist << additionalBuiltins;

    // None, True and False should be a keyword, not a builtin, remove them...
    builtinlist.removeAll("None");
    builtinlist.removeAll("True");
    builtinlist.removeAll("False");

    QString builtin = "([^.'\"\\#]\\b|^)" + any("builtin", builtinlist) + "\\b";
    QString builtin_fct = any("builtin_fct", QStringList("_{2}[a-zA-Z0-9_]*_{2}"));

    // a comment is a hash (#) followed by any character but a newline (^\\n negates the newline)
    QString comment = any("comment", QStringList("#[^\\n]*"));

    // the instance keyword is either self or cls (word boundary on both sides)
    QString instance = any("instance", QStringList("\\bself\\b") << "\\bcls\\b");

    /* decorators:

    - each decorator is in its own line
    - there can be an arbitrary number of spaces (or tabs) at the beginning
    - then the @ (at) sign follows
    - then there must be an identifier.

    An identifier is defined by the expression in the rounded brackets.

    Officially, it is also allowed to have spaces / tabs between @ and
    identifier, however we don't allow this here in order to reserve
    this for the @ operator (like +, -, /, *...).
    */
    QString decorator = any("decorator", QStringList("^[\\s]*@([a-zA-Z_][\\w\\.]*)?"));

    //some fixes: 1. numbers with postfix l or L does not exist any more in python 3;
    // 2. the order of the following number entries is relevant
    // 3. if dots are included in the regex, no \\b can be used, since the dot is also a word boundary

    //using lookbehind
    // trailing +/- is not part of the number
    QString number = any("number", QStringList() <<
            "(?<![a-zA-Z\\)\\]\\}\\.0-9_])\\.?([0-9](_?[0-9])*)(?:\\.([0-9]?(_?[0-9])*))?(?:[eE][+-]?([0-9](_?[0-9])*))?[jJ]?" << // float + complex
            "\\b0[xX](?:_?[0-9A-Fa-f])+" << // hex integer
            "\\b0[oO](?:_?[0-7])+" <<       // oct integer
            "\\b0[bB](?:_?[01])+" <<        // bin integer
            "\\b[1-9](?:_?[0-9])*" <<       // decimal integer (non-zero)
            "\\b0(?:_?0)*")                 // decimal integer (zero)
        + "\\b";

    QString prefix = "r|u|R|U|f|F|fr|Fr|fR|FR|rf|rF|Rf|RF|b|B|br|Br|bR|BR|rb|rB|Rb|RB";
                                                                              //"(\\b(b|u))?'[^'\\\\\\n]*(\\\\.[^'\\\\\\n]*)*'?"
    QString sqstring =     QString("(\\b(%1))?'[^'\\\\\\n]*(\\\\.[^'\\\\\\n]*)*'?").arg(prefix); //"(\\b(%1))?'[^'\\\\\\n]*(\\\\.[^'\\\\\\n]*)*'?";
    QString dqstring =     QString("(\\b(%1))?\"[^\"\\\\\\n]*(\\\\.[^\"\\\\\\n]*)*\"?").arg(prefix); //"(\\b(%1))?\"[^\"\\\\\\n]*(\\\\.[^\"\\\\\\n]*)*\"?";
    QString uf_sqstring =  QString("(\\b(%1))?'[^'\\\\\\n]*(\\\\.[^'\\\\\\n]*)*(\\\\)\\n(?!')$").arg(prefix);
    QString uf_dqstring =  QString("(\\b(%1))?\"[^\"\\\\\\n]*(\\\\.[^\"\\\\\\n]*)*(\\\\)\\n(?!\")$").arg(prefix);
    QString sq3string =    QString("(\\b(%1))?'''[^'\\\\]*((\\\\.|'(?!''))[^'\\\\]*)*(''')?").arg(prefix);
    QString dq3string =    QString("(\\b(%1))?\"\"\"[^\"\\\\]*((\\\\.|\"(?!\"\"))[^\"\\\\]*)*(\"\"\")?").arg(prefix);
    QString uf_sq3string = QString("(\\b(%1))?'''[^'\\\\]*((\\\\.|'(?!''))[^'\\\\]*)*(\\\\)?(?!''')$").arg(prefix);
    QString uf_dq3string = QString("(\\b(%1))?\"\"\"[^\"\\\\]*((\\\\.|\"(?!\"\"))[^\"\\\\]*)*(\\\\)?(?!\"\"\")$").arg(prefix);
    QString string = any("string", QStringList() << sq3string << dq3string << sqstring << dqstring);
    //string = any("string", QStringList() << sq3string << sqstring);
    QString ufstring1 = any("uf_sqstring", QStringList(uf_sqstring));
    QString ufstring2 = any("uf_dqstring", QStringList(uf_dqstring));
    QString ufstring3 = any("uf_sq3string", QStringList(uf_sq3string));
    QString ufstring4 = any("uf_dq3string", QStringList(uf_dq3string));

#if 0 //QT_VERSION > MIN_QT_REGULAREXPRESSION_VERSION
    QStringList all;
    all << instance << decorator << kw << kw_namespace << builtin << word_operators << builtin_fct << comment;
    all << ufstring1 << ufstring2 << ufstring3 << ufstring4 << string << number << any("SNYC", QStringList("\\n"));
    QRegularExpression regExp = QRegularExpression(all.join("|"));
    regExpressions.append(NamedRegExp(regExp.namedCaptureGroups(), regExp));
#else
    regExpressions.append(NamedRegExp("instance", QRegularExpression(instance)));
    regExpressions.append(NamedRegExp("decorator", QRegularExpression(decorator)));
    regExpressions.append(NamedRegExp("keyword", QRegularExpression(kw)));
    regExpressions.append(NamedRegExp("namespace", QRegularExpression(kw_namespace)));
    regExpressions.append(NamedRegExp("builtin", QRegularExpression(builtin)));
    regExpressions.append(NamedRegExp("operator_word", QRegularExpression(word_operators)));
    regExpressions.append(NamedRegExp("builtin_fct", QRegularExpression(builtin_fct)));
    regExpressions.append(NamedRegExp("comment", QRegularExpression(comment)));
    regExpressions.append(NamedRegExp("uf_sqstring", QRegularExpression(ufstring1)));
    regExpressions.append(NamedRegExp("uf_dqstring", QRegularExpression(ufstring2)));
    regExpressions.append(NamedRegExp("uf_sq3string", QRegularExpression(ufstring3)));
    regExpressions.append(NamedRegExp("uf_dq3string", QRegularExpression(ufstring4)));
    regExpressions.append(NamedRegExp("string", QRegularExpression(string)));
    regExpressions.append(NamedRegExp("number", QRegularExpression(number)));
    regExpressions.append(NamedRegExp("SYNC", QRegularExpression(any("SYNC", QStringList("\\n")))));
#endif

    return regExpressions;
}




} //end namespace ito
