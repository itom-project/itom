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

#include "abstractCodeEditorWidget.h"
#include "../global.h"
#include "../AppManagement.h"
#include "../helper/guiHelper.h"
#include "../codeEditor/foldDetector/indentFoldDetector.h"
#include "../codeEditor/syntaxHighlighter/pythonSyntaxHighlighter.h"
#include "../codeEditor/modes/occurrences.h"
#include "../codeEditor/managers/modesManager.h"
#include "../codeEditor/modes/pyAutoIndent.h"
#include "../codeEditor/modes/indenter.h"
#include "../codeEditor/syntaxHighlighter/codeEditorStyle.h"

#include <qstring.h>
#include <qsettings.h>
#include <qdebug.h>
#include <qcolor.h>
#include <qfont.h>
#include <qtooltip.h>
#include <qclipboard.h>
#include <qapplication.h>
#include <qmimedata.h>
#include <qregularexpression.h>



namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
AbstractCodeEditorWidget::AbstractCodeEditorWidget(QWidget* parent) :
    CodeEditor(parent),
    m_userSelectionState(selNo)
{
    init();
    reloadSettings();

    connect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(reloadSettings()));
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractCodeEditorWidget::~AbstractCodeEditorWidget()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractCodeEditorWidget::init()
{
    m_editorStyle = QSharedPointer<CodeEditorStyle>(new CodeEditorStyle());
    //add python syntax highlighter
    m_pythonSyntaxHighlighter = QSharedPointer<SyntaxHighlighterBase>(new PythonSyntaxHighlighter(document(), "PythonSyntaxHighlighter", m_editorStyle));
    m_pythonSyntaxHighlighter->setFoldDetector(QSharedPointer<FoldDetector>(new IndentFoldDetector()));
    modes()->append(m_pythonSyntaxHighlighter.dynamicCast<Mode>());

    OccurrencesHighlighterMode *occHighlighterMode = new OccurrencesHighlighterMode("OccurrencesHighlighterMode");
    occHighlighterMode->setBackground(Qt::green);
    occHighlighterMode->setCaseSensitive(true);
    occHighlighterMode->setSelectOnDoubleClick(true);
    occHighlighterMode->setDelay(100);
    modes()->append(Mode::Ptr(occHighlighterMode));

    m_codeCompletionMode = QSharedPointer<CodeCompletionMode>(new CodeCompletionMode("CodeCompletionMode"));
    modes()->append(Mode::Ptr(m_codeCompletionMode.dynamicCast<Mode>()));

    m_calltipsMode = QSharedPointer<PyCalltipsMode>(new PyCalltipsMode("CalltipsMode", "", this));
    modes()->append(Mode::Ptr(m_calltipsMode.dynamicCast<Mode>()));

    m_pyAutoIndentMode = QSharedPointer<PyAutoIndentMode>(new PyAutoIndentMode("PyAutoIndentMode"));
    modes()->append(Mode::Ptr(m_pyAutoIndentMode.dynamicCast<Mode>()));

    modes()->append(Mode::Ptr(new IndenterMode("IndenterMode")));

    m_symbolMatcher = QSharedPointer<SymbolMatcherMode>(new SymbolMatcherMode("SymbolMatcherMode"));
    modes()->append(m_symbolMatcher.dynamicCast<Mode>());

    m_caretLineHighlighter = QSharedPointer<CaretLineHighlighterMode>(new CaretLineHighlighterMode("CaretLineHighlighterMode"));
    modes()->append(m_caretLineHighlighter.dynamicCast<Mode>());
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractCodeEditorWidget::loadSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    bool updateSyntaxHighlighter = false;

    CodeEditorStyle defaultStyle;

    // ------------ general  --------------------------------------------------------

    //TODO:
    /*QString eolMode = settings.value("eolMode", "EolUnix").toString();

    if (eolMode == "EolUnix")
    {
        setEolMode(QsciScintilla::EolUnix);
    }
    else if (eolMode == "EolWindows")
    {
        setEolMode(QsciScintilla::EolWindows);
    }
    else
    {
        setEolMode(QsciScintilla::EolMac);
    }*/

    QSharedPointer<PyAutoIndentMode> pyAutoIndentMode = modes()->get("PyAutoIndentMode").dynamicCast<PyAutoIndentMode>();

    if (pyAutoIndentMode)
    {
        //always enable=true, control the two functionalities via
        //enableAutoIndent and setAutoStripTrailingSpacesAfterReturn
        pyAutoIndentMode->setEnabled(true);
        pyAutoIndentMode->enableAutoIndent(settings.value("autoIndent", true).toBool()); //auto indentation
    }

    setUseSpacesInsteadOfTabs(!settings.value("indentationUseTabs", false).toBool()); //tabs (true) or whitespace (false)
    setTabLength(settings.value("indentationWidth", 4).toInt()); //numbers of whitespaces
    setShowIndentationGuides(settings.value("showIndentationGuides", true).toBool());

    // ------------ calltips --------------------------------------------------------
    m_calltipsMode->setEnabled(settings.value("calltipsEnabled",true).toBool());

    // ------------ auto completion --------------------------------------------------------
    m_codeCompletionMode->setEnabled(settings.value("autoComplEnabled", true).toBool());
    m_codeCompletionMode->setCaseSensitive(settings.value("autoComplCaseSensitive", false).toBool());
    m_codeCompletionMode->setTriggerLength(settings.value("autoComplThreshold", 2).toInt());
    m_codeCompletionMode->setShowTooltips(settings.value("autoComplShowTooltips", true).toBool());
    m_codeCompletionMode->setFilterMode((ito::CodeCompletionMode::FilterMode)settings.value("autoComplFilterMode", CodeCompletionMode::FilterFuzzy).toInt());

    // --------------- styles ------------------------------------------------------------

    if (m_pythonSyntaxHighlighter)
    {
        setBackground(QColor(settings.value("paperBackgroundColor", QColor(Qt::white)).toString()));
        m_pythonSyntaxHighlighter->editorStyle()->setBackground(QColor(settings.value("paperBackgroundColor", QColor(Qt::white)).toString()));

        QTextCharFormat keyWhitespaceFormat = m_pythonSyntaxHighlighter->editorStyle()->format(StyleItem::KeyWhitespace);

        if (keyWhitespaceFormat.background() != QColor(settings.value("whitespaceBackgroundColor", QColor(Qt::white)).toString()))
        {
            m_pythonSyntaxHighlighter->editorStyle()->rformat(StyleItem::KeyWhitespace).setBackground(QColor(settings.value("whitespaceBackgroundColor", QColor(Qt::white)).toString()));
            updateSyntaxHighlighter = true;
        }

        if (keyWhitespaceFormat.foreground().color() != QColor(settings.value("whitespaceForegroundColor", QColor(Qt::black)).toString()))
        {
            m_pythonSyntaxHighlighter->editorStyle()->rformat(StyleItem::KeyWhitespace).setForeground(QColor(settings.value("whitespaceForegroundColor", QColor(Qt::black)).toString()));
            updateSyntaxHighlighter = true;
        }
    }

    if (m_symbolMatcher)
    {
        m_symbolMatcher->setMatchBackground(QColor(settings.value("matchedBraceBackgroundColor", QColor(Qt::white)).toString()));
        m_symbolMatcher->setMatchForeground(QColor(settings.value("matchedBraceForegroundColor", QColor(Qt::red)).toString()));

        m_symbolMatcher->setUnmatchBackground(QColor(settings.value("unmatchedBraceBackgroundColor", QColor(Qt::white)).toString()));
        m_symbolMatcher->setUnmatchForeground(QColor(settings.value("unmatchedBraceForegroundColor", QColor(Qt::red)).toString()));
    }

    m_caretLineHighlighter->setBackground(QColor(settings.value("caretBackgroundColor", QColor(Qt::white)).toString()));
    m_caretLineHighlighter->setEnabled(settings.value("caretBackgroundShow", false).toBool());
    //todo
    setForeground(QColor(settings.value("caretForegroundColor", QColor(Qt::black)).toString())); //caret color


    Mode::Ptr mode = modes()->get("OccurrencesHighlighterMode");
    if (mode)
    {
        OccurrencesHighlighterMode* occHighlighterMode = (OccurrencesHighlighterMode*)(mode.data());
        occHighlighterMode->setBackground(QColor(settings.value("markerSameStringBackgroundColor", QColor(Qt::green)).toString()));
    }

    setSelectionBackground(QColor(settings.value("selectionBackgroundColor", QColor(51, 153, 255)).toString()));
    setSelectionForeground(QColor(settings.value("selectionForegroundColor", QColor(Qt::white)).toString()));

    settings.endGroup();

    // ------------ styles ---------------------------------------------------------------
    //set font for line numbers (equal to font of default style number)


    //TODO
    //QFont marginFont = qSciLex->font(qSciLex->defaultStyle());
    //setMarginsFont(marginFont);

    QTextCharFormat defaultFormat;
    QTextCharFormat currentFormat;


    foreach (StyleItem::StyleType styleType, StyleItem::availableStyleTypes())
    {
        if (styleType == StyleItem::KeyWhitespace)
        {
            continue; //this will be handled separately
        }

        StyleItem &item = m_editorStyle->at(styleType);
        defaultFormat =  defaultStyle[styleType].format();
        currentFormat = item.format();

        if (item.isValid())
        {
            settings.beginGroup("PythonLexerStyle" + QString().setNum(item.type()));

            QColor bgColor = settings.value("backgroundColor", background()).toString();
            if (bgColor.isValid())
            {
                bgColor.setAlpha(settings.value("backgroundColorAlpha", 255).toInt());
                if (currentFormat.background().color() != bgColor)
                {
                    item.rformat().setBackground(bgColor);
                    updateSyntaxHighlighter = true;
                }
            }

            QColor fgColor = settings.value("foregroundColor", defaultFormat.foreground().color()).toString();
            if (fgColor.isValid())
            {
                fgColor.setAlpha(settings.value("foregroundColorAlpha", 255).toInt());

                if (currentFormat.foreground().color() != fgColor)
                {
                    item.rformat().setForeground(fgColor);
                    updateSyntaxHighlighter = true;
                }
            }

            QString fontFamily = settings.value("fontFamily", "").toString();
            if (fontFamily != "")
            {
                if (currentFormat.fontFamily() != fontFamily)
                {
                    item.rformat().setFontFamily(fontFamily);
                    updateSyntaxHighlighter = true;
                }
            }

            int fontPointSize = settings.value("pointSize", 0).toInt();
            if (fontPointSize > 0)
            {
                if (currentFormat.fontPointSize() != fontPointSize)
                {
                    item.rformat().setFontPointSize(fontPointSize);
                    updateSyntaxHighlighter = true;
                }
            }

            int fontWeight = settings.value("weight", defaultFormat.fontWeight()).toInt();
            if (currentFormat.fontWeight() != fontWeight)
            {
                item.rformat().setFontWeight(fontWeight);
                updateSyntaxHighlighter = true;
            }

            bool fontItalic = settings.value("italic", defaultFormat.fontItalic()).toBool();
            if (currentFormat.fontItalic() != fontItalic)
            {
                item.rformat().setFontItalic(fontItalic);
                updateSyntaxHighlighter = true;
            }

            settings.endGroup();
        }

        if (styleType == StyleItem::KeyDefault)
        {
            currentFormat = item.format();

            if (item.isValid())
            {
                //set font of whitespace to default
                QTextCharFormat &whitespaceFormat = m_editorStyle->rformat(ito::StyleItem::KeyWhitespace);
                if (whitespaceFormat.font() != currentFormat.font())
                {
                    whitespaceFormat.setFont(currentFormat.font());
                    updateTabStopAndIndentationWidth();
                    updateSyntaxHighlighter = true;
                }

                if (fontName() != currentFormat.fontFamily())
                {
                    setFontName(currentFormat.fontFamily());
                    updateSyntaxHighlighter = true;
                }
                if (fontSize() != currentFormat.fontPointSize())
                {
                    setFontSize(currentFormat.fontPointSize());
                    updateSyntaxHighlighter = true;
                }
            }
            else
            {
                if (fontName() != defaultFormat.fontFamily())
                {
                    setFontName(defaultFormat.fontFamily());
                    updateSyntaxHighlighter = true;
                }
                if (fontSize() != defaultFormat.fontPointSize())
                {
                    setFontSize(defaultFormat.fontPointSize());
                    updateSyntaxHighlighter = true;
                }
            }
        }
    }

    if (updateSyntaxHighlighter && m_pythonSyntaxHighlighter)
    {
        m_pythonSyntaxHighlighter->refreshEditor(m_editorStyle);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QPixmap AbstractCodeEditorWidget::loadMarker(const QString &name, int sizeAt96dpi) const
{
    int dpi = GuiHelper::getScreenLogicalDpi();
    QPixmap px(name);

    if (dpi != 96 || px.height() != sizeAt96dpi)
    {
        int newSize = sizeAt96dpi * dpi / 96;
        px = px.scaled(newSize, newSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    return px;
}

//----------------------------------------------------------------------------------------------------------------------------------
QString AbstractCodeEditorWidget::getWordAtPosition(const int &line, const int &index) const
{
    return wordAtPosition(line, index, true);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! counts the numbers of leading tabs or spaces of a string
/*!
    \return number of leading tabs or spaces
*/
int AbstractCodeEditorWidget::getSpaceTabCount(const QString &text) const
{
    int res = 0;
    if (text.mid(res, 1).indexOf(QRegularExpression("[\t]")) > -1 || text.mid(res, 1) == " ")
    {
        do
        {
            ++res;
        }
        while (text.mid(res, 1).indexOf(QRegularExpression("[\t]")) > -1 || text.mid(res, 1) == " ");
    }

    return res;
}

//-------------------------------------------------------------------------------------
//! removes parts of the possible indentation of the given text from line 2 until the end and returns the re-formatted text.
/*
This method splits the given text using the \n endline character.

If the given or returned text is empty, an empty string is returned and lineCount is set to 0.

If the given text contains only one line, it is returned (trimmed if ``trimText`` is true)
and ``lineCount`` is 1.

Else, the current indentation level of each non-empty line is checked and the minimum
indentation level is denoted as minIndentLevel. Afterwards every line is tried to be
unindented by minIndentLevel.

\param text is the original text
\param lineCount contains the number of lines in the given text after having called this method
\param trimText defines if the returned text should be trimmed (leading and trailing
    spaces and tabs are removed) or if not.

\returns the modified string
*/
QString AbstractCodeEditorWidget::formatCodeBeforeInsertion(const QString &text, int &lineCount, bool trimText /*= false*/, const QString &newIndent /*= ""*/) const
{
    QString res = "";
    lineCount = 0;

    if (text != "")
    {
        const QString endline = "\n";

        QStringList commandList = text.split(endline);
        lineCount = commandList.size();

        if (lineCount == 1)
        {
            res = text;
        }
        else if (lineCount > 1)
        {
			//if the first line starts does not start with a space or tab,
			//it is assumed, that the cursor just starts at the first real
			//character and subsequent lines might possibly be indented. Then
			//do not change or consider the first line for indentation detection
			//and / or removal. Else: consider it.
			const QString &firstLine = commandList[0];
			const QString &firstLineLeftStrip = Utils::lstrip(firstLine);

			bool firstLineIndented = firstLine.size() > firstLineLeftStrip.size();

			if (!firstLineIndented && firstLineLeftStrip.size() > 0)
			{
				//it can be that the first line starts with a possible keyword
				//for a subsequent indented block. If so, also consider the first line to
				//be indented (such that it is considered for the minIndentLevel.
				QList<QString> blockKeywords;
				blockKeywords << "def" << "if" << "elif" << "else" << "while" << \
					"for" << "try" << "except" << "finally" << "with" << "class" << "async";

				if (blockKeywords.contains(firstLineLeftStrip.split(" ")[0]))
				{
					firstLineIndented = true;
				}
			}

			int minIndentLevel = 1e6;
			int startLine = firstLineIndented ? 0 : 1;

			for (int i = startLine; i < lineCount; ++i)
			{
                if (commandList[i].trimmed() != "" && commandList[i][0]!='#')
				{
					minIndentLevel = std::min(minIndentLevel, getSpaceTabCount(commandList[i]));
				}
			}

			if (minIndentLevel == 0 && newIndent == "")
			{
				res = text;
			}
			else
			{
				if (!firstLineIndented)
				{
					res += firstLine;
				}
				else
				{
					res += firstLine.mid(minIndentLevel);
				}

				for (int i = 1; i < lineCount; ++i)
				{
                    if (commandList[i][0] != '#')
                    {
                        res += endline + newIndent + commandList[i].mid(minIndentLevel);
                    }
                    else
                    {
                        res += endline + newIndent + commandList[i]; // do not remove indent from comment line
                    }
				}
			}
        }
    }

    if (trimText)
    {
        res = res.trimmed();
    }

    if (res == "")
    {
        lineCount = 0;
    }

    return res;
}

//-------------------------------------------------------------------------------------
//! this method modifies a code string such before copying it to the clipboard or a mimedata.
/*
If the text contains less than two lines, nothing is changed.
Else, the prependedTextInFirstLine can contain the text before the code. If this text
only contains whitespaces, it is prepended to the code, such that the entire code
has a proper indentation.
*/
QString AbstractCodeEditorWidget::formatCodeForClipboard(const QString &code, const QString &prependedTextInFirstLine) const
{
    if (prependedTextInFirstLine == "" || prependedTextInFirstLine.trimmed() != "")
    {
        return code;
    }

    QStringList lines = code.split("\n");

    if (lines.size() < 2)
    {
        return code;
    }
    else
    {
        return prependedTextInFirstLine + code;
    }
}

//-------------------------------------------------------------------------------------
//! copy selected code to the clipboard
/*
Depending on the settings, the text, that is copied will be
pre-formatted, such that it better fits to the current column
of the cursor to keep a consistent indentation level.

Make sure to call this method only if the copy operation is allowed.
*/
void AbstractCodeEditorWidget::copy()
{
    CodeEditor::copy();

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");
    bool formatCopyCode = settings.value("formatCopyCutCode", true).toBool();
    settings.endGroup();

    if (formatCopyCode)
    {
        QClipboard *clipboard = QApplication::clipboard();

        const QMimeData *mimeData = clipboard->mimeData();

        if (mimeData && mimeData->hasText())
        {
            int lineFrom, lineTo, columnFrom, columnTo;
            getSelection(&lineFrom, &columnFrom, &lineTo, &columnTo);

            if (lineFrom > lineTo)
            {
                std::swap(lineFrom, lineTo);
                std::swap(columnFrom, columnTo);
            }
            else if ((lineFrom == lineTo) && (columnFrom > columnTo))
            {
                std::swap(columnFrom, columnTo);
            }

            if (lineFrom == -1)
            {
                getCursorPosition(&lineFrom, &columnFrom);
            }

            QString prependText;

            if (lineFrom >= 0 && columnFrom >= 0)
            {
                prependText = lineText(lineFrom).left(columnFrom);
            }

            QString modifiedText = formatCodeForClipboard(mimeData->text(), prependText);

            // from Win11 on, directly reassigning a text to the clipboard
            // might lead to errors (similar to https://bugreports.qt.io/browse/QTBUG-27097).
            // A workaround seems to be to change the clipboard with a small
            // delay, however the event loop has to be run at least one time.
            // Therefore, a simple sleep / delay does not work.
#ifdef _WIN32
            QTimer::singleShot(25, [=]()
            {
                clipboard->setText(modifiedText);
            });
#else
            //clipboard->clear(); // mimeData is now invalid!
            //clipboard->setText(modifiedText);
#endif
        }
    }
}

//-------------------------------------------------------------------------------------
//! paste code from the clipboard at the current cursor position
/*
Depending on the settings, the text, that should be pasted,
is pre-formatted, such that it better fits to the current column
of the cursor to keep a consistent indentation level.

Make sure that the cursor is at the desired position before calling this method.
*/
void AbstractCodeEditorWidget::paste()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");
    bool formatPasteCode = settings.value("formatPasteAndDropCode", true).toBool();
    settings.endGroup();

    if (formatPasteCode)
    {
        // the strategy is to get the text from the clipboard,
        // adapt this text, set it again to the clipboard and
        // use the ordinary paste method to insert it. Afterwards,
        // the original text will be set again to the clipboard.
        QClipboard *clipboard = QApplication::clipboard();
        QString currentClipboardText = "";

        if (clipboard->mimeData()->hasText())
        {
            int lineIdx, column;
            int lineToIdx, columnTo;
            getSelection(&lineIdx, &column, &lineToIdx, &columnTo);

            if (lineIdx == -1)
            {
                // no selection, get the current cursor position
                getCursorPosition(&lineIdx, &column);
            }
            else if (lineIdx > lineToIdx)
            {
                // the start of the selection is at the end of the selection. swap it.
                lineIdx = lineToIdx;
                column = columnTo;
            }
            else if (lineIdx == lineToIdx && column > columnTo)
            {
                // the start of the selection is at the end of the selection. swap it.
                lineIdx = lineToIdx;
                column = columnTo;
            }

            QString indent;

            // if this is a console widget and the current line starts with >>,
            // the column should be subtracted by the length of this special start string.
            column -= std::max(0, startLineOffset(lineIdx));

            if (useSpacesInsteadOfTabs())
            {
                indent = QString(column, ' ');
            }
            else
            {
                indent = QString(column, '\t');
            }

            currentClipboardText = clipboard->text();
            int lineCount;
            clipboard->setText(formatCodeBeforeInsertion(currentClipboardText, lineCount, false, indent));
        }

        CodeEditor::paste();

        if (currentClipboardText != "")
        {
            clipboard->setText(currentClipboardText);
        }
    }
    else
    {
        CodeEditor::paste();
    }
}

//-------------------------------------------------------------------------------------
//! cut selected code and puts it into the clipboard
/*
Depending on the settings, the text, that is cut will be
pre-formatted, such that it better fits to the current column
of the cursor to keep a consistent indentation level.

Make sure to call this method only if the cut operation is allowed or
possibly adjust the selected text to a valid section.
*/
void AbstractCodeEditorWidget::cut()
{
    CodeEditor::cut();

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");
    bool formatCopyCode = settings.value("formatCopyCutCode", true).toBool();
    settings.endGroup();

    if (formatCopyCode)
    {
        QClipboard *clipboard = QApplication::clipboard();

        const QMimeData *mimeData = clipboard->mimeData();

        if (mimeData && mimeData->hasText())
        {
            int lineFrom, lineTo, columnFrom, columnTo;
            getSelection(&lineFrom, &columnFrom, &lineTo, &columnTo);

            if (lineFrom > lineTo)
            {
                std::swap(lineFrom, lineTo);
                std::swap(columnFrom, columnTo);
            }
            else if ((lineFrom == lineTo) && (columnFrom > columnTo))
            {
                std::swap(columnFrom, columnTo);
            }

            if (lineFrom == -1)
            {
                getCursorPosition(&lineFrom, &columnFrom);
            }

            QString prependText;

            if (lineFrom >= 0 && columnFrom >= 0)
            {
                prependText = lineText(lineFrom).left(columnFrom);
            }

            QString modifiedText = formatCodeForClipboard(mimeData->text(), prependText);

            // from Win11 on, directly reassigning a text to the clipboard
// might lead to errors (similar to https://bugreports.qt.io/browse/QTBUG-27097).
// A workaround seems to be to change the clipboard with a small
// delay, however the event loop has to be run at least one time.
// Therefore, a simple sleep / delay does not work.
#ifdef _WIN32
            QTimer::singleShot(25, [=]()
            {
                clipboard->setText(modifiedText);
            });
#else
            //clipboard->clear(); // mimeData is now invalid!
            //clipboard->setText(modifiedText);
#endif
        }
    }
}


} //end namespace ito
