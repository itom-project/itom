/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#include "abstractPyScintillaWidget.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qstring.h>
#include <qsettings.h>
#include <qdebug.h>
#include <qcolor.h>
#include <qfont.h>
#include <qtooltip.h>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
AbstractPyScintillaWidget::AbstractPyScintillaWidget(QWidget* parent):
    QsciScintilla(parent),
    qSciLex(NULL),
    m_userSelectionState(selNo),
    m_textIndicatorActive(false),
    m_textIndicatorNr(-1)
{
    qDebug("abstractPyScintillaWidget constructor start");
    init();
    reloadSettings();

    connect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(reloadSettings()));
    connect(this, SIGNAL(selectionChanged()), this, SLOT(selectionChanged()));
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractPyScintillaWidget::~AbstractPyScintillaWidget()
{
    DELETE_AND_SET_NULL(qSciLex);
    m_pApiManager = NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractPyScintillaWidget::init()
{
    DELETE_AND_SET_NULL(qSciLex);
    qSciLex = new QsciLexerPython(this);
    m_pApiManager = ito::QsciApiManager::getInstance();
    qSciLex->setAPIs(m_pApiManager->getQsciAPIs());
    setLexer(qSciLex);

    setUtf8(true); //usually the default encoding is latin1, but the scintilla editor is still utf8 since qscintilla < 2.8 does not accept special characters in the keypressevent coming from non utf8 ecodings.

    m_textIndicatorNr = indicatorDefine(QsciScintilla::RoundBoxIndicator);
    setIndicatorForegroundColor(Qt::green, m_textIndicatorNr);
    setIndicatorDrawUnder(true, m_textIndicatorNr);
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractPyScintillaWidget::loadSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");

    // ------------ general  --------------------------------------------------------

    QString eolMode = settings.value("eolMode", "EolUnix").toString();

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
    }

    setAutoIndent(settings.value("autoIndent", true).toBool());                  //auto indentation
    setIndentationsUseTabs(settings.value("indentationUseTabs", false).toBool()); //tabs (true) or whitespace (false)
    setIndentationWidth(settings.value("indentationWidth", 4).toInt());          //numbers of whitespaces
    setIndentationGuides(settings.value("showIndentationGuides", true).toBool());

    QString indentationWarning = settings.value("indentationWarning", "Inconsistent").toString();

    if (eolMode == "Inconsistent")
    {
        qSciLex->setIndentationWarning(QsciLexerPython::Inconsistent);
    }
    else if (eolMode == "NoWarning")
    {
        qSciLex->setIndentationWarning(QsciLexerPython::NoWarning);
    }
    else if (eolMode == "TabsAfterSpaces")
    {
        qSciLex->setIndentationWarning(QsciLexerPython::TabsAfterSpaces);
    }
    else if (eolMode == "Spaces")
    {
        qSciLex->setIndentationWarning(QsciLexerPython::Spaces);
    }
    else //Tabs
    {
        qSciLex->setIndentationWarning(QsciLexerPython::Tabs);
    }

    // ------------ API --------------------------------------------------------

    //do not change api here, since this is directly done by property dialog and qsciApiManager

    //!< add commands to autoCompletion:
    /*styles:
        qSciApi->add("methodName(param1,param2) Description);
        qSciApi->add("name");
    */

    // ------------ calltips --------------------------------------------------------
    bool calltipsEnabled = settings.value("calltipsEnabled",true).toBool();

    if (calltipsEnabled)
    {
        QString style = settings.value("calltipsStyle","NoContext").toString();

        if (style == "NoContext")
        {
            setCallTipsStyle(QsciScintilla::CallTipsNoContext);
        }
        else if (style == "NoAutoCompletionContext")
        {
            setCallTipsStyle(QsciScintilla::CallTipsNoAutoCompletionContext);
        }
        else if (style == "Context")
        {
            setCallTipsStyle(QsciScintilla::CallTipsContext);
        }
        else
        {
            setCallTipsStyle(QsciScintilla::CallTipsNone);
        }

        setCallTipsVisible(settings.value("calltipsNoVisible",3).toInt()); //show 3 call tips before using arrows (up/down)
    }
    else
    {
        setCallTipsStyle(QsciScintilla::CallTipsNone);
    }

    // ------------ auto completion --------------------------------------------------------
    bool acEnabled = settings.value("autoComplEnabled", true).toBool();
    QString source = settings.value("autoComplSource", "AcsAPIs").toString();

    if (acEnabled)
    {
        if (source == "AcsAll")
        {
            setAutoCompletionSource(QsciScintilla::AcsAll);
        }
        else if (source == "AcsDocument")
        {
            setAutoCompletionSource(QsciScintilla::AcsDocument);
        }
        else if (source == "AcsAPIs")
        {
            setAutoCompletionSource(QsciScintilla::AcsAPIs);
        }
        else
        {
            setAutoCompletionSource(QsciScintilla::AcsNone);
        }
    }
    else
    {
        setAutoCompletionSource(QsciScintilla::AcsNone);
    }

    setAutoCompletionThreshold(settings.value("autoComplThreshold", 3).toInt());
    setAutoCompletionFillupsEnabled(settings.value("autoComplFillUps", true).toBool()); //Enable the use of fill-up characters, either those explicitly set or those set by a lexer.
    setAutoCompletionCaseSensitivity(settings.value("autoComplCaseSensitive", false).toBool());
    setAutoCompletionReplaceWord(settings.value("autoComplReplaceWord", false).toBool());
    setAutoCompletionShowSingle(settings.value("autoComplShowSingle", false).toBool());

    settings.endGroup();

    // ------------ styles ---------------------------------------------------------------
    //set font for line numbers (equal to font of default style number)
    QFont marginFont = qSciLex->font(qSciLex->defaultStyle());
    setMarginsFont(marginFont);

    int noOfStyles = qSciLex->styleBitsNeeded();

    for (int i = 0; i < (2 << noOfStyles); i++)
    {
        if (!qSciLex->description(i).isEmpty())
        {
            settings.beginGroup("PyScintilla_LexerStyle" + QString().setNum(i));

            QColor bgColor = settings.value("backgroundColor", qSciLex->defaultPaper(i)).toString();
            if (bgColor.isValid())
            {
                bgColor.setAlpha(settings.value("backgroundColorAlpha", 255).toInt());
                qSciLex->setPaper(bgColor,i);
            }

            QColor fgColor = settings.value("foregroundColor", qSciLex->defaultColor(i)).toString();
            if (fgColor.isValid())
            {
                fgColor.setAlpha(settings.value("foregroundColorAlpha", 255).toInt());
                qSciLex->setColor(fgColor, i);
            }

            QFont font = QFont(settings.value("fontFamily", "").toString(), settings.value("pointSize", 0).toInt(), settings.value("weight", 0).toInt(), settings.value("italic", false).toBool());
            if (font.pointSize() > 0 && font.family() != "")
            {
                qSciLex->setFont(font, i);
            }
            else
            {
                qSciLex->setFont(qSciLex->defaultFont(i),i);
            }

            qSciLex->setEolFill(settings.value("fillToEOL", qSciLex->defaultEolFill(i)).toBool(), i);
            settings.endGroup();
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//void AbstractPyScintillaWidget::mouseReleaseEvent(QMouseEvent * event)
//{
//    //QsciScintilla::mouseReleaseEvent(event);
//    //checkUserSelectionState();
//    event->accept();
//}
//
//void AbstractPyScintillaWidget::keyReleaseEvent(QKeyEvent * event)
//{
//    //QsciScintilla::keyPressEvent(event);
//    checkUserSelectionState();
//    event->ignore();
//}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractPyScintillaWidget::checkUserSelectionState()
{
    int lineFrom, indexFrom, lineTo, indexTo;
    bool sel = true;

    getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
    if (lineFrom == -1)
    {
        sel = false;
    }

// signale in scriptEditorOrganizer annehmen und gebuendelt an Replace senden!
    switch(m_userSelectionState)
    {
    case selNo:
        if (sel)
        {
            m_userSelectionState = selRange;
            //emit signal, since selection is ready now
            emit userSelectionChanged(lineFrom, indexFrom, lineTo, indexTo);
        }
        break;
    case selRange:
        if (!sel)
        {
            m_userSelectionState = selNo;
            //emit signal, since selection is gone
            emit userSelectionChanged(-1, -1, -1, -1);
        }
        else
        {
            //m_userSelectionState = selRange; //remains the same, but other selection
            //emit signal, since selection is changed
            emit userSelectionChanged(lineFrom, indexFrom, lineTo, indexTo);
        }
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractPyScintillaWidget::selectionChanged()
{
    int nrOfLines = lines();
    int lengthOfLastLine = text(nrOfLines-1).length();

    if (m_textIndicatorActive)
    {
        m_textIndicatorActive = false;
        clearIndicatorRange(0, 0, nrOfLines, lengthOfLastLine, m_textIndicatorNr);
    }

    QString selection;
    QString lineText;
    int lineLength;
    int lineFrom, indexFrom, lineTo, indexTo;
    int j;
    int index;
    int selLength;

    if (hasSelectedText())
    {
        selection = selectedText();
        if (selection.trimmed() == "")
        {
            return;
        }
        selLength = selection.length();
        getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);

        if (lineFrom == lineTo)
        {
            for (int i = 0; i < nrOfLines; i++)
            {
                lineText = text(i);
                lineLength = lineText.length() - selLength;
                j = 0;

                while(j <= lineLength)
                {
                    index = lineText.indexOf(selection, j, Qt::CaseInsensitive);
                    if (index >= 0)
                    {
                        if (i != lineFrom || (index < indexFrom || index >= indexTo))
                        {
                            m_textIndicatorActive = true;
                            fillIndicatorRange(i, index, i, index + selLength, m_textIndicatorNr);
                        }
                        j = index + selLength;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QString AbstractPyScintillaWidget::getWordAtPosition(const int &line, const int &index)
{
    if (line < 0 || line >= lines())
    {
        return "";
    }

    if (index < 0 || index >= lineLength(line))
    {
        return "";
    }

    long pos = positionFromLineIndex(line, index);
    long start = SendScintilla(QsciScintilla::SCI_WORDSTARTPOSITION, pos, true);
    long end = SendScintilla(QsciScintilla::SCI_WORDENDPOSITION, pos, true);

    if (start != end)
    {
        char *bytes = new char[(end-start) + 1];
        SendScintilla(QsciScintilla::SCI_GETTEXTRANGE, start, end, bytes);
        QString word = QString(bytes);
        delete[] bytes;
        return word;
    }

    return "";
}

//----------------------------------------------------------------------------------------------------------------------------------
//! counts the numbers of leading tabs or spaces of a string
/*!
    \return number of leading tabs or spaces
*/
int AbstractPyScintillaWidget::getSpaceTabCount(const QString &s)
{
    int res = 0;
    if (s.mid(res, 1).indexOf(QRegExp("[\t]")) > -1 || s.mid(res, 1) == " ")
    {
        do
        {
            ++res;
        }
        while (s.mid(res, 1).indexOf(QRegExp("[\t]")) > -1 || s.mid(res, 1) == " ");
    }

    return res;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! checks if text line contains a colon sign as last valid character (only comments or spaces are allowed after the colon)
/*!
    This method is necessary in order to verify if the following text lines must be indented with respect
    to this line in Python syntax.

    \return true if colon is last valid sign, else false
*/
bool AbstractPyScintillaWidget::haveToIndention(QString s)
{
    s = s.trimmed();
    s.replace("'''", "\a");
    s.replace("\"\"\"", "\a");
    int count1 = s.count("\a");
    int count2 = s.count("#");

    if (count1 + count2 > 0)
    {
        if (count1 == 0)
        {
            s = s.mid(1, s.indexOf("#"));
        }
        else if (count2 == 0)
        {
            bool comment = (count1 % 2 == 1);
            if (comment)
            {
                s = s.mid(1, s.lastIndexOf("\a") - 1);
                s = s.trimmed();
                --count1;
            }

            while (count1 > 0)
            {
                int pos1 = s.indexOf("\a");
                int pos2 = pos1 + 1;
                while (s.mid(pos2, 1) != "\a")
                {
                    ++pos2;
                }
                s = s.mid(0, pos1) + s.mid(pos2 + 1);
                --count1;
                --count1;
            }
        }
        else
        {
            s = s.mid(1, s.indexOf("#"));
            s = s.trimmed();

            bool comment = ((count1 & 2) == 1);
            if (comment)
            {
                s = s.mid(1, s.lastIndexOf("\a"));
                s = s.trimmed();
                --count1;
            }

            while (count1 > 0)
            {
                int pos1 = s.indexOf("\a");
                int pos2 = pos1 + 1;
                while (s.mid(pos2, 1) != "\a")
                {
                    ++pos2;
                }
                s = s.mid(1, pos1) + s.mid(pos2 + 1);
                --count1;
                --count1;
            }
        }
    }

    s = s.trimmed();
    return s.mid(s.size() - 1, 1) == ":";
}

//----------------------------------------------------------------------------------------------------------------------------------
QString AbstractPyScintillaWidget::formatPhytonCodePart(const QString &text, int &lineCount)
{
    QString res = "";
    lineCount = 0;
    if (text.trimmed() != "")
    {
        QString endlineRegExp = "[\n]";
        QString endline = "\n";

        QStringList commandList = text.split(QRegExp(endlineRegExp));
        lineCount = commandList.size();
        if (lineCount == 1)
        {
            res = text.trimmed();
        }
        else
        {
            int i = 1;
            while (i < lineCount && commandList[i].trimmed() == "")
            {
                ++i;
            }

            if (i < lineCount)
            {
                int spaceTabCount1 = getSpaceTabCount(commandList[i]);
                int spaceTabCount2 = 0;
                int tmp = 0;
                i = 2;
                while (i < lineCount && spaceTabCount2 == 0)
                {
                    tmp = getSpaceTabCount(commandList[i]);
                    if (tmp != spaceTabCount1)
                    {
                        spaceTabCount2 = tmp;
                    }
                    ++i;
                }

                int delCount = 0;
                if (haveToIndention(commandList[0]))
                {
                    int spaceTabDifCount = 0;
                    if (spaceTabCount2 != 0)
                    {
                        if (spaceTabCount1 > spaceTabCount2)
                        {
                            spaceTabDifCount = spaceTabCount1 - spaceTabCount2;
                        }
                        else
                        {
                            spaceTabDifCount = spaceTabCount2 - spaceTabCount1;
                        }
                    }
                    else
                    {
                        if (spaceTabCount1 == 0 || spaceTabCount1 % 4 == 0)
                        {
                            spaceTabDifCount = 4;
                        }
                        else if (spaceTabCount1 % 3 == 0)
                        {
                            spaceTabDifCount = 3;
                        }
                        else if (spaceTabCount1 % 2 == 0)
                        {
                            spaceTabDifCount = 2;
                        }
                        else
                        {
                            spaceTabDifCount = 1;
                        }
                    }

                    delCount = spaceTabCount1 - spaceTabDifCount;
                }
                else
                {
                    delCount = spaceTabCount1;
                }

                res = commandList[0].trimmed() + endline;
                for (i = 1; i < lineCount; ++i)
                {
                    commandList[i].remove(0, delCount);
                    res += commandList[i] + endline;
                }
            }
            else
            {
                res = text.trimmed(); 
            }
        }
    }

    return res;
}

//----------------------------------------------------------------------------------------------------------------------------------
//bool AbstractPyScintillaWidget::event (QEvent * event)
//{
//    if (event->type() == QEvent::ToolTip && !QToolTip::isVisible())
//    {
//        //see http://www.riverbankcomputing.com/pipermail/qscintilla/2008-November/000381.html
//        QHelpEvent *evt = static_cast<QHelpEvent*>(event);
//        QPoint point = evt->pos();
//        long pos = SendScintilla(QsciScintilla::SCI_POSITIONFROMPOINTCLOSE, point.x(), point.y());
//
//        if (pos >= 0)
//        {
//            long start = SendScintilla(QsciScintilla::SCI_WORDSTARTPOSITION, pos, true);
//            long end = SendScintilla(QsciScintilla::SCI_WORDENDPOSITION, pos, true);
//
//            if (start != end)
//            {
//                char *bytes = new char[(end-start) + 1];
//                SendScintilla(QsciScintilla::SCI_GETTEXTRANGE, start, end, bytes);
//                QString word = QString(bytes);
//                delete[] bytes;
//
//                long x_start = SendScintilla(QsciScintilla::SCI_POINTXFROMPOSITION, 0, start);
//                long y_start = SendScintilla(QsciScintilla::SCI_POINTYFROMPOSITION, 0, start);
//                long x_end = SendScintilla(QsciScintilla::SCI_POINTXFROMPOSITION, 0, end);
//                long line = SendScintilla(QsciScintilla::SCI_LINEFROMPOSITION, start);
//                long height = SendScintilla(QsciScintilla::SCI_TEXTHEIGHT, line);
//                QRect rect = QRect(x_start, y_start, x_end - x_start, height);
//                QToolTip::showText(evt->globalPos(), word, this->viewport(), rect);
//            }
//        }
//    }
//
//    return QsciScintilla::event(event);
//}

} //end namespace ito
