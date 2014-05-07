/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

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

// signale in scriptEditorOrganizer annehmen und geb�ndelt an Replace senden!
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
        clearIndicatorRange(0,0,nrOfLines,lengthOfLastLine,m_textIndicatorNr);
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
                j=0;
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
