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

#include "widgetPropEditorStyles.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qcolordialog.h>
#include <qfontdialog.h>
#include <qfiledialog.h>
#include <qpalette.h>
#include <qsettings.h>
#include <qxmlstream.h>
#include <qmessagebox.h>

namespace ito
{

const int PAPERCOLOR = 1;
const int FOLDMARGINCOLOR = 2;
const int MARGINCOLOR = 3;
const int WHITESPACECOLOR = 4;
const int UNMATCHEDBRACECOLOR = 5;
const int MATCHEDBRACECOLOR = 6;
const int MARKERERRORCOLOR = 7;
const int MARKERCURRENTCOLOR = 8;
const int MARKERINPUTCOLOR = 9;
const int CARETCOLOR = 10;
const int SELECTIONCOLOR = 11;
const int MARKERSAMESTRINGCOLOR = 12;

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorStyles::WidgetPropEditorStyles(QWidget *parent) :
    AbstractPropertyPageWidget(parent),
    m_changing(false)
{
    ui.setupUi(this);

    //ui.lblSampleText->setBackgroundRole(QPalette::Highlight);
    ui.lblSampleText->setAutoFillBackground(false);

    qSciLex = new QsciLexerPython(this);

    int noOfStyles = qSciLex->styleBitsNeeded();

    for (int i = 0; i < (2 << noOfStyles); i++)
    {
        if (!qSciLex->description(i).isEmpty())
        {
            StyleNode entry;
            entry.m_index = i;
            entry.m_name = qSciLex->description(i);
            entry.m_fillToEOL = qSciLex->defaultEolFill(entry.m_index);
            entry.m_backgroundColor = qSciLex->defaultPaper(entry.m_index);
            entry.m_foregroundColor = qSciLex->defaultColor(entry.m_index);
            entry.m_font = qSciLex->defaultFont(entry.m_index);

            ui.listWidget->addItem(entry.m_name);

            m_styles.push_back(entry);
        }
    }

    QListWidgetItem *separator = new QListWidgetItem("------------------------------", NULL, 1000);
    separator->setFlags(Qt::ItemIsEnabled);
    ui.listWidget->addItem(separator);

    ui.listWidget->addItem(new QListWidgetItem(tr("Paper color"), NULL, PAPERCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Caret color (Foreground: cursor color, Background: color of current line)"), NULL, CARETCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Fold margin color"), NULL, FOLDMARGINCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Margin color"), NULL, MARGINCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Whitespace color (if whitespace characters are visible)"), NULL, WHITESPACECOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Unmatched brace color"), NULL, UNMATCHEDBRACECOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Matched brace color"), NULL, MATCHEDBRACECOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Command line: Background for error messages"), NULL, MARKERERRORCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Command line: Background for currently executed line"), NULL, MARKERCURRENTCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Command line: Background for python input"), NULL, MARKERINPUTCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Background color and text color of current selection"), NULL, SELECTIONCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Background color of words equal to the currently selected string"), NULL, MARKERSAMESTRINGCOLOR));

    ui.btnForegroundColor->setEnabled(false);
    ui.btnBackgroundColor->setEnabled(false);
    ui.btnFont->setEnabled(false);
    ui.checkFillEOL->setEnabled(false);
    ui.checkShowCaretBackground->setVisible(false);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorStyles::~WidgetPropEditorStyles()
{
    DELETE_AND_SET_NULL(qSciLex);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::readSettings()
{
    readSettingsInternal(AppManagement::getSettingsFile());
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::writeSettings()
{
    writeSettingsInternal(AppManagement::getSettingsFile());
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::writeSettingsInternal(const QString &filename)
{
    QSettings settings(filename, QSettings::IniFormat);

    StyleNode entry;
    foreach(entry, m_styles)
    {
        settings.beginGroup("PyScintilla_LexerStyle" + QString().setNum(entry.m_index));
        settings.setValue("backgroundColor", entry.m_backgroundColor.name());
        settings.setValue("backgroundColorAlpha", entry.m_backgroundColor.alpha());
        settings.setValue("foregroundColor", entry.m_foregroundColor.name());
        settings.setValue("foregroundColorAlpha", entry.m_foregroundColor.alpha());
        settings.setValue("fillToEOL", entry.m_fillToEOL);
        settings.setValue("fontFamily", entry.m_font.family()),
            settings.setValue("pointSize", entry.m_font.pointSize()),
            settings.setValue("weight", entry.m_font.weight()),
            settings.setValue("italic", entry.m_font.italic());
        settings.endGroup();
    }

    settings.beginGroup("PyScintilla");
    settings.setValue("paperBackgroundColor", m_paperBgcolor);
    settings.setValue("marginBackgroundColor", m_marginBgcolor);
    settings.setValue("marginForegroundColor", m_marginFgcolor);
    settings.setValue("caretBackgroundColor", m_caretBgcolor);
    settings.setValue("caretForegroundColor", m_caretFgcolor);
    settings.setValue("caretBackgroundShow", ui.checkShowCaretBackground->isChecked());
    settings.setValue("foldMarginBackgroundColor", m_foldMarginBgcolor);
    settings.setValue("foldMarginForegroundColor", m_foldMarginFgcolor);
    settings.setValue("markerCurrentBackgroundColor", m_markerCurrentBgcolor);
    settings.setValue("markerInputForegroundColor", m_markerInputBgcolor);
    settings.setValue("markerErrorForegroundColor", m_markerErrorBgcolor);
    settings.setValue("whitespaceForegroundColor", m_whitespaceFgcolor);
    settings.setValue("unmatchedBraceBackgroundColor", m_unmatchedBraceBgcolor);
    settings.setValue("unmatchedBraceForegroundColor", m_unmatchedBraceFgcolor);
    settings.setValue("matchedBraceBackgroundColor", m_matchedBraceBgcolor);
    settings.setValue("matchedBraceForegroundColor", m_matchedBraceFgcolor);
    settings.setValue("selectionBackgroundColor", m_selectionBgcolor);
    settings.setValue("selectionForegroundColor", m_selectionFgcolor);
    settings.setValue("markerSameStringBackgroundColor", m_markerSameStringBgcolor);
    

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::readSettingsInternal(const QString &filename)
{
    QSettings settings(filename, QSettings::IniFormat);

    for (int i = 0; i < m_styles.size(); i++)
    {
        settings.beginGroup("PyScintilla_LexerStyle" + QString().setNum(m_styles[i].m_index));
        m_styles[i].m_backgroundColor = QColor(settings.value("backgroundColor", m_styles[i].m_backgroundColor.name()).toString());
        m_styles[i].m_backgroundColor.setAlpha(settings.value("backgroundColorAlpha", m_styles[i].m_backgroundColor.alpha()).toInt());
        m_styles[i].m_foregroundColor = QColor(settings.value("foregroundColor", m_styles[i].m_foregroundColor.name()).toString());
        m_styles[i].m_backgroundColor.setAlpha(settings.value("foregroundColorAlpha", m_styles[i].m_foregroundColor.alpha()).toInt());
        m_styles[i].m_fillToEOL = settings.value("fillToEOL", m_styles[i].m_fillToEOL).toBool();
        m_styles[i].m_font = QFont(settings.value("fontFamily", m_styles[i].m_font.family()).toString(), settings.value("pointSize", m_styles[i].m_font.pointSize()).toInt(), settings.value("weight", m_styles[i].m_font.weight()).toInt(), settings.value("italic", m_styles[i].m_font.italic()).toBool());
        settings.endGroup();
    }

    settings.beginGroup("PyScintilla");
    //the following default values are also written in on_btnReset_clicked()
    m_paperBgcolor = QColor(settings.value("paperBackgroundColor", QColor(Qt::white)).toString());
    m_marginBgcolor = QColor(settings.value("marginBackgroundColor", QColor(224, 224, 224)).toString());
    m_marginFgcolor = QColor(settings.value("marginForegroundColor", QColor(Qt::black)).toString());
    m_foldMarginBgcolor = QColor(settings.value("foldMarginBackgroundColor", QColor(Qt::white)).toString());
    m_foldMarginFgcolor = QColor(settings.value("foldMarginForegroundColor", QColor(233, 233, 233)).toString());
    m_markerCurrentBgcolor = QColor(settings.value("markerCurrentBackgroundColor", QColor(255, 255, 128)).toString());
    m_markerInputBgcolor = QColor(settings.value("markerInputForegroundColor", QColor(179, 222, 171)).toString());
    m_markerErrorBgcolor = QColor(settings.value("markerErrorForegroundColor", QColor(255, 192, 192)).toString());
    m_whitespaceFgcolor = QColor(settings.value("whitespaceForegroundColor", QColor(Qt::black)).toString());
    m_unmatchedBraceBgcolor = QColor(settings.value("unmatchedBraceBackgroundColor", QColor(Qt::white)).toString());
    m_unmatchedBraceFgcolor = QColor(settings.value("unmatchedBraceForegroundColor", QColor(128, 0, 0)).toString());
    m_matchedBraceBgcolor = QColor(settings.value("matchedBraceBackgroundColor", QColor(Qt::white)).toString());
    m_matchedBraceFgcolor = QColor(settings.value("matchedBraceForegroundColor", QColor(Qt::red)).toString());
    m_caretBgcolor = QColor(settings.value("caretBackgroundColor", QColor(Qt::white)).toString());
    m_caretFgcolor = QColor(settings.value("caretForegroundColor", QColor(Qt::black)).toString());
    ui.checkShowCaretBackground->setChecked(settings.value("caretBackgroundShow", false).toBool());
    m_selectionBgcolor = QColor(settings.value("selectionBackgroundColor", QColor(51, 153, 255)).toString());
    m_selectionFgcolor = QColor(settings.value("selectionForegroundColor", QColor(Qt::white)).toString());
    m_markerSameStringBgcolor = QColor(settings.value("markerSameStringBackgroundColor", QColor(Qt::green)).toString());

    settings.endGroup();

    on_listWidget_currentItemChanged(ui.listWidget->currentItem(), NULL);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_listWidget_currentItemChanged(QListWidgetItem *current, QListWidgetItem * /* previous */)
{
    m_changing = true;
    if (current)
    {
        if (current->type() == 0)
        {
            int index = ui.listWidget->currentIndex().row();
            ui.checkFillEOL->setChecked(m_styles[index].m_fillToEOL);
            ui.btnBackgroundColor->setColor(m_styles[index].m_backgroundColor);
            ui.btnForegroundColor->setColor(m_styles[index].m_foregroundColor);

            ui.lblSampleText->setText(tr("Sample Text"));
            ui.lblSampleText->setFont(m_styles[index].m_font);
            ui.lblSampleText->setStyleSheet(QString("color: %1; background-color: %2;").arg(m_styles[index].m_foregroundColor.name()).arg(m_styles[index].m_backgroundColor.name()));
            ui.lblSampleText->repaint();
            
            ui.btnForegroundColor->setEnabled(true);
            ui.btnBackgroundColor->setEnabled(true);
            ui.btnFont->setEnabled(true);
            ui.checkFillEOL->setEnabled(true);
            ui.checkShowCaretBackground->setVisible(false);
        }
        else if (current->type() < 1000)
        {
            ui.checkFillEOL->setEnabled(false);
            ui.checkFillEOL->setChecked(false);
            ui.btnFont->setEnabled(false);
            ui.btnBackgroundColor->setEnabled(true);
            ui.checkShowCaretBackground->setVisible(false);
            QColor bg;
            QColor fg(Qt::black);

            switch (current->type())
            {
            case PAPERCOLOR:
                bg = m_paperBgcolor;
                ui.btnForegroundColor->setEnabled(false);
                break;
            case FOLDMARGINCOLOR:
                bg = m_foldMarginBgcolor;
                fg = m_foldMarginFgcolor;
                ui.btnForegroundColor->setEnabled(true);
                break;
            case MARGINCOLOR:
                bg = m_marginBgcolor;
                fg = m_marginFgcolor;
                ui.btnForegroundColor->setEnabled(true);
                break;
            case WHITESPACECOLOR:
                ui.btnBackgroundColor->setEnabled(false);
                fg = m_whitespaceFgcolor;
                ui.btnForegroundColor->setEnabled(true);
                break;
            case UNMATCHEDBRACECOLOR:
                bg = m_unmatchedBraceBgcolor;
                fg = m_unmatchedBraceFgcolor;
                ui.btnForegroundColor->setEnabled(true);
                break;
            case MATCHEDBRACECOLOR:
                bg = m_matchedBraceBgcolor;
                fg = m_matchedBraceFgcolor;
                ui.btnForegroundColor->setEnabled(true);
                break;
            case MARKERERRORCOLOR:
                bg = m_markerErrorBgcolor;
                ui.btnForegroundColor->setEnabled(false);
                break;
            case MARKERCURRENTCOLOR:
                bg = m_markerCurrentBgcolor;
                ui.btnForegroundColor->setEnabled(false);
                break;
            case MARKERINPUTCOLOR:
                bg = m_markerInputBgcolor;
                ui.btnForegroundColor->setEnabled(false);
                break;
            case CARETCOLOR:
                bg = m_caretBgcolor;
                fg = m_caretFgcolor;
                ui.btnForegroundColor->setEnabled(true);
                ui.checkShowCaretBackground->setVisible(true);
                break;
            case SELECTIONCOLOR:
                bg = m_selectionBgcolor;
                fg = m_selectionFgcolor;
                ui.btnForegroundColor->setEnabled(true);
                break;
            case MARKERSAMESTRINGCOLOR:
                bg = m_markerSameStringBgcolor;
                ui.btnForegroundColor->setEnabled(false);
                break;

            }

            if (ui.btnForegroundColor->isEnabled())
            {
                ui.lblSampleText->setText(tr("Sample Text"));
                ui.btnForegroundColor->setColor(fg);
            }
            else
            {
                ui.lblSampleText->setText(tr(""));
                ui.btnForegroundColor->setColor(fg);
            }

            if (ui.btnBackgroundColor->isEnabled())
            {
                ui.lblSampleText->setStyleSheet(tr("color: %1; background-color: %2;").arg(fg.name()).arg(bg.name()));
                ui.btnBackgroundColor->setColor(bg);
            }
            else
            {
                ui.lblSampleText->setStyleSheet(tr("color: %1; background-color: %2;").arg(fg.name()).arg(m_paperBgcolor.name()));
                ui.btnBackgroundColor->setColor(m_paperBgcolor);
            }
            ui.lblSampleText->repaint();
        }
        else
        {
            ui.lblSampleText->setText(tr(""));
            ui.btnForegroundColor->setEnabled(false);
            ui.btnBackgroundColor->setEnabled(false);
            ui.btnFont->setEnabled(false);
            ui.checkFillEOL->setEnabled(false);
        }
    }
    m_changing = false;
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_btnBackgroundColor_colorChanged(QColor color)
{
    if (!m_changing)
    {
        QListWidgetItem *current = ui.listWidget->currentItem();
        int index = ui.listWidget->currentIndex().row();
        if (index >= 0 && current->type() == 0)
        {
            if (color.isValid())
            {
                m_styles[index].m_backgroundColor = color;
                on_listWidget_currentItemChanged(current, NULL);
            }
        }
        else if (current->type() > 0)
        {
            switch (current->type())
            {
            case PAPERCOLOR:
                m_paperBgcolor = color;
                break;
            case FOLDMARGINCOLOR:
                m_foldMarginBgcolor = color;
                break;
            case MARGINCOLOR:
                m_marginBgcolor = color;
                break;
            case UNMATCHEDBRACECOLOR:
                m_unmatchedBraceBgcolor = color;
                break;
            case MATCHEDBRACECOLOR:
                m_matchedBraceBgcolor = color;
                break;
            case MARKERERRORCOLOR:
                m_markerErrorBgcolor = color;
                break;
            case MARKERCURRENTCOLOR:
                m_markerCurrentBgcolor = color;
                break;
            case MARKERINPUTCOLOR:
                m_markerInputBgcolor = color;
                break;
            case CARETCOLOR:
                m_caretBgcolor = color;
                break;
            case SELECTIONCOLOR:
                m_selectionBgcolor = color;
                break;
            case MARKERSAMESTRINGCOLOR:
                m_markerSameStringBgcolor = color;
                break;
            }

            on_listWidget_currentItemChanged(current, NULL);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_btnForegroundColor_colorChanged(QColor color)
{
    if (!m_changing)
    {
        QListWidgetItem *current = ui.listWidget->currentItem();
        int index = ui.listWidget->currentIndex().row();
        if (index >= 0 && current->type() == 0)
        {
            if (color.isValid())
            {
                m_styles[index].m_foregroundColor = color;
                on_listWidget_currentItemChanged(current, NULL);
            }
        }
        else if (current->type() > 0)
        {
            switch (current->type())
            {
            case FOLDMARGINCOLOR:
                m_foldMarginFgcolor = color;
                break;
            case MARGINCOLOR:
                m_marginFgcolor = color;
                break;
            case WHITESPACECOLOR:
                m_whitespaceFgcolor = color;
                break;
            case UNMATCHEDBRACECOLOR:
                m_unmatchedBraceFgcolor = color;
                break;
            case MATCHEDBRACECOLOR:
                m_matchedBraceFgcolor = color;
                break;
            case CARETCOLOR:
                m_caretFgcolor = color;
                break;
            case SELECTIONCOLOR:
                m_selectionFgcolor = color;
                break;
            }

            on_listWidget_currentItemChanged(current, NULL);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_btnFont_clicked()
{
    int index = ui.listWidget->currentIndex().row();
    if (index >= 0 && index < m_styles.size())
    {
        QFont font = m_styles[index].m_font;
        bool ok;
        font = QFontDialog::getFont(&ok, font, this);

        if (ok)
        {
            m_styles[index].m_font = font;
            on_listWidget_currentItemChanged(ui.listWidget->currentItem(), NULL);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_checkFillEOL_stateChanged(int state)
{
    int index = ui.listWidget->currentIndex().row();
    if (index >= 0 && index < m_styles.size())
    {
        m_styles[index].m_fillToEOL = (state != Qt::Unchecked);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::setFontSizeGeneral(const int fontSizeAdd)
{
    int selectedRow = ui.listWidget->currentIndex().row();

    for (int i = 0; i < m_styles.size(); i++)
    {
        m_styles[i].m_font.setPointSize(m_styles[i].m_font.pointSize() + fontSizeAdd);

        if (i == selectedRow)
        {
            on_listWidget_currentItemChanged(ui.listWidget->currentItem(), NULL);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_btnFontSizeDec_clicked()
{
    setFontSizeGeneral(-1);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_btnFontSizeInc_clicked()
{
    setFontSizeGeneral(1);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_btnReset_clicked()
{
    qSciLex = new QsciLexerPython(this);
    int selectedRow = ui.listWidget->currentIndex().row();
    int noOfStyles = qSciLex->styleBitsNeeded();
    int pos = 0;

    for (int i = 0; i < (2 << noOfStyles); i++)
    {
        if (!qSciLex->description(i).isEmpty())
        {
            m_styles[pos].m_fillToEOL = qSciLex->defaultEolFill(i);
            m_styles[pos].m_backgroundColor = qSciLex->defaultPaper(i);
            m_styles[pos].m_foregroundColor = qSciLex->defaultColor(i);
            m_styles[pos].m_font = qSciLex->defaultFont(i);
            ++pos;
        }
    }

    m_paperBgcolor = QColor(Qt::white);
    m_marginBgcolor = QColor(224, 224, 224);
    m_marginFgcolor = QColor(Qt::black);
    m_foldMarginBgcolor = QColor(Qt::white);
    m_foldMarginFgcolor = QColor(233, 233, 233);
    m_markerCurrentBgcolor = QColor(255, 255, 128);
    m_markerInputBgcolor = QColor(179, 222, 171);
    m_markerErrorBgcolor = QColor(255, 192, 192);
    m_whitespaceFgcolor = QColor(Qt::black);
    m_unmatchedBraceBgcolor = QColor(Qt::white);
    m_unmatchedBraceFgcolor = QColor(128, 0, 0);
    m_matchedBraceBgcolor = QColor(Qt::white);
    m_matchedBraceFgcolor = QColor(Qt::red);
    m_caretBgcolor = QColor(Qt::white);
    m_caretFgcolor = QColor(Qt::black);
    m_selectionBgcolor = QColor(51, 153, 255);
    m_selectionFgcolor = QColor(Qt::white);
    m_markerSameStringBgcolor = QColor(Qt::green);
    ui.checkShowCaretBackground->setChecked(false);

    on_listWidget_currentItemChanged(ui.listWidget->currentItem(), NULL);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_btnImport_clicked()
{
    static QString importFilePath;

    QString filename = QFileDialog::getOpenFileName(this, tr("Import style file"), importFilePath, "All supported style files (*.ini *.xml);;itom style file (*.ini);;Notepad++ styles (*.xml)");

    if (!filename.isEmpty())
    {
        QFileInfo fileinfo(filename);

        if (!fileinfo.exists())
        {
            QMessageBox::critical(this, tr("File does not exist."), tr("The file '%1' does not exist.").arg(filename));
        }
        else if (fileinfo.suffix() == "ini")
        {
            //read the style from a QSettings ini file
            readSettingsInternal(filename);
        }
        else
        {
            //read the style from a notepad++ xml style file
            QFile file(filename);
            if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            {
                QMessageBox::critical(this, tr("File does readable."), tr("The file '%1' cannot be opened.").arg(filename));
            }
            else
            {
                QXmlStreamReader xml(&file);

                if (xml.atEnd())
                {
                    QMessageBox::critical(this, tr("Error reading xml file."), tr("The content of the file '%1' seems to be corrupt.").arg(filename));
                }
                else
                {
                    bool LexerStylesFound = false;
                    bool PythonLexerFound = false;
                    bool GlobalStylesFound = false;
                    QVector<QXmlStreamAttributes> pythonStyles;
                    QVector<QXmlStreamAttributes> globalStyles;

                    if (xml.readNextStartElement())
                    {
                        if (xml.name() == "NotepadPlus")
                        {
                            while (xml.readNextStartElement())
                            {
                                if (xml.name() == "LexerStyles")
                                {
                                    LexerStylesFound = true;
                                    
                                    while (xml.readNextStartElement())
                                    {
                                        if (xml.name() == "LexerType" && xml.attributes().value("name") == "python")
                                        {
                                            PythonLexerFound = true;

                                            while (xml.readNextStartElement())
                                            {
                                                if (xml.name() == "WordsStyle")
                                                {
                                                    pythonStyles.append(xml.attributes());
                                                }

                                                xml.skipCurrentElement();
                                            }

                                            break;
                                        }

                                        xml.skipCurrentElement();
                                    }

                                }
                                else if (xml.name() == "GlobalStyles")
                                {
                                    GlobalStylesFound = true;

                                    while (xml.readNextStartElement())
                                    {
                                        if (xml.name() == "WidgetStyle")
                                        {
                                            globalStyles.append(xml.attributes());
                                        }

                                        xml.skipCurrentElement();
                                    }
                                    break;
                                }

                                xml.skipCurrentElement();
                            }

                            if (!LexerStylesFound)
                            {
                                xml.raiseError(tr("Missing node 'LexerStyles' as child of 'NotepadPlus' in xml file."));
                            }

                            if (!PythonLexerFound)
                            {
                                xml.raiseError(tr("Missing node 'LexerType' with name 'python' as child of 'LexerStyles' in xml file."));
                            }

                            if (!GlobalStylesFound)
                            {
                                xml.raiseError(tr("Missing node 'GlobalStyles' as child of 'NotepadPlus' in xml file."));
                            }
                        }
                        else
                        {
                            xml.raiseError(tr("The file is not a Notepad++ style file."));
                        }
                    }

                    QXmlStreamReader::Error err = xml.error();
                    if (err != QXmlStreamReader::NoError)
                    {
                        QMessageBox::critical(this, tr("Error reading xml file."), tr("The content of the file '%1' could not be properly analyzed (%2): %3").arg(filename).arg(err).arg(xml.errorString()));
                    }
                    else
                    {
                        QFont globalOverrideFont;
                        QColor globalForegroundColor;
                        QColor globalBackgroundColor;

                        foreach(const QXmlStreamAttributes &attr, globalStyles)
                        {
                            if (attr.hasAttribute("name") && attr.value("name") == "Global override")
                            {
                                globalForegroundColor = QColor(QString("#%1").arg(attr.value("fgColor").toString()));
                                
                                if (attr.hasAttribute("fontStyle"))
                                {
                                    globalOverrideFont.setBold(attr.value("fontStyle").toString().toInt() & 1);
                                    globalOverrideFont.setItalic(attr.value("fontStyle").toString().toInt() & 2);
                                    globalOverrideFont.setUnderline(attr.value("fontStyle").toString().toInt() & 4);
                                }
                                if (attr.hasAttribute("fontName"))
                                {
                                    globalOverrideFont.setFamily(attr.value("fontName").toString());
                                }
                                if (attr.hasAttribute("fontSize"))
                                {
                                    globalOverrideFont.setPointSize(attr.value("fontSize").toString().toInt());
                                }
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == "Default Style")
                            {
                                m_paperBgcolor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                                globalBackgroundColor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                                m_selectionFgcolor = globalBackgroundColor;
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == "Line number margin")
                            {
                                m_marginBgcolor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                                m_marginFgcolor = QColor(QString("#%1").arg(attr.value("fgColor").toString()));
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == "Fold margin")
                            {
                                if (attr.hasAttribute("bgColor") && !attr.value("bgColor").isEmpty())
                                {
                                    m_foldMarginBgcolor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                                }
                                else
                                {
                                    m_foldMarginBgcolor = m_marginBgcolor;
                                }
                                m_foldMarginFgcolor = QColor(QString("#%1").arg(attr.value("fgColor").toString()));
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == "Brace highlight style")
                            {
                                m_matchedBraceBgcolor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                                m_matchedBraceFgcolor = QColor(QString("#%1").arg(attr.value("fgColor").toString()));
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == "Bad brace colour")
                            {
                                m_unmatchedBraceBgcolor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                                m_unmatchedBraceFgcolor = QColor(QString("#%1").arg(attr.value("fgColor").toString()));
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == "White space symbol")
                            {
                                m_whitespaceFgcolor = QColor(QString("#%1").arg(attr.value("fgColor").toString()));
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == "Caret colour")
                            {
                                m_caretFgcolor = QColor(QString("#%1").arg(attr.value("fgColor").toString()));
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == "Current line background colour")
                            {
                                if (attr.hasAttribute("bgColor") && !attr.value("bgColor").isEmpty())
                                {
                                    m_caretBgcolor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                                }
                                else
                                {
                                    m_caretBgcolor = globalBackgroundColor;
                                }
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == "Selected text colour")
                            {
                                m_selectionBgcolor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                            }
                            
                        }

                        QVector<int> stylesFound;
                        bool ok;
                        foreach(const QXmlStreamAttributes &attr, pythonStyles)
                        {
                            for (int i = 0; i < m_styles.size(); ++i)
                            {
                                if (m_styles[i].m_index == attr.value("styleID").toString().toInt(&ok) && ok)
                                {
                                    stylesFound << m_styles[i].m_index;
                                    m_styles[i].m_fillToEOL = false;
                                    if (attr.hasAttribute("bgColor") && !attr.value("bgColor").isEmpty())
                                    {
                                        m_styles[i].m_backgroundColor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                                    }
                                    else
                                    {
                                        m_styles[i].m_backgroundColor = globalBackgroundColor;
                                    }

                                    if (attr.hasAttribute("fgColor") && !attr.value("fgColor").isEmpty())
                                    {
                                        m_styles[i].m_foregroundColor = QColor(QString("#%1").arg(attr.value("fgColor").toString()));
                                    }
                                    else
                                    {
                                        m_styles[i].m_foregroundColor = globalForegroundColor;
                                    }

                                    if (attr.hasAttribute("fontStyle"))
                                    {
                                        m_styles[i].m_font.setBold(attr.value("fontStyle").toString().toInt() & 1);
                                        m_styles[i].m_font.setItalic(attr.value("fontStyle").toString().toInt() & 2);
                                        m_styles[i].m_font.setUnderline(attr.value("fontStyle").toString().toInt() & 4);
                                    }
                                    if (attr.hasAttribute("fontName") && !attr.value("fontName").isEmpty())
                                    {
                                        m_styles[i].m_font.setFamily(attr.value("fontName").toString());
                                    }
                                    else
                                    {
                                        m_styles[i].m_font.setFamily(globalOverrideFont.family());
                                    }

                                    if (attr.hasAttribute("fontSize") && !attr.value("fontSize").isEmpty())
                                    {
                                        m_styles[i].m_font.setPointSize(attr.value("fontSize").toString().toInt());
                                    }
                                    else
                                    {
                                        m_styles[i].m_font.setPointSize(globalOverrideFont.pointSizeF());
                                    }
                                }
                            }
                        }

                        //set all unfound styles to the global overwrite style
                        for (int i = 0; i < m_styles.size(); ++i)
                        {
                            if (!stylesFound.contains(m_styles[i].m_index))
                            {
                                m_styles[i].m_fillToEOL = false;
                                m_styles[i].m_backgroundColor = globalBackgroundColor;
                                m_styles[i].m_foregroundColor = globalForegroundColor;
                                m_styles[i].m_font = globalOverrideFont.family();
                            }
                        }

                        on_listWidget_currentItemChanged(ui.listWidget->currentItem(), NULL);
                    }

                    xml.clear();
                }

                file.close();
            } 
        }

        importFilePath = QFileInfo(filename).canonicalPath();
    }
}


//-----------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_btnExport_clicked()
{
    static QString exportFilePath;

    QString filename = QFileDialog::getSaveFileName(this, tr("Export style data"), exportFilePath, "itom style file (*.ini)");

    if (filename != "")
    {
        writeSettingsInternal(filename);
        exportFilePath = QFileInfo(filename).canonicalPath();
    }
}

} //end namespace ito
