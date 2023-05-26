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
// const int FOLDMARGINCOLOR = 2; -> does not exist any more, changed via stylesheet
// const int MARGINCOLOR = 3; -> doest not exist any more, changed via stylesheet
const int WHITESPACECOLOR = 4;
const int UNMATCHEDBRACECOLOR = 5;
const int MATCHEDBRACECOLOR = 6;
const int MARKERERRORCOLOR = 7;
const int MARKERCURRENTCOLOR = 8;
const int MARKERINPUTCOLOR = 9;
const int CARETCOLOR = 10;
const int SELECTIONCOLOR = 11;
const int MARKERSAMESTRINGCOLOR = 12;
const int MARKERSCRIPTERRORCOLOR = 13;

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorStyles::WidgetPropEditorStyles(QWidget *parent) :
    AbstractPropertyPageWidget(parent),
    m_changing(false),
    m_pCodeEditorStyle(new CodeEditorStyle())
{
    ui.setupUi(this);

    //ui.lblSampleText->setBackgroundRole(QPalette::Highlight);
    ui.lblSampleText->setAutoFillBackground(false);

    QList<int> styleKeys = m_pCodeEditorStyle->styleKeys();
    int noOfStyles = m_pCodeEditorStyle->numStyles();
    StyleItem styleItem;

    for (int i = 0; i < noOfStyles; i++)
    {
        styleItem = (*m_pCodeEditorStyle)[(StyleItem::StyleType)styleKeys[i]];
        if (styleItem.name() != "")
        {
            StyleNode entry;
            entry.m_index = styleItem.type();
            entry.m_name = styleItem.name();
            entry.m_backgroundColor = styleItem.format().background().color();
            entry.m_foregroundColor = styleItem.format().foreground().color();
            entry.m_font = styleItem.format().font();

            ui.listWidget->addItem(entry.m_name);

            m_styles.push_back(entry);
        }
    }

    QListWidgetItem *separator = new QListWidgetItem("------------------------------", NULL, 1000);
    separator->setFlags(Qt::ItemIsEnabled);
    ui.listWidget->addItem(separator);

    ui.listWidget->addItem(new QListWidgetItem(tr("Paper color"), NULL, PAPERCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Caret color (Foreground: cursor color, Background: color of current line)"), NULL, CARETCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Whitespace color (if whitespace characters are visible)"), NULL, WHITESPACECOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Unmatched brace color"), NULL, UNMATCHEDBRACECOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Matched brace color"), NULL, MATCHEDBRACECOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Command line: Background for error messages"), NULL, MARKERERRORCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Command line: Background for currently executed line"), NULL, MARKERCURRENTCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Command line: Background for python input"), NULL, MARKERINPUTCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Script: Background for erroneous line"), NULL, MARKERSCRIPTERRORCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Background color and text color of current selection"), NULL, SELECTIONCOLOR));
    ui.listWidget->addItem(new QListWidgetItem(tr("Background color of words equal to the currently selected string"), NULL, MARKERSAMESTRINGCOLOR));

    ui.btnForegroundColor->setEnabled(false);
    ui.btnBackgroundColor->setEnabled(false);
    ui.btnFont->setEnabled(false);
    ui.checkShowCaretBackground->setVisible(false);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorStyles::~WidgetPropEditorStyles()
{
    DELETE_AND_SET_NULL(m_pCodeEditorStyle);
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
        settings.beginGroup("PythonLexerStyle" + QString().setNum(entry.m_index));
        settings.setValue("backgroundColor", entry.m_backgroundColor.name());
        settings.setValue("backgroundColorAlpha", entry.m_backgroundColor.alpha());
        settings.setValue("foregroundColor", entry.m_foregroundColor.name());
        settings.setValue("foregroundColorAlpha", entry.m_foregroundColor.alpha());
        settings.setValue("fontFamily", entry.m_font.family()),
        settings.setValue("pointSize", entry.m_font.pointSize()),
        settings.setValue("weight", entry.m_font.weight()),
        settings.setValue("italic", entry.m_font.italic());
        settings.endGroup();
    }

    settings.beginGroup("CodeEditor");
    settings.setValue("paperBackgroundColor", m_paperBgcolor);
    settings.setValue("caretBackgroundColor", m_caretBgcolor);
    settings.setValue("caretForegroundColor", m_caretFgcolor);
    settings.setValue("caretBackgroundShow", ui.checkShowCaretBackground->isChecked());
    settings.setValue("markerScriptErrorBackgroundColor", m_markerScriptErrorBgcolor);
    settings.setValue("markerCurrentBackgroundColor", m_markerCurrentBgcolor);
    settings.setValue("markerInputForegroundColor", m_markerInputBgcolor);
    settings.setValue("markerErrorForegroundColor", m_markerErrorBgcolor);
    settings.setValue("whitespaceForegroundColor", m_whitespaceFgcolor);
    settings.setValue("whitespaceBackgroundColor", m_whitespaceBgcolor);
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
        settings.beginGroup("PythonLexerStyle" + QString().setNum(m_styles[i].m_index));
        m_styles[i].m_backgroundColor = QColor(settings.value("backgroundColor", m_styles[i].m_backgroundColor.name()).toString());
        m_styles[i].m_backgroundColor.setAlpha(settings.value("backgroundColorAlpha", m_styles[i].m_backgroundColor.alpha()).toInt());
        m_styles[i].m_foregroundColor = QColor(settings.value("foregroundColor", m_styles[i].m_foregroundColor.name()).toString());
        m_styles[i].m_foregroundColor.setAlpha(settings.value("foregroundColorAlpha", m_styles[i].m_foregroundColor.alpha()).toInt());
        m_styles[i].m_font = QFont(
            settings.value("fontFamily", m_styles[i].m_font.family()).toString(),
            settings.value("pointSize", m_styles[i].m_font.pointSize()).toInt(),
            settings.value("weight", m_styles[i].m_font.weight()).toInt(),
            settings.value("italic", m_styles[i].m_font.italic()).toBool()
        );
        settings.endGroup();
    }

    settings.beginGroup("CodeEditor");
    //the following default values are also written in on_btnReset_clicked()
    m_paperBgcolor = QColor(settings.value("paperBackgroundColor", QColor(Qt::white)).toString());
    m_markerScriptErrorBgcolor = QColor(settings.value("markerScriptErrorBackgroundColor", QColor(255, 192, 192)).toString());
    m_markerCurrentBgcolor = QColor(settings.value("markerCurrentBackgroundColor", QColor(255, 255, 128)).toString());
    m_markerInputBgcolor = QColor(settings.value("markerInputForegroundColor", QColor(179, 222, 171)).toString());
    m_markerErrorBgcolor = QColor(settings.value("markerErrorForegroundColor", QColor(255, 192, 192)).toString());
    m_whitespaceFgcolor = QColor(settings.value("whitespaceForegroundColor", QColor(Qt::black)).toString());
    m_whitespaceBgcolor = QColor(settings.value("whitespaceBackgroundColor", QColor(Qt::white)).toString());
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
void WidgetPropEditorStyles::on_btnTextBackgroundsTransparent_clicked()
{
    for (int i = 0; i < m_styles.size(); i++)
    {
        m_styles[i].m_backgroundColor = Qt::transparent;
    }

    m_markerCurrentBgcolor = Qt::transparent;
    m_whitespaceBgcolor = Qt::transparent;
    m_unmatchedBraceBgcolor = Qt::transparent;
    m_matchedBraceBgcolor = Qt::transparent;
    m_caretBgcolor = Qt::transparent;

    on_listWidget_currentItemChanged(ui.listWidget->currentItem(), NULL);
}

//----------------------------------------------------------------------------------------------------------------------------------
QString WidgetPropEditorStyles::colorStringMixedWithPaperBgColor(const QColor &color)
{
    if (color.alpha() == 255)
    {
        return color.name();
    }
    else
    {
        float sum = m_paperBgcolor.alphaF() + color.alphaF();
        QColor mixedColor = Qt::transparent;

        if (sum > 0)
        {
            float r = m_paperBgcolor.alphaF() / (m_paperBgcolor.alphaF() + color.alphaF());

            mixedColor = QColor(
                color.red() * (1 - r) + m_paperBgcolor.red() * r,
                color.green() * (1 - r) + m_paperBgcolor.green() * r,
                color.blue() * (1 - r) + m_paperBgcolor.blue() * r,
                255
            );
        }

        return QString("rgba(%1,%2,%3,%4);"). \
            arg(mixedColor.red()). \
            arg(mixedColor.green()).arg(mixedColor.blue()).arg(mixedColor.alpha());
    }
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

            QColor bgColor = m_styles[index].m_backgroundColor;
            QColor fgColor = m_styles[index].m_foregroundColor;
            ui.btnBackgroundColor->setColor(bgColor);
            ui.btnForegroundColor->setColor(fgColor);

            ui.lblSampleText->setText(tr("Sample Text"));
            ui.lblSampleText->setFont(m_styles[index].m_font);

            ui.lblSampleText->setStyleSheet(QString("color: %1; background-color: %2;"). \
                    arg(fgColor.name()).arg(colorStringMixedWithPaperBgColor(bgColor)));

            ui.lblSampleText->repaint();

            ui.btnForegroundColor->setEnabled(true);
            ui.btnBackgroundColor->setEnabled(true);
            ui.btnFont->setEnabled(true);
            ui.checkShowCaretBackground->setVisible(false);
        }
        else if (current->type() < 1000)
        {
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
            case WHITESPACECOLOR:
                fg = m_whitespaceFgcolor;
                bg = m_whitespaceBgcolor;
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
            case MARKERSCRIPTERRORCOLOR:
                bg = m_markerScriptErrorBgcolor;
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
                ui.lblSampleText->setStyleSheet(tr("color: %1; background-color: %2;").arg(fg.name()).arg(colorStringMixedWithPaperBgColor(bg)));
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
            case UNMATCHEDBRACECOLOR:
                m_unmatchedBraceBgcolor = color;
                break;
            case MATCHEDBRACECOLOR:
                m_matchedBraceBgcolor = color;
                break;
            case MARKERERRORCOLOR:
                m_markerErrorBgcolor = color;
                break;
            case MARKERSCRIPTERRORCOLOR:
                m_markerScriptErrorBgcolor = color;
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
            case WHITESPACECOLOR:
                m_whitespaceBgcolor = color;
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

        /* workaround:

        If the current language of itom does not correspond to the OS language (see with Windows),
        the style list in the dialog is translated to the OS language, however the styleName of the
        font is in the itom language (e.g. english). Then, the styles are not correctly selected.
        Avoid this by testing all available styles of the font family and compare the resulting font with
        the given font.
        */
        QFontDatabase fdb;

        QStringList possibleStyles = fdb.styles(font.family());
        if (!possibleStyles.contains(font.styleName()))
        {
            foreach(const QString &s, possibleStyles)
            {
                QFont f = fdb.font(font.family(), s, font.pointSize());

                if (f.italic() == font.italic() &&
                    f.weight() == font.weight() &&
                    f.strikeOut() == font.strikeOut() &&
                    f.underline() == font.underline())
                {
                    font = f;
                    break;
                }
            }
        }

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
    int selectedRow = ui.listWidget->currentIndex().row();

    QList<int> styleKeys = m_pCodeEditorStyle->styleKeys();
    int noOfStyles = m_pCodeEditorStyle->numStyles();
    StyleItem styleItem;
    m_styles.clear();

    for (int i = 0; i < noOfStyles; i++)
    {
        styleItem = (*m_pCodeEditorStyle)[(StyleItem::StyleType)styleKeys[i]];
        if (styleItem.name() != "")
        {
            StyleNode entry;
            entry.m_index = styleItem.type();
            entry.m_name = styleItem.name();
            entry.m_backgroundColor = styleItem.format().background().color();
            entry.m_foregroundColor = styleItem.format().foreground().color();
            entry.m_font = styleItem.format().font();

            m_styles.push_back(entry);
        }
    }

    m_paperBgcolor = QColor(Qt::white);
    m_markerSameStringBgcolor = QColor(255, 192, 192);
    m_markerCurrentBgcolor = QColor(255, 255, 128);
    m_markerInputBgcolor = QColor(179, 222, 171);
    m_markerErrorBgcolor = QColor(255, 192, 192);
    m_markerScriptErrorBgcolor = QColor(255, 192, 192);
    m_whitespaceFgcolor = QColor(Qt::black);
    m_whitespaceBgcolor = QColor(Qt::white);
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

    if (importFilePath.isNull())
    {
        QDir basePath = QCoreApplication::applicationDirPath();
        if (basePath.cd("styles/editorThemes"))
        {
            importFilePath = basePath.canonicalPath();
        }
    }

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
                        if (xml.name() == QLatin1String("NotepadPlus"))
                        {
                            while (xml.readNextStartElement())
                            {
                                if (xml.name() == QLatin1String("LexerStyles"))
                                {
                                    LexerStylesFound = true;

                                    while (xml.readNextStartElement())
                                    {
                                        if (xml.name() == QLatin1String("LexerType") && xml.attributes().value("name") == QLatin1String("python"))
                                        {
                                            PythonLexerFound = true;

                                            while (xml.readNextStartElement())
                                            {
                                                if (xml.name() == QLatin1String("WordsStyle"))
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
                                else if (xml.name() == QLatin1String("GlobalStyles"))
                                {
                                    GlobalStylesFound = true;

                                    while (xml.readNextStartElement())
                                    {
                                        if (xml.name() == QLatin1String("WidgetStyle"))
                                        {
                                            globalStyles.append(xml.attributes());
                                        }

                                        xml.skipCurrentElement();
                                    }
                                    break;
                                }

                                xml.skipCurrentElement();
                            }

                            if (xml.error() != QXmlStreamReader::NoError)
                            {
                                //xml syntax error
                            }
                            else if (!LexerStylesFound)
                            {
                                xml.raiseError(tr("Missing node 'LexerStyles' as child of 'NotepadPlus' in xml file."));
                            }
                            else if (!PythonLexerFound)
                            {
                                xml.raiseError(tr("Missing node 'LexerType' with name 'python' as child of 'LexerStyles' in xml file."));
                            }
                            else if (!GlobalStylesFound)
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
                        if (err == QXmlStreamReader::CustomError)
                        {
                            QMessageBox::critical(this, tr("Error reading xml file."), tr("The content of the file '%1' could not be properly analyzed (%2): %3"). \
                                arg(filename).arg(err).arg(xml.errorString()));
                        }
                        else
                        {
                            QMessageBox::critical(this, tr("Error reading xml file."), tr("The content of the file '%1' could not be properly analyzed (%2): %3 in line %4, column %5"). \
                                arg(filename).arg(err). \
                                arg(xml.errorString().toHtmlEscaped()).arg(xml.lineNumber()).arg(xml.columnNumber()));
                        }
                    }
                    else
                    {
                        QFont globalOverrideFont;
                        QColor globalForegroundColor;
                        QColor globalBackgroundColor;

                        foreach(const QXmlStreamAttributes &attr, globalStyles)
                        {
                            if (attr.hasAttribute("name") && attr.value("name") == QLatin1String("Global override"))
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
                            else if (attr.hasAttribute("name") && attr.value("name") == QLatin1String("Default Style"))
                            {
                                m_paperBgcolor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                                globalBackgroundColor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                                m_selectionFgcolor = globalBackgroundColor;
                                m_whitespaceBgcolor = globalBackgroundColor;
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == QLatin1String("Line number margin"))
                            {
                                // pass
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == QLatin1String("Fold margin"))
                            {
                                // pass
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == QLatin1String("Brace highlight style"))
                            {
                                m_matchedBraceBgcolor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                                m_matchedBraceFgcolor = QColor(QString("#%1").arg(attr.value("fgColor").toString()));
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == QLatin1String("Bad brace colour"))
                            {
                                m_unmatchedBraceBgcolor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                                m_unmatchedBraceFgcolor = QColor(QString("#%1").arg(attr.value("fgColor").toString()));
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == QLatin1String("White space symbol"))
                            {
                                m_whitespaceFgcolor = QColor(QString("#%1").arg(attr.value("fgColor").toString()));
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == QLatin1String("Caret colour"))
                            {
                                m_caretFgcolor = QColor(QString("#%1").arg(attr.value("fgColor").toString()));
                            }
                            else if (attr.hasAttribute("name") && attr.value("name") == QLatin1String("Current line background colour"))
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
                            else if (attr.hasAttribute("name") && attr.value("name") == QLatin1String("Selected text colour"))
                            {
                                m_selectionBgcolor = QColor(QString("#%1").arg(attr.value("bgColor").toString()));
                            }
                        }

                        QMap<StyleItem::StyleType, int> mapIdx;
                        //Default = 0,
                        //Comment = 1,
                        //Number = 2,
                        //DoubleQuotedString = 3,
                        //SingleQuotedString = 4,
                        //Keyword = 5,
                        //TripleSingleQuotedString = 6,
                        //TripleDoubleQuotedString = 7,
                        //ClassName = 8,
                        //FunctionMethodName = 9,
                        //Operator = 10,
                        //Identifier = 11,
                        //CommentBlock = 12,
                        //UnclosedString = 13,
                        //HighlightedIdentifier = 14,
                        //Decorator = 15


                        //the numbers are the lexer style ids of QScintilla, python lexer
                        mapIdx[StyleItem::KeyDefault] = 0; //Default
                        mapIdx[StyleItem::KeyComment] = 1; //Comment
                        mapIdx[StyleItem::KeyNumber] = 2; //Number
                        mapIdx[StyleItem::KeyString] = 4; //SingleQuotedString
                        mapIdx[StyleItem::KeyKeyword] = 5; //Keyword
                        mapIdx[StyleItem::KeyDocstring] = 6; //TripleSingleQuotedString
                        mapIdx[StyleItem::KeyClass] = 8; //ClassName
                        mapIdx[StyleItem::KeyFunction] = 9; //FunctionMethodName
                        mapIdx[StyleItem::KeyOperator] = 10; //Operator
                        mapIdx[StyleItem::KeyDecorator] = 15; //Decorator
                        mapIdx[StyleItem::KeyNamespace] = 5; //Keyword
                        mapIdx[StyleItem::KeyType] = 0; //Default
                        mapIdx[StyleItem::KeyKeywordReserved] = 0; //Default
                        mapIdx[StyleItem::KeyBuiltin] = 11; //Identifier
                        mapIdx[StyleItem::KeyDefinition] = 9; //FunctionMethodName
                        mapIdx[StyleItem::KeyInstance] = 0; //Default
                        mapIdx[StyleItem::KeyTag] = 0; //Default
                        mapIdx[StyleItem::KeySelf] = 11; //Identifier
                        mapIdx[StyleItem::KeyPunctuation] = 10; //Operator
                        mapIdx[StyleItem::KeyConstant] = 9; //FunctionMethodName
                        mapIdx[StyleItem::KeyOperatorWord] = 10; //Operator
                        mapIdx[StyleItem::KeyStreamOutput] = 0; //Default
                        mapIdx[StyleItem::KeyStreamError] = 4; //SingleQuotedString

                        QVector<int> stylesFound;
                        int styleId;
                        bool ok;
                        foreach(const QXmlStreamAttributes &attr, pythonStyles)
                        {
                            for (int i = 0; i < m_styles.size(); ++i)
                            {
                                styleId = attr.value("styleID").toString().toInt(&ok);

                                if (ok && mapIdx.contains(m_styles[i].m_index) && styleId == mapIdx[m_styles[i].m_index])
                                {
                                    stylesFound << m_styles[i].m_index;
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
                                m_styles[i].m_backgroundColor = globalBackgroundColor;
                                m_styles[i].m_foregroundColor = globalForegroundColor;
                                m_styles[i].m_font = globalOverrideFont;
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

    if (exportFilePath.isNull())
    {
        QDir basePath = QCoreApplication::applicationDirPath();
        if (basePath.cd("styles/editorThemes"))
        {
            exportFilePath = basePath.canonicalPath();
        }
    }

    QString filename = QFileDialog::getSaveFileName(this, tr("Export style data"), exportFilePath, "itom style file (*.ini)");

    if (filename != "")
    {
        writeSettingsInternal(filename);
        exportFilePath = QFileInfo(filename).canonicalPath();
    }
}

} //end namespace ito
