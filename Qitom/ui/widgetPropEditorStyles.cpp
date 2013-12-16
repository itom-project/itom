/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
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

#include "widgetPropEditorStyles.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qcolordialog.h>
#include <qfontdialog.h>
#include <qpalette.h>

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorStyles::WidgetPropEditorStyles(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
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
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorStyles::~WidgetPropEditorStyles()
{
    DELETE_AND_SET_NULL(qSciLex);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);

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
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);

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
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_listWidget_currentItemChanged(QListWidgetItem *current, QListWidgetItem * /* previous */)
{
    if (current)
    {
        int index = ui.listWidget->currentIndex().row();

        ui.checkFillEOL->setChecked(m_styles[index].m_fillToEOL);
        ui.lblSampleText->setFont(m_styles[index].m_font);
        QPalette pl = ui.lblSampleText->palette();
        pl.setColor(ui.lblSampleText->foregroundRole(), m_styles[index].m_foregroundColor);
        pl.setColor(ui.lblSampleText->backgroundRole(), m_styles[index].m_backgroundColor);
        ui.lblSampleText->setPalette(pl);
        ui.lblSampleText->repaint();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_btnBackgroundColor_clicked()
{
    int index = ui.listWidget->currentIndex().row();
    if (index >= 0)
    {
        QColor color = m_styles[index].m_backgroundColor;
        color = QColorDialog::getColor(color, this, tr("choose background color"), QColorDialog::ShowAlphaChannel);

        if (color.isValid())
        {
            m_styles[index].m_backgroundColor = color;
            on_listWidget_currentItemChanged(ui.listWidget->currentItem(), NULL);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_btnFont_clicked()
{
    int index = ui.listWidget->currentIndex().row();
    if (index >= 0)
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
void WidgetPropEditorStyles::on_btnForegroundColor_clicked()
{
    int index = ui.listWidget->currentIndex().row();
    if (index >= 0)
    {
        QColor color = m_styles[index].m_foregroundColor;
        color = QColorDialog::getColor(color, this, tr("choose foreground color"), QColorDialog::ShowAlphaChannel);

        if (color.isValid())
        {
            m_styles[index].m_foregroundColor = color;
            on_listWidget_currentItemChanged(ui.listWidget->currentItem(), NULL);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorStyles::on_checkFillEOL_stateChanged(int state)
{
    int index = ui.listWidget->currentIndex().row();
    if (index >= 0)
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

