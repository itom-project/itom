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

#include "dialogReplace.h"
#include "../AppManagement.h"
#include <qcompleter.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
DialogReplace::DialogReplace(QWidget *parent) :
    QDialog(parent),
    m_pCompleter(NULL)
{
    int size = 0;
    ui.setupUi(this);

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("dialogReplace");
    size = settings.beginReadArray("lastFindText");
    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);
        ui.comboBoxFindText->addItem(settings.value("find", QString()).toString());
    }
    settings.endArray();

    size = settings.beginReadArray("lastReplacedText");
    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);
        ui.comboBoxReplacedText->addItem(settings.value("replace", QString()).toString());
    }
    settings.endArray();

    ui.checkBoxWholeWord->setChecked(settings.value("WholeWord").toBool());
    ui.checkBoxCase->setChecked(settings.value("Case").toBool());
    ui.checkBoxWrapAround->setChecked(settings.value("WrapAround").toBool());
    ui.checkBoxRegular->setChecked(settings.value("Regular").toBool());
    ui.checkBoxReplaceWith->setChecked(settings.value("ReplaceWith").toBool());

    if (settings.value("Direction").toString() == "Up")
    {
        ui.radioButtonUp->setChecked(true);
    }
    else
    {
        ui.radioButtonDown->setChecked(true);
    }

    settings.endGroup();

    on_checkBoxReplaceWith_clicked();
    ui.groupBoxOptions->setVisible(false);
    setMaximumHeight(100);

    m_pCompleter = new QCompleter(this);
    m_pCompleter->setCompletionMode(QCompleter::InlineCompletion);
    m_pCompleter->setCaseSensitivity(Qt::CaseSensitive);
    ui.comboBoxFindText->setCompleter(m_pCompleter);
    ui.comboBoxReplacedText->setCompleter(m_pCompleter);
}

//----------------------------------------------------------------------------------------------------------------------------------
int DialogReplace::comboBoxGetIndex(const QString &text, QComboBox *comboBox) const
{
    Qt::CaseSensitivity cs = Qt::CaseInsensitive;
    if (ui.checkBoxCase->isChecked())
    {
        cs = Qt::CaseSensitive;
    }

    int ret = 0;
    while (ret < comboBox->count() && text.compare(comboBox->itemText(ret), cs) != 0)
    {
        ret++;
    }

    if (ret == comboBox->count())
    {
        ret = -1;
    }

    return ret;
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogReplace::comboBoxAddItem(const QString &text, QComboBox *comboBox)
{
    if (text != "")
    {
        Qt::CaseSensitivity cs = Qt::CaseInsensitive;
        if (ui.checkBoxCase->isChecked())
        {
            cs = Qt::CaseSensitive;
        }

        int x = 0;
        while (x < comboBox->count() && text.compare(comboBox->itemText(x), cs) != 0)
        {
            x++;
        }

        if (x == comboBox->count())
        {
            if (comboBox->count() == 0)
            {
                comboBox->addItem(text);
            }
            else
            {
                comboBox->insertItem(0, text);
                if (comboBox->count() < 20)
                {
                    comboBox->removeItem(21);
                }
            }
        }
        else
        {
            if (x > 0)
            {
                comboBox->removeItem(x);
                comboBox->insertItem(0, text);
            }
        }

        comboBox->setCurrentIndex(0);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogReplace::on_checkBoxReplaceWith_clicked()
{
    bool isChecked = ui.checkBoxReplaceWith->isChecked();

    ui.comboBoxReplacedText->setEnabled(isChecked);
    ui.pushButtonReplace->setEnabled(isChecked);
    ui.pushButtonReplaceAll->setEnabled(isChecked);
    ui.checkBoxReplaceInSelection->setEnabled(isChecked);
}

//----------------------------------------------------------------------------------------------------------------------------------
//void DialogReplace::setData(const QString &defaultText, const int &lineFrom, const int &indexFrom, const int &lineTo, const int &indexTo)
void DialogReplace::setData(const QString &defaultText, const bool &rowSelected)
{
//    if (lineTo == lineFrom)
/*    if (rowSelected)
    {
//        m_lineFrom = -1;
        ui.comboBoxFindIn->setCurrentIndex(0);
    }
    else
    {
//        m_lineFrom = lineFrom;
        ui.comboBoxFindIn->setCurrentIndex(1);
    }*/
/*    m_indexFrom = indexFrom;
    m_lineTo = lineTo;
    m_indexTo = indexTo;*/

    int index = comboBoxGetIndex(defaultText, ui.comboBoxFindText);
    if (index != -1)
    {
        ui.comboBoxFindText->setCurrentIndex(index);
    }
    else
    {
        if (defaultText != "")
        {
            ui.comboBoxFindText->setEditText(defaultText);
        }
        else
        {
            ui.comboBoxFindText->setCurrentIndex(0);
        }
    }
    this->activateWindow();
    ui.comboBoxFindText->setFocus();
    ui.comboBoxFindText->lineEdit()->selectAll();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogReplace::closeEvent(QCloseEvent * event)
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("dialogReplace");
    settings.beginWriteArray("lastFindText");
    for (int i = 0; i < ui.comboBoxFindText->count(); i++)
    {
        settings.setArrayIndex(i);
        settings.setValue("find", ui.comboBoxFindText->itemText(i));
    }
    settings.endArray();

    settings.beginWriteArray("lastReplacedText");
    for (int i = 0; i < ui.comboBoxReplacedText->count(); i++)
    {
        settings.setArrayIndex(i);
        settings.setValue("replace", ui.comboBoxReplacedText->itemText(i));
    }
    settings.endArray();

    settings.setValue("WholeWord", ui.checkBoxWholeWord->isChecked());
    settings.setValue("Case", ui.checkBoxCase->isChecked());
    settings.setValue("WrapAround", ui.checkBoxWrapAround->isChecked());
    settings.setValue("Regular", ui.checkBoxRegular->isChecked());
    settings.setValue("ReplaceWith", ui.checkBoxReplaceWith->isChecked());

    if (ui.radioButtonUp->isChecked())
    {
        settings.setValue("Direction", "Up");
    }
    else
    {
        settings.setValue("Direction", "Down");
    }

    settings.endGroup();

    event->accept();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogReplace::on_pushButtonFindNext_clicked()
{
    comboBoxAddItem(ui.comboBoxFindText->currentText(), ui.comboBoxFindText);

    bool regExpr = ui.checkBoxRegular->isChecked();
    bool caseSensitive = ui.checkBoxCase->isChecked();
    bool wholeWord = ui.checkBoxWholeWord->isChecked();
    bool wrap = ui.checkBoxWrapAround->isChecked();
    bool forward = ui.radioButtonDown->isChecked();
    emit findNext(ui.comboBoxFindText->currentText(), regExpr, caseSensitive, wholeWord, wrap, forward, false);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogReplace::on_pushButtonReplace_clicked()
{
    comboBoxAddItem(ui.comboBoxFindText->currentText(), ui.comboBoxFindText);
    comboBoxAddItem(ui.comboBoxReplacedText->currentText(), ui.comboBoxReplacedText);

    bool regExpr = ui.checkBoxRegular->isChecked();
    bool caseSensitive = ui.checkBoxCase->isChecked();
    bool wholeWord = ui.checkBoxWholeWord->isChecked();
    bool wrap = ui.checkBoxWrapAround->isChecked();
    bool forward = ui.radioButtonDown->isChecked();

    emit replaceSelection(ui.comboBoxFindText->currentText(), ui.comboBoxReplacedText->currentText());
    emit findNext(ui.comboBoxFindText->currentText(), regExpr, caseSensitive, wholeWord, wrap, forward, false);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogReplace::on_pushButtonReplaceAll_clicked()
{
    comboBoxAddItem(ui.comboBoxFindText->currentText(), ui.comboBoxFindText);
    comboBoxAddItem(ui.comboBoxReplacedText->currentText(), ui.comboBoxReplacedText);

    bool regExpr = ui.checkBoxRegular->isChecked();
    bool caseSensitive = ui.checkBoxCase->isChecked();
    bool wholeWord = ui.checkBoxWholeWord->isChecked();
    bool findInSel = ui.checkBoxReplaceInSelection->isChecked();
    //    int findIn = ui.comboBoxFindIn->currentIndex();

    emit replaceAll(ui.comboBoxFindText->currentText(), ui.comboBoxReplacedText->currentText(), regExpr, caseSensitive, wholeWord, findInSel);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogReplace::on_pushButtonExpand_clicked()
{
    if (ui.groupBoxOptions->isVisible())
    {
        QRect rect = geometry();
        ui.groupBoxOptions->setVisible(false);
        ui.pushButtonExpand->setText(tr("Expand"));

        setVisible(false);
        setMaximumHeight(100);
        setVisible(true);
        setGeometry(rect);
    }
    else
    {
        ui.groupBoxOptions->setVisible(true);
        ui.pushButtonExpand->setText(tr("Collapse"));
    }
}

} //end namespace ito
