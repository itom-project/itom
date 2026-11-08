/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut für Technische Optik (ITO),
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

#include "dialogGoto.h"

namespace ito
{

DialogGoto::DialogGoto(const int maxLine, const int curLine, const int maxChar, const int curChar, QWidget *parent) :
    QDialog(parent)
{
    ui.setupUi(this);

    setWindowModality(Qt::WindowModal);

    ui.radioLine->setChecked(true);
    m_captionLine = QString( tr("Line number (1 - %1, current: %2):")).arg(maxLine).arg(curLine);
    m_captionChar = QString( tr("Character number (0 - %1, current: %2):")).arg(maxChar).arg(curChar);
    m_maxLine = maxLine;
    m_maxChar = maxChar;
    m_maxColumn = 1;

    ui.spinColumn->setMinimum(1);
    ui.spinColumn->setMaximum(1);
    ui.spinColumn->setValue(1);

    if(m_maxChar == 0) //disable option to goto character
    {
        ui.radioChar->setVisible(false);
    }

    on_radioLine_clicked();

    ui.spinValue->setFocus();
    ui.spinValue->selectAll();
}

void DialogGoto::on_radioLine_clicked()
{
    ui.lblCaption->setText(m_captionLine);
    ui.spinValue->setMinimum(1);
    ui.spinValue->setMaximum(m_maxLine);
    ui.lblColumn->setVisible(true);
    ui.spinColumn->setVisible(true);
}

void DialogGoto::on_radioChar_clicked()
{
    ui.lblCaption->setText(m_captionChar);
    ui.spinValue->setMinimum(0);
    ui.spinValue->setMaximum(m_maxChar);
    ui.lblColumn->setVisible(false);
    ui.spinColumn->setVisible(false);
}

void DialogGoto::setColumn(const int maxColumn, const int curColumn)
{
    // the maximum is the length of the current line. A column beyond the length of the
    // target line is clipped by the editor.
    m_maxColumn = qMax(1, maxColumn);
    ui.lblColumn->setText(tr("Column (1 - %1, current: %2):").arg(m_maxColumn).arg(curColumn));
    ui.spinColumn->setMinimum(1);
    ui.spinColumn->setMaximum(qMax(m_maxColumn, 100000)); // other lines may be longer
    ui.spinColumn->setValue(qBound(1, curColumn, 100000));
}

int DialogGoto::getColumn() const
{
    return ui.radioLine->isChecked() ? ui.spinColumn->value() : 1;
}

void DialogGoto::getData(bool &lineNotChar, int &curValue)
{
    lineNotChar = ui.radioLine->isChecked();
    curValue = ui.spinValue->value();
}

} //end namespace ito
