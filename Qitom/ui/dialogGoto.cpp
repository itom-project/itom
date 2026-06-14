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
}

void DialogGoto::on_radioChar_clicked()
{
    ui.lblCaption->setText(m_captionChar);
    ui.spinValue->setMinimum(0);
    ui.spinValue->setMaximum(m_maxChar);
}

void DialogGoto::getData(bool &lineNotChar, int &curValue)
{
    lineNotChar = ui.radioLine->isChecked();
    curValue = ui.spinValue->value();
}

} //end namespace ito
