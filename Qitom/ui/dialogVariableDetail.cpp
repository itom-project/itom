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

#include "dialogVariableDetail.h"

#include <qclipboard.h>
#include <qfont.h>
#include <qfontdatabase.h>

namespace ito
{

//----------------------------------------------------------------------------------------------
DialogVariableDetail::DialogVariableDetail(const QString &name, const QString &type, const QString &value, QWidget *parent) :
    QDialog(parent)
{
    // show maximize button
    setWindowFlags(windowFlags() |
        Qt::CustomizeWindowHint |
        Qt::WindowMaximizeButtonHint |
        Qt::WindowCloseButtonHint);

    ui.setupUi(this);

    // get a default monospace font for the fields
    const QFont fixedFont = QFontDatabase::systemFont(QFontDatabase::FixedFont);
    ui.txtName->setFont(fixedFont);
    ui.txtType->setFont(fixedFont);
    ui.txtValue->setFont(fixedFont);

    ui.txtName->setText(name);
    ui.txtType->setText(type);

    // if the text contains a line break, disable any wrapping, else
    // wrap at widget level.
    if (value.contains("\n"))
    {
        ui.txtValue->setLineWrapMode(QPlainTextEdit::NoWrap);
    }
    else
    {
        ui.txtValue->setLineWrapMode(QPlainTextEdit::WidgetWidth);
    }

    ui.txtValue->setPlainText(value);
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetail::on_btnCopyClipboard_clicked()
{
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(ui.txtName->text(), QClipboard::Clipboard);
}

} //end namespace ito
