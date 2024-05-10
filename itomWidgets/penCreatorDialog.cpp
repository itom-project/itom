/* ********************************************************************
itom measurement system
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2020, Institut fuer Technische Optik (ITO),
Universitaet Stuttgart, Germany

This file is part of itom.

itom is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

itom is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "penCreatorDialog.h"
#include <QPen>
#include <QMetaEnum>
#include <QDebug>

PenCreatorDialog::PenCreatorDialog(QPen &inputPen,bool colorEditable, QWidget *parent) :
QDialog(parent),
pen(inputPen)
{

    ui.setupUi(this);
    ui.colorBtn->setEnabled(colorEditable);
    //connect(ui.buttonBox, SIGNAL(accepted()), this, SLOT(close()));

    //fill the combo boxes of the gui

        const QMetaObject *mo = qt_getEnumMetaObject(Qt::DashLine);//pen style
        QMetaEnum me = mo->enumerator(mo->indexOfEnumerator("PenStyle"));

        int i;
        for (i = 0; i < me.keyCount(); ++i)
        {
            ui.styleCombo->addItem(me.key(i), me.value(i)); //add pen styles
        }

        mo = qt_getEnumMetaObject(Qt::SquareCap); //cap style
        me = mo->enumerator(mo->indexOfEnumerator("PenCapStyle"));
        for (i = 0; i < me.keyCount(); ++i)
        {
            ui.capCombo->addItem(me.key(i), me.value(i)); //add cap styles
        }

        mo = qt_getEnumMetaObject(Qt::BevelJoin); //join style
        me = mo->enumerator(mo->indexOfEnumerator("PenJoinStyle"));

        for (i = 0; i < me.keyCount(); ++i)
        {
            ui.joinCombo->addItem(me.key(i), me.value(i)); //add join styles
        }
        synchronizeGUI();
}

//-----------------------------------------------------------------------------
PenCreatorDialog::~PenCreatorDialog()
{

}

//-----------------------------------------------------------------------------
void PenCreatorDialog::setPen(const QPen &pen)
{
    this->pen = pen;
    synchronizeGUI();
}

void setCurrentData(QComboBox *comboBox, int data)
{
    for (int i = 0; i < comboBox->count(); ++i)
    {
        if (comboBox->itemData(i, Qt::UserRole).toInt() == data)
        {
            comboBox->setCurrentIndex(i);
            break;
        }
    }
}

//-----------------------------------------------------------------------------
void PenCreatorDialog::synchronizeGUI()
{
    ui.colorBtn->setColor(pen.color());
    ui.widthSpin->setValue(pen.widthF());
    setCurrentData(ui.styleCombo, (int)pen.style());
    setCurrentData(ui.capCombo, (int)pen.capStyle());
    setCurrentData(ui.joinCombo, (int)pen.joinStyle());
}
//-----------------------------------------------------------------------------
void PenCreatorDialog::updatePen()
{
    pen.setColor(ui.colorBtn->color());
    pen.setWidthF(ui.widthSpin->value());
    pen.setStyle((Qt::PenStyle)(ui.styleCombo->itemData(ui.styleCombo->currentIndex(), Qt::UserRole).toInt()));
    pen.setCapStyle((Qt::PenCapStyle)(ui.capCombo->itemData(ui.capCombo->currentIndex(), Qt::UserRole).toInt()));
    pen.setJoinStyle((Qt::PenJoinStyle)(ui.joinCombo->itemData(ui.joinCombo->currentIndex(), Qt::UserRole).toInt()));
}

//-----------------------------------------------------------------------------
QPen PenCreatorDialog::getPen()
{
    return pen;
}

//-----------------------------------------------------------------------------
void PenCreatorDialog::on_buttonBox_clicked(QAbstractButton* btn)
{

        QDialogButtonBox::ButtonRole role = ui.buttonBox->buttonRole(btn);

        if (role == QDialogButtonBox::RejectRole)
        {
            reject(); //close dialog with reject
        }
        else if (role == QDialogButtonBox::AcceptRole)
        {
            updatePen(); //since the ok btn was cklicked we create a new pen with the adjusted properties
            accept(); //AcceptRole
        }
}
