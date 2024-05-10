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

#include "brushCreatorDialog.h"
#include <QMetaEnum>
#include <QDebug>


BrushCreatorDialog::BrushCreatorDialog(QBrush &inputBrush, QWidget *parent) :
    QDialog(parent),
    brush(inputBrush)
{
    ui.setupUi(this);
    //connect(ui.buttonBox, SIGNAL(accepted()), this, SLOT(close()));

    //fill the combo boxes of the gui

    const QMetaObject *mo = qt_getEnumMetaObject(Qt::SolidPattern); //style

    QMetaEnum me = mo->enumerator(mo->indexOfEnumerator("BrushStyle"));

    int i;
    for (i = 0; i < me.keyCount(); ++i)
    {

        if (strcmp(me.key(i) ,"TexturePattern") && strcmp(me.key(i), "RadialGradientPattern") && strcmp(me.key(i), "ConicalGradientPattern") && strcmp(me.key(i), "LinearGradientPattern"))
            ui.brushCombo->addItem(me.key(i), QVariant()); //add pen styles
    }
    synchronizeGUI();
}
//-----------------------------------------------------------------------------
BrushCreatorDialog::~BrushCreatorDialog()
{

}
//-----------------------------------------------------------------------------
void BrushCreatorDialog::setBrush(const QBrush &brush)
{
    this->brush = brush;
    synchronizeGUI();
}

//-----------------------------------------------------------------------------
void BrushCreatorDialog::setShowAlphaChannel(bool showAlphaChannel)
{
    ColorPickerButton::ColorDialogOptions options = ui.colorBtn->dialogOptions();

    if (showAlphaChannel)
    {
        options |= ColorPickerButton::ShowAlphaChannel;
    }
    else
    {
        options ^= ColorPickerButton::ShowAlphaChannel;
    }

    ui.colorBtn->setDialogOptions(options);
}

//-----------------------------------------------------------------------------
void BrushCreatorDialog::synchronizeGUI()
{
    ui.colorBtn->setColor(brush.color());
    ui.brushCombo->setCurrentIndex((int)brush.style());
}
//-----------------------------------------------------------------------------
void BrushCreatorDialog::updateBrush()
{
    brush.setColor(ui.colorBtn->color());
    brush.setStyle((Qt::BrushStyle)ui.brushCombo->currentIndex());
}
//-----------------------------------------------------------------------------
QBrush BrushCreatorDialog::getBrush()
{
    return brush;
}
//-----------------------------------------------------------------------------
void BrushCreatorDialog::on_buttonBox_clicked(QAbstractButton* btn)
{

    QDialogButtonBox::ButtonRole role = ui.buttonBox->buttonRole(btn);

    if (role == QDialogButtonBox::RejectRole)
    {
        reject(); //close dialog with reject
    }
    else if (role == QDialogButtonBox::AcceptRole)
    {
        updateBrush(); //since the ok btn was cklicked we create a new pen with the adjusted properties
        accept(); //AcceptRole
    }
}
