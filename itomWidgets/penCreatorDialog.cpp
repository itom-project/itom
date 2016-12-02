/* ********************************************************************
itom measurement system
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#if QT_VERSION < 0x050500
//workaround for qt_getEnumMetaObject
//see: https://forum.qt.io/topic/644/global-qmetaobject/2
struct StaticQtMetaObject : public QObject
{
    static inline const QMetaObject& get() { return staticQtMetaObject; }
};
#endif

PenCreatorDialog::PenCreatorDialog(QPen &inputPen,QWidget *parent) :
QDialog(parent),
pen(inputPen)
{
    ui.setupUi(this);
    //connect(ui.buttonBox, SIGNAL(accepted()), this, SLOT(close()));

    //fill the combo boxes of the gui

#if QT_VERSION >= 0x050500
        const QMetaObject *mo = qt_getEnumMetaObject(Qt::DashLine);//pen style
#else
        const QMetaObject mo_ = StaticQtMetaObject::get();
        const QMetaObject *mo = &mo_;
#endif
        QMetaEnum me = mo->enumerator(mo->indexOfEnumerator("PenStyle"));

        int i;
        for (i = 0; i < me.keyCount(); ++i)
        {
            ui.styleCombo->addItem(me.key(i), QVariant()); //add pen styles
        }
#if QT_VERSION >= 0x050500
        mo = qt_getEnumMetaObject(Qt::SquareCap); //cap style
#else
        const QMetaObject mo_ = StaticQtMetaObject::get();
        const QMetaObject *mo = &mo_;
#endif
        me = mo->enumerator(mo->indexOfEnumerator("PenCapStyle"));
        for (i = 0; i < me.keyCount(); ++i)
        {
            ui.capCombo->addItem(me.key(i), QVariant()); //add cap styles
        }
#if QT_VERSION >= 0x050500
        mo = qt_getEnumMetaObject(Qt::BevelJoin); //join style
#else
        const QMetaObject mo_ = StaticQtMetaObject::get();
        const QMetaObject *mo = &mo_;
#endif
        me = mo->enumerator(mo->indexOfEnumerator("PenJoinStyle"));
        for (i = 0; i < me.keyCount(); ++i)
        {
            ui.joinCombo->addItem(me.key(i), QVariant()); //add join styles
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
//-----------------------------------------------------------------------------
void PenCreatorDialog::synchronizeGUI()
{
    ui.colorBtn->setColor(pen.color());
    ui.widthSpin->setValue(pen.widthF());
    ui.styleCombo->setCurrentIndex((int)pen.style());
    ui.capCombo->setCurrentIndex((int)pen.capStyle());
    ui.joinCombo->setCurrentIndex((int)pen.joinStyle());
}
//-----------------------------------------------------------------------------
void PenCreatorDialog::updatePen()
{
    pen.setColor(ui.colorBtn->color());
    pen.setWidthF(ui.widthSpin->value());
    pen.setStyle((Qt::PenStyle)ui.styleCombo->currentIndex());
    pen.setCapStyle((Qt::PenCapStyle)ui.capCombo->currentIndex());
    pen.setJoinStyle((Qt::PenJoinStyle)ui.joinCombo->currentIndex());
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