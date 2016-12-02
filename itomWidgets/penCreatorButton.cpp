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
#include <penCreatorButton.h>
#include <qpen.h>
#include <QStyle>
#include <QPainter>
#include <QIcon>
#include <QStylePainter>
#include <QStyleOptionButton>
#include <penCreatorDialog.h>
#include <QDebug>


class PenCreatorButtonPrivate
{
    Q_DECLARE_PUBLIC(PenCreatorButton);
protected:
    PenCreatorButton* const q_ptr;
public:
    PenCreatorButtonPrivate(PenCreatorButton& object, QPen inputPen = QPen(QBrush(Qt::black), 2, Qt::SolidLine));
    QPen pen;
    QIcon icon;
    void init();
    void computeButtonIcon();
    PenCreatorDialog *dialog;
};
//---------------------------------------------------------------------------------------------------------------------------------------------------------
PenCreatorButtonPrivate::PenCreatorButtonPrivate(PenCreatorButton& object, QPen inputPen)
: q_ptr(&object),
dialog(NULL),
pen(inputPen)
{

}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
void PenCreatorButtonPrivate::init()
{
    Q_Q(PenCreatorButton);
    q->setCheckable(true);
    QObject::connect(q, SIGNAL(toggled(bool)),q, SLOT(onToggled(bool)));
    computeButtonIcon();
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
void PenCreatorButtonPrivate::computeButtonIcon()
{
    Q_Q(PenCreatorButton);
    //q->setFixedSize(200, 200);
    int iconSize = q->style()->pixelMetric(QStyle::PM_LargeIconSize);
    QPixmap pix(iconSize, iconSize);
    pix.fill(this->pen.color().isValid() ? q->palette().button().color() : Qt::transparent);
    QPainter p(&pix);
    p.setPen(pen);
    qDebug() << pen.color().name();
    //p.setBrush(this->pen.color().isValid() ?this->pen.color() : QBrush(Qt::NoBrush));
    //p.drawRect(2, 2, pix.width() - 1, pix.height() - 5);
    p.drawLine(2, 2 + ((pix.height() - 5) / 2), 2 + pix.width() - 1, 2 + ((pix.height() - 5) / 2));
    icon = QIcon(pix);

}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
PenCreatorButton::PenCreatorButton(QWidget* _parent)
    : QPushButton(_parent),
    d_ptr(new PenCreatorButtonPrivate(*this))
{
    Q_D(PenCreatorButton);
    d->init();

        
};
//---------------------------------------------------------------------------------------------------------------------------------------------------------
PenCreatorButton::PenCreatorButton(QPen pen, QWidget* parent) :
QPushButton(parent),
d_ptr(new PenCreatorButtonPrivate(*this, pen))
{
    Q_D(PenCreatorButton);
    d->init();
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
void PenCreatorButton::paintEvent(QPaintEvent* event)
{
    Q_D(PenCreatorButton);
    QStylePainter p(this);
    QStyleOptionButton option;
    this->initStyleOption(&option);
    option.icon = d->icon;
    p.drawControl(QStyle::CE_PushButton, option);
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
void PenCreatorButton::onToggled(bool change/*= true*/)
{
    if (change)
    {
        changePen();
        setChecked(false);
    }

}
//-----------------------------------------------------------------------------
void PenCreatorButton::setPen(const QPen &pen)
{
    Q_D(PenCreatorButton);
    d->pen = pen;
    d->computeButtonIcon();
}
//-----------------------------------------------------------------------------
void PenCreatorButton::changePen()
{
    Q_D(PenCreatorButton);
    if(!d->dialog)
    { 
        d->dialog = new PenCreatorDialog(d->pen,this);
    }
    else
    {
        d->dialog->synchronizeGUI();
    }
    if (d->dialog->exec())
    {
        d->computeButtonIcon();
    }
    

}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
PenCreatorButton::~PenCreatorButton()
{

}