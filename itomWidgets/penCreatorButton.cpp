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


class PenCreatorButtonPrivate
{
    Q_DECLARE_PUBLIC(PenCreatorButton);
protected:
    PenCreatorButton* const q_ptr;
public:
    PenCreatorButtonPrivate(PenCreatorButton& object);
    QPen pen;
    QIcon icon;
    void init();
    void computeButtonIcon();
};
//---------------------------------------------------------------------------------------------------------------------------------------------------------
PenCreatorButtonPrivate::PenCreatorButtonPrivate(PenCreatorButton& object)
: q_ptr(&object),
pen(QBrush(Qt::gray), 2, Qt::SolidLine)
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
    int iconSize = q->style()->pixelMetric(QStyle::PM_SmallIconSize);
    QPixmap pix(iconSize, iconSize);
    pix.fill(this->pen.color().isValid() ? q->palette().button().color() : Qt::transparent);
    QPainter p(&pix);
    p.setPen(QPen(Qt::gray));
    p.setBrush(this->pen.color().isValid() ?this->pen.color() : QBrush(Qt::NoBrush));
    p.drawRect(2, 2, pix.width() - 5, pix.height() - 5);
    p.drawLine(2 + ((pix.height() - 5) / 2), 2, 2 + ((pix.height() - 5) / 2), 2 + pix.width() - 5);
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

}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
PenCreatorButton::~PenCreatorButton()
{

}