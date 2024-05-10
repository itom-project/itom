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
    QIcon m_icon;
    QSize m_iconSizeCache;
    bool m_colorEditable;
    mutable QSize m_sizeHintCache;

    void init();
    PenCreatorDialog *dialog;
};
//---------------------------------------------------------------------------------------------------------------------------------------------------------
PenCreatorButtonPrivate::PenCreatorButtonPrivate(PenCreatorButton& object, QPen inputPen)
: q_ptr(&object),
    dialog(NULL),
    pen(inputPen),
    m_colorEditable(true)
{

}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
void PenCreatorButtonPrivate::init()
{
    Q_Q(PenCreatorButton);
    q->setCheckable(true);
    QObject::connect(q, SIGNAL(toggled(bool)),q, SLOT(onToggled(bool)));
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
    option.text = "";

    QSize desiredIconSize;
    int iconSize = this->style()->pixelMetric(QStyle::PM_SmallIconSize);
    desiredIconSize.setHeight(iconSize);
    desiredIconSize.setWidth(option.rect.width() - (option.rect.height() - iconSize));

    if (d->m_icon.isNull() || d->m_iconSizeCache != desiredIconSize)
    {
        QPixmap pix(desiredIconSize);
        //pix.fill(d->pen.color().isValid() ? palette().button().color() : Qt::transparent);
        pix.fill(Qt::transparent);
        QPainter p(&pix);
        p.setPen(d->pen);


        p.drawLine(2, 2 + ((pix.height() - 5) / 2), pix.width()-2, 2 + ((pix.height() - 5) / 2));
        d->m_icon = QIcon(pix);
        d->m_iconSizeCache = desiredIconSize;
    }

    option.icon = d->m_icon;
    option.iconSize = desiredIconSize;
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

    if (d->pen != pen)
    {
        d->pen = pen;
        d->m_icon = QIcon(); //invalidate icon to be recreated in computeButtonIcon
    }
}

//-----------------------------------------------------------------------------
QPen PenCreatorButton::getPen() const
{
    Q_D(const PenCreatorButton);
    return d->pen;
}


//-----------------------------------------------------------------------------
void PenCreatorButton::changePen()
{
    Q_D(PenCreatorButton);

    d->dialog = new PenCreatorDialog(d->pen, d->m_colorEditable , this);
    if (d->dialog->exec())
    {
        d->m_icon = QIcon();

    }
    delete d->dialog;
    d->dialog = NULL;



}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
QSize PenCreatorButton::sizeHint() const
{
    Q_D(const PenCreatorButton);
    if (d->m_sizeHintCache.isValid())
    {
        return d->m_sizeHintCache;
    }

    // If no text, the sizehint is a QToolButton sizeHint
    QStyleOptionButton pushButtonOpt;
    this->initStyleOption(&pushButtonOpt);
    pushButtonOpt.text = "";
    int iconSize = this->style()->pixelMetric(QStyle::PM_SmallIconSize);

    QStyleOptionToolButton opt;
    (&opt)->QStyleOption::operator=(pushButtonOpt);
    opt.arrowType = Qt::NoArrow;
    opt.icon = d->m_icon;
    opt.iconSize = QSize(iconSize, iconSize);
    opt.rect.setSize(opt.iconSize); // PM_MenuButtonIndicator depends on the height
#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
    d->m_sizeHintCache = this->style()
                             ->sizeFromContents(QStyle::CT_ToolButton, &opt, opt.iconSize, this);
#else
    d->m_sizeHintCache = this->style()
                             ->sizeFromContents(QStyle::CT_ToolButton, &opt, opt.iconSize, this)
                             .expandedTo(QApplication::globalStrut());
#endif
    return d->m_sizeHintCache;
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
void PenCreatorButton::setColorState(const bool &val)
{
    Q_D(PenCreatorButton);
    d->m_colorEditable = val;


}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
bool PenCreatorButton::getColorState() const
{
    Q_D(const PenCreatorButton);
    return d->m_colorEditable;
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
PenCreatorButton::~PenCreatorButton()
{

}
