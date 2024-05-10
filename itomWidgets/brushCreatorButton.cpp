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
#include <brushCreatorButton.h>
#include <qbrush.h>
#include <QStyle>
#include <QPainter>
#include <QIcon>
#include <QStylePainter>
#include <QStyleOptionButton>
#include <brushCreatorDialog.h>
#include <QDebug>


class BrushCreatorButtonPrivate
{
    Q_DECLARE_PUBLIC(BrushCreatorButton);
protected:
    BrushCreatorButton* const q_ptr;
public:
    BrushCreatorButtonPrivate(BrushCreatorButton& object, QBrush inputBrush = QBrush(Qt::black, Qt::SolidPattern));
    QBrush brush;
    bool m_showAlphaChannel;
    QIcon m_icon;
    QSize m_iconSizeCache;
    mutable QSize m_sizeHintCache;

    void init();
    BrushCreatorDialog *dialog;
};
//---------------------------------------------------------------------------------------------------------------------------------------------------------
BrushCreatorButtonPrivate::BrushCreatorButtonPrivate(BrushCreatorButton& object, QBrush inputBrush)
    : q_ptr(&object),
    dialog(NULL),
    brush(inputBrush),
    m_showAlphaChannel(false)
{

}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
void BrushCreatorButtonPrivate::init()
{
    Q_Q(BrushCreatorButton);
    q->setCheckable(true);
    QObject::connect(q, SIGNAL(toggled(bool)), q, SLOT(onToggled(bool)));
}

//---------------------------------------------------------------------------------------------------------------------------------------------------------
BrushCreatorButton::BrushCreatorButton(QWidget* _parent) :
	QPushButton(_parent),
    d_ptr(new BrushCreatorButtonPrivate(*this))
{
    Q_D(BrushCreatorButton);
    d->init();


};
//---------------------------------------------------------------------------------------------------------------------------------------------------------
BrushCreatorButton::BrushCreatorButton(QBrush brush, QWidget* parent) :
	QPushButton(parent),
	d_ptr(new BrushCreatorButtonPrivate(*this, brush))
{
    Q_D(BrushCreatorButton);
    d->init();
}

//---------------------------------------------------------------------------------------------------------------------------------------------------------
BrushCreatorButton::~BrushCreatorButton()
{
}

//---------------------------------------------------------------------------------------------------------------------------------------------------------
void BrushCreatorButton::paintEvent(QPaintEvent* event)
{
    Q_D(BrushCreatorButton);
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
        //p.setPen(d->pen);
        p.setBrush(d->brush);
        p.drawRect(2, 2, pix.width() - 4, pix.height()-4);
        d->m_icon = QIcon(pix);
        d->m_iconSizeCache = desiredIconSize;
    }

    option.icon = d->m_icon;
    option.iconSize = desiredIconSize;
    p.drawControl(QStyle::CE_PushButton, option);
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
void BrushCreatorButton::onToggled(bool change/*= true*/)
{
    if (change)
    {
        changeBrush();
        setChecked(false);
    }

}

//-----------------------------------------------------------------------------
void BrushCreatorButton::setBrush(const QBrush &brush)
{
    Q_D(BrushCreatorButton);

    if (d->brush != brush)
    {
        d->brush = brush;
        d->m_icon = QIcon(); //invalidate icon to be recreated in computeButtonIcon
    }
}

//-----------------------------------------------------------------------------
QBrush BrushCreatorButton::getBrush() const
{
    Q_D(const BrushCreatorButton);
    return d->brush;
}


//-----------------------------------------------------------------------------
bool BrushCreatorButton::getShowAlphaChannel() const
{
    Q_D(const BrushCreatorButton);
    return d->m_showAlphaChannel;
}

//-----------------------------------------------------------------------------
void BrushCreatorButton::setShowAlphaChannel(bool showAlphaChannel)
{
    Q_D(BrushCreatorButton);

    if (d->m_showAlphaChannel != showAlphaChannel)
    {
        d->m_showAlphaChannel = showAlphaChannel;
    }
}

//-----------------------------------------------------------------------------
void BrushCreatorButton::changeBrush()
{
    Q_D(BrushCreatorButton);
    if (!d->dialog)
    {
        d->dialog = new BrushCreatorDialog(d->brush, this);
    }
    else
    {
        d->dialog->synchronizeGUI();
    }

    d->dialog->setShowAlphaChannel(d->m_showAlphaChannel);

    if (d->dialog->exec())
    {
        d->m_icon = QIcon();
    }


}
//---------------------------------------------------------------------------------------------------------------------------------------------------------
QSize BrushCreatorButton::sizeHint() const
{
    Q_D(const BrushCreatorButton);
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
