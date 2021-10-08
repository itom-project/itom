/*=========================================================================

  Library:   CTK

  Copyright (c) Kitware Inc.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0.txt

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

=========================================================================*/

// Qt includes
#include <QApplication>
#include <QColorDialog>
#include <QDebug>
#include <QIcon>
#include <QPainter>
#include <QPixmap>
#include <QStyle>
#include <QStyleOptionButton>
#include <QStylePainter>

// CTK includes
#include "colorDialog.h"
#include "colorPickerButton.h"

class ColorPickerButtonPrivate
{
  Q_DECLARE_PUBLIC(ColorPickerButton);
protected:
  ColorPickerButton* const q_ptr;
public:
  ColorPickerButtonPrivate(ColorPickerButton& object);
  void init();
  void computeIcon();
  QString text()const;

  QIcon  Icon;
  QColor Color;
  QString ColorName;
  bool   DisplayColorName;
  ColorPickerButton::ColorDialogOptions DialogOptions;
  mutable QSize CachedSizeHint;
};

//-----------------------------------------------------------------------------
ColorPickerButtonPrivate::ColorPickerButtonPrivate(ColorPickerButton& object)
  : q_ptr(&object)
{
  this->Color = Qt::black;
  this->ColorName = QString();
  this->DisplayColorName = true;
  this->DialogOptions = ColorPickerButton::ColorDialogOption();
}

//-----------------------------------------------------------------------------
void ColorPickerButtonPrivate::init()
{
  Q_Q(ColorPickerButton);
  q->setCheckable(true);
  QObject::connect(q, SIGNAL(toggled(bool)),
                   q, SLOT(onToggled(bool)));
  this->computeIcon();
}

//-----------------------------------------------------------------------------
void ColorPickerButtonPrivate::computeIcon()
{
  Q_Q(ColorPickerButton);
  int _iconSize = q->style()->pixelMetric(QStyle::PM_SmallIconSize);
  QPixmap pix(_iconSize, _iconSize);
  pix.fill(this->Color.isValid() ?
    q->palette().button().color() : Qt::transparent);
  QPainter p(&pix);
  p.setPen(QPen(Qt::gray));
  p.setBrush(this->Color.isValid() ?
    this->Color : QBrush(Qt::NoBrush));
  p.drawRect(2, 2, pix.width() - 5, pix.height() - 5);

  this->Icon = QIcon(pix);
}

//-----------------------------------------------------------------------------
QString ColorPickerButtonPrivate::text()const
{
  Q_Q(const ColorPickerButton);
  if (!this->DisplayColorName)
    {
    return q->text();
    }
  if (this->ColorName.isEmpty())
    {
      if (this->DialogOptions & ColorPickerButton::ShowAlphaChannel)
      {
          return QString("#%1%2%3%4").arg(this->Color.alpha(), 2, 16, QLatin1Char('0')).arg(this->Color.red(), 2, 16, QLatin1Char('0')) \
              .arg(this->Color.green(), 2, 16, QLatin1Char('0')).arg(this->Color.blue(), 2, 16, QLatin1Char('0'));
      }
      else
      {
        return this->Color.name();
      }
    }
  else
    {
    return this->ColorName;
    }
}

//-----------------------------------------------------------------------------
ColorPickerButton::ColorPickerButton(QWidget* _parent)
  : QPushButton(_parent)
  , d_ptr(new ColorPickerButtonPrivate(*this))
{
  Q_D(ColorPickerButton);
  d->init();
}

//-----------------------------------------------------------------------------
ColorPickerButton::ColorPickerButton(const QString& _text, QWidget* _parent)
  : QPushButton(_text, _parent)
  , d_ptr(new ColorPickerButtonPrivate(*this))
{
  Q_D(ColorPickerButton);
  d->init();
}

//-----------------------------------------------------------------------------
ColorPickerButton::ColorPickerButton(const QColor& _color,
                                           const QString& _text,
                                           QWidget* _parent)
  : QPushButton(_text, _parent)
  , d_ptr(new ColorPickerButtonPrivate(*this))
{
  Q_D(ColorPickerButton);
  d->init();
  this->setColor(_color);
}

//-----------------------------------------------------------------------------
ColorPickerButton::~ColorPickerButton()
{
}

//-----------------------------------------------------------------------------
void ColorPickerButton::changeColor()
{
  Q_D(ColorPickerButton);
  QColor newColor;
  QString newColorName;
  QColorDialog::ColorDialogOptions options;
  options |= QColorDialog::ColorDialogOption(
    static_cast<int>(d->DialogOptions & ShowAlphaChannel));
  options |= QColorDialog::ColorDialogOption(
    static_cast<int>(d->DialogOptions & NoButtons));
  options |= QColorDialog::ColorDialogOption(
    static_cast<int>(d->DialogOptions & DontUseNativeDialog));
  if (d->DialogOptions & UseColorDialog)
    {
    newColor = ColorDialog::getColor(d->Color, this, QString(""),options);
    newColorName = ColorDialog::getColorName();
    }
  else
    {
    newColor = QColorDialog::getColor(d->Color, this, QString(""), options);
    }
  if (newColor.isValid())
    {
    this->setColor(newColor);
    this->setColorName(newColorName);
    }
}

//-----------------------------------------------------------------------------
void ColorPickerButton::onToggled(bool change)
{
  if (change)
    {
    this->changeColor();
    this->setChecked(false);
    }
}
//-----------------------------------------------------------------------------
void ColorPickerButton::setDisplayColorName(bool displayColorName)
{
  Q_D(ColorPickerButton);
  d->DisplayColorName = displayColorName;
  d->CachedSizeHint = QSize();
  this->update();
  this->updateGeometry();
}

//-----------------------------------------------------------------------------
bool ColorPickerButton::displayColorName()const
{
  Q_D(const ColorPickerButton);
  return d->DisplayColorName;
}

//-----------------------------------------------------------------------------
void ColorPickerButton::setDialogOptions(const ColorDialogOptions& options)
{
  Q_D(ColorPickerButton);
  d->DialogOptions = options;
}

//-----------------------------------------------------------------------------
const ColorPickerButton::ColorDialogOptions& ColorPickerButton::dialogOptions()const
{
  Q_D(const ColorPickerButton);
  return d->DialogOptions;
}

//-----------------------------------------------------------------------------
void ColorPickerButton::setColor(const QColor& newColor)
{
  Q_D(ColorPickerButton);
  if (newColor == d->Color)
    {
    return;
    }

  d->Color = newColor;
  d->computeIcon();

  this->update();
  emit colorChanged(d->Color);
}

//-----------------------------------------------------------------------------
QColor ColorPickerButton::color()const
{
  Q_D(const ColorPickerButton);
  return d->Color;
}

//-----------------------------------------------------------------------------
void ColorPickerButton::setColorName(const QString& newColorName)
{
  Q_D(ColorPickerButton);
  if (newColorName == d->ColorName)
    {
    return;
    }

  d->ColorName = newColorName;
  d->CachedSizeHint = QSize();
  this->update();
  this->updateGeometry();
  emit colorNameChanged(d->ColorName);
}

//-----------------------------------------------------------------------------
QString ColorPickerButton::colorName()const
{
  Q_D(const ColorPickerButton);
  return d->ColorName;
}

//-----------------------------------------------------------------------------
void ColorPickerButton::paintEvent(QPaintEvent *)
{
  Q_D(ColorPickerButton);
  QStylePainter p(this);
  QStyleOptionButton option;
  this->initStyleOption(&option);
  option.text = d->text();
  option.icon = d->Icon;
  p.drawControl(QStyle::CE_PushButton, option);
}

//-----------------------------------------------------------------------------
QSize ColorPickerButton::sizeHint()const
{
  Q_D(const ColorPickerButton);
  if (!d->DisplayColorName && !this->text().isEmpty())
    {
    return this->QPushButton::sizeHint();
    }
  if (d->CachedSizeHint.isValid())
    {
    return d->CachedSizeHint;
    }

  // If no text, the sizehint is a QToolButton sizeHint
  QStyleOptionButton pushButtonOpt;
  this->initStyleOption(&pushButtonOpt);
  pushButtonOpt.text = d->text();
  int iconSize = this->style()->pixelMetric(QStyle::PM_SmallIconSize);
  if (pushButtonOpt.text == QString())
    {
    QStyleOptionToolButton opt;
    (&opt)->QStyleOption::operator=(pushButtonOpt);
    opt.arrowType = Qt::NoArrow;
    opt.icon = d->Icon;
    opt.iconSize = QSize(iconSize, iconSize);
    opt.rect.setSize(opt.iconSize); // PM_MenuButtonIndicator depends on the height
#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
    d->CachedSizeHint = this->style()
                            ->sizeFromContents(QStyle::CT_ToolButton, &opt, opt.iconSize, this);
#else
    d->CachedSizeHint = this->style()
                            ->sizeFromContents(QStyle::CT_ToolButton, &opt, opt.iconSize, this)
                            .expandedTo(QApplication::globalStrut());
#endif
    }
  else
    {
    pushButtonOpt.icon = d->Icon;
    pushButtonOpt.iconSize = QSize(iconSize, iconSize);
    pushButtonOpt.rect.setSize(pushButtonOpt.iconSize); // PM_MenuButtonIndicator depends on the height
#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
    d->CachedSizeHint =
        (style()
             ->sizeFromContents(QStyle::CT_PushButton, &pushButtonOpt, pushButtonOpt.iconSize, this));
#else
    d->CachedSizeHint =
        (style()
             ->sizeFromContents(QStyle::CT_PushButton, &pushButtonOpt, pushButtonOpt.iconSize, this)
             .expandedTo(QApplication::globalStrut()));
#endif
    }
  return d->CachedSizeHint;
}
