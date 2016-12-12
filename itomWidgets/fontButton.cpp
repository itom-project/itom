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
#include <QDebug>
#include <QFontDialog>
#include <QMetaEnum>
#include <QMetaObject>

// CTK includes
#include "fontButton.h"

//-----------------------------------------------------------------------------
class FontButtonPrivate
{
  Q_DECLARE_PUBLIC(FontButton);

protected:
  FontButton* const q_ptr;

public:
  FontButtonPrivate(FontButton& object);
  void init();
  void updateText();
  QString fullNameWeight()const;

  QFont Font;
  QString FontTextFormat;
};

//-----------------------------------------------------------------------------
FontButtonPrivate::FontButtonPrivate(FontButton& object)
  :q_ptr(&object)
{
  this->Font = qApp->font();
  this->FontTextFormat = "fff-sss";
}

//-----------------------------------------------------------------------------
void FontButtonPrivate::init()
{
  Q_Q(FontButton);
  QObject::connect(q, SIGNAL(clicked()), q, SLOT(browseFont()));

  this->updateText();
}


//-----------------------------------------------------------------------------
QString FontButtonPrivate::fullNameWeight()const
{
  /** \todo: QFont:Weight is not yet in the meta object system
  QMetaObject meta = QFont::staticMetaObject;
  for (int i = 0; i < meta.enumeratorCount(); ++i)
    {
    QMetaEnum metaEnum = meta.enumerator(i);
    if (metaEnum.name() == "Weight")
      {
      return QString(metaEnum.valueToKey(this->Font.weight()));
      }
    }
  */
  switch (this->Font.weight())
    {
    case QFont::Light:
      return QString("Light");
    case QFont::Normal:
      return QString("Normal");
    case QFont::DemiBold:
      return QString("DemiBold");
    case QFont::Bold:
      return QString("Bold");
    case QFont::Black:
      return QString("Black");
    default:
      return QString();
    }
  return QString();
}

//-----------------------------------------------------------------------------
void FontButtonPrivate::updateText()
{
  Q_Q(FontButton);
  QString text = this->FontTextFormat;
  text.replace("fff", this->Font.family());
  text.replace("sss", QString("%1pt").arg(this->Font.pointSize()));
  text.replace("ss", QString("%1").arg(this->Font.pointSize()));
  text.replace("www", this->fullNameWeight());
  text.replace("ww", QString("%1").arg(this->Font.weight()));
  text.replace("biu", QString("%1%2%3")
    .arg(this->Font.bold() ? 'b' : '-')
    .arg(this->Font.italic() ? 'i' : '-')
    .arg(this->Font.underline() ? 'u' : '-'));
  text.replace("bbb", this->Font.bold() ? "bold" : "");
  text.replace("bb", this->Font.bold() ? "b" : "");
  text.replace("iii", this->Font.italic() ? "italic" : "");
  text.replace("ii", this->Font.italic() ? "i" : "");
  text.replace("uuu", this->Font.underline() ? "underline" : "");
  text.replace("uu", this->Font.underline() ? "u" : "");
  q->setText(text);
}

//-----------------------------------------------------------------------------
FontButton::FontButton(QWidget * parentWidget)
  : QPushButton(parentWidget)
  , d_ptr(new FontButtonPrivate(*this))
{
  Q_D(FontButton);
  d->init();
}

//-----------------------------------------------------------------------------
FontButton::FontButton(const QFont& font,
                             QWidget * parentWidget)
  : QPushButton(parentWidget)
  , d_ptr(new FontButtonPrivate(*this))
{
  Q_D(FontButton);
  d->init();
  this->setCurrentFont(font);
}

//-----------------------------------------------------------------------------
FontButton::~FontButton()
{
}

//-----------------------------------------------------------------------------
void FontButton::setCurrentFont(const QFont& newFont)
{
  Q_D(FontButton);

  if (d->Font == newFont)
    {
    return;
    }

  d->Font = newFont;

  this->setFont(newFont);
  d->updateText();

  emit currentFontChanged(newFont);
}

//-----------------------------------------------------------------------------
QFont FontButton::currentFont()const
{
  Q_D(const FontButton);
  return d->Font;
}

//-----------------------------------------------------------------------------
void FontButton::browseFont()
{
  Q_D(FontButton);
  bool ok = false;
  QFont newFont = QFontDialog::getFont(&ok, d->Font, this);
  if (ok)
    {
    this->setCurrentFont(newFont);
    }
}

//-----------------------------------------------------------------------------
void FontButton::setFontTextFormat(const QString& fontTextFormat)
{
  Q_D(FontButton);
  d->FontTextFormat = fontTextFormat;
  d->updateText();
}

//-----------------------------------------------------------------------------
QString FontButton::fontTextFormat()const
{
  Q_D(const FontButton);
  return d->FontTextFormat;
}
