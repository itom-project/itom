/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.

    This file is a port and modified version of the
    CTK Common Toolkit (http://www.commontk.org)
*********************************************************************** */

// Qt includes
#include <QApplication>
#include <QDebug>
#include <QChildEvent>
#include <QMouseEvent>
#include <QStylePainter>
#include <QStyleOptionGroupBox>
#include <QStyle>

//  includes
#include "collapsibleGroupBox.h"

#if QT_VERSION >= 0x040600
#include "proxyStyle.h"

//-----------------------------------------------------------------------------
class CollapsibleGroupBoxStyle:public ProxyStyle
{
public:
  typedef ProxyStyle Superclass;
  CollapsibleGroupBoxStyle(QStyle* style = 0, QObject* parent =0)
    : Superclass(style, parent)
  {
  }
  virtual void drawPrimitive(PrimitiveElement pe, const QStyleOption * opt, QPainter * p, const QWidget * widget = 0) const
  {
    if (pe == QStyle::PE_IndicatorCheckBox)
      {
      const CollapsibleGroupBox* groupBox= qobject_cast<const CollapsibleGroupBox*>(widget);
      if (groupBox)
        {
        this->Superclass::drawPrimitive(groupBox->isChecked() ? QStyle::PE_IndicatorArrowDown : QStyle::PE_IndicatorArrowRight, opt, p, widget);
        return;
        }
      }
    this->Superclass::drawPrimitive(pe, opt, p, widget);
  }
  virtual int pixelMetric(PixelMetric metric, const QStyleOption * option, const QWidget * widget) const
  {
    if (metric == QStyle::PM_IndicatorHeight)
      {
      const CollapsibleGroupBox* groupBox= qobject_cast<const CollapsibleGroupBox*>(widget);
      if (groupBox)
        {
        return groupBox->fontMetrics().height();
        }
      }
    return this->Superclass::pixelMetric(metric, option, widget);
  }
};
#endif

//-----------------------------------------------------------------------------
class CollapsibleGroupBoxPrivate
{
  Q_DECLARE_PUBLIC(CollapsibleGroupBox);
protected:
  CollapsibleGroupBox* const q_ptr;
public:
  CollapsibleGroupBoxPrivate(CollapsibleGroupBox& object);
  void init();
  void setChildVisibility(QWidget* childWidget);

  /// Size of the widget for collapsing
  QSize OldSize;
  /// Maximum allowed height
  int   MaxHeight;
  int   CollapsedHeight;

  /// We change the visibility of the chidren in setChildrenVisibility
  /// and we track when the visibility is changed to force it back to possibly
  /// force the child to be hidden. To prevent infinite loop we need to know
  /// who is changing children's visibility.
  bool     ForcingVisibility;
  /// Sometimes the creation of the widget is not done inside setVisible,
  /// as we need to do special processing the first time the groupBox is
  /// setVisible, we track its created state with the variable
  bool     IsStateCreated;

#if QT_VERSION >= 0x040600
  /// Pointer to keep track of the proxy style
  CollapsibleGroupBoxStyle* GroupBoxStyle;
#endif
};

//-----------------------------------------------------------------------------
CollapsibleGroupBoxPrivate::CollapsibleGroupBoxPrivate(
  CollapsibleGroupBox& object)
  :q_ptr(&object)
{
  this->ForcingVisibility = false;
  this->IsStateCreated = false;
  this->MaxHeight = 0;
  this->CollapsedHeight = 14;
#if QT_VERSION >= 0x040600
  this->GroupBoxStyle = 0;
#endif
}

//-----------------------------------------------------------------------------
void CollapsibleGroupBoxPrivate::init()
{
  Q_Q(CollapsibleGroupBox);
  q->setCheckable(true);
  QObject::connect(q, SIGNAL(toggled(bool)), q, SLOT(expand(bool)));

  this->MaxHeight = q->maximumHeight();
#if QT_VERSION >= 0x040600
  QWidget* parent = q->parentWidget();
  QStyle* parentStyle = (parent) ? parent->style() : qApp->style();
  this->GroupBoxStyle = new CollapsibleGroupBoxStyle(parentStyle, qApp);
  q->setStyle(this->GroupBoxStyle);
  this->GroupBoxStyle->ensureBaseStyle();
#else
  this->setStyleSheet(
    "CollapsibleGroupBox::indicator:checked{"
    "image: url(:/Icons/expand-up.png);}"
    "CollapsibleGroupBox::indicator:unchecked{"
    "image: url(:/Icons/expand-down.png);}");
#endif
}
//-----------------------------------------------------------------------------
void CollapsibleGroupBoxPrivate::setChildVisibility(QWidget* childWidget)
{
  Q_Q(CollapsibleGroupBox);
  // Don't hide children while the widget is not yet created (before show() is
  // called). If we hide them (but don't set ExplicitShowHide), they would be
  // shown anyway when they will be created (because ExplicitShowHide is not set).
  // If we set ExplicitShowHide, then calling setVisible(false) on them would
  // be a no (because they are already hidden and ExplicitShowHide is set).
  // So we don't hide/show the children until the widget is created.
  if (!q->testAttribute(Qt::WA_WState_Created))
    {
    return;
    }
  this->ForcingVisibility = true;

  bool visible= !q->collapsed();
  // if the widget has been explicity hidden, then hide it.
  if (childWidget->property("visibilityToParent").isValid()
      && !childWidget->property("visibilityToParent").toBool())
    {
    visible = false;
    }

  // Setting Qt::WA_WState_Visible to true during child construction can have
  // undesirable side effects.
  if (childWidget->testAttribute(Qt::WA_WState_Created) ||
      !visible)
    {
    childWidget->setVisible(visible);
    }

  // setVisible() has set the ExplicitShowHide flag, restore it as we don't want
  // to make it like it was an explicit visible set because we want
  // to allow any children to be explicitly hidden by the user.
  if ((!childWidget->property("visibilityToParent").isValid() ||
      childWidget->property("visibilityToParent").toBool()))
    {
    childWidget->setAttribute(Qt::WA_WState_ExplicitShowHide, false);
    }
  this->ForcingVisibility = false;
}

//-----------------------------------------------------------------------------
CollapsibleGroupBox::CollapsibleGroupBox(QWidget* _parent)
  :QGroupBox(_parent)
  , d_ptr(new CollapsibleGroupBoxPrivate(*this))
{
  Q_D(CollapsibleGroupBox);
  d->init();
}

//-----------------------------------------------------------------------------
CollapsibleGroupBox::CollapsibleGroupBox(const QString& title, QWidget* _parent)
  :QGroupBox(title, _parent)
  , d_ptr(new CollapsibleGroupBoxPrivate(*this))
{
  Q_D(CollapsibleGroupBox);
  d->init();
}

//-----------------------------------------------------------------------------
CollapsibleGroupBox::~CollapsibleGroupBox()
{

}

//-----------------------------------------------------------------------------
void CollapsibleGroupBox::setCollapsedHeight(int heightInPixels)
{
  Q_D(CollapsibleGroupBox);
  d->CollapsedHeight = heightInPixels;
}

//-----------------------------------------------------------------------------
int CollapsibleGroupBox::collapsedHeight()const
{
  Q_D(const CollapsibleGroupBox);
  return d->CollapsedHeight;
}

//-----------------------------------------------------------------------------
void CollapsibleGroupBox::expand(bool _expand)
{
  Q_D(CollapsibleGroupBox);
  if (!_expand)
    {
    d->OldSize = this->size();
    }

  // Update the visibility of all the children
  // We can't use findChildren as it would return the grandchildren
  foreach(QObject* childObject, this->children())
    {
    if (childObject->isWidgetType())
      {
      d->setChildVisibility(qobject_cast<QWidget*>(childObject));
      }
    }

  if (_expand)
    {
    this->setMaximumHeight(d->MaxHeight);
    this->resize(d->OldSize);
    }
  else
    {
    d->MaxHeight = this->maximumHeight();
    QStyleOptionGroupBox option;
    this->initStyleOption(&option);
    QRect labelRect = this->style()->subControlRect(
      QStyle::CC_GroupBox, &option, QStyle::SC_GroupBoxLabel, this);
    this->setMaximumHeight(labelRect.height() + d->CollapsedHeight);
    }
}

#if QT_VERSION < 0x040600
//-----------------------------------------------------------------------------
void CollapsibleGroupBox::paintEvent(QPaintEvent* e)
{
  this->QGroupBox::paintEvent(e);

  QStylePainter paint(this);
  QStyleOptionGroupBox option;
  initStyleOption(&option);
  option.activeSubControls &= ~QStyle::SC_GroupBoxCheckBox;
  paint.drawComplexControl(QStyle::CC_GroupBox, option);

}

//-----------------------------------------------------------------------------
void CollapsibleGroupBox::mousePressEvent(QMouseEvent *event)
{
    if (event->button() != Qt::LeftButton) {
        event->ignore();
        return;
    }
    // no animation
}

//-----------------------------------------------------------------------------
void CollapsibleGroupBox::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() != Qt::LeftButton) {
        event->ignore();
        return;
    }

    QStyleOptionGroupBox box;
    initStyleOption(&box);
    box.activeSubControls &= !QStyle::SC_GroupBoxCheckBox;
    QStyle::SubControl released = style()->hitTestComplexControl(QStyle::CC_GroupBox, &box,
                                                                 event->pos(), this);
    bool toggle = this->isCheckable() && (released == QStyle::SC_GroupBoxLabel
                                   || released == QStyle::SC_GroupBoxCheckBox);
    if (toggle)
      {
      this->setChecked(!this->isChecked());
      }
}

#endif

//-----------------------------------------------------------------------------
void CollapsibleGroupBox::childEvent(QChildEvent* c)
{
  Q_D(CollapsibleGroupBox);
  QObject* child = c->child();
  if (c && c->type() == QEvent::ChildAdded &&
      child && child->isWidgetType())
    {
    QWidget *childWidget = qobject_cast<QWidget*>(c->child());
    // Handle the case where the child has already it's visibility set before
    // being added to the widget
    if (childWidget->testAttribute(Qt::WA_WState_ExplicitShowHide) &&
        childWidget->testAttribute(Qt::WA_WState_Hidden))
      {
      // if the widget has explicitly set to hidden, then mark it as such
      childWidget->setProperty("visibilityToParent", false);
      }
    // We want to catch all the child's Show/Hide events.
    child->installEventFilter(this);

    //crash for adding QSpinBox, QDoubleSpinBox, QDateTimeEdit... if calling setChildVisibility during add-operation (see https://github.com/commontk/CTK/commit/36f72607d964e8216ea967e7ae68af92fca00f1c)
    {
        // If the child is added while CollapsibleButton is collapsed, then we
        // need to hide the child.
        ////d->setChildVisibility(childWidget); <-- activate this line once the bug has been fixed
    }
    }
  this->QGroupBox::childEvent(c);
}

//-----------------------------------------------------------------------------
void CollapsibleGroupBox::setVisible(bool show)
{
  Q_D(CollapsibleGroupBox);
  // calling QWidget::setVisible() on CollapsibleGroupBox will eventually
  // call QWidget::showChildren() or hideChildren() which will generate
  // ShowToParent/HideToParent events but we want to ignore that case in
  // eventFilter().
  d->ForcingVisibility = true;
  this->QGroupBox::setVisible(show);
  d->ForcingVisibility = false;
  // We have been ignoring setChildVisibility() while the collapsible button
  // is not yet created, now that it is created, ensure that the children
  // are correctly shown/hidden depending on their explicit visibility and
  // the collapsed property of the button.
  if (!d->IsStateCreated && this->testAttribute(Qt::WA_WState_Created))
    {
    d->IsStateCreated = true;
    foreach(QObject* child, this->children())
      {
      QWidget* childWidget = qobject_cast<QWidget*>(child);
      if (childWidget)
        {
        d->setChildVisibility(childWidget);
        }
      }
    }
}

//-----------------------------------------------------------------------------
bool CollapsibleGroupBox::eventFilter(QObject* child, QEvent* e)
{
  Q_D(CollapsibleGroupBox);
  Q_ASSERT(child && e);
  // Make sure the Show/QHide events are not generated by one of our
  // CollapsibleButton function.
  if (d->ForcingVisibility)
    {
    return false;
    }
  // When we are here, it's because somewhere (not in CollapsibleButton),
  // someone explicitly called setVisible() on a child widget.
  // If the collapsible button is collapsed/closed, then even if someone
  // request the widget to be visible, we force it back to be hidden because
  // they meant to be hidden to its parent, the collapsible button. However the
  // child will later be shown when the button will be expanded/opened.
  // On the other hand, if the user explicitly hide the child when the button
  // is collapsed/closed, then we want to keep it hidden next time the
  // collapsible button is expanded/opened.
  if (e->type() == QEvent::ShowToParent)
    {
    child->setProperty("visibilityToParent", true);
    Q_ASSERT(qobject_cast<QWidget*>(child));
    // force the widget to be hidden if the button is collapsed.
    d->setChildVisibility(qobject_cast<QWidget*>(child));
    }
  else if(e->type() == QEvent::HideToParent)
    {
    // we don't need to force the widget to be visible here.
    child->setProperty("visibilityToParent", false);
    }
  return this->QGroupBox::eventFilter(child, e);
}
