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
    Common framework (http://www.commontk.org)
*********************************************************************** */

// Qt includes
#include <QApplication>
#include <QEvent>
#include <QHeaderView>
#include <QKeyEvent>
#include <QLayout>
#include <QScrollBar>
#include <QMouseEvent>
#include <QModelIndex>
#include <QStack>
#include <QTreeView>
#include <QDebug>
#include <QScreen>
#include <QWindow>

// CTK includes
#include "treeComboBox.h"

// -------------------------------------------------------------------------
class TreeComboBoxPrivate
{
  Q_DECLARE_PUBLIC(TreeComboBox);
protected:
  TreeComboBox* const q_ptr;
public:
  TreeComboBoxPrivate(TreeComboBox& object);
  int computeWidthHint()const;

  bool SkipNextHide;
  bool RootSet;
  bool SendCurrentItem;
  QPersistentModelIndex Root;
  int VisibleModelColumn;
};

// -------------------------------------------------------------------------
TreeComboBoxPrivate::TreeComboBoxPrivate(TreeComboBox& object)
  :q_ptr(&object)
{
  this->SkipNextHide = false;
  this->RootSet = false;
  this->SendCurrentItem = false;
  this->VisibleModelColumn = -1; // all visible by default
}

// -------------------------------------------------------------------------
int TreeComboBoxPrivate::computeWidthHint()const
{
  Q_Q(const TreeComboBox);
  return q->view()->sizeHintForColumn(q->modelColumn());
}

// -------------------------------------------------------------------------
TreeComboBox::TreeComboBox(QWidget* _parent):Superclass(_parent)
  , d_ptr(new TreeComboBoxPrivate(*this))
{
  QTreeView* treeView = new QTreeView(this);
  treeView->setHeaderHidden(true);
  this->setView(treeView);
  // we install the filter AFTER the QComboBox installed it.
  // so that our eventFilter will be called first
  this->view()->viewport()->installEventFilter(this);
  connect(treeView, SIGNAL(collapsed(QModelIndex)),
          this, SLOT(resizePopup()));
  connect(treeView, SIGNAL(expanded(QModelIndex)),
          this, SLOT(resizePopup()));
}

// -------------------------------------------------------------------------
TreeComboBox::~TreeComboBox()
{
}

// -------------------------------------------------------------------------
int TreeComboBox::visibleModelColumn()const
{
  Q_D(const TreeComboBox);
  return d->VisibleModelColumn;
}

// -------------------------------------------------------------------------
void TreeComboBox::setVisibleModelColumn(int index)
{
  Q_D(TreeComboBox);
  d->VisibleModelColumn = index;
}

// -------------------------------------------------------------------------
bool TreeComboBox::eventFilter(QObject* object, QEvent* _event)
{
  Q_D(TreeComboBox);
  Q_UNUSED(object);
  bool res = false;
  d->SendCurrentItem = false;
  switch (_event->type())
    {
    default:
      break;
    case QEvent::ShortcutOverride:
      switch (static_cast<QKeyEvent*>(_event)->key())
        {
        case Qt::Key_Enter:
        case Qt::Key_Return:
        case Qt::Key_Select:
          d->SendCurrentItem = true;
          break;
        default:
          break;
        }
      break;
    case QEvent::MouseButtonRelease:
      QMouseEvent* mouseEvent = dynamic_cast<QMouseEvent*>(_event);
      QModelIndex index = this->view()->indexAt(mouseEvent->pos());
      // do we click the branch (+ or -) or the item itself ?
      if (this->view()->model()->hasChildren(index) &&
          (index.flags() & Qt::ItemIsSelectable) &&
          !this->view()->visualRect(index).contains(mouseEvent->pos()))
        {//qDebug() << "Set skip on";
        // if the branch is clicked, then we don't want to close the
        // popup. (we don't want to select the item, just expand it.)
        // of course, all that doesn't apply with unselectable items, as
        // they won't close the popup.
        d->SkipNextHide = true;
        }

      // we want to get rid of an odd behavior.
      // If the user highlight a selectable item and then
      // click on the branch of an unselectable item while keeping the
      // previous selection. The popup would be normally closed in that
      // case. We don't want that.
      if ( this->view()->model()->hasChildren(index) &&
           !(index.flags() & Qt::ItemIsSelectable) &&
           !this->view()->visualRect(index).contains(mouseEvent->pos()))
        {//qDebug() << "eat";
        // eat the event, don't go to the QComboBox event filters.
        res = true;
        }

      d->SendCurrentItem = this->view()->rect().contains(mouseEvent->pos()) &&
        this->view()->currentIndex().isValid() &&
        (this->view()->currentIndex().flags() & Qt::ItemIsEnabled) &&
        (this->view()->currentIndex().flags() & Qt::ItemIsSelectable);
      break;
    }
  return res;
}

// -------------------------------------------------------------------------
void TreeComboBox::showPopup()
{
  Q_D(TreeComboBox);
  QHeaderView* header = qobject_cast<QTreeView*>(this->view())->header();
  for (int i = 0; i < header->count(); ++i)
    {
    header->setSectionHidden(i, d->VisibleModelColumn != -1 &&
                                i != d->VisibleModelColumn);
    }
  this->QComboBox::showPopup();
  emit this->popupShow();
}

// -------------------------------------------------------------------------
void TreeComboBox::hidePopup()
{
  Q_D(TreeComboBox);

  if (d->SkipNextHide)
    {// don't hide the popup if the selected item is a parent.
    d->SkipNextHide = false;
    //this->setCurrentIndex(-1);
    //qDebug() << "skip";
    //this->QComboBox::showPopup();
    }
  else
    {
    //QModelIndex _currentIndex = this->view()->currentIndex();
    //qDebug() << "TreeComboBox::hidePopup() " << _currentIndex << " " << _currentIndex.row();
    //qDebug() << "before: " << this->currentIndex() << this->view()->currentIndex();
    this->QComboBox::hidePopup();
    //qDebug() << "after: " << this->currentIndex() << this->view()->currentIndex();
    //this->setRootModelIndex(_currentIndex.parent());
    //this->setCurrentIndex(_currentIndex.row());
    if (d->SendCurrentItem)
      {
      d->SendCurrentItem = false;
      QKeyEvent event(QEvent::ShortcutOverride, Qt::Key_Enter, Qt::NoModifier);
      QApplication::sendEvent(this->view(), &event);
      }
    emit this->popupHide();
    //qDebug() << "after2: " << this->currentIndex() << this->view()->currentIndex();
    }
}

// -------------------------------------------------------------------------
void TreeComboBox::paintEvent(QPaintEvent *p)
{
  //qDebug() << __FUNCTION__ << " " << this->currentText() << " " << this->currentIndex() ;
  //qDebug() << this->itemText(0) << this->itemText(1);
  this->QComboBox::paintEvent(p);
}

// -------------------------------------------------------------------------
QTreeView* TreeComboBox::treeView()const
{
  return qobject_cast<QTreeView*>(this->view());
}

// -------------------------------------------------------------------------
QWindow* windowFromWidget(const QWidget* widget)
{
    // https://phabricator.kde.org/D22379
    QWindow* windowHandle = widget->windowHandle();
    if (windowHandle)
    {
        return windowHandle;
    }

    const QWidget* nativeParent = widget->nativeParentWidget();
    if (nativeParent)
    {
        return nativeParent->windowHandle();
    }

    return nullptr;
}

// -------------------------------------------------------------------------
QScreen* screenFromWidget(const QWidget* widget)
{
    // https://phabricator.kde.org/D22379
    const QWindow* windowHandle = windowFromWidget(widget);
    if (windowHandle && windowHandle->screen())
    {
        return windowHandle->screen();
    }

    return QGuiApplication::primaryScreen();
}

// -------------------------------------------------------------------------
void TreeComboBox::resizePopup()
{
  // copied from QComboBox.cpp
  Q_D(TreeComboBox);

  QStyle * const style = this->style();
  QWidget* container = qobject_cast<QWidget*>(this->view()->parent());

  QStyleOptionComboBox opt;
  this->initStyleOption(&opt);
  QRect listRect(style->subControlRect(QStyle::CC_ComboBox, &opt,
                                       QStyle::SC_ComboBoxListBoxPopup, this));
  QRect screen = screenFromWidget(this)->geometry();
  QPoint below = this->mapToGlobal(listRect.bottomLeft());
  int belowHeight = screen.bottom() - below.y();
  QPoint above = this->mapToGlobal(listRect.topLeft());
  int aboveHeight = above.y() - screen.y();
  bool boundToScreen = !this->window()->testAttribute(Qt::WA_DontShowOnScreen);

  const bool usePopup = style->styleHint(QStyle::SH_ComboBox_Popup, &opt, this);
    {
    int listHeight = 0;
    int count = 0;
    QStack<QModelIndex> toCheck;
    toCheck.push(this->view()->rootIndex());
#ifndef QT_NO_TREEVIEW
    QTreeView *treeView = qobject_cast<QTreeView*>(this->view());
    if (treeView && treeView->header() && !treeView->header()->isHidden())
      listHeight += treeView->header()->height();
#endif
    while (!toCheck.isEmpty())
      {
      QModelIndex parent = toCheck.pop();
      for (int i = 0; i < this->model()->rowCount(parent); ++i)
        {
        QModelIndex idx = this->model()->index(i, this->modelColumn(), parent);
        if (!idx.isValid())
          {
          continue;
          }
        listHeight += this->view()->visualRect(idx).height(); /* + container->spacing() */;
#ifndef QT_NO_TREEVIEW
        if (this->model()->hasChildren(idx) && treeView && treeView->isExpanded(idx))
          {
          toCheck.push(idx);
          }
#endif
        ++count;
        if (!usePopup && count > this->maxVisibleItems())
          {
          toCheck.clear();
          break;
          }
        }
      }
    listRect.setHeight(listHeight);
    }
      {
      // add the spacing for the grid on the top and the bottom;
      int heightMargin = 0;//2*container->spacing();

      // add the frame of the view
      QMargins margins = this->view()->contentsMargins();
      heightMargin += margins.top() + margins.bottom();

      listRect.setHeight(listRect.height() + heightMargin);
      }

      // Add space for margin at top and bottom if the style wants it.
      if (usePopup)
        {
        listRect.setHeight(listRect.height() + style->pixelMetric(QStyle::PM_MenuVMargin, &opt, this) * 2);
        }

      // Make sure the popup is wide enough to display its contents.
      if (usePopup)
        {
        const int diff = d->computeWidthHint() - this->width();
        if (diff > 0)
          {
          listRect.setWidth(listRect.width() + diff);
          }
        }

      //we need to activate the layout to make sure the min/maximum size are set when the widget was not yet show
      container->layout()->activate();
      //takes account of the minimum/maximum size of the container
      listRect.setSize( listRect.size().expandedTo(container->minimumSize())
                        .boundedTo(container->maximumSize()));

      // make sure the widget fits on screen
      if (boundToScreen)
        {
        if (listRect.width() > screen.width() )
          {
          listRect.setWidth(screen.width());
          }
        if (this->mapToGlobal(listRect.bottomRight()).x() > screen.right())
          {
          below.setX(screen.x() + screen.width() - listRect.width());
          above.setX(screen.x() + screen.width() - listRect.width());
          }
        if (this->mapToGlobal(listRect.topLeft()).x() < screen.x() )
          {
          below.setX(screen.x());
          above.setX(screen.x());
          }
        }

      if (usePopup)
        {
        // Position horizontally.
        listRect.moveLeft(above.x());

#ifndef Q_WS_S60
        // Position vertically so the curently selected item lines up
        // with the combo box.
        const QRect currentItemRect = this->view()->visualRect(this->view()->currentIndex());
        const int offset = listRect.top() - currentItemRect.top();
        listRect.moveTop(above.y() + offset - listRect.top());
#endif

      // Clamp the listRect height and vertical position so we don't expand outside the
      // available screen geometry.This may override the vertical position, but it is more
      // important to show as much as possible of the popup.
        const int height = !boundToScreen ? listRect.height() : qMin(listRect.height(), screen.height());
#ifdef Q_WS_S60
        //popup needs to be stretched with screen minimum dimension
        listRect.setHeight(qMin(screen.height(), screen.width()));
#else
        listRect.setHeight(height);
#endif

        if (boundToScreen)
          {
          if (listRect.top() < screen.top())
            {
            listRect.moveTop(screen.top());
            }
          if (listRect.bottom() > screen.bottom())
            {
            listRect.moveBottom(screen.bottom());
            }
          }
#ifdef Q_WS_S60
        if (screen.width() < screen.height())
          {
          // in portait, menu should be positioned above softkeys
          listRect.moveBottom(screen.bottom());
          }
        else
          {
          TRect staConTopRect = TRect();
          AknLayoutUtils::LayoutMetricsRect(AknLayoutUtils::EStaconTop, staConTopRect);
          listRect.setWidth(listRect.height());
          //by default popup is centered on screen in landscape
          listRect.moveCenter(screen.center());
          if (staConTopRect.IsEmpty())
            {
            // landscape without stacon, menu should be at the right
            (opt.direction == Qt::LeftToRight) ? listRect.setRight(screen.right()) :
              listRect.setLeft(screen.left());
            }
          }
#endif
        }
      else if (!boundToScreen || listRect.height() <= belowHeight)
        {
        listRect.moveTopLeft(below);
        }
      else if (listRect.height() <= aboveHeight)
        {
        listRect.moveBottomLeft(above);
        }
      else if (belowHeight >= aboveHeight)
        {
        listRect.setHeight(belowHeight);
        listRect.moveTopLeft(below);
        }
      else
        {
        listRect.setHeight(aboveHeight);
        listRect.moveBottomLeft(above);
        }

      QScrollBar *sb = this->view()->horizontalScrollBar();
      Qt::ScrollBarPolicy policy = this->view()->horizontalScrollBarPolicy();
      bool needHorizontalScrollBar =
        (policy == Qt::ScrollBarAsNeeded || policy == Qt::ScrollBarAlwaysOn)
        && sb->minimum() < sb->maximum();
      if (needHorizontalScrollBar)
        {
        listRect.adjust(0, 0, 0, sb->height());
        }
      container->setGeometry(listRect);
}
