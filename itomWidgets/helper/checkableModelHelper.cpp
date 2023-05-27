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
#include <QAbstractItemModel>
#include <QDebug>
#include <QStandardItemModel>
#include <QPointer>

// CTK includes
#include "checkableModelHelper.h"

//-----------------------------------------------------------------------------
class CheckableModelHelperPrivate
{
  Q_DECLARE_PUBLIC(CheckableModelHelper);
protected:
  CheckableModelHelper* const q_ptr;
public:
  CheckableModelHelperPrivate(CheckableModelHelper& object);
  ~CheckableModelHelperPrivate();

  void init();
  /// Set index checkstate and call propagate
  void setIndexCheckState(const QModelIndex& index, Qt::CheckState checkState);
  /// Return the depth in the model tree of the index.
  /// -1 if the index is the root element a header or a header, 0 if the index
  /// is a toplevel index, 1 if its parent is toplevel, 2 if its grandparent is
  /// toplevel, etc.
  int indexDepth(const QModelIndex& modelIndex)const;
  /// Set the checkstate of the index based on its children and grand children
  void updateCheckState(const QModelIndex& modelIndex);
  /// Set the check state of the index to all its children and grand children
  void propagateCheckStateToChildren(const QModelIndex& modelIndex);

  Qt::CheckState checkState(const QModelIndex& index, bool *checkable)const;
  void setCheckState(const QModelIndex& index, Qt::CheckState newCheckState);

  void forceCheckability(const QModelIndex& index);

  QPointer<QAbstractItemModel> Model;
  QModelIndex         RootIndex;
  Qt::Orientation     Orientation;
  bool                HeaderIsUpdating;
  bool                ItemsAreUpdating;
  bool                ForceCheckability;
  /// 0 means no propagation
  /// -1 means unlimited propagation
  /// 1 means propagate to top-level indexes
  /// 2 means propagate to top-level and their children
  /// ...
  int                 PropagateDepth;
  Qt::CheckState      DefaultCheckState;
};

//----------------------------------------------------------------------------
CheckableModelHelperPrivate::CheckableModelHelperPrivate(CheckableModelHelper& object)
  : q_ptr(&object)
{
  this->HeaderIsUpdating = false;
  this->ItemsAreUpdating = false;
  this->ForceCheckability = false;
  this->PropagateDepth = -1;
  this->DefaultCheckState = Qt::Unchecked;
}

//-----------------------------------------------------------------------------
CheckableModelHelperPrivate::~CheckableModelHelperPrivate()
{
}

//----------------------------------------------------------------------------
void CheckableModelHelperPrivate::init()
{
}

//----------------------------------------------------------------------------
Qt::CheckState CheckableModelHelperPrivate::checkState(
  const QModelIndex& index, bool *checkable)const
{
  Q_Q(const CheckableModelHelper);
  if (!q->model())
    {
    qWarning() << "Model has not been set.";
    return q->defaultCheckState();
    }
  QVariant indexCheckState = index != q->rootIndex() ?
    q->model()->data(index, Qt::CheckStateRole):
    q->model()->headerData(0, q->orientation(), Qt::CheckStateRole);
  return static_cast<Qt::CheckState>(indexCheckState.toInt(checkable));
}

//----------------------------------------------------------------------------
void CheckableModelHelperPrivate::setCheckState(
  const QModelIndex& modelIndex, Qt::CheckState newCheckState)
{
  Q_Q(CheckableModelHelper);
  if (!q->model())
    {
    qWarning() << "Model has not been set.";
    return;
    }
  else if (modelIndex != q->rootIndex())
    {
    q->model()->setData(modelIndex, static_cast<int>(newCheckState),
                        Qt::CheckStateRole);
    }
  else
    {
    q->model()->setHeaderData(0, q->orientation(), static_cast<int>(newCheckState),
                              Qt::CheckStateRole);
    }
}

//----------------------------------------------------------------------------
void CheckableModelHelperPrivate::setIndexCheckState(
  const QModelIndex& index, Qt::CheckState checkState)
{
  bool checkable = false;
  this->checkState(index, &checkable);
  if (!checkable && !this->ForceCheckability)
    {
    // The index is not checkable and we don't want to force checkability
    return;
    }
  this->setCheckState(index, checkState);
  this->propagateCheckStateToChildren(index);
}

//-----------------------------------------------------------------------------
int CheckableModelHelperPrivate::indexDepth(const QModelIndex& modelIndex)const
{
  int depth = -1;
  QModelIndex parentIndex = modelIndex;
  while (parentIndex.isValid())
    {
    ++depth;
    parentIndex = parentIndex.parent();
    }
  return depth;
}

//-----------------------------------------------------------------------------
void CheckableModelHelperPrivate
::updateCheckState(const QModelIndex& modelIndex)
{
  Q_Q(CheckableModelHelper);
  bool checkable = false;
  int oldCheckState = this->checkState(modelIndex, &checkable);
  if (!checkable)
    {
    return;
    }

  Qt::CheckState newCheckState = Qt::PartiallyChecked;
  bool firstCheckableChild = true;
  const int rowCount = q->orientation() == Qt::Horizontal ?
    q->model()->rowCount(modelIndex) : 1;
  const int columnCount = q->orientation() == Qt::Vertical ?
    q->model()->columnCount(modelIndex) : 1;
  for (int r = 0; r < rowCount; ++r)
    {
    for (int c = 0; c < columnCount; ++c)
      {
      QModelIndex child = q->model()->index(r, c, modelIndex);
      QVariant childCheckState = q->model()->data(child, Qt::CheckStateRole);
      int childState = childCheckState.toInt(&checkable);
      if (!checkable)
        {
        continue;
        }
      if (firstCheckableChild)
        {
        newCheckState = static_cast<Qt::CheckState>(childState);
        firstCheckableChild = false;
        }
      if (newCheckState != childState)
        {
        newCheckState = Qt::PartiallyChecked;
        }
      if (newCheckState == Qt::PartiallyChecked)
        {
        break;
        }
      }
    if (!firstCheckableChild && newCheckState == Qt::PartiallyChecked)
      {
      break;
      }
    }
  if (oldCheckState == newCheckState)
    {
    return;
    }
  this->setCheckState(modelIndex, newCheckState);
  if (modelIndex != q->rootIndex())
    {
    this->updateCheckState(modelIndex.parent());
    }
}

//-----------------------------------------------------------------------------
void CheckableModelHelperPrivate
::propagateCheckStateToChildren(const QModelIndex& modelIndex)
{
  Q_Q(CheckableModelHelper);
  int indexDepth = this->indexDepth(modelIndex);
  if (this->PropagateDepth == 0 ||
      !(indexDepth < this->PropagateDepth || this->PropagateDepth == -1))
    {
    return;
    }

  bool checkable = false;
  Qt::CheckState checkState = this->checkState(modelIndex, &checkable);
  if (!checkable || checkState == Qt::PartiallyChecked)
    {
    return;
    }

  while (this->ForceCheckability && q->model()->canFetchMore(modelIndex))
    {
    q->model()->fetchMore(modelIndex);
    }

  const int rowCount = q->orientation() == Qt::Horizontal ?
    q->model()->rowCount(modelIndex) : 1;
  const int columnCount = q->orientation() == Qt::Vertical ?
    q->model()->columnCount(modelIndex) : 1;
  for (int r = 0; r < rowCount; ++r)
    {
    for (int c = 0; c < columnCount; ++c)
      {
      QModelIndex child = q->model()->index(r, c, modelIndex);
      this->setIndexCheckState(child, checkState);
      }
    }
}

//-----------------------------------------------------------------------------
void CheckableModelHelperPrivate
::forceCheckability(const QModelIndex& modelIndex)
{
  Q_Q(CheckableModelHelper);
  if (!this->ForceCheckability)
    {
    return;
    }
  this->setCheckState(modelIndex, this->DefaultCheckState);
  // Apparently (not sure) some views require the User-checkable
  // flag to be set to be able to show the checkboxes
  if (qobject_cast<QStandardItemModel*>(q->model()))
    {
    QStandardItem* item = modelIndex != q->rootIndex() ?
      qobject_cast<QStandardItemModel*>(q->model())->itemFromIndex(modelIndex) :
      (q->orientation() == Qt::Horizontal ?
         qobject_cast<QStandardItemModel*>(q->model())->horizontalHeaderItem(0) :
         qobject_cast<QStandardItemModel*>(q->model())->verticalHeaderItem(0));
    item->setCheckable(true);
    }
}

//----------------------------------------------------------------------------
CheckableModelHelper::CheckableModelHelper(
  Qt::Orientation orient, QObject* objectParent)
  : QObject(objectParent)
  , d_ptr(new CheckableModelHelperPrivate(*this))
{
  Q_D(CheckableModelHelper);
  d->Orientation = orient;
  d->init();
}

//-----------------------------------------------------------------------------
CheckableModelHelper::~CheckableModelHelper()
{
}

//-----------------------------------------------------------------------------
Qt::Orientation CheckableModelHelper::orientation()const
{
  Q_D(const CheckableModelHelper);
  return d->Orientation;
}

//-----------------------------------------------------------------------------
QAbstractItemModel* CheckableModelHelper::model()const
{
  Q_D(const CheckableModelHelper);
  return d->Model.isNull() ? 0 : d->Model.data();
}

//-----------------------------------------------------------------------------
void CheckableModelHelper::setModel(QAbstractItemModel *newModel)
{
  Q_D(CheckableModelHelper);
  QAbstractItemModel *current = this->model();
  if (current == newModel)
    {
    return;
    }
  if(current)
    {
    this->disconnect(
      current, SIGNAL(headerDataChanged(Qt::Orientation,int,int)),
      this, SLOT(onHeaderDataChanged(Qt::Orientation,int,int)));
    this->disconnect(
      current, SIGNAL(dataChanged(QModelIndex,QModelIndex)),
      this, SLOT(onDataChanged(QModelIndex,QModelIndex)));
    this->disconnect(
      current, SIGNAL(columnsInserted(QModelIndex,int,int)),
      this, SLOT(onColumnsInserted(QModelIndex,int,int)));
    this->disconnect(
      current, SIGNAL(rowsInserted(QModelIndex,int,int)),
      this, SLOT(onRowsInserted(QModelIndex,int,int)));
    }
  d->Model = newModel;
  if(newModel)
    {
    this->connect(
      newModel, SIGNAL(headerDataChanged(Qt::Orientation,int,int)),
      this, SLOT(onHeaderDataChanged(Qt::Orientation,int,int)));
    if (d->PropagateDepth != 0)
      {
      this->connect(
        newModel, SIGNAL(dataChanged(QModelIndex,QModelIndex)),
        this, SLOT(onDataChanged(QModelIndex,QModelIndex)));
      }
    this->connect(
      newModel, SIGNAL(columnsInserted(QModelIndex,int,int)),
      this, SLOT(onColumnsInserted(QModelIndex,int,int)));
    this->connect(
      newModel, SIGNAL(rowsInserted(QModelIndex,int,int)),
      this, SLOT(onRowsInserted(QModelIndex,int,int)));

    if (d->ForceCheckability)
      {
      foreach(QModelIndex index, newModel->match(newModel->index(0,0), Qt::CheckStateRole, QVariant(), -1,Qt::MatchRecursive))
        {
        d->forceCheckability(index);
        }
      d->forceCheckability(this->rootIndex());
      }
    this->updateHeadersFromItems();
    }
}

//-----------------------------------------------------------------------------
QModelIndex CheckableModelHelper::rootIndex()const
{
  Q_D(const CheckableModelHelper);
  return d->RootIndex;
}

//-----------------------------------------------------------------------------
void CheckableModelHelper::setRootIndex(const QModelIndex &index)
{
  Q_D(CheckableModelHelper);
  d->RootIndex = index;
  if (d->PropagateDepth != 0)
    {
    this->updateHeadersFromItems();
    }
}

//-----------------------------------------------------------------------------
void CheckableModelHelper::setPropagateDepth(int depth)
{
  Q_D(CheckableModelHelper);
  if (d->PropagateDepth == depth)
    {
    return;
    }
  d->PropagateDepth = depth;
  if (!this->model())
    {
    return;
    }
  if (depth != 0)
    {
    this->connect(
      this->model(), SIGNAL(dataChanged(QModelIndex,QModelIndex)),
      this, SLOT(onDataChanged(QModelIndex,QModelIndex)), Qt::UniqueConnection);
    this->updateHeadersFromItems();
    }
  else
    {
    this->disconnect(
      this->model(), SIGNAL(dataChanged(QModelIndex,QModelIndex)),
      this, SLOT(onDataChanged(QModelIndex,QModelIndex)));
    }
}

//-----------------------------------------------------------------------------
int CheckableModelHelper::propagateDepth()const
{
  Q_D(const CheckableModelHelper);
  return d->PropagateDepth;
}

//-----------------------------------------------------------------------------
void CheckableModelHelper::setForceCheckability(bool force)
{
  Q_D(CheckableModelHelper);
  if (d->ForceCheckability == force)
    {
    return;
    }
  d->ForceCheckability = force;
  if (this->model())
    {
    d->propagateCheckStateToChildren(this->rootIndex());
    }
}

//-----------------------------------------------------------------------------
bool CheckableModelHelper::forceCheckability()const
{
  Q_D(const CheckableModelHelper);
  return d->ForceCheckability;
}

//-----------------------------------------------------------------------------
void CheckableModelHelper::setDefaultCheckState(Qt::CheckState defaultCheckState)
{
  Q_D(CheckableModelHelper);
  d->DefaultCheckState = defaultCheckState;
}

//-----------------------------------------------------------------------------
Qt::CheckState CheckableModelHelper::defaultCheckState()const
{
  Q_D(const CheckableModelHelper);
  return d->DefaultCheckState;
}

//-----------------------------------------------------------------------------
void CheckableModelHelper::setHeaderCheckState(int section, Qt::CheckState checkState)
{
  QAbstractItemModel *current = this->model();
  if(current == 0)
    {
    return;
    }
  current->setHeaderData(section, this->orientation(),
                         static_cast<int>(checkState), Qt::CheckStateRole);
}

//-----------------------------------------------------------------------------
void CheckableModelHelper::setCheckState(const QModelIndex& index, Qt::CheckState checkState)
{
  QAbstractItemModel *current = this->model();
  if(current == 0)
    {
    return;
    }
  current->setData(index, static_cast<int>(checkState), Qt::CheckStateRole);
}
//-----------------------------------------------------------------------------
void CheckableModelHelper::toggleCheckState(const QModelIndex& modelIndex)
{
  // If the section is checkable, toggle the check state.
  if(!this->isCheckable(modelIndex))
    {
    return;
    }
  // I've no strong feeling to turn the state checked or unchecked when the
  // state is PartiallyChecked.
  this->setCheckState(modelIndex,
    this->checkState(modelIndex) == Qt::Checked ? Qt::Unchecked : Qt::Checked);
}

//-----------------------------------------------------------------------------
void CheckableModelHelper::toggleHeaderCheckState(int section)
{
  // If the section is checkable, toggle the check state.
  if(!this->isHeaderCheckable(section))
    {
    return;
    }
  // I've no strong feeling to turn the state checked or unchecked when the
  // state is PartiallyChecked.
  this->setHeaderCheckState(
    section, this->headerCheckState(section) == Qt::Checked ?
    Qt::Unchecked : Qt::Checked);
}

//-----------------------------------------------------------------------------
void CheckableModelHelper::onHeaderDataChanged(Qt::Orientation orient,
                                              int firstSection,
                                              int lastSection)
{
  Q_D(CheckableModelHelper);
  Q_UNUSED(firstSection);
  Q_UNUSED(lastSection);
  if(orient != this->orientation())
    {
    return;
    }
  bool oldItemsAreUpdating = d->ItemsAreUpdating;
  if (!d->ItemsAreUpdating)
    {
    d->ItemsAreUpdating = true;
    d->propagateCheckStateToChildren(this->rootIndex());
    }
  d->ItemsAreUpdating = oldItemsAreUpdating;
}

//-----------------------------------------------------------------------------
void CheckableModelHelper::updateHeadersFromItems()
{
  Q_D(CheckableModelHelper);
  QAbstractItemModel *currentModel = this->model();
  if (!currentModel)
    {
    return;
    }
  d->updateCheckState(QModelIndex());
}

//-----------------------------------------------------------------------------
void CheckableModelHelper::onDataChanged(const QModelIndex & topLeft,
                                           const QModelIndex & bottomRight)
{
  Q_UNUSED(bottomRight);
  Q_D(CheckableModelHelper);
  if(d->ItemsAreUpdating || d->PropagateDepth == 0)
    {
    return;
    }
  bool checkable = false;
  d->checkState(topLeft, &checkable);
  if (!checkable)
    {
    return;
    }
  d->ItemsAreUpdating = true;
  // TODO: handle topLeft "TO bottomRight"
  d->propagateCheckStateToChildren(topLeft);
  d->updateCheckState(topLeft.parent());

  d->ItemsAreUpdating = false;
}

//-----------------------------------------------------------------------------
void CheckableModelHelper::onColumnsInserted(const QModelIndex &parentIndex,
  int start, int end)
{
  Q_D(CheckableModelHelper);
  if (this->orientation() == Qt::Horizontal)
    {
    if (start == 0)
      {
      this->updateHeadersFromItems();
      }
    }
  else
    {
    if (d->ForceCheckability)
      {
      for (int i = start; i <= end; ++i)
        {
        QModelIndex index = this->model()->index(0, i, parentIndex);
        d->forceCheckability(index);
        }
      }
    this->onDataChanged(this->model()->index(0, start, parentIndex),
                        this->model()->index(0, end, parentIndex));
    }
}

//-----------------------------------------------------------------------------
void CheckableModelHelper::onRowsInserted(const QModelIndex &parentIndex,
  int start, int end)
{
  Q_D(CheckableModelHelper);
  if (this->orientation() == Qt::Vertical)
    {
    if (start == 0)
      {
      this->updateHeadersFromItems();
      }
    }
  else
    {
    if (d->ForceCheckability)
      {
      for (int i = start; i <= end; ++i)
        {
        QModelIndex index = this->model()->index(i, 0, parentIndex);
        d->forceCheckability(index);
        }
      }
    this->onDataChanged(this->model()->index(start, 0, parentIndex),
                        this->model()->index(end, 0, parentIndex));
    }
}

//-----------------------------------------------------------------------------
bool CheckableModelHelper::isHeaderCheckable(int section)const
{
  if (!this->model())
    {
    qWarning() << "CheckableModelHelper::isHeaderCheckable : Model has not been set";
    return (this->forceCheckability() && section == 0);
    }
  return !this->model()->headerData(section, this->orientation(), Qt::CheckStateRole).isNull();
}

//-----------------------------------------------------------------------------
bool CheckableModelHelper::isCheckable(const QModelIndex& index)const
{
  if (!this->model())
    {
    qWarning() << "CheckableModelHelper::isCheckable : Model has not been set";
    return (this->forceCheckability() && index.column() == 0);
    }
  return !this->model()->data(index, Qt::CheckStateRole).isNull();
}

//-----------------------------------------------------------------------------
Qt::CheckState CheckableModelHelper::headerCheckState(int section)const
{
  if (!this->model())
    {
    qWarning() << "CheckableModelHelper::headerCheckState : Model has not been set";
    return this->defaultCheckState();
    }
  return static_cast<Qt::CheckState>(
    this->model()->headerData(section, this->orientation(), Qt::CheckStateRole).toInt());
}

//-----------------------------------------------------------------------------
Qt::CheckState CheckableModelHelper::checkState(const QModelIndex& index)const
{
  if (!this->model())
    {
    qWarning() << "CheckableModelHelper::checkState : Model has not been set";
    return this->defaultCheckState();
    }
  return static_cast<Qt::CheckState>(
    this->model()->data(index, Qt::CheckStateRole).toInt());
}

//-----------------------------------------------------------------------------
bool CheckableModelHelper::headerCheckState(int section, Qt::CheckState& checkState)const
{
  bool checkable = false;
  if (!this->model())
    {
    qWarning() << "CheckableModelHelper::headerCheckState : Model has not been set";
    return (this->forceCheckability() && section == 0);
    }
  checkState = static_cast<Qt::CheckState>(
    this->model()->headerData(section, this->orientation(), Qt::CheckStateRole).toInt(&checkable));
  return checkable;
}

//-----------------------------------------------------------------------------
bool CheckableModelHelper::checkState(const QModelIndex& index, Qt::CheckState& checkState)const
{
  bool checkable = false;
  if (!this->model())
    {
    qWarning() << "CheckableModelHelper::checkState : Model has not been set";
    return (this->forceCheckability() && index.column() == 0);
    }
  checkState = static_cast<Qt::CheckState>(
    this->model()->data(index, Qt::CheckStateRole).toInt(&checkable));
  return checkable;
}
