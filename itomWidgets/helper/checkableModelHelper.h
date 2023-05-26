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

#ifndef CHECKABLEMODELHELPER_H
#define CHECKABLEMODELHELPER_H

// Qt includes
#include <QModelIndex>
#include <QObject>
class QAbstractItemModel;


class CheckableModelHelperPrivate;

/// \ingroup Widgets
///
/// CheckableModelHelper expose functions to handle checkable models
class CheckableModelHelper : public QObject
{
  Q_OBJECT;
  Q_PROPERTY(bool forceCheckability READ forceCheckability WRITE setForceCheckability);
  Q_PROPERTY(int propagateDepth READ propagateDepth WRITE setPropagateDepth);
  Q_PROPERTY(Qt::CheckState defaultCheckState READ defaultCheckState WRITE setDefaultCheckState);

public:
  CheckableModelHelper(Qt::Orientation orientation, QObject *parent=0);
  virtual ~CheckableModelHelper();

  Qt::Orientation orientation()const;


  ///
  /// When setting the model, if PropagateToItems is true (by default), the check
  /// state of the checkable headers is updated from the check state of the items
  /// If you want to make sure of the check state of a header, after setting the
  /// (done by myView.setHeader(myCheckableModelHelper)), you can call
  /// myModel.setHeaderData(0, Qt::Horizontal, Qt::Checked, Qt::CheckStateRole)
  /// or myCheckableModelHelper->setCheckState(0, Qt::Checked)
  QAbstractItemModel *model()const;
  virtual void setModel(QAbstractItemModel *model);

  /// Reimplemented for internal reasons
  QModelIndex rootIndex()const;
  virtual void setRootIndex(const QModelIndex &index);

  ///
  /// A section is checkable if its CheckStateRole data is non null.
  /// One can access the same value through the model:
  /// model->headerData(orientation, section, Qt::CheckStateRole).isEmpty()
  bool isHeaderCheckable(int section)const;
  bool isCheckable(const QModelIndex& index)const;

  ///
  /// Utility function that returns the checkState of the section.
  /// One can access the same value through the model:
  /// model->headerData(orientation, section, Qt::CheckStateRole)
  Qt::CheckState headerCheckState(int section)const;
  Qt::CheckState checkState(const QModelIndex&)const;

  ///
  /// Utility function that returns the checkState of the section.
  /// One can access the same value through the model:
  /// model->headerData(orientation, section, Qt::CheckStateRole)
  bool headerCheckState(int section, Qt::CheckState& checkState )const;
  bool checkState(const QModelIndex&, Qt::CheckState& checkState )const;

  /// How deep in the model(tree) do you want the check state to be propagated
  /// A value of -1 correspond to the deepest level of the model.
  /// -1 by default
  void setPropagateDepth(int depth);
  int  propagateDepth()const;

  /// When true, the new items are automatically set to checkable
  void setForceCheckability(bool force);
  bool forceCheckability()const;

  Qt::CheckState defaultCheckState()const;
  void setDefaultCheckState(Qt::CheckState);

public Q_SLOTS:
  void setCheckState(const QModelIndex& modelIndex, Qt::CheckState checkState);
  ///
  /// Warning, setting the check state automatically set the
  /// header section checkable
  void setHeaderCheckState(int section, Qt::CheckState checkState);

  /// Utility function to toggle the checkstate of an index
  void toggleCheckState(const QModelIndex& modelIndex);
  void toggleHeaderCheckState(int section);

private Q_SLOTS:
  void onHeaderDataChanged(Qt::Orientation orient, int first, int last);

  void onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight);
  void updateHeadersFromItems();
  void onColumnsInserted(const QModelIndex& parent, int start, int end);
  void onRowsInserted(const QModelIndex& parent, int start, int end);

protected:
  QScopedPointer<CheckableModelHelperPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(CheckableModelHelper);
  Q_DISABLE_COPY(CheckableModelHelper);
};

#endif
