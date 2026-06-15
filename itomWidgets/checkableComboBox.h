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

#ifndef CHECKABLECOMBOBOX_H
#define CHECKABLECOMBOBOX_H

// Qt includes
#include <QComboBox>

#include "commonWidgets.h"

class CheckableModelHelper;
class CheckableComboBoxPrivate;

/// \ingroup Widgets
/// Description
/// CheckableComboBox is a QComboBox that allow its items to be checkable
class ITOMWIDGETS_EXPORT CheckableComboBox : public QComboBox
{
    Q_OBJECT

    Q_PROPERTY(QVector<int> checkedIndices READ getCheckedIndices WRITE setCheckedIndices);

public:
  CheckableComboBox(QWidget *parent = 0);
  virtual ~CheckableComboBox();

  /// Use setCheckableModel instead of setModel()
  QAbstractItemModel* checkableModel()const;
  void setCheckableModel(QAbstractItemModel *model);

  /// Returns an up-to-date list of all the checked indexes.
  QModelIndexList checkedIndexes()const;
  /// Returns true if all the indexes are checked, false otherwise
  bool allChecked()const;
  /// Returns true if none of the indexes is checked, false otherwise
  bool noneChecked()const;

  /// Utility function to conveniently check the state of an index
  void setCheckState(const QModelIndex& index, Qt::CheckState check);
  /// Utility function to return the check state of a model index
  Qt::CheckState checkState(const QModelIndex& index)const;

  /// Returns a pointer to the checkable model helper to give a direct access
  /// to the check manager.
  CheckableModelHelper* checkableModelHelper()const;

  /// Reimplemented for internal reasons
  bool eventFilter(QObject *o, QEvent *e);

  void setCheckedIndices(const QVector<int> &indices);
  QVector<int> getCheckedIndices() const;


Q_SIGNALS:
  void checkedIndexesChanged();

public Q_SLOTS:
	void setIndexState(int index, bool state);

protected Q_SLOTS:
  void onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight);

protected:
  /// Reimplemented for internal reasons
  virtual void paintEvent(QPaintEvent*);

protected:
  QScopedPointer<CheckableComboBoxPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(CheckableComboBox);
  Q_DISABLE_COPY(CheckableComboBox);
};

#endif
