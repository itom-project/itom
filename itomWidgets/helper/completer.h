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

#ifndef COMPLETER_H
#define COMPLETER_H

// Qt includes
#include <QCompleter>
#include <QMetaType>

#include "commonWidgets.h"

// CTK includes
class CompleterPrivate;

/// \ingroup Widgets
/// Completer is a QCompleter that allows different way of filtering
/// the model, not just by filtering strings that start with the
/// \sa completionPrefix (default behavior).
/// Completer is a bit hackish as it reimplements a methods (splitPath)
/// from QCompleter in a way that is not intended.
/// Disclaimer, it might not work in all contexts, but seems to work
/// fine with a QLineEdit.
/// e.g.:
/// QStringList model;
/// model << "toto tata tutu";
/// model << "tata toto tutu";
/// Completer completer(model);
/// completer.setModelFiltering(Completer::FilterWordStartsWith);
/// QLineEdit lineEdit;
/// lineEdit.setCompleter(&completer);
/// ...
/// If the user types "ta", both entries will show up in the completer
/// If the user types "ot", no entries will show up in the completer
/// however using \sa FilterContains would have shown both.
class ITOMWIDGETS_EXPORT Completer: public QCompleter
{
  Q_OBJECT
  Q_ENUMS(ModelFiltering)
  /// FilterStartsWith is the default behavior (same as QCompleter).The
  /// completer filters out strings that don't start with \sa completionPrefix
  /// FilterContains is the most permissive filter, the completer filters out
  /// only strings that don't contain the characters from \sa completionPrefix
  /// FilterWordStartsWith is useful when strings contain space separated words
  /// and \sa completionPrefix applies to the beginnig of any of the words in the
  /// string.
  Q_PROPERTY(ModelFiltering modelFiltering READ modelFiltering WRITE setModelFiltering)

public:
  Completer(QObject* parent = 0);
  Completer(QAbstractItemModel* model, QObject* parent = 0);
  Completer(const QStringList& list, QObject* parent = 0 );
  virtual ~Completer();

  enum ModelFiltering
    {
    FilterStartsWith=0,
    FilterContains,
    FilterWordStartsWith
    };

  ModelFiltering modelFiltering()const;
  void setModelFiltering(ModelFiltering filter);

  virtual QStringList splitPath(const QString& s)const;

  /// Completer::model() might return a filtered model
  /// (QSortFilterAbstractModel) different from the one that was set.
  /// QCompleter::setModel should not be used and setSourceModel used
  /// instead.
  QAbstractItemModel* sourceModel()const;
  void setSourceModel(QAbstractItemModel* model);

protected:
  QScopedPointer<CompleterPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(Completer);
  Q_DISABLE_COPY(Completer);
};

Q_DECLARE_METATYPE(Completer::ModelFiltering)

#endif // __Completer_h
