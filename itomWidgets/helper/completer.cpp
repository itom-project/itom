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
#include <QDebug>
#include <QSortFilterProxyModel>
#include <QStringList>

// CTK includes
#include "completer.h"

// -------------------------------------------------------------------------
class CompleterPrivate
{
  Q_DECLARE_PUBLIC(Completer);
protected:
  Completer* const q_ptr;

public:
  CompleterPrivate(Completer& object);
  ~CompleterPrivate();
  void init();

  QStringList splitPath(const QString& path);
  void updateSortFilterProxyModel();

  Completer::ModelFiltering Filtering;
  QSortFilterProxyModel* SortFilterProxyModel;
protected:
  Q_DISABLE_COPY(CompleterPrivate);
};

// -------------------------------------------------------------------------
CompleterPrivate::CompleterPrivate(Completer& object)
  :q_ptr(&object)
{
  qRegisterMetaType<Completer::ModelFiltering>("Completer::ModelFiltering");
  this->Filtering = Completer::FilterStartsWith;
  this->SortFilterProxyModel = 0;
}

// -------------------------------------------------------------------------
CompleterPrivate::~CompleterPrivate()
{
  delete this->SortFilterProxyModel;
}

// -------------------------------------------------------------------------
void CompleterPrivate::init()
{
  this->SortFilterProxyModel = new QSortFilterProxyModel(0);
}

// -------------------------------------------------------------------------
QStringList CompleterPrivate::splitPath(const QString& s)
{
  Q_Q(Completer);
  QStringList paths;
  switch(q->modelFiltering())
    {
    default:
    case Completer::FilterStartsWith:
      paths = q->QCompleter::splitPath(s);
      break;
    case Completer::FilterContains:
      this->updateSortFilterProxyModel();
      this->SortFilterProxyModel->setFilterWildcard(s);
      paths = QStringList();
      break;
    case Completer::FilterWordStartsWith:
      {
      this->updateSortFilterProxyModel();
#if (QT_VERSION >= QT_VERSION_CHECK(5, 12, 0))
      QRegularExpression regexp = QRegularExpression(QRegularExpression::escape(s));

      if (!q->caseSensitivity())
      {
          regexp.setPatternOptions(QRegularExpression::CaseInsensitiveOption);
      }

      this->SortFilterProxyModel->setFilterRegularExpression(regexp);
#else
      QRegExp regexp = QRegExp(QRegExp::escape(s));
      regexp.setCaseSensitivity(q->caseSensitivity());
      this->SortFilterProxyModel->setFilterRegExp(regexp);
#endif
      paths = QStringList();
      break;
      }
    }
  return paths;
}

// -------------------------------------------------------------------------
void CompleterPrivate::updateSortFilterProxyModel()
{
  Q_Q(Completer);
  this->SortFilterProxyModel->setFilterCaseSensitivity(q->caseSensitivity());
  this->SortFilterProxyModel->setFilterKeyColumn(q->completionColumn());
}

// -------------------------------------------------------------------------
Completer::Completer(QObject* parent)
  : QCompleter(parent)
  , d_ptr(new CompleterPrivate(*this))
{
  Q_D(Completer);
  d->init();
}

// -------------------------------------------------------------------------
Completer::Completer(QAbstractItemModel* model, QObject* parent)
  : QCompleter(model, parent)
  , d_ptr(new CompleterPrivate(*this))
{
  Q_D(Completer);
  d->init();
}

// -------------------------------------------------------------------------
Completer::Completer(const QStringList& list, QObject* parent)
  : QCompleter(list, parent)
  , d_ptr(new CompleterPrivate(*this))
{
  Q_D(Completer);
  d->init();
}

// -------------------------------------------------------------------------
Completer::~Completer()
{
}

// -------------------------------------------------------------------------
Completer::ModelFiltering Completer::modelFiltering() const
{
  Q_D(const Completer);
  return d->Filtering;
}

// -------------------------------------------------------------------------
void Completer::setModelFiltering(ModelFiltering filter)
{
  Q_D(Completer);
  if (filter == d->Filtering)
    {
    return;
    }
  QAbstractItemModel* source = this->sourceModel();
  d->Filtering = filter;
  this->setSourceModel(source);
  Q_ASSERT(this->sourceModel());
  // Update the filtering
  this->setCompletionPrefix(this->completionPrefix());
}

// -------------------------------------------------------------------------
QStringList Completer::splitPath(const QString& s)const
{
  Q_D(const Completer);
  return const_cast<CompleterPrivate*>(d)->splitPath(s);
}

// -------------------------------------------------------------------------
QAbstractItemModel* Completer::sourceModel()const
{
  Q_D(const Completer);
  if (d->Filtering != Completer::FilterStartsWith)
    {
    return d->SortFilterProxyModel->sourceModel();
    }
  return this->QCompleter::model();
}

// -------------------------------------------------------------------------
void Completer::setSourceModel(QAbstractItemModel* source)
{
  Q_D(Completer);
  QAbstractItemModel* model = source;
  if (d->Filtering != Completer::FilterStartsWith)
    {
    d->SortFilterProxyModel->setSourceModel(source);
    if (source && source->parent() == this)
      {
      source->setParent(d->SortFilterProxyModel);
      }
    model = d->SortFilterProxyModel;
    }
  else if (source && source->parent() == d->SortFilterProxyModel)
    {
    source->setParent(this);
    }
  this->setModel(model);
}
