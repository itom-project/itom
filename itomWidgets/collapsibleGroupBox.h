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

#ifndef __CollapsibleGroupBox_h
#define __CollapsibleGroupBox_h

// Qt includes
#include <QGroupBox>

#include "commonWidgets.h"

class CollapsibleGroupBoxPrivate;

/// \ingroup Widgets
/// A QGroupBox with an arrow indicator that shows/hides the groupbox contents
/// when clicked. It responds to the slot QGroupBox::setChecked(bool) or
/// CollapsibleGroupBox::setCollapsed(bool)
/// When checked is true, the groupbox is expanded
/// When checked is false, the groupbox is collapsed
class ITOMWIDGETS_EXPORT CollapsibleGroupBox : public QGroupBox
{
  Q_OBJECT
  Q_PROPERTY(bool collapsed READ collapsed WRITE setCollapsed)

  /// This property holds the height in pixels of the contents (excludes the title)
  /// when the box is collapsed.
  /// 14px by default, it is the smallest height that fit Mac Style.
  Q_PROPERTY(int collapsedHeight READ collapsedHeight WRITE setCollapsedHeight)

public:
  CollapsibleGroupBox(QWidget* parent = 0);
  CollapsibleGroupBox(const QString& title, QWidget* parent = 0);
  virtual ~CollapsibleGroupBox();

  /// Utility function to collapse the groupbox
  /// Collapse(close) the group box if collapse is true, expand(open)
  /// it otherwise.
  /// \sa QGroupBox::setChecked(bool)
  inline void setCollapsed(bool collapse);

  /// Return the collapse state of the groupbox
  /// true if the groupbox is collapsed (closed), false if it is expanded(open)
  inline bool collapsed()const;

  /// Set the height of the collapsed box. Does not include the title height.
  virtual void setCollapsedHeight(int heightInPixels);
  int collapsedHeight()const;

  /// Reimplemented for internal reasons
  /// Catch when a child widget's visibility is externally changed
  virtual bool eventFilter(QObject* child, QEvent* e);

  /// Reimplemented for internal reasons
  virtual void setVisible(bool show);
protected slots:
  /// called when the arrow indicator is clicked
  /// users can call it programatically by calling setChecked(bool)
  virtual void expand(bool expand);

protected:
  QScopedPointer<CollapsibleGroupBoxPrivate> d_ptr;
  /// reimplemented for internal reasons
  virtual void childEvent(QChildEvent*);

#if QT_VERSION < 0x040600
  virtual void paintEvent(QPaintEvent*);
  virtual void mousePressEvent(QMouseEvent*);
  virtual void mouseReleaseEvent(QMouseEvent*);
#endif

private:
  Q_DECLARE_PRIVATE(CollapsibleGroupBox);
  Q_DISABLE_COPY(CollapsibleGroupBox);
};

//----------------------------------------------------------------------------
bool CollapsibleGroupBox::collapsed()const
{
  return !this->isChecked();
}

//----------------------------------------------------------------------------
void CollapsibleGroupBox::setCollapsed(bool collapse)
{
  this->setChecked(!collapse);
}

#endif
