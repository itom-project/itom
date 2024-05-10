/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2021, Institut fuer Technische Optik (ITO),
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
*********************************************************************** */

#ifndef STRINGLISTEDITOR_H
#define STRINGLISTEDITOR_H

#include <qevent.h>
#include <qlineedit.h>
#include <qstringlist.h>
#include <qtoolbutton.h>
#include <qwidget.h>

class StringListEditor : public QWidget
{
    Q_OBJECT
public:
    StringListEditor(QWidget* parent = 0);
    virtual ~StringListEditor();

    QStringList value() const;
    void setValue(QStringList stringList);

private:
    QStringList m_stringList;
    QLineEdit* m_textEdit;
    QToolButton* m_toolBtn;

protected:
    // void focusOutEvent ( QFocusEvent * event );

signals:
    /** slot that is being called by the editor widget */
    void stringListChanged(QStringList stringList);

private slots:
    void btnClicked();
};
#endif
