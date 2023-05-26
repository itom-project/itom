// *************************************************************************************************
//
// QPropertyEditor v 0.3
//
// --------------------------------------
// Copyright (C) 2007 Volker Wiendl
// Acknowledgements to Roman alias banal from qt-apps.org for the Enum enhancement
//
//
// The QPropertyEditor Library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by the Free Software
// Foundation; either version 2 of the License, or (at your option) any later
// version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with
// this program; if not, write to the Free Software Foundation, Inc., 59 Temple
// Place - Suite 330, Boston, MA 02111-1307, USA, or go to
// http://www.gnu.org/copyleft/lesser.txt.
//
// *************************************************************************************************

#ifndef COLORCOMBO_H_
#define COLORCOMBO_H_

#include <qcombobox.h>

class ColorCombo : public QComboBox
{
    Q_OBJECT
public:
    ColorCombo(QWidget* parent = 0);
    virtual ~ColorCombo();

    QColor color() const;
    void setColor(QColor c);

signals:
    /** slot that is being called by the editor widget */
    void colorChanged(QColor c);

private slots:
    void currentChanged(int index);



private:
    QColor    m_init;

};
#endif
