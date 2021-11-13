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

#include "FontEditor.h"

#include <qfontdialog.h>
#include <qlayout.h>

//-------------------------------------------------------------------------------------
FontEditor::FontEditor(QWidget* parent /*= 0*/) : QWidget(parent)
{
    m_textEdit = new QLineEdit(this);
    m_textEdit->setReadOnly(true);

    m_toolBtn = new QToolButton(this);
    connect(m_toolBtn, SIGNAL(clicked()), this, SLOT(btnClicked()));

    QHBoxLayout* layout = new QHBoxLayout(this);
    layout->addWidget(m_textEdit);
    layout->addWidget(m_toolBtn);

    setLayout(layout);
    layout->setSpacing(0);
    layout->setContentsMargins(0, 0, 0, 0);
    ;
    setContentsMargins(0, 0, 0, 0);

    setMinimumHeight(15);
    setFocusProxy(
        m_toolBtn); // this is very important: see http://qt-project.org/forums/viewthread/3860/
}

//-------------------------------------------------------------------------------------
FontEditor::~FontEditor()
{
}

//-------------------------------------------------------------------------------------
QFont FontEditor::value() const
{
    return m_font;
}

//-------------------------------------------------------------------------------------
void FontEditor::setValue(QFont font)
{
    m_font = font;
    m_textEdit->setText(QString("[%1, %2]").arg(m_font.family()).arg(m_font.pointSize()));
}

//-------------------------------------------------------------------------------------
void FontEditor::btnClicked()
{
    bool ok;
    QFont font = QFontDialog::getFont(&ok, m_font, this, "select font");

    if (ok)
    {
        setValue(font);
        emit fontChanged(font);
    }
}
