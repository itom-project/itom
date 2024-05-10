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

#include "stringListEditor.h"

#include "stringListDialog.h"

#include <qfontdialog.h>
#include <qlayout.h>


//-------------------------------------------------------------------------------------
StringListEditor::StringListEditor(QWidget* parent /*= 0*/) : QWidget(parent)
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
StringListEditor::~StringListEditor()
{
}

//-------------------------------------------------------------------------------------
QStringList StringListEditor::value() const
{
    return m_stringList;
}

//-------------------------------------------------------------------------------------
void StringListEditor::setValue(QStringList stringList)
{
    m_stringList = stringList;
    switch (m_stringList.count())
    {
    case 0:
        m_textEdit->setText("[]");
        break;
    case 1:
        m_textEdit->setText(QString("[%1]").arg(m_stringList[0]));
        break;
    case 2:
        m_textEdit->setText(QString("[%1; %2]").arg(m_stringList[0]).arg(m_stringList[1]));
        break;
    case 3:
        m_textEdit->setText(
            QString("[%1; %2; %3]").arg(m_stringList[0]).arg(m_stringList[1]).arg(m_stringList[2]));
        break;
    default:
        m_textEdit->setText(QString("[%1; %2; %3; ...]")
                                .arg(m_stringList[0])
                                .arg(m_stringList[1])
                                .arg(m_stringList[2]));
        break;
    }
}

//-------------------------------------------------------------------------------------
void StringListEditor::btnClicked()
{
    StringListDialog* dialog = new StringListDialog(m_stringList, this);
    dialog->setNewItemText(tr("New Item"));
    if (dialog->exec())
    {
        QStringList stringlist = dialog->getStringList();
        setValue(stringlist);
        emit stringListChanged(stringlist);
    }
}
