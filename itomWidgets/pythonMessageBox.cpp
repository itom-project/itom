/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2017, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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

#include "qdatetime.h"
#include "pythonMessageBox.h"
#include <windows.h>
#include <QtWidgets/qmessagebox.h>

//---------------------------------------------------------------------------------------------------------
void PythonMessageBox::removeFirstItem()
{
    QListWidgetItem* delItem = this->takeItem(0);
    if (delItem)
    {
        this->removeItemWidget(delItem);
    }
}

//---------------------------------------------------------------------------------------------------------
PythonMessageBox::PythonMessageBox(QWidget* parent /*= NULL*/) :
    QListWidget(parent),
    m_newItem(NULL),
    m_maxMessages(0)
{
//    m_mainWindowHandle = FindWindow(NULL, L"itom");

/*
    QMessageBox msgBox;
    msgBox.setText(QString("HWND '%1'").arg((int)m_mainWindowHandle));
    msgBox.setIcon(QMessageBox::Critical);
    msgBox.exec();
*/
}

//---------------------------------------------------------------------------------------------------------
void PythonMessageBox::addNewMessage(QString newMessage)
{
    if (newMessage.compare("\n") == 0)
    {
        QString str = "";

        if (!m_newItem)
        {
            QDate date(QDate::currentDate());
            QTime time(QTime::currentTime());
            str = date.toString(Qt::ISODate) + " " + time.toString(Qt::ISODate) + " ";
            m_newItem = new QListWidgetItem(str, this);

            if (m_maxMessages > 0 && this->count() > m_maxMessages)
            {
                removeFirstItem();
            }
        }
        else
        {
            str = m_newItem->text();
        }

        newMessage.replace("\n", "");
        str = str + newMessage;
        m_newItem->setText(str);

        this->scrollToItem(m_newItem);
    }
    else
    {
        delete m_newItem;
        m_newItem = NULL;
    }
}

//---------------------------------------------------------------------------------------------------------
void PythonMessageBox::setMaxMessages(const int newMaxMessages)
{
    m_maxMessages = newMaxMessages;

    while (m_maxMessages > 0 && this->count() > m_maxMessages)
    {
        removeFirstItem();
    }
}
