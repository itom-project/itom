/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2017, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut f�r Technische
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

#ifndef PYTHONMESSAGEWIDGET_H
#define PYTHONMESSAGEWIDGET_H

#ifdef __APPLE__
extern "C++" {
#endif

#include "../common/commonGlobal.h"
#include "../common/typeDefs.h"

#include "commonWidgets.h"
#include <qlistwidget.h>
//#include "../../itom/Qitom/widgets/abstractDockWidget.h"

class ITOMWIDGETS_EXPORT PythonMessageBox : public QListWidget
{
    Q_OBJECT

    Q_PROPERTY(int maxMessages READ getMaxMessages WRITE setMaxMessages NOTIFY maxMessagesChanged)

    public Q_SLOTS:
        void setMaxMessages(const int newMaxMessages);

    Q_SIGNALS:
        void maxMessagesChanged(const int newMaxMessages);


    public:
        PythonMessageBox(QWidget* parent = NULL);
        int getMaxMessages()const { return m_maxMessages; };

    protected:
//        void createActions();

    private:
        QListWidgetItem *m_newItem;
        int m_maxMessages;
        HWND m_mainWindowHandle;
        void removeFirstItem();

    public slots:
        void addNewMessage(QString newMessage);

    private slots:
};

#ifdef __APPLE__
}
#endif

#endif // PYTHONMESSAGEWIDGET_H