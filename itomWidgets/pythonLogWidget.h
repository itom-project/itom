/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    University of Stuttgart, Germany

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

#ifndef PYTHONLOGWIDGET_H
#define PYTHONLOGWIDGET_H

#ifdef __APPLE__
extern "C++" {
#endif

#include "../common/commonGlobal.h"

#include "common/abstractApiWidget.h"

#include "../common/typeDefs.h"

#include "commonWidgets.h"

#include <qscopedpointer.h>

class PythonLogWidgetPrivate;

class ITOMWIDGETS_EXPORT PythonLogWidget : public ito::AbstractApiWidget
{
    Q_OBJECT

    Q_PROPERTY(int maxMessages READ getMaxMessages WRITE setMaxMessages NOTIFY maxMessagesChanged)
    Q_PROPERTY(bool outputStream READ getOutputStream WRITE setOutputStream)
    Q_PROPERTY(bool errorStream READ getErrorStream WRITE setErrorStream)
    Q_PROPERTY(int verticalSizeHint READ getVerticalSizeHint WRITE setVerticalSizeHint)
    Q_PROPERTY(bool autoScroll READ getAutoScroll WRITE setAutoScroll)

    WIDGET_ITOM_API

    public Q_SLOTS:
        void setMaxMessages(const int newMaxMessages);
        ito::RetVal setOutputStream(bool enabled);
        ito::RetVal setErrorStream(bool enabled);
        void setVerticalSizeHint(int value);
        void clear();
        void setAutoScroll(bool autoScroll);


    Q_SIGNALS:
        void maxMessagesChanged(const int newMaxMessages);

    public:
        explicit PythonLogWidget(QWidget* parent = NULL);
        ~PythonLogWidget();

        int getMaxMessages() const;
        bool getOutputStream() const;
        bool getErrorStream() const;
        int getVerticalSizeHint() const;
        bool getAutoScroll() const;

    protected:
//        void createActions();
        virtual ito::RetVal init();
        QSize sizeHint() const;

        QScopedPointer<PythonLogWidgetPrivate> d_ptr;

    public slots:
        void messageReceived(QString message, ito::tStreamMessageType messageType);

    private Q_SLOTS:
        void showContextMenu(const QPoint &pt);

    private:
        Q_DECLARE_PRIVATE(PythonLogWidget);
        Q_DISABLE_COPY(PythonLogWidget);
};

#ifdef __APPLE__
}
#endif

#endif // PYTHONLOGWIDGET_H
