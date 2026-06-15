/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef USERUIDIALOG
#define USERUIDIALOG

#include "../common/sharedStructuresQt.h"
#include "../api/apiFunctionsGraph.h"
#include "../global.h"

#include <qdialog.h>

#include <qstring.h>
#include <qmap.h>
#include <QBoxLayout>
#include <qdialogbuttonbox.h>
#include <qabstractbutton.h>

#include <qstringlist.h>
#include <qmetaobject.h>

namespace ito
{

class UserUiDialog : public QDialog
{
    Q_OBJECT
public:
    enum tButtonBarType /*!< enumeration describing the availability and position of an automatic created button bar*/
    {
        bbTypeNo            = 0x0000,
        bbTypeHorizontal    = 0x0001,
        bbTypeVertical      = 0x0002
    };

    UserUiDialog(
        const QString& filename,
        tButtonBarType buttonBarType,
        const StringMap& dialogButtons,
        RetVal& retValue,
        QWidget* parent = NULL,
        Qt::WindowFlags f = Qt::WindowFlags());
    UserUiDialog(
        QWidget* contentWidget,
        tButtonBarType buttonBarType,
        const StringMap& dialogButtons,
        RetVal& retValue,
        QWidget* parent = NULL,
        Qt::WindowFlags f = Qt::WindowFlags());

    ~UserUiDialog();

protected:
    RetVal init(const QString &filename, tButtonBarType buttonBarType, const StringMap &dialogButtons);
    RetVal init(QWidget *contentWidget, tButtonBarType buttonBarType, const StringMap &dialogButtons);

private:
    QDialogButtonBox::ButtonRole getButtonRole(const QString &role);

    QBoxLayout *m_boxLayout;
    QDialogButtonBox *m_dialogBtnBox;
    QWidget *m_uiWidget;

signals:

public slots:

private slots:
    void dialogButtonClicked ( QAbstractButton * button );
};

} //end namespace ito

#endif
