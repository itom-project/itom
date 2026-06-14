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

#ifndef DIALOGICONBROWSER_H
#define DIALOGICONBROWSER_H

#include <qdialog.h>
#include <qtreewidget.h>
#include <qfuturewatcher.h>
#include <qlist.h>

#include "ui_dialogIconBrowser.h"

class QTreeWidgetItem; //forward declaration

namespace ito
{

class DialogIconBrowser : public QDialog
{
    Q_OBJECT

public:
    DialogIconBrowser(QWidget *parent = NULL);
    ~DialogIconBrowser();

protected:
    Ui::DialogIconBrowser ui;

    QList<QTreeWidgetItem*> loadIcons();

private:
    QFutureWatcher<QList<QTreeWidgetItem*> > m_loadWatcher;

signals:
    void sendIconBrowserText(QString iconLink);

private slots:
//    void on_applyButton_clicked();    //!< Write the current settings to the internal paramsVals and sent them to the grabber
    void on_treeWidget_currentItemChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous);
    void on_pushButtonClipboard_clicked(bool value);
    void on_pushButtonInsert_clicked(bool value);
    void on_txtFilter_textChanged(const QString &text);

    void loadFinished();
};

} //end namespace ito

#endif
