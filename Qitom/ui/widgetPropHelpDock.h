/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#ifndef WIDGETPROPHELPDOCK_H
#define WIDGETPROPHELPDOCK_H

#include "abstractPropertyPageWidget.h"

#include <qwidget.h>
#include <qstring.h>

#include "ui_widgetPropHelpDock.h"
#include "helper/fileDownloader.h"
#include <qlist.h>
#include <qdir.h>
#include <qxmlstream.h>

namespace ito
{

class WidgetPropHelpDock: public AbstractPropertyPageWidget
{
    Q_OBJECT

public:
    WidgetPropHelpDock(QWidget *parent = NULL);
    ~WidgetPropHelpDock();

protected:
    bool event (QEvent * event);

private:
    Ui::WidgetPropHelpDock ui;

    enum UpdateState
    {
        stateUnknown = 0,
        stateUpToDate,
        stateUpdateAvailable,
        stateDownloadAvailable,
        stateWrongScheme,
    };

    struct DatabaseInfo
    {
        bool        isChecked;
        UpdateState updateState;
        int         version;
        int         schemeID;
        QString     name;
        QString     date;
        QFileInfo   path;
        QUrl        url;
    };

    // Functions
    void readSettings();
    void writeSettings();
    void getHelpList();
    void refreshExistingDBs();
    void initMenus();
    void compareDatabaseVersions();
    void updateCheckedIdList();
    void setExistingDBsChecks();
    QPair<int, DatabaseInfo> parseFile(QXmlStreamReader& xml);
    void updateTreeWidget();
    void refreshUpdatableDBs();
    void setUpdateColumnText(QTreeWidgetItem *widget);
    void showInGraphicalShell(const QString & filePath);
    void showErrorMessage(const QString &error);



    // Variables
    QString m_pdbPath;
    bool m_listChanged;
    bool m_treeIsUpdating;
    bool m_downloadTimeoutReached;
    int  m_downloadTimeout;
    QUrl m_serverAdress;

    // Lists and Co
    QMap< int, DatabaseInfo > existingDBs;
    QMap< int, DatabaseInfo > updatableDBs;
    QList< int > checkedIdList;

    // Menu
    QMenu* m_pContextMenu;
    std::map<QString,QAction*> contextMenuActions;

    // Consts
    static const int m_urID = Qt::UserRole + 1; // ID
    static const int m_urUD = Qt::UserRole + 2; // UpdateState
    static const int m_urFD = Qt::UserRole + 3; // Path (FileDir)
    static const int SCHEME_ID = 1; // Update available
    QString m_xmlFileName;

signals:

public slots:


private slots:
    void on_treeWidgetDB_itemChanged(QTreeWidgetItem * item, int column);
    void on_checkModules_stateChanged (int state);
    void on_checkFilters_stateChanged (int state);
    void on_checkWidgets_stateChanged (int state);
    void on_checkDataIO_stateChanged (int state);
    void on_spinTimeout_valueChanged(int i);
    void refreshButtonClicked();
    void mnuDownloadUpdate();
    void mnuLocateOnDisk();
    void mnuRemoveDatabase();
    void treeWidgetContextMenuRequested(const QPoint &pos);
    void downloadTimeoutReached();
};

} //end namespace ito

#endif
