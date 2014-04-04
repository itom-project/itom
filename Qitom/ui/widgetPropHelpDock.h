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

#include <QtGui>

#include "ui_widgetPropHelpDock.h"
#include "helper\fileDownloader.h"
#include <qlist.h>

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

    struct databaseInfo
    {
        bool    isChecked;
        int     updateState;
        QString name;
        QString version;
        QString date;
        QString schemeID;
        QString path;
    };

    enum updateState
    {
        unknown = 0,
        upToDate,
        updateAvailable,
        downloadAvailable,
        wrongScheme,
    };


    // Functions
    void readSettings();
    void writeSettings();
    void getHelpList();
    void refreshExistingDBs();
    void contextMenuEvent (QContextMenuEvent * event);
    void initMenus();
    void preShowContextMenuMargin(); 
    void compareDatabaseVersions();
    void updateCheckedIdList();
    void setExistingDBsChecks();
    QPair<int, databaseInfo> parseFile(QXmlStreamReader& xml);
    void updateTreeWidget();
    void refreshUpdatableDBs();
    void setUpdateColumnText(QTreeWidgetItem *widget);


    // Variables
    QString m_pdbPath;
    bool m_listChanged;
    std::map<QString,QAction*> updateMenuActions;
    QMenu *updateMenu;
    FileDownloader *m_pXmlCtrl;
    FileDownloader *m_pSqlCtrl;
    bool m_treeIsUpdating;

    // Lists and Co
    QMap< int, databaseInfo > existingDBs;
    QMap< int, databaseInfo > updatableDBs;
    QList< int > checkedIdList;

    // Consts
    static const int m_urID = Qt::UserRole + 1; // ID
    static const int m_urUD = Qt::UserRole + 2; // Update available
    static const int SCHEME_ID = 1; // Update available

signals:

public slots:
    

private slots:
    void on_treeWidgetDB_itemChanged(QTreeWidgetItem * item, int column);
    void on_checkModules_stateChanged (int state);
    void on_checkFilters_stateChanged (int state);
    void xmlDownloaded();
    void refreshButtonClicked();
};

#endif
