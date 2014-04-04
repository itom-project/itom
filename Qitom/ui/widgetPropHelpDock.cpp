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

#include "widgetPropHelpDock.h"
//#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>
#include <qurl.h>
#include "helper\fileDownloader.h"
#include <QtSql/qsqldatabase.h>
#include <QtSql/qsqlquery.h>
#include <qxmlstream.h>
#include <global.h>



WidgetPropHelpDock::WidgetPropHelpDock(QWidget *parent) :
    AbstractPropertyPageWidget(parent),
    m_pdbPath(qApp->applicationDirPath()+"/help/")
{
    ui.setupUi(this);
    m_listChanged = false;
    m_treeIsUpdating = false;
    ui.label->hide();
    initMenus();
    connect(ui.pushRefreshUpdates, SIGNAL(clicked()), this, SLOT(refreshButtonClicked()));
    QStringList headers;
    headers << tr("Database") << tr("Version") << tr("Date") << tr("Updates and Downloads") ;
    ui.treeWidgetDB->setHeaderLabels(headers);
    ui.treeWidgetDB->setColumnWidth(0, 200);
    ui.treeWidgetDB->setColumnWidth(1,  50);
    ui.treeWidgetDB->setColumnWidth(2,  60);
}

WidgetPropHelpDock::~WidgetPropHelpDock()
{

}

void WidgetPropHelpDock::on_checkModules_stateChanged (int state)
{
    if (ui.checkModules->isChecked())
        ui.treeWidgetDB->setEnabled(true);
    else
        ui.treeWidgetDB->setEnabled(false);
}

void WidgetPropHelpDock::on_checkFilters_stateChanged (int state)
{
    m_listChanged = true;
    ui.label->show();
}

void WidgetPropHelpDock::on_treeWidgetDB_itemChanged(QTreeWidgetItem* item, int column)
{
    if (!m_treeIsUpdating && item->data(0, m_urID).isValid())
    {
        updateCheckedIdList();
        setExistingDBsChecks();
        m_listChanged = true;
        ui.label->show();
    }
}

void WidgetPropHelpDock::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("HelpScriptReference"); //keep this name fix here!

    ui.checkExtLinks->setChecked( settings.value("openExtLinks", true).toBool() );
    ui.checkPlaintext->setChecked( settings.value("plaintext", false).toBool() );
    ui.checkFilters->setChecked( settings.value("showFilters", false).toBool() );
    ui.checkWidgets->setChecked( settings.value("showWidgets", false).toBool() );
    ui.checkDataIO->setChecked( settings.value("showDataIO", false).toBool() );
    ui.checkModules->setChecked( settings.value("showModules", false).toBool() );
    ui.lineEdit->setText( settings.value("serverAdress", "").toString());
    if (ui.checkModules->isChecked())
        ui.treeWidgetDB->setEnabled(true);
    else
        ui.treeWidgetDB->setEnabled(false);

    // Read the List of Databases and if they are checked ($ = checked, % = unchecked)
    int size = settings.beginReadArray("Databases");
    checkedIdList.clear();
    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);
        int id = settings.value("DB", QString()).toInt();
        checkedIdList.append(id);
        settings.remove("DB");
    }
    settings.endGroup();
    refreshButtonClicked();
}

void WidgetPropHelpDock::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("HelpScriptReference");

    settings.setValue("openExtLinks", ui.checkExtLinks->isChecked());
    settings.setValue("plaintext"   , ui.checkPlaintext->isChecked());
    settings.setValue("reLoadDBs"   , m_listChanged );
    settings.setValue("showFilters" , ui.checkFilters->isChecked());
    settings.setValue("showWidgets" , ui.checkWidgets->isChecked());
    settings.setValue("showDataIO"  , ui.checkDataIO->isChecked());
    settings.setValue("showModules" , ui.checkModules->isChecked());
    settings.setValue("serverAdress", ui.lineEdit->text());

    // Write the checkstate with the List into the ini File
    settings.beginWriteArray("Databases");
    QMap<int, WidgetPropHelpDock::databaseInfo>::iterator it;
    int i = 0;
    for (it = existingDBs.begin(); it != existingDBs.end(); ++it)
    {
        settings.setArrayIndex(i);
        if (it->isChecked)
        {
            settings.setValue("DB", it.key());
        }
        ++i;
    }
    settings.endArray();
    settings.endGroup();
}



// All about the Databases and online updates

void WidgetPropHelpDock::refreshButtonClicked()
{// This button is also automatically clicked when the option dialog is started
    
    // Update both Databases
    refreshExistingDBs();
    refreshUpdatableDBs();

    // if the new db is downloaded, funcktions are called as folowed:
    // => xmlDownloaded()
    // ===> compareDatabaseVersions()
    // =====> updateTreeWidget()
}

// refreshes the existingDBs Map
void WidgetPropHelpDock::refreshExistingDBs()
{
    QDirIterator it(m_pdbPath, QStringList("*.db"), QDir::Files | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
    QString fileName;
    existingDBs.clear();
    while(it.hasNext())
    {
        QString path = it.next();
        fileName = path.right(path.length() - path.lastIndexOf('/') - 1);
        // open SQL-Database and take a look into the info-table to read further information
        QFile f(path);
        WidgetPropHelpDock::databaseInfo *item = new WidgetPropHelpDock::databaseInfo();
        if (f.exists())
        {
            QSqlDatabase database = QSqlDatabase::addDatabase("QSQLITE", path); //important to have variables database and query in local scope such that removeDatabase (outside of this scope) can securly free all resources! -> see docs about removeDatabase
            database.setDatabaseName(path);
            bool ok = database.open();
            if (ok)
            {
                QSqlQuery query("SELECT id, name, version, date, itomMinVersion FROM databaseInfo ORDER BY id", database);
                query.exec();
                query.next();
                int id                = query.value(0).toInt();
                item->name            = query.value(1).toString();
                item->version         = query.value(2).toString();
                item->date            = query.value(3).toString();
                item->schemeID        = query.value(4).toString();
                item->path            = path;
                item->updateState     = updateState::unknown;
                // add to Map
                existingDBs.insert(id, *item);
            }
            database.close();
        }  
        QSqlDatabase::removeDatabase(path);
    }
    setExistingDBsChecks();
}

// This block downlaods and refreshes the updatableDBs Map
void WidgetPropHelpDock::refreshUpdatableDBs()
{
    // Get information of online DBs
    m_pXmlCtrl = new FileDownloader(QUrl(ui.lineEdit->text()), this);
    connect(m_pXmlCtrl, SIGNAL(downloaded()), SLOT(xmlDownloaded()));
}
void WidgetPropHelpDock::xmlDownloaded()
{
    // first of all, clear the old List
    updatableDBs.clear();

    // now start parsing the new List
    QXmlStreamReader xml;
    xml.addData(m_pXmlCtrl->downloadedData());
    while(!xml.atEnd() && !xml.hasError()) 
    {
        QXmlStreamReader::TokenType token = xml.readNext();
        /* If token is just StartDocument, we'll go to next.*/
        if(token == QXmlStreamReader::StartDocument) 
        {
            continue;
        }
        /* If token is StartElement, we'll see if we can read it.*/
        if(token == QXmlStreamReader::StartElement) 
        {
            /* If it's named persons, we'll go to the next.*/
            if(xml.name() == "databases") 
            {
                continue;
            }
            
            if(xml.name() == "file") 
            {
                QPair <int, WidgetPropHelpDock::databaseInfo> p = this->parseFile(xml);
                updatableDBs.insert(p.first, p.second);
            }
        }
    }
    // Now Compare the two Lists in
    compareDatabaseVersions();
}
QPair<int, WidgetPropHelpDock::databaseInfo> WidgetPropHelpDock::parseFile(QXmlStreamReader& xml) 
{
    QPair<int, WidgetPropHelpDock::databaseInfo> file;
    int id = 0;
    /* Let's check that we're really getting a person. */
    if(xml.tokenType() != QXmlStreamReader::StartElement && xml.name() == "file") 
    {
        return file;
    }
    WidgetPropHelpDock::databaseInfo *appendedItem = new WidgetPropHelpDock::databaseInfo();
    appendedItem->updateState = updateState::unknown;

    // Check for ID-Attribute
    QXmlStreamAttributes attributes = xml.attributes();
    if(attributes.hasAttribute("id")) 
    {
        id = attributes.value("id").toString().toInt();
    }
    /* Next element... */
    xml.readNext();
    while(!(xml.tokenType() == QXmlStreamReader::EndElement && xml.name() == "file")) 
    {
        if(xml.tokenType() == QXmlStreamReader::StartElement) 
        {
            if(xml.name() == "name")
            {
                xml.readNext();
                appendedItem->name = xml.text().toString();
            }
            else if(xml.name() == "version")
            {
                xml.readNext();
                appendedItem->version = xml.text().toString();
            }
            else if(xml.name() == "date")
            {
                xml.readNext();
                appendedItem->date = xml.text().toString();
            }
            else if(xml.name() == "schemeID")
            {
                xml.readNext();
                appendedItem->schemeID = xml.text().toString();
            }
            else if(xml.name() == "path")
            {
                xml.readNext();
                appendedItem->path = xml.text().toString();
            }
        }
        /* ...and next... */
        xml.readNext();
    }
    file.first  = id;
    file.second = *appendedItem;
    return file;
}

void WidgetPropHelpDock::setUpdateColumnText(QTreeWidgetItem *widget)
{
    QString infoText = "";
    int ID = widget->data(0, m_urID).toInt();
    switch(widget->data(0, m_urUD).toInt()) 
    {
        case  updateState::unknown: 
        {
            widget->setText(3, "unknown (lokal build Database)");
            widget->setIcon(0, QIcon(":/helpTreeDockWidget/pluginRawIO.png"));
            break;
        }
        case  updateState::upToDate: 
        {
            widget->setText(3, "Up to date");
            widget->setIcon(0, QIcon(":/helpTreeDockWidget/masterProject.png"));
            break;
        }
        case  updateState::updateAvailable: 
        {
            widget->setText(3, "Update to version: " + updatableDBs[ID].version + " (" + updatableDBs[ID].date + ")");
            widget->setIcon(0, QIcon(":/application/dialog-error-4.png"));
            break;
        }
        case  updateState::downloadAvailable: 
        {
            widget->setText(3, "Download version: " + updatableDBs[ID].version + " (" + updatableDBs[ID].date + ")");
            widget->setIcon(0, QIcon(":/helpTreeDockWidget/downloadUpdate.png"));
            break;
        }
        case  updateState::wrongScheme: 
        {
            widget->setText(3, "wrong Scheme: " + updatableDBs[ID].schemeID + " (your Scheme "+existingDBs[ID].schemeID+")");
            widget->setIcon(0, QIcon(":/application/dialog-error-4.png"));
            break;
        }
        default: 
        {
            widget->setText(3, "");
            break;
        }
    }
}

void WidgetPropHelpDock::updateTreeWidget()
{
    // surpress the itemChanged() event of the tree View
    m_treeIsUpdating = true;
    
    // to avoid the loss of the checkboxes state
    setExistingDBsChecks();
    ui.treeWidgetDB->clear();

    // new Toplevel-Item (local)
    QTreeWidgetItem *topItem = new QTreeWidgetItem();
    topItem->setText(0, "Local");
    if (topItem->flags().testFlag(Qt::ItemIsUserCheckable))
    {
        topItem->setFlags( topItem->flags() ^ Qt::ItemIsUserCheckable);
    }
    //topItem->setIcon(0, QIcon(":/helpTreeDockWidget/pluginAdda"));
    ui.treeWidgetDB->addTopLevelItem(topItem);

    // Create Local-Subitems
    QMap<int, WidgetPropHelpDock::databaseInfo>::iterator i;
    for (i = existingDBs.begin(); i != existingDBs.end(); ++i)
    {
        QTreeWidgetItem *newWidget = new QTreeWidgetItem();
        newWidget->setText(0, i->name);
        newWidget->setText(1, i->version);
        newWidget->setText(2, i->date);
        newWidget->setData(0, m_urID, i.key());
        newWidget->setData(0, m_urUD, i->updateState);
        if (i->isChecked == true)
        {
            newWidget->setCheckState(0, Qt::Checked);
        }
        else
        {
            newWidget->setCheckState(0, Qt::Unchecked);
        }
        // Set update Column in separate function
        setUpdateColumnText(newWidget);
        // Add item to Parent
        topItem->addChild(newWidget);
    }
    ui.treeWidgetDB->addTopLevelItem(topItem);

    // new Toplevel-Item (remote)
    topItem = new QTreeWidgetItem();
    topItem->setText(0, "Remote");
    if (topItem->flags().testFlag(Qt::ItemIsUserCheckable))
    {
        topItem->setFlags( topItem->flags() ^ Qt::ItemIsUserCheckable);
    }

    // Create Remote-Subitems
    for (i = updatableDBs.begin(); i != updatableDBs.end(); ++i)
    {
        if (i->updateState == updateState::downloadAvailable)
        {
            QTreeWidgetItem *newWidget = new QTreeWidgetItem();
            newWidget->setFlags(newWidget->flags() ^ Qt::ItemIsUserCheckable);
            newWidget->setText(0, i->name);
            newWidget->setText(1, i->version);
            newWidget->setText(2, i->date);
            newWidget->setData(0, m_urID, i.key());
            newWidget->setData(0, m_urUD, i->updateState);
            // Set update Column in separate function
            setUpdateColumnText(newWidget);
            // Add item to Parent
            topItem->addChild(newWidget);
        }
    }
    ui.treeWidgetDB->addTopLevelItem(topItem);
    ui.treeWidgetDB->expandAll();

    // release the surpress of the itemChanged() event of the tree View
    m_treeIsUpdating = false;
}

void WidgetPropHelpDock::compareDatabaseVersions()
{
    // All existingDBs-elements are set to unknown
    // now compare them all to the downloadlist an decide which state they get
    QMap<int, WidgetPropHelpDock::databaseInfo>::iterator i;
    for (i = updatableDBs.begin(); i != updatableDBs.end(); ++i)
    {
        if (existingDBs.contains(i.key()))
        {
            if (i->schemeID.toDouble() == SCHEME_ID)
            {
                if (i->version.toDouble() > existingDBs.value(i.key()).version.toDouble())
                {
                    existingDBs[i.key()].updateState = updateState::updateAvailable;
                }
                else
                {
                    existingDBs[i.key()].updateState = updateState::upToDate;
                }
            }
            else
            {
                i->updateState = updateState::wrongScheme;
            }
        }
        else
        {
            if (updatableDBs.value(i.key()).schemeID.toDouble() == SCHEME_ID)
            {
                i->updateState = updateState::downloadAvailable;
            }
            else
            {
                i->updateState = updateState::wrongScheme;
            }
        }
    }
    // update the ListWidget
    updateTreeWidget();
}

//----------------------------------------------------------------------------------------------------------------------------------
// This function saves the id of each checked entry of the TreeWidget in the 
// checkedIdList. Don´t modify that list manually. 
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::updateCheckedIdList()
{
    checkedIdList.clear();
    for (int i = 0; i < ui.treeWidgetDB->topLevelItem(0)->childCount() ; ++i)
    {
        if (ui.treeWidgetDB->topLevelItem(0)->child(i)->checkState(0) == Qt::Checked)
        {
            checkedIdList.append(ui.treeWidgetDB->topLevelItem(0)->child(i)->data(0, m_urID).toInt());
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::setExistingDBsChecks()
{
    QMap<int, WidgetPropHelpDock::databaseInfo>::iterator i;
    for (i = existingDBs.begin(); i != existingDBs.end(); ++i)
    {
        if (checkedIdList.contains(i.key()))
        {
            i->isChecked = true;
        }
        else
        {
            i->isChecked = false;
        }
    }
}



// Hint-Information (path and url)
bool WidgetPropHelpDock::event (QEvent * event)
{
    if (false)//(event->type() == QEvent::ToolTip)
    {
        QHelpEvent *evt = static_cast<QHelpEvent*>(event);
        // TODO zuviele point
        QPoint point  = evt->pos();
        QPoint point2 = ui.treeWidgetDB->mapFrom(this, point);
        QTreeWidgetItem *selItem = ui.treeWidgetDB->itemAt(point2);
        if (selItem != NULL)
        {
            QString message = "Path: " + existingDBs.value(selItem->data(0, m_urID).toInt()).path
                            +"\nUrl: " + updatableDBs.value(selItem->data(0, m_urID).toInt()).path; 
            QToolTip::showText(evt->globalPos(), message, this);
        }
    }
    return AbstractPropertyPageWidget::event(event);
}



// Update-Context-Menu


void WidgetPropHelpDock::initMenus()
{
   /* updateMenu = new QMenu(this);
    updateMenuActions["update"] = updateMenu->addAction(QIcon(":/bookmark/icons/bookmarkToggle.png"), tr("&update"), this, SLOT(updateHelpFiles(QStringList files)));
    updateMenuActions["update all"] = updateMenu->addAction(QIcon(":/bookmark/icons/bookmarkNext.png"), tr("update all"), this, SLOT(updateHelpFiles(QStringList files)));
    updateMenuActions["info"] = updateMenu->addAction(QIcon(":/bookmark/icons/bookmarkNext.png"), tr("info"), this, SLOT(updateHelpFiles(QStringList files)));

    connect(updateMenu, SIGNAL(aboutToShow()), this, SLOT(preShowContextMenuMargin()));*/
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::preShowContextMenuMargin()
{
    //// if this item is updatable
    //updateMenuActions["update"]->setEnabled(true);
    //
    //// if List has more than one entry
    //updateMenuActions["update all"]->setEnabled(true);

    //// Always
    //updateMenuActions["info"]->setEnabled(true);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::contextMenuEvent (QContextMenuEvent * event)
{
    /*event->accept();*/
}
