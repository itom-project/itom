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
#include "../AppManagement.h"

#include <qsettings.h>
#include <qurl.h>
#include "helper\fileDownloader.h"
#include <QtSql/qsqldatabase.h>
#include <QtSql/qsqlquery.h>
#include <qxmlstream.h>
#include <global.h>

// Constructor
//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropHelpDock::WidgetPropHelpDock(QWidget *parent) :
    AbstractPropertyPageWidget(parent),
    m_pdbPath(qApp->applicationDirPath()+"/help/"),
    m_downloadTimeout(10000),
    m_downloadTimeoutReached(false)
{
    ui.setupUi(this);
    m_listChanged = false;
    m_treeIsUpdating = false;
    ui.label->hide();
    
    // init consts
     m_xmlFileName = ""; //updateInfo.xml";

    initMenus();
    connect(ui.pushRefreshUpdates, SIGNAL(clicked()), this, SLOT(refreshButtonClicked()));
    QStringList headers;
    headers << tr("Database") << tr("Version") << tr("Date") << tr("Updates and Downloads") ;
    ui.treeWidgetDB->setHeaderLabels(headers);
    ui.treeWidgetDB->setColumnWidth(0, 200);
    ui.treeWidgetDB->setColumnWidth(1,  50);
    ui.treeWidgetDB->setColumnWidth(2,  60);
    ui.treeWidgetDB->setContextMenuPolicy(Qt::CustomContextMenu);

    connect(ui.treeWidgetDB, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(treeWidgetContextMenuRequested(const QPoint &)));
    connect(ui.spinTimeout, SIGNAL(valueChanged(int)), this, SLOT(on_spinTimeout_valueChanged(int)));
}

// Destructor
//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropHelpDock::~WidgetPropHelpDock()
{

}

// Checkbox changed 
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::on_checkModules_stateChanged (int state)
{
    if (ui.checkModules->isChecked())
        ui.treeWidgetDB->setEnabled(true);
    else
        ui.treeWidgetDB->setEnabled(false);
}

// Checkbox changed
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::on_checkFilters_stateChanged (int state)
{
    ui.label->show();
}

// Checkbox changed
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::on_treeWidgetDB_itemChanged(QTreeWidgetItem* item, int column)
{
    if (!m_treeIsUpdating && item->data(0, m_urID).isValid())
    {
        m_listChanged = true;
        updateCheckedIdList();
        setExistingDBsChecks();
        ui.label->show();
    }
}

// get settings from ini
//----------------------------------------------------------------------------------------------------------------------------------
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
    m_downloadTimeout = settings.value("downloadTimeout", 10000).toInt();
    ui.spinTimeout->setValue(m_downloadTimeout/1000);

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
        QString nameID = settings.value("DB", QString()).toString();
        int id = nameID.mid(nameID.indexOf("§")+1, -1).toInt();
        checkedIdList.append(id);
        settings.remove("DB");
    }
    settings.endGroup();
    //refreshButtonClicked();
}

// set settings from ini
//----------------------------------------------------------------------------------------------------------------------------------
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
    settings.setValue("downloadTimeout", m_downloadTimeout);

    // Write the checkstate with the List into the ini File
    settings.beginWriteArray("Databases");
    QMap<int, WidgetPropHelpDock::DatabaseInfo>::iterator it;
    int i = 0;
    for (it = existingDBs.begin(); it != existingDBs.end(); ++it)
    {
        if (it->isChecked)
        {
            settings.setArrayIndex(i);
            settings.setValue("DB", it->name + "§" + QString::number(it.key()));
            ++i;
        }
    }
    settings.endArray();
    settings.endGroup();
}

// ui Spinbox for timeout changed
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::on_spinTimeout_valueChanged(int i)
{
    m_downloadTimeout = i*1000;
}

// All about the Databases and online updates
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::refreshButtonClicked()
{// This button is also automatically clicked when the option dialog is started
    m_serverAdress.setUrl(ui.lineEdit->text());

    // Update both Databases
    refreshExistingDBs();
    refreshUpdatableDBs();

    // if the new db is downloaded, funcktions are called as folowed:
    // => xmlDownloaded()
    // ===> compareDatabaseVersions()
    // =====> updateTreeWidget()
}

// refreshes the existingDBs Map
//----------------------------------------------------------------------------------------------------------------------------------
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
        WidgetPropHelpDock::DatabaseInfo *item = new WidgetPropHelpDock::DatabaseInfo();
        if (f.exists())
        {
            QSqlDatabase database = QSqlDatabase::addDatabase("QSQLITE", path); //important to have variables database and query in local scope such that removeDatabase (outside of this scope) can securly free all resources! -> see docs about removeDatabase
            database.setDatabaseName(path);
            bool ok = database.open();
            if (ok)
            {
                QSqlQuery query("SELECT id, name, version, date, itomMinVersion FROM DatabaseInfo ORDER BY id", database);
                query.exec();
                query.next();
                int id                = query.value(0).toInt();
                item->name            = query.value(1).toString();
                item->version         = query.value(2).toInt();
                item->date            = query.value(3).toString();
                item->schemeID        = query.value(4).toInt();
                item->path            = QFileInfo(path + QDir::separator());
                item->url             = QUrl();
                item->updateState     = stateUnknown;
                // add to Map
                existingDBs.insert(id, *item);
            }
            database.close();
        }  
        QSqlDatabase::removeDatabase(path);
    }
    setExistingDBsChecks();
}

// if download takes too long, this slot is called and the while loop is broken up
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::downloadTimeoutReached()
{
    m_downloadTimeoutReached = true;
}

// This block downlaods and refreshes the updatableDBs Map
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::refreshUpdatableDBs()
{
    QProgressDialog progress("Remote database update...", tr("Cancel"), 0, 101, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);

    FileDownloader *downloader = new FileDownloader(QUrl(m_serverAdress.toString() + m_xmlFileName), 5, this);
    FileDownloader::Status status = FileDownloader::sRunning; //0: still running, 1: ok, 2: error, 3: cancelled
    QString errorMsg;
    bool dbFound = false;
    QString dbError;

    // Download-timeout
    QTimer *timeoutTimer = new QTimer(this);
    connect(timeoutTimer, SIGNAL(timeout()), this, SLOT(downloadTimeoutReached()));
    timeoutTimer->setSingleShot(true);
    m_downloadTimeoutReached = false;
    timeoutTimer->start(m_downloadTimeout);

    while(status == FileDownloader::sRunning)
    {
        if (progress.wasCanceled())
        {
            downloader->abortDownload();
            status = FileDownloader::sAborted;
        }
        else
        {
            progress.setValue(downloader->getDownloadProgress());
            status = downloader->getStatus(errorMsg);
        }

        QCoreApplication::processEvents();

        if (m_downloadTimeoutReached)
        {
            status = FileDownloader::sError;
            errorMsg = "Timeout: Server is not responding in time";
            break;
        }
    }
    timeoutTimer->stop();
    timeoutTimer->deleteLater();

    // first of all, clear the old List
    updatableDBs.clear();

    // now start parsing the new List
    if (status == FileDownloader::sFinished)
    {
        QXmlStreamReader xml;
        xml.addData(downloader->downloadedData());
        while(!xml.atEnd() && !xml.hasError()) 
        {
            QXmlStreamReader::TokenType token = xml.readNext();
            if(token == QXmlStreamReader::StartDocument) 
            {
                continue;
            }
            if(token == QXmlStreamReader::StartElement) 
            {
                if(xml.name() == "databases") 
                {
                    if (xml.attributes().hasAttribute("type"))
                    {
                        if (xml.attributes().value("type") == "itom.repository.helpDatabase")
                        {
                            dbFound = true;
                        }
                        else
                        {
                            dbError = tr("Invalid type attribute of xml file");
                        }
                    }
                    else
                    {
                        dbError = tr("Type attribute node 'database' of xml file is missing.");
                    }
                }
                else if(dbFound && xml.name() == "file") 
                {
                    QPair <int, WidgetPropHelpDock::DatabaseInfo> p = this->parseFile(xml);
                    if (updatableDBs.contains(p.first))
                    {
                        if (p.second.schemeID == SCHEME_ID)
                        {
                            if (updatableDBs[p.first].version < p.second.version)
                            {
                                updatableDBs.insert(p.first, p.second);
                            }
                        }
                    }   
                    else
                    {
                        updatableDBs.insert(p.first, p.second);
                    }
                }
            }
        }

        if (xml.hasError())
        {
            dbFound = false;
            dbError = tr("xml parsing error: %1").arg(xml.errorString());
        }
        else if (!dbFound && dbError == "")
        {
            dbError = tr("xml error: node 'database' is missing.");
        }

        if (!dbFound)
        {
            status = FileDownloader::sError;
            errorMsg = dbError;
        } 
    }

    if (status == FileDownloader::sError)
    {
        showErrorMessage(errorMsg);
    }
    
    // finish progressWindow
    progress.setValue(101);

    // Clean up
    downloader->deleteLater();

    // Now continue to Compare the two Lists in
    compareDatabaseVersions();
}

// Parse the xml Elements and return each as pair
//----------------------------------------------------------------------------------------------------------------------------------
QPair<int, WidgetPropHelpDock::DatabaseInfo> WidgetPropHelpDock::parseFile(QXmlStreamReader& xml) 
{
    QPair<int, WidgetPropHelpDock::DatabaseInfo> file;
    int id = 0;
    if(xml.tokenType() != QXmlStreamReader::StartElement && xml.name() == "file") 
    {
        return file;
    }
    WidgetPropHelpDock::DatabaseInfo *appendedItem = new WidgetPropHelpDock::DatabaseInfo();
    appendedItem->updateState = stateUnknown;

    // Check for ID-Attribute
    QXmlStreamAttributes attributes = xml.attributes();
    if(attributes.hasAttribute("id")) 
    {
        id = attributes.value("id").toString().toInt();
    }
    /* Next element... */
    xml.readNext();
    int timeOut = 10;
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
                appendedItem->version = xml.text().toString().toInt();
            }
            else if(xml.name() == "date")
            {
                xml.readNext();
                appendedItem->date = xml.text().toString();
            }
            else if(xml.name() == "schemeID")
            {
                xml.readNext();
                appendedItem->schemeID = xml.text().toString().toInt();
            }
            else if(xml.name() == "url")
            {
                xml.readNext();
                appendedItem->url = QUrl(xml.text().toString());
            }
        }
        else if(timeOut <= 0)
        {
            break;
        }
        else
        {
            --timeOut;
        }
        /* ...and next... */
        xml.readNext();
    }
    file.first  = id;
    file.second = *appendedItem;
    return file;
}

// Sets the text in the last column of the tree (update available etc)
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::setUpdateColumnText(QTreeWidgetItem *widget)
{
    QString infoText = "";
    int ID = widget->data(0, m_urID).toInt();
    switch(widget->data(0, m_urUD).toInt()) 
    {
        case  stateUnknown: 
        {
            widget->setText(3, "unknown (lokal build Database)");
            widget->setIcon(3, QIcon(":/helpTreeDockWidget/localDatabase")); // OK!
            break;
        }
        case  stateUpToDate: 
        {
            widget->setText(3, tr("Up to date"));
            widget->setIcon(3, QIcon(":/helpTreeDockWidget/upToDate"));
            break;
        }
        case  stateUpdateAvailable: 
        {
            widget->setText(3, tr("Update to version: %1 (%2)").arg(QString::number(updatableDBs[ID].version)).arg(updatableDBs[ID].date));
            widget->setIcon(3, QIcon(":/helpTreeDockWidget/downloadUpdate"));
            break;
        }
        case  stateDownloadAvailable: 
        {
            widget->setText(3, tr("Download version: %1 (%2)").arg(QString::number(updatableDBs[ID].version)).arg(updatableDBs[ID].date));
            widget->setIcon(3, QIcon(":/helpTreeDockWidget/downloadUpdate"));
            break;
        }
        case  stateWrongScheme: 
        {
            widget->setText(3, tr("wrong Scheme: %1 (your scheme%2)").arg(QString::number(existingDBs[ID].schemeID)).arg(QString::number(SCHEME_ID)));
            widget->setIcon(3, QIcon(":/helpTreeDockWidget/wrongScheme"));
            widget->setFlags(widget->flags() ^ Qt::ItemIsEnabled);
            break;
        }
        default: 
        {
            widget->setText(3, "");
            break;
        }
    }
}

// Updates the treeWidget from the two maps (existingDBs / updatableDBs)
//----------------------------------------------------------------------------------------------------------------------------------
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
    topItem->setIcon(0, QIcon(":/helpTreeDockWidget/localDatabase"));
    if (topItem->flags().testFlag(Qt::ItemIsUserCheckable))
    {
        topItem->setFlags( topItem->flags() ^ Qt::ItemIsUserCheckable);
    }
    //topItem->setIcon(0, QIcon(":/helpTreeDockWidget/pluginAdda"));
    ui.treeWidgetDB->addTopLevelItem(topItem);

    // Create Local-Subitems
    QMap<int, WidgetPropHelpDock::DatabaseInfo>::iterator i;
    for (i = existingDBs.begin(); i != existingDBs.end(); ++i)
    {
        QTreeWidgetItem *newWidget = new QTreeWidgetItem();
        newWidget->setText(0, i->name);
        newWidget->setText(1, QString::number(i->version));
        newWidget->setText(2, i->date);
        newWidget->setData(0, m_urID, i.key());
        newWidget->setData(0, m_urUD, i->updateState);
        newWidget->setData(0, m_urFD, i->path.path());
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
    topItem->setText(0, tr("Remote"));
    topItem->setIcon(0, QIcon(":/helpTreeDockWidget/downloadUpdate"));
    if (topItem->flags().testFlag(Qt::ItemIsUserCheckable))
    {
        topItem->setFlags( topItem->flags() ^ Qt::ItemIsUserCheckable);
    }

    // Create Remote-Subitems
    for (i = updatableDBs.begin(); i != updatableDBs.end(); ++i)
    {
        if (i->updateState == stateDownloadAvailable)
        { // This code snipet added to the if would add wrong schemes, too:  "|| i->updateState == updateState::wrongScheme"
            QTreeWidgetItem *newWidget = new QTreeWidgetItem();
            newWidget->setFlags(newWidget->flags() ^ Qt::ItemIsUserCheckable);
            newWidget->setText(0, i->name);
            newWidget->setText(1, QString::number(i->version));
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

// Compares the two maps and sets their update/download status
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::compareDatabaseVersions()
{
    // All existingDBs-elements are set to unknown
    // now compare them all to the downloadlist an decide which state they get
    QMap<int, WidgetPropHelpDock::DatabaseInfo>::iterator i;
    for (i = updatableDBs.begin(); i != updatableDBs.end(); ++i)
    {
        if (existingDBs.contains(i.key()))
        {
            if (i->schemeID == SCHEME_ID)
            {
                if (i->version > existingDBs.value(i.key()).version)
                {
                    existingDBs[i.key()].updateState = stateUpdateAvailable;
                }
                else
                {
                    existingDBs[i.key()].updateState = stateUpToDate;
                }
            }
            else
            { // Version with other Scheme is available, but has wrong Scheme
                existingDBs[i.key()].updateState = stateUpToDate;
            }
        }
        else
        {
            if (updatableDBs.value(i.key()).schemeID == SCHEME_ID)
            {
                i->updateState = stateDownloadAvailable;
            }
            else
            { // Not Downloadable because of wrong Scheme ... don´t display
                i->updateState = stateWrongScheme;
            }
        }
    }
    // detect all the unknown DBs with wrong Scheme
    for (i = existingDBs.begin(); i != existingDBs.end(); ++i)
    {
        if (i->updateState == stateUnknown && i->schemeID != SCHEME_ID)
        {
            i->updateState = stateWrongScheme;
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

// Checkstate of a DB changes
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::setExistingDBsChecks()
{
    QMap<int, WidgetPropHelpDock::DatabaseInfo>::iterator i;
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
//----------------------------------------------------------------------------------------------------------------------------------
bool WidgetPropHelpDock::event (QEvent * event)
{
    if (event->type() == QEvent::ToolTip)
    {
        QHelpEvent *evt = static_cast<QHelpEvent*>(event);
        // Maybe this mapping is too much, but it works!
        QPoint pos = ui.treeWidgetDB->viewport()->mapFromGlobal(evt->globalPos());
        QPoint global = ui.treeWidgetDB->viewport()->mapToGlobal(pos);
        QTreeWidgetItem *selItem = ui.treeWidgetDB->itemAt(ui.treeWidgetDB->viewport()->mapFromGlobal(global));
        if (selItem != NULL)
        {
            QString message = selItem->data(0, m_urFD).toString();
            QToolTip::showText(global, message, this);
        }
    }
    return AbstractPropertyPageWidget::event(event);
}

// Update-Context-Menu

// Show error Message
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::showErrorMessage(const QString &error)
{
    QMessageBox::warning(this, tr("download error"), error);
}

// init DropdownMenu
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::initMenus()
{
    m_pContextMenu = new QMenu(this);
    contextMenuActions["update"] = m_pContextMenu->addAction(QIcon(":/helpTreeDockWidget/downloadUpdate"), tr("&update"), this, SLOT(mnuDownloadUpdate()));
    contextMenuActions["locateOnDisk"] = m_pContextMenu->addAction(QIcon(":/files/icons/browser.png"), tr("locate on disk"), this, SLOT(mnuLocateOnDisk()));
    contextMenuActions["removeDatabase"] = m_pContextMenu->addAction(QIcon(":/helpTreeDockWidget/deleteDatabase"), tr("remove from disk"), this, SLOT(mnuRemoveDatabase()));


    connect(m_pContextMenu, SIGNAL(aboutToShow()), this, SLOT(preShowContextMenuMargin()));
}

// This Block downloads/updates a single database
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::mnuDownloadUpdate()
{
    QTreeWidgetItem *item = ui.treeWidgetDB->selectedItems().at(0);
    if (item != NULL)
    {
        QFileInfo oldFile = existingDBs[item->data(0, m_urID).toInt()].path;
        QProgressDialog progress(tr("Remote database update..."), tr("Cancel"), 0, 101, this);
        progress.setWindowModality(Qt::WindowModal);
        progress.setMinimumDuration(0);

        QUrl url = /*m_serverAdress + */updatableDBs[item->data(0, m_urID).toInt()].url;
        QString Test1 = url.toString();
        QString Test3 = oldFile.absolutePath();
        QString Test4 = oldFile.absoluteFilePath();
        FileDownloader *downloader = new FileDownloader(url, 5, this);
        FileDownloader::Status status = FileDownloader::sRunning; //0: still running, 1: ok, 2: error, 3: cancelled
        QString errorMsg;

        // Download-timeout
        QTimer *timeoutTimer = new QTimer(this);
        connect(timeoutTimer, SIGNAL(timeout()), this, SLOT(downloadTimeoutReached()));
        timeoutTimer->setSingleShot(true);
        m_downloadTimeoutReached = false;
        timeoutTimer->start(m_downloadTimeout);

        while(status == FileDownloader::sRunning)
        {
            if (progress.wasCanceled())
            {
                downloader->abortDownload();
                status = FileDownloader::sAborted;
            }
            else
            {
                progress.setValue(downloader->getDownloadProgress());
                status = downloader->getStatus(errorMsg);
            }
            
            QCoreApplication::processEvents();
        
            if (m_downloadTimeoutReached)
            {
                status = FileDownloader::sError;
                errorMsg = tr("Timeout: Server is not responding in time");
                break;
            }
        }
        timeoutTimer->stop();
        timeoutTimer->deleteLater();

        // Replace old Database with Downloaded
        if (status == FileDownloader::sFinished)
        {
            QString newLocalPath;
            
            //url = url.left(url.lastIndexOf("/"));
            // Downlaod Finished, Safe File
            if (oldFile.exists())
            {
                newLocalPath = oldFile.path() + existingDBs[item->data(0, m_urID).toInt()].name + ".db";//url.right(url.length()-url.lastIndexOf("/"));
                if (!QFile::remove(oldFile.path()))
                {
                    showErrorMessage(tr("Could not delete old local version of Database"));
                }
            }
            else 
            {
                newLocalPath = m_pdbPath + updatableDBs[item->data(0, m_urID).toInt()].name + ".db";//right(url.length()-url.lastIndexOf("/")-1);
            }
            QFile file(newLocalPath);
            file.open(QIODevice::WriteOnly);
            // save new Database to old Path
            file.write(downloader->downloadedData());
            file.close();
            // Sucessful update, refresh TreeWidget:
            refreshButtonClicked();
        }
        else if (status == FileDownloader::sError)
        {
            showErrorMessage(errorMsg);
        }

        // finish progressWindow
        progress.setValue(101);

        // CleanUp
        downloader->deleteLater();
    }
}

// Locate on disc button on Database clicked
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::mnuLocateOnDisk()
{
    QTreeWidgetItem *item = ui.treeWidgetDB->selectedItems().at(0);
    if (item != NULL)
    {
        showInGraphicalShell(item->data(0, m_urFD).toString());
    }
}

void WidgetPropHelpDock::mnuRemoveDatabase()
{
    QTreeWidgetItem *item = ui.treeWidgetDB->selectedItems().at(0);
    if (item != NULL)
    {
        QFile::remove(item->data(0, m_urFD).toString());
        refreshButtonClicked();
    }
}

// Highlights a file in the explorer when mnuLocateOnDisc is clicked
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::showInGraphicalShell(const QString & filePath)
{

    #ifdef Q_WS_MAC
    QStringList args;
    args << "-e";
    args << "tell application \"Finder\"";
    args << "-e";
    args << "activate";
    args << "-e";
    args << "select POSIX file \""+filePath+"\"";
    args << "-e";
    args << "end tell";
    QProcess::startDetached("osascript", args);
#endif

#ifdef Q_WS_WIN
    QStringList args;
    args << "/select," << QDir::toNativeSeparators(filePath);
    QProcess::startDetached("explorer", args);
#endif
}

// ContextMenu pops up
//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropHelpDock::treeWidgetContextMenuRequested(const QPoint &pos)
{
    //pos is relative to viewport of treeWidget
    QPoint global = ui.treeWidgetDB->viewport()->mapToGlobal(pos); //map pos to global, screen coordinates
    QTreeWidgetItem *selItem = ui.treeWidgetDB->itemAt(pos); //itemAt requires position relative to viewport
     //event->globalPos());
    if (selItem != NULL)
    {
        if (selItem->data(0, m_urUD) == stateDownloadAvailable)
        {
            contextMenuActions["update"]->setEnabled(true);
            contextMenuActions["update"]->setText("downlaod");
            contextMenuActions["locateOnDisk"]->setEnabled(false);
            contextMenuActions["removeDatabase"]->setEnabled(false);
            m_pContextMenu->exec(global);
        }
        else if (selItem->data(0, m_urUD) == stateUpdateAvailable)
        {
            contextMenuActions["update"]->setEnabled(true);
            contextMenuActions["update"]->setText("update");
            contextMenuActions["locateOnDisk"]->setEnabled(true);
            contextMenuActions["removeDatabase"]->setEnabled(true);
            m_pContextMenu->exec(global);
        }
        else if (selItem->data(0, m_urUD) == stateUnknown ||
                 selItem->data(0, m_urUD) == stateUpdateAvailable ||
                 selItem->data(0, m_urUD) == stateUpToDate)
        {
            contextMenuActions["update"]->setEnabled(false);
            contextMenuActions["update"]->setText("update");
            contextMenuActions["locateOnDisk"]->setEnabled(true);
            contextMenuActions["removeDatabase"]->setEnabled(true);
            m_pContextMenu->exec(global);
        }
    }

    // event am ende wieder loeschen
}