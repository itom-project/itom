#include "helpTreeDockWidget.h"

// #include <qdebug.h>
#include <qdesktopservices.h>
#include <qdiriterator.h>
#include <qfile.h>
#include <qfiledialog.h>
#include <qmessagebox.h>
#include <qpainter.h>
#include <qregexp.h>
#include <qsortfilterproxymodel.h>
#include <qstandarditemmodel.h>
#include <qstringlistmodel.h>
#include <qtconcurrentrun.h>
#include <qtextdocument.h>
#include <qtextstream.h>
#include <QThread>
#include <qtimer.h>
#include <qtreeview.h>
#include <stdio.h>

#include "../widgets/helpDockWidget.h"
#include "../models/leafFilterProxyModel.h"
#include "../AppManagement.h"

//----------------------------------------------------------------------------------------------------------------------------------
// on_start
HelpTreeDockWidget::HelpTreeDockWidget(QWidget *parent, ito::AbstractDockWidget *dock, Qt::WFlags flags)
    : QWidget(parent, flags),
	m_historyIndex(-1),
	m_pMainModel(NULL),
	m_dbPath(qApp->applicationDirPath() + "/help"),
	m_pParent(dock)
{
    ui.setupUi(this);

	connect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(propertiesChanged()));

	// Initialize Variables
	m_treeVisible = false;

	connect(&dbLoaderWatcher, SIGNAL(resultReadyAt(int)), this, SLOT(dbLoaderFinished(int)));

    m_pMainFilterModel = new LeafFilterProxyModel(this);
    m_pMainModel = new QStandardItemModel(this);
    m_pMainFilterModel->setFilterCaseSensitivity(Qt::CaseInsensitive);

	//Install Eventfilter
    ui.commandLinkButton->setVisible(false);
	//ui.commandLinkButton->installEventFilter(this);
	ui.treeView->installEventFilter(this);
	ui.textBrowser->installEventFilter(this);

	m_previewMovie = new QMovie(":/application/icons/loader32x32trans.gif", QByteArray(), this);
    ui.lblProcessMovie->setMovie(m_previewMovie);
    ui.lblProcessMovie->setVisible(false);
    ui.lblProcessText->setVisible(false);

	loadIni();
	m_forced = true;
	propertiesChanged();
	//reloadDB();

    QStringList iconAliases;
    iconAliases << "class" << "const" << "routine" << "module" << "package" << "unknown" << "link_unknown" << "link_class" << "link_const" << "link_module" << "link_package" << "link_routine";
    foreach(const QString &icon, iconAliases)
    {
        m_iconGallery[icon] = QIcon(":/helpTreeDockWidget/" + icon);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// GUI-on_close
HelpTreeDockWidget::~HelpTreeDockWidget()
{
	saveIni();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Filter the events for showing and hiding the treeview
bool HelpTreeDockWidget::eventFilter(QObject *obj, QEvent *event)
{
	// = qobject_cast<ito::AbstractDockWidget*>(parent());

	if (obj == ui.commandLinkButton && event->type() == QEvent::Enter)
	{
		showTreeview();
	}
	else if (obj == ui.treeView && event->type() == QEvent::Enter)
	{	
		if (m_pParent && !m_pParent->isFloating())
		{
			showTreeview();
		}
	}
	else if (obj == ui.textBrowser && event->type() == QEvent::Enter)
	{
		if (m_pParent && !m_pParent->isFloating())
		{
			unshowTreeview();
			return true;
		}	
	}
	return QObject::eventFilter(obj, event);
 }

//----------------------------------------------------------------------------------------------------------------------------------
// Save Gui positions to Main-ini-File
void HelpTreeDockWidget::saveIni()
{
	QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup(objectName());
	settings.setValue("percWidthVi", m_percWidthVi);
	settings.setValue("percWidthUn", m_percWidthUn);
	settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Load Gui positions to Main-ini-File
void HelpTreeDockWidget::loadIni()
{
	QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup(objectName());
	m_percWidthVi = settings.value("percWidthVi", "50").toDouble();
    m_percWidthUn = settings.value("percWidthUn", "50").toDouble();
	settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Load SQL-DatabasesList in m_ Variable when properties changed
void HelpTreeDockWidget::propertiesChanged()
{ // Load the new list of DBs with checkstates from the INI-File
	
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup(objectName());
	// Read the other Options
	m_openLinks = settings.value("OpenExtLinks", true).toBool();
	m_plaintext = settings.value("Plaintext", false).toBool();

	// if the setting of the loaded DBs has changed:
	// This setting exists only from the time when the property dialog was open till this routine is done!
	if (settings.value("reLoadDBs", false).toBool() | m_forced)
	{
		// Read the List
		m_includedDBs.clear();
		int size = settings.beginReadArray("Databases");
		for (int i = 0; i < size; ++i)
		{
			settings.setArrayIndex(i);
			QString dbName = settings.value("DB", QString()).toString();
			if (dbName.startsWith("$"))	
			{// This was checked and will be used
				dbName.remove(0, 2);
				//Add to m_pMainlist
				m_includedDBs.append(dbName);
			}
		}
        settings.endArray();
		reloadDB();
	}
	settings.remove("reLoadDBs");
	settings.endGroup();
	m_forced = false;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Build Tree - Bekommt das Model, das zuletzt erstellte Item und eine Liste mit dem Pfad
/*static*/ void HelpTreeDockWidget::createItemRek(QStandardItemModel* model, QStandardItem& parent, const QString parentPath, QStringList &items, const QMap<QString,QIcon> *iconGallery)
{
    QString firstItem;
    QString path;
    QString name;
    QStringList splitt;
    int MyR = Qt::UserRole;

    while(items.count() > 0)
    {
        firstItem = items[0];
        splitt = firstItem.split(':');

        // split firstItem into path and name (bla.bla.bla.name) where bla.bla.bla is the path
        int li = splitt[1].lastIndexOf(".");
        if(li >= 0)
        {
            path = splitt[1].left(li);
            name = splitt[1].mid(li + 1);
        }
        else
        {
            path = "";
            name = splitt[1];
        }
        if(path == parentPath) //first item is direct child of parent
        {
            items.takeFirst();
            QStandardItem *node = new QStandardItem(name);
			if (splitt[0].startsWith("link_"))
			{
				// diese Zeile könnte man auch durch Code ersetzen der das Link Icon automatisch zeichnet... das waere flexibler
				node->setIcon(iconGallery->value(splitt[0]));
			}
			else
			{ // Kein Link Normales Bild
				node->setIcon(iconGallery->value(splitt[0])); //Don't load icons here from file since operations on QPixmap are not allowed in another thread
			}
			node->setEditable(false);
            node->setData(splitt[1], MyR + 1);
            node->setToolTip(splitt[1]);
            createItemRek(model, *node, splitt[1], items, iconGallery);
            parent.appendRow(node);
        }
        else if(path.indexOf(parentPath) == 0) //parentPath is the first part of path
        {
            items.takeFirst();
            int li = path.lastIndexOf(".");
            QStandardItem *node = new QStandardItem(path.mid(li + 1));
			if (splitt[0].startsWith("link_")) // Siehe 19 Zeilen vorher
			{ //ist ein Link (vielleicht wie oben Icon dynamisch zeichnen lassen
				node->setIcon(iconGallery->value(splitt[0]));
			}
			else
			{ // Kein Link Normales Bild
				node->setIcon(iconGallery->value(splitt[0]));
			}
			node->setEditable(false);
            node->setData(path, MyR + 1);                
            createItemRek(model, *node, path, items, iconGallery);  
            parent.appendRow(node);
        }
        else
        {
            break;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Get Data from SQL File and store it in a table
/*static*/ ito::RetVal HelpTreeDockWidget::readSQL(/*QList<QSqlDatabase> &DBList,*/ const QString &filter, const QString &file, QList<QString> &items)
{
	ito::RetVal retval = ito::retOk;
	QFile f(file);
  
	if(f.exists())
	{
        QSqlDatabase database = QSqlDatabase::addDatabase("QSQLITE", file); //important to have variables database and query in local scope such that removeDatabase (outside of this scope) can securly free all resources! -> see docs about removeDatabase
	    database.setDatabaseName(file);
		bool ok = database.open();
		if(ok)
		{
			QSqlQuery query("SELECT type, prefix, prefixL, name FROM itomCTL ORDER BY prefix", database);
			query.exec();
			while (query.next())
			{
				items.append(query.value(0).toString() + QString(":") + query.value(1).toString());
			}

			//DBList.append(database);
		}
		else
		{
			retval += ito::RetVal::format(ito::retWarning, 0, tr("Database %s could not be opened").toAscii().data(), file.toAscii().data());
		}
		database.close();
	}
	else
	{
		retval += ito::RetVal::format(ito::retWarning, 0, tr("Database %s could not be found").toAscii().data(), file.toAscii().data());
	}	
	QSqlDatabase::removeDatabase(file);
	return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Reload Database and clear search-edit and start the new Thread
void HelpTreeDockWidget::reloadDB()
{
	
	//Create and Display Mainmodel
	m_pMainModel->clear();
    ui.treeView->reset();
	
	m_pMainFilterModel->setSourceModel(NULL);
	m_previewMovie->start();
    ui.lblProcessMovie->setVisible(true);
    ui.lblProcessText->setVisible(true);
	ui.treeView->setVisible(false);
	ui.splitter->setVisible(false);
    ui.lblProcessText->setText(tr("Help database is loading..."));

	// THREAD START QtConcurrent::run
	QFuture<ito::RetVal> f1 = QtConcurrent::run(loadDBinThread, m_dbPath, m_includedDBs, m_pMainModel/*, m_pDBList*/, &m_iconGallery);
	dbLoaderWatcher.setFuture(f1);
	//f1.waitForFinished();
	// THREAD END
}

//----------------------------------------------------------------------------------------------------------------------------------
void HelpTreeDockWidget::dbLoaderFinished(int /*index*/)
{
    ito::RetVal retval = dbLoaderWatcher.future().resultAt(0);
	// Ende Neuer Code
	m_pMainFilterModel->setSourceModel(m_pMainModel);

	//model has been 
	ui.treeView->setModel(m_pMainFilterModel);

	m_previewMovie->stop();
    ui.lblProcessMovie->setVisible(false);

    if (m_includedDBs.size() > 0)
    {
        ui.lblProcessText->setVisible(false);
	    ui.treeView->setVisible(true);
	    ui.splitter->setVisible(true);
    }
    else
    {
        ui.lblProcessText->setVisible(true);
	    ui.treeView->setVisible(false);
	    ui.splitter->setVisible(false);
        ui.lblProcessText->setText(tr("No help database available"));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Load the Database in different Thread
/*static*/ ito::RetVal HelpTreeDockWidget::loadDBinThread(const QString &path, const QStringList &includedDBs, QStandardItemModel *mainModel, const QMap<QString,QIcon> *iconGallery)
{
	QStringList sqlList;
	ito::RetVal retval;
    for (int i = 0; i < includedDBs.length(); i++)
    {
		sqlList.clear();
		QString temp;
		temp = path+'/'+includedDBs.at(i);
		retval = readSQL(/*DBList,*/ "", temp, sqlList);

		QCoreApplication::processEvents();

		if (!retval.containsWarningOrError())
		{
			createItemRek(mainModel, *(mainModel->invisibleRootItem()), "", sqlList, iconGallery);
		}
		else
		{/* The Database named: m_pIncludedDBs[i] is not available anymore!!! show Error*/}
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Highlight (parse) the Helptext to make it nice and readable for non docutils Docstrings
// ERROR decides whether it´s already formatted by docutils (Error = 0) or it must be parsed by this function (Error != 0)
QTextDocument* HelpTreeDockWidget::highlightContent(const QString &helpText, const QString &prefix, const QString &name, const QString &param, const QString &shortDesc, const QString &error)
{
    QString errorS = error.left(error.indexOf(" ",0));
    int errorCode = errorS.toInt();
	QStringList errorList;

    /*********************************/
    // Allgemeine HTML sachen anfügen /
    /*********************************/ 
    QString rawContent = helpText;
    QString html =	"<html><head>"
					"<link rel='stylesheet' type='text/css' href='help_style.css'>"
					"</head><body>"
					"</body></html>";

    if (errorCode == 1)
    {
        ui.label->setText(tr("Parser: martin1-Parser"));
    }
    else if (errorCode == 0)
	{
        ui.label->setText(tr("Parser: docutils"));
		
		// REGEX muss noch an den html code angepasst werden . ... und dann noch der betreffende teil aus dem helptext ausgeschnitten werden!
		/*QRegExp docError("System Message: ERROR/\\d \\(.+\\).*\\.");
		errorList = docError.capturedTexts();
        QStringListModel *listM = new QStringListModel();
        listM->setStringList(errorList);*/
	}
    else if (errorCode == -1)
    {
        ui.label->setText(tr("Parser: No Help available"));
    }

	//CSS File als QString einlesen
	// -------------------------------------
	// Crate a QTextDocument with the defined HTML, CSS and the images
	QTextDocument *doc = new QTextDocument(this);
	QFile file(":/helpTreeDockWidget/help_style");
	file.open(QIODevice::ReadOnly);
	QByteArray cssArray = file.readAll();
	QString cssFile = QString(cssArray);
	doc->addResource(QTextDocument::StyleSheetResource, QUrl("help_style.css"), cssFile);

	if (errorCode != 0)
    {
        // Zeilenumbrüche ersetzen
        // -------------------------------------
        rawContent.replace('\n',"<br/>");
        // Shortdescription einfügen
        // -------------------------------------
        //if (ShortDesc != "-")
			
            //rawContent.insert(0,ShortDesc+""); // ShortDescription mit ID versehen: id=\"sDesc\" um getrennt zu highlighten
        // Parameter formatieren
        // -------------------------------------
	}
	else if (errorCode == 0)
	{
		rawContent.replace("h1", "h2");
	}

    // Überschrift (Funktionsname) einfuegen
    // -------------------------------------
    rawContent.insert(0,"<h1 id=\"FunctionName\">"+name+param+"</h1>"+"");

    // Prefix als Navigations-Links einfuegen
    // -------------------------------------
    QStringList splittedLink = prefix.split(".");
    rawContent.insert(0,">>"+splittedLink[splittedLink.length()-1]);
    for (int i = splittedLink.length()-2; i > -1; i--)
    {
        QString linkPath;
        for (int j = 0; j<=i; j++)
            linkPath.append(splittedLink.mid(0,i+1)[j]+".");
		if (linkPath.right(1) == ".")
			linkPath = linkPath.left(linkPath.length()-1);
        rawContent.insert(0,">> <a id=\"HiLink\" href=\"itom://"+linkPath+"\">"+splittedLink[i]+"</a>");
    }

    if (errorCode != 0)
    {
        // Variables Declaration
		//--------------------------------------
        QStringList sections; 
        sections  <<  "<h2 id=\"Sections\">"  <<  "</h2>"  <<  "Parameters"  <<  "Returns"  <<  "Attributes" <<  "Examples"  <<  "Notes"  <<  "Args"  <<   "Raises"  <<  "See Also"  <<  "References"; // <<   <<   <<  ... Hier alle regex für Keywords der 1. Überschrift eintragen
        int pos = 0;
        
		// Sections Highlighten
        // -------------------------------------
		QRegExp reg("([a-zA-Z0-9 -.,:;]+)<br/>\\W{,7}-{3,}\\W{,7}<br/>|([a-zA-Z0-9 -.,:;]+)<br/>\\W{,7}={3,}\\W{,7}<br/>|([a-zA-Z0-9 -.,:;]+)<br/>\\W{,7}~{3,}\\W{,7}<br/>");
        reg.setMinimal(true);
        while ((pos = reg.indexIn(rawContent, pos)) != -1)
        {
            if (pos == -1) {
                break;}
            QString content = rawContent.mid(pos, reg.matchedLength());
        
            QRegExp keywordReg("<br/>");
            keywordReg.setMinimal(false);
            int pos2 = keywordReg.indexIn(content, 0);
            content = content.left(pos2);

            rawContent.remove(pos, reg.matchedLength());
            rawContent.insert(pos, sections[0] + content + sections[1]);
            pos += QString(sections[0] + sections[1]).length()+content.length();
        }

        // Enumerations bei folgenden Section setzen 
        // -------------------------------------
        // Section[2] = Parameters, Section[3] = Returns,... Attributes
        for (int i = 2; i<5; i++)
        {
            int headPos = 0;
            int pos = 0;
            int bottomPos = 1;
            QRegExp head(QString("<h2 id=\"Sections\">.*"+sections[i]+".*</h2>"));
            QRegExp bottom("<h2 id=\"Sections\">");
            head.setMinimal(true);
            // find multiple occurences of one heading
            while (((headPos = head.indexIn(rawContent, headPos)) != -1) && (pos < bottomPos))
            {
                if (headPos == -1) {
                    break;}
                // beginning of the bullets
                rawContent.insert(headPos + head.matchedLength(),"<ul>");
                pos = headPos;            
                bottomPos = bottom.indexIn(rawContent, headPos+head.matchedLength());
                if (bottomPos == -1)
                {
                    rawContent.append("</ul>");
                    bottomPos = rawContent.length()-1;
                }
                else
                    rawContent.insert(bottomPos,"</ul>");

                // search for: "x : int" for example
                QRegExp line("[a-zA-Z _,.-]*: [^<br/>]*");            
            
                while (((pos = line.indexIn(rawContent,pos)) != -1) && (pos < bottomPos))
                {
                    if (pos == -1) {
                        break;}
                    rawContent.insert(pos,"<li>");
                    pos += line.matchedLength()+4;
                }
                // end of the Bullets
            
                headPos += head.matchedLength() + 5;
            }
        }
    }

    // Alles zusammenführen
    // -------------------------------------
	html.insert(86, rawContent);
	doc->setHtml(html);
    return doc;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Display the Help-Text
ito::RetVal HelpTreeDockWidget::displayHelp(const QString &path, const int newpage)
{ 
	ito::RetVal retval = ito::retOk;

	ui.textBrowser->clear();
	bool ok = false;
    bool found = false;

	// Das ist ein kleiner workaround mit dem if 5 Zeilen später. Man könnt euahc direkt über die includeddbs list iterieren
	// dann wäre folgende Zeile hinfällig
	QDirIterator it(m_dbPath, QStringList("*.db"), QDir::Files | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);

    while(it.hasNext() && !found)
    {
		QString temp = it.next();
		if (m_includedDBs.contains(temp.right(temp.length()-m_dbPath.length()-1)))
		{
			QFile file(temp);
		
			if (file.exists())
			{
				{ //important to have variables database and query in local scope such that removeDatabase (outside of this scope) can securly free all resources! -> see docs about removeDatabase
					// display the help: Run through all the files in the directory
					QSqlDatabase database = QSqlDatabase::addDatabase("QSQLITE",temp);
					database.setDatabaseName(temp);
					ok = database.open();
					if (ok)
					{
						QSqlQuery query("SELECT type, prefix, prefixL, name, param, sdesc, doc, htmlERROR  FROM itomCTL WHERE prefixL IS '"+path.toUtf8().toLower()+"'", database);
						query.exec();
						found = query.next();
						if (found)
						{
							QByteArray docCompressed = query.value(6).toByteArray();
							QString doc;
							if (docCompressed.size() > 0)
							{
								doc = qUncompress(docCompressed);
							}

							if (!m_plaintext)
								ui.textBrowser->setDocument(highlightContent(doc, query.value(1).toString(), query.value(3).toString(), query.value(4).toString(), query.value(5).toString(), query.value(7).toString()));
							else
							{
								QString output = QString(highlightContent(doc, query.value(1).toString(), query.value(3).toString(), query.value(4).toString(), query.value(5).toString(), query.value(7).toString())->toHtml());
								output.replace("<br/>","<br/>\n");
								ui.textBrowser->document()->setPlainText(output);             
							}
							if (newpage == 1)
							{
								m_historyIndex++;
								m_history.insert(m_historyIndex, path.toUtf8());
								for (int i = m_history.length(); i > m_historyIndex; i--)
								{
									m_history.removeAt(i);
								}
							}
						}
						database.close();
					}
				}
				QSqlDatabase::removeDatabase(temp);
			}
		}
    }

	return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
// finds a Modelindex belonging to an Itemname
QModelIndex HelpTreeDockWidget::findIndexByName(const QString &modelName)
{
	QStringList path = modelName.split('.');
	QStandardItem *current = m_pMainModel->invisibleRootItem();
    QStandardItem *temp;
    int counts;
    QString tempString;
    QString firstPath;
    bool found;

	while (path.length() > 0)
	{
        firstPath = path.takeFirst();
		counts = current->rowCount();
        found = false;

		for (int j = 0; j < counts; ++j)
		{
            temp = current->child(j,0);
            tempString = temp->data().toString();

            if (tempString.endsWith(firstPath) && tempString.split(".").last() == firstPath) //fast implementation, first compare will mostly fail, therefore the split is only executed few times
            {
                current = temp;
                found = true;
                break;
            }

            if (!found)
            {
                return QModelIndex(); //nothing found
            }
		}
	}

	return current->index();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Filter the mainmodel
void HelpTreeDockWidget::liveFilter(const QString &filterText)
{
	showTreeview();
    m_pMainFilterModel->setFilterRegExp(filterText);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Returns a list containing the protocol[0] and the real link[1]
// http://thisisthelink
// prot|||....link.....        
QStringList HelpTreeDockWidget::separateLink(const QUrl &link)
{
	//qDebug()  <<  "THE REALLINK: "  <<  link.toEncoded()  <<  "The LinkCaption: "  <<  link.userInfo();
	QStringList result;

	QRegExp maillink(QString("^(mailto):(.*)"));
	QRegExp itomlink(QString("^([A-Za-z0-9]+)://(.*)"));
	
	if (maillink.indexIn(link.toEncoded(),0) != -1)
	{
		result.append(maillink.cap(1));
		result.append(link.toEncoded());
	}
	else if (itomlink.indexIn(link.toEncoded(),0) != -1)
	{
		result.append(itomlink.cap(1));
		result.append(itomlink.cap(2));
	}
	else
		result.append("-1");
	return result;
}


/*************************************************************/
/*****************GUI-Bezogene-Funktionen*********************/
/*************************************************************/

//----------------------------------------------------------------------------------------------------------------------------------
// Expand all TreeNodes
void HelpTreeDockWidget::expandTree()
{
	ui.treeView->expandAll();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Collapse all TreeNodes
void HelpTreeDockWidget::collapseTree()
{
	ui.treeView->collapseAll();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Link inside Textbrowser is clicked
void HelpTreeDockWidget::on_textBrowser_anchorClicked(const QUrl & link)
{
    QStringList parts = separateLink(link.toString());
    if (parts[0] == "itom")
    {
		//qDebug()  <<  "OnTreeClickedPfad: "  <<  parts[1];
		displayHelp(parts[1], 1);

        QModelIndex filteredIndex = m_pMainFilterModel->mapFromSource(findIndexByName(parts[1]));
		ui.treeView->setCurrentIndex(filteredIndex);
    }
    else if (parts[0] == "http")
    {
		QDesktopServices::openUrl(link);
    }
	else if (parts[0] == "mailto")
    {
		QDesktopServices::openUrl(parts[1]);
    }
	else if (parts[0] == "-1")
    {
		ui.label->setText(tr("invalid Link"));
    }
	else
	{
		ui.label->setText(tr("unknown protocol"));
		QMessageBox msgBox;
		msgBox.setText(tr("The protocol of the link is unknown. "));
		msgBox.setInformativeText(tr("Do you want to try with the external browser?"));
		msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
		msgBox.setDefaultButton(QMessageBox::Yes);
		int ret = msgBox.exec();
		switch (ret) 
		{
			case QMessageBox::Yes:
				QDesktopServices::openUrl(link);
			case QMessageBox::No:
				break;
		}
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
// Saves the position of the splitter depending on the use of the tree or the textbox
void HelpTreeDockWidget::on_splitter_splitterMoved (int pos, int index)
{
	double width = ui.splitter->width();
	if (m_treeVisible == true)
		m_percWidthVi = pos/width*100;
	else
		m_percWidthUn = pos/width*100;
	if (m_percWidthVi < m_percWidthUn)
		m_percWidthVi = m_percWidthUn+10;
	if (m_percWidthVi == 0)
		m_percWidthVi = 30;
	// Verhaltnis testweise anzeigen lassen
	//ui.label->setText(QString("vi %1 un %2").arg(percWidthVi).arg(percWidthUn));
}

//----------------------------------------------------------------------------------------------------------------------------------
// Show the Help in the right Memo
void HelpTreeDockWidget::on_treeView_clicked(QModelIndex i)
{
    int MyR = Qt::UserRole;
    //qDebug()  <<  "OnTreeClickedPfad: "  <<  QString(i.data(MyR+1).toString());
	displayHelp(QString(i.data(MyR+1).toString()), 1);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Back-Button
void HelpTreeDockWidget::navigateBackwards()
{
    if (m_historyIndex > 0)
    {
        m_historyIndex--;
        displayHelp(m_history.at(m_historyIndex), 0);

		// Highlight the entry in the tree
        QModelIndex filteredIndex = m_pMainFilterModel->mapFromSource(findIndexByName(m_history.at(m_historyIndex)));
		ui.treeView->setCurrentIndex(filteredIndex);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Forward-Button
void HelpTreeDockWidget::navigateForwards()
{
    if (m_historyIndex < m_history.length()-1)
    {
        m_historyIndex++;
        displayHelp(m_history.at(m_historyIndex), 0);

		// Highlight the entry in the tree
		QModelIndex filteredIndex = m_pMainFilterModel->mapFromSource(findIndexByName(m_history.at(m_historyIndex)));
		ui.treeView->setCurrentIndex(filteredIndex);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Tree einblenden
void HelpTreeDockWidget::showTreeview()
{
	m_treeVisible = true;
	QList<int> intList;
	intList  <<  ui.splitter->width()*m_percWidthVi/100  <<  ui.splitter->width()*(100-m_percWidthVi)/100;
	ui.splitter->setSizes(intList);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Tree ausblenden
void HelpTreeDockWidget::unshowTreeview()
{
	m_treeVisible = false;
	QList<int> intList;
	intList  <<  ui.splitter->width()*m_percWidthUn/100  <<  ui.splitter->width()*(100-m_percWidthUn)/100;
	ui.splitter->setSizes(intList);
}
