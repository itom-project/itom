#include "helpTreeDockWidget.h"

#include <qfiledialog.h>
#include <qfile.h>
#include <qmessagebox.h>
#include <qtreeview.h>
#include <qtextdocument.h>
#include <qstandarditemmodel.h>
#include <qtextstream.h>
#include <qregexp.h>
#include <stdio.h>
#include <qdiriterator.h>
#include <qfile.h>
#include <qtimer.h>
#include <qsortfilterproxymodel.h>
#include <qdesktopservices.h>
#include <qstringlistmodel.h>
#include "../widgets/helpDockWidget.h"
#include "../models/leafFilterProxyModel.h"
#include <qpainter.h>


// Debug includes
#include <qdebug.h>




// GUI-on_start
HelpTreeDockWidget::HelpTreeDockWidget(QWidget *parent, Qt::WFlags flags)
    : QWidget(parent, flags),
	m_pHistoryIndex(-1),
	m_pMainModel(NULL),
    m_pMainFilterModel(NULL),
    m_pHistory(NULL),
	m_dbPath(qApp->applicationDirPath()+"\\help")
{
    ui.setupUi(this);
	//connect(ui.commandLinkButton, SIGNAL(moveOver()), this, SLOT(showTreeview()));

	// Initialize Variables
	TreeCloseTimer = new QTimer(this);
	connect(TreeCloseTimer, SIGNAL(timeout()), this, SLOT(unshowTreeview()));
	TreeCloseTimer->setSingleShot(true);
	TreeCloseTimer->setInterval(5000);

    m_pMainFilterModel = new LeafFilterProxyModel(this);
    m_pMainModel = new QStandardItemModel(this);
    m_pHistory = new QStringList();
    m_pMainFilterModel->setFilterCaseSensitivity(Qt::CaseInsensitive);

	reloadDB();

	//Install Eventfilter
	ui.commandLinkButton->installEventFilter(this);
	ui.treeView->installEventFilter(this);
	ui.textBrowser->installEventFilter(this);
}


// Event when the cursor leaves the widget
void HelpTreeDockWidget::leaveEvent( QEvent * event )
{
	TreeCloseTimer->start();
} 

// Event when the cursor enters/reenters the widget
void HelpTreeDockWidget::enterEvent( QEvent * event )
{
	TreeCloseTimer->stop();
}


// Filter the events for showing and hiding the treeview
bool HelpTreeDockWidget::eventFilter(QObject *obj, QEvent *event)
{
	if (obj == ui.commandLinkButton && event->type() == QEvent::Enter)
	{
		showTreeview();
		return true;
	}
	else if (obj == ui.textBrowser && event->type() == QEvent::Enter)
	{
		unshowTreeview();
	}
	return QObject::eventFilter(obj, event);
 }


// GUI-on_close
HelpTreeDockWidget::~HelpTreeDockWidget()
{
	while(m_DBList.length() > 0)
	{
		m_DBList.first().close();
		m_DBList.first().removeDatabase(m_DBList.first().databaseName());
		m_DBList.removeFirst();
	}
}


// Build Tree - Bekommt das Model, das zuletzt erstellte Item und eine Liste mit dem Pfad
void HelpTreeDockWidget::CreateItemRek(QStandardItemModel& model, QStandardItem& parent, const QString parentPath, QList<QString> &items)
{
    QString firstItem;
    QString path;
    QString name;
    QStringList splitt;
    int MyR = Qt::UserRole;

    while( items.count() > 0)
    {
        firstItem = items[0];
        splitt = firstItem.split(':');

        // split firstItem into path and name (bla.bla.bla.name) where bla.bla.bla is the path
        int li = splitt[1].lastIndexOf(".");
        if(li >= 0)
        {
            path = splitt[1].left(li);
            name = splitt[1].mid(li+1);
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
			if (splitt[0][0] != 'l')
			{
				node->setIcon(QIcon(":/helpTreeDockWidget/"+splitt[0]));
			}
			else
			{
				// Wenn man in der nächsten Zeile vor die letzte Klammer eine -1 setzt, kann man das L am ANfang für Link wegschneiden
				//name = splitt[0].right(splitt[0].length());
				node->setIcon(QIcon(":/helpTreeDockWidget/"+splitt[0]));
			}
			node->setEditable(false);
            node->setData(splitt[1],MyR+1);
            node->setToolTip(splitt[1]);
            CreateItemRek(model, *node, splitt[1], items);
            parent.appendRow(node);
        }
        else if(path.indexOf( parentPath ) == 0) //parentPath is the first part of path
        {
            items.takeFirst();
            int li = path.lastIndexOf(".");
            QStandardItem *node = new QStandardItem(path.mid(li+1));
			if (splitt[0][0] != 'l')
			{
                node->setIcon(QIcon(":/helpTreeDockWidget/"+splitt[0]+".png"));
			}
			else
			{
				// Wenn man in der nächsten Zeile vor die letzte Klammer eine -1 setzt, kann man das L am ANfang für Link wegschneiden
				//name = splitt[0].right(splitt[0].length());
				node->setIcon(QIcon(":/helpTreeDockWidget/"+splitt[0]));
			}
			node->setEditable(false);
            node->setData(path,MyR+1);                
            CreateItemRek(model, *node, path, items);  
            parent.appendRow(node);
        }
        else
        {
            break;
        }
    }
}


// Get Data from SQL File and store it in a table
ito::RetVal HelpTreeDockWidget::readSQL(const QString &filter, const QString &file, QList<QString> &items)
{
	ito::RetVal retval = ito::retOk;
	QSqlDatabase database = QSqlDatabase::addDatabase("QSQLITE",file);
	database.setDatabaseName(file);

	//QList<QString> items;
	QFile f( file );
  
	if( f.exists() )
	{
		bool ok = database.open();
		if(ok)
		{
			QSqlQuery query("SELECT type, prefix, prefixL, name FROM itomCTL ORDER BY prefix", database);
			query.exec();
			while (query.next())
			{
				items.append(query.value(0).toString() + QString(":") + query.value(1).toString());
			}

			m_DBList.append(database);
		}
		else
		{
			retval += ito::RetVal::format(ito::retWarning,0,"Database %s could not be opened", file.toAscii().data());
		}
		database.close();
	}
	else
	{
		retval += ito::RetVal::format(ito::retWarning,0,"Database %s could not be found", file.toAscii().data());
	}	
	QSqlDatabase::removeDatabase(file);
	return retval;
}


// Highlight (parse) the Helptext to make it nice and readable for non docutils Docstrings
// ERROR decides whether it´s already formatted by docutils (Error = 0) or it must be parsed by this function (Error != 0)
QTextDocument* HelpTreeDockWidget::HighlightContent(const QString &Helptext, const QString &Prefix , const QString &Name , const QString &Param , const QString &ShortDesc, const QString &Error)
{
    QString ErrorS = Error.left(Error.indexOf(" ",0));
    int Errorcode = ErrorS.toInt();
	QStringList ErrorList;

	

    /*********************************/
    // Allgemeine HTML sachen anfügen /
    /*********************************/ 
    QString rawContent = Helptext;
    QString html =	"<html><head>"
					"<link rel='stylesheet' type='text/css' href='help_style.css'>"
					"</head><body>"
					"</body></html>";

    if (Errorcode == 1)
        ui.label->setText("Parser: martin1-Parser");  
    else if (Errorcode == 0)
	{
        ui.label->setText("Parser: docutils");
		
		// REGEX muss noch an den html code angepasst werden . ... und dann noch der betreffende teil aus dem helptext ausgeschnitten werden!
		QRegExp docError("System Message: ERROR/\\d \\(.+\\).*\\.");
		ErrorList = docError.capturedTexts();
        QStringListModel *listM = new QStringListModel();
        listM->setStringList(ErrorList);
		//Helptext.replace(docError, "");
		// ...
	}
    else if (Errorcode == -1)
        ui.label->setText("Parser: No Help available");


	//CSS File als QString einlesen
	// -------------------------------------
	// Crate a QTextDocument with the defined HTML, CSS and the images
	QTextDocument *doc = new QTextDocument;
	QFile file(":/helpTreeDockWidget/help_style");
	file.open(QIODevice::ReadOnly);
	QByteArray cssArray = file.readAll();
	QString cssFile = QString(cssArray);
	doc->addResource( QTextDocument::StyleSheetResource, QUrl( "help_style.css" ), cssFile );

	if (Errorcode != 0)
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
	else if (Errorcode == 0)
	{
		rawContent.replace("h1", "h2");
	}


    // Überschrift (Funktionsname) einfuegen
    // -------------------------------------
    rawContent.insert(0,"<h1 id=\"FunctionName\">"+Name+Param+"</h1>"+"");

    // Prefix als Navigations-Links einfuegen
    // -------------------------------------
    QStringList splittedLink = Prefix.split(".");
    rawContent.insert(0,">>"+splittedLink[splittedLink.length()-1]);
    for (int i = splittedLink.length()-2; i > -1; i--)
    {
        QString linkpath;
        for (int j = 0; j<=i; j++)
            linkpath.append(splittedLink.mid(0,i+1)[j]+".");
		if (linkpath.right(1) == ".")
			linkpath = linkpath.left(linkpath.length()-1);
        rawContent.insert(0,">> <a id=\"HiLink\" href=\"itom://"+linkpath+"\">"+splittedLink[i]+"</a>");
    }

    if (Errorcode != 0)
    {
        // Variables Declaration
		//--------------------------------------
        QStringList Sections; 
        Sections << "<h2 id=\"Sections\">" << "</h2>" << "Parameters" << "Returns" << "Attributes"<< "Examples" << "Notes" << "Args" <<  "Raises" << "See Also" << "References"; //<< << << ... Hier alle regex für Keywords der 1. Überschrift eintragen
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
            rawContent.insert(pos, Sections[0] + content + Sections[1]);
            pos += QString(Sections[0] + Sections[1]).length()+content.length();
        }

        // Enumerations bei folgenden Section setzen 
        // -------------------------------------
        // Section[2] = Parameters, Section[3] = Returns,... Attributes
        for (int i = 2; i<5; i++)
        {
            int headpos = 0;
            int pos = 0;
            int bottompos = 1;
            QRegExp head(QString("<h2 id=\"Sections\">.*"+Sections[i]+".*</h2>"));
            QRegExp bottom("<h2 id=\"Sections\">");
            head.setMinimal(true);
            // find multiple occurences of one heading
            while ( ((headpos = head.indexIn(rawContent, headpos)) != -1) && (pos < bottompos) )
            {
                if (headpos == -1) {
                    break;}
                // beginning of the bullets
                rawContent.insert(headpos + head.matchedLength(),"<ul>");
                pos = headpos;            
                bottompos = bottom.indexIn(rawContent, headpos+head.matchedLength());
                if (bottompos == -1)
                {
                    rawContent.append("</ul>");
                    bottompos = rawContent.length()-1;
                }
                else
                    rawContent.insert(bottompos,"</ul>");

                // search for: "x : int" for example
                QRegExp line("[a-zA-Z _,.-]*: [^<br/>]*");            
            
                while ( ((pos = line.indexIn(rawContent,pos)) != -1) && (pos < bottompos) )
                {
                    if (pos == -1) {
                        break;}
                    rawContent.insert(pos,"<li>");
                    pos += line.matchedLength()+4;
                }
                // end of the Bullets
            
                headpos += head.matchedLength() + 5;
            }
        }
    }

    // Alles zusammenführen
    // -------------------------------------
	html.insert(86, rawContent);
	doc->setHtml( html );
    return doc;
}


// Display the Help-Text
ito::RetVal HelpTreeDockWidget::DisplayHelp(const QString &path, const int newpage)
{ 
	ito::RetVal retval = ito::retOk;

	ui.textBrowser->clear();
	bool ok = false;

	QDirIterator it(m_dbPath, QStringList("*.db"), QDir::Files | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);

    while(it.hasNext())
    {
		QString temp = it.next();
		QSqlDatabase database = QSqlDatabase::addDatabase("QSQLITE",temp);
		if (!retval.containsWarningOrError())
		{// display the help: Run through all the files in the directory
			database.setDatabaseName(temp);
			ok = database.open();
			QSqlQuery query("SELECT type, prefix, prefixL, name, param, sdesc, doc, htmlERROR  FROM itomCTL WHERE prefixL IS '"+path.toUtf8().toLower()+"'", database);
			query.exec();
			bool found = query.next();
			if (ok && found)
			{
				QByteArray docCompressed = query.value(6).toByteArray();
				QString doc;
				if (docCompressed.size() > 0)
				{
					doc = qUncompress(docCompressed);
				}

				if (ui.checkBox->checkState() == false)
					ui.textBrowser->setDocument(HighlightContent(doc, query.value(1).toString(), query.value(3).toString(), query.value(4).toString(), query.value(5).toString(), query.value(7).toString()));
				else
				{
					QString output = QString(HighlightContent(doc, query.value(1).toString(), query.value(3).toString(), query.value(4).toString(), query.value(5).toString(), query.value(7).toString())->toHtml());
					output.replace("<br/>","<br/>\n");
					ui.textBrowser->document()->setPlainText(output);             
				}
				if (newpage == 1)
				{
					m_pHistoryIndex++;
					m_pHistory->insert(m_pHistoryIndex, path.toUtf8());
					for (int i = m_pHistory->length(); i > m_pHistoryIndex; i--)
					{
						m_pHistory->removeAt(i);
					}
				}
				QSqlDatabase::removeDatabase(temp);
				break;
			}
			QSqlDatabase::removeDatabase(temp);
		}
    }

	return retval;
}


// finds a Modelindex belonging to an Itemname
QModelIndex HelpTreeDockWidget::FindIndexByName(const QString Modelname)
{
	QStringList path = Modelname.split('.');
	QStandardItem *current = m_pMainModel->invisibleRootItem();
	while (path.length() > 0)
	{
		int z = current->rowCount();
		for (int j = 0; j < current->rowCount(); j++)
		{
			QString Test = current->child(j,0)->data().toString().split(".").last();
			if ((Test) == path[0])
			{
				current = current->child(j,0);
				break;
			}
		}
		path.takeFirst();
	}
	QString Test2 = current->data().toString().split(".").last();
	return current->index();
}

// Filter the mainmodel
void HelpTreeDockWidget::liveFilter(const QString &filtertext)
{
	showTreeview();
    m_pMainFilterModel->setFilterRegExp(filtertext);
}


// Returns a list containing the protocol[0] and the real link[1]
// http://thisisthelink
// prot|||....link.....        
QStringList HelpTreeDockWidget::SeparateLink(const QUrl &link)
{
	qDebug() << "THE REALLINK: " << link.toEncoded() << "The LinkCaption: " << link.userInfo();
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

// Expand all TreeNodes
void HelpTreeDockWidget::expandTree()
{
	ui.treeView->expandAll();
}

// Collapse all TreeNodes
void HelpTreeDockWidget::collapseTree()
{
	ui.treeView->collapseAll();
}

// Link inside Textbrowser is clicked
void HelpTreeDockWidget::on_textBrowser_anchorClicked(const QUrl & link)
{
    QStringList parts = SeparateLink(link.toString());
    if (parts[0] == "itom")
    {
		qDebug() << "OnTreeClickedPfad: " << parts[1];
		DisplayHelp( parts[1], 1);
		ui.treeView->setCurrentIndex(FindIndexByName(parts[1]));
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
		ui.label->setText("invalid Link");
	else
	{
		ui.label->setText("unknown protocol");
		QMessageBox msgBox;
		msgBox.setText("The protocol of the link is unknown. ");
		msgBox.setInformativeText("Do you want to try with the external browser?");
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

// Show the Help in the right Memo
void HelpTreeDockWidget::on_treeView_clicked(QModelIndex i)
{
    int MyR = Qt::UserRole;
    qDebug() << "OnTreeClickedPfad: " << QString(i.data(MyR+1).toString());
	DisplayHelp(QString(i.data(MyR+1).toString()), 1);

}

// Back-Button
void HelpTreeDockWidget::navigateBackwards()
{
    if (m_pHistoryIndex > 0)
    {
        m_pHistoryIndex--;
        DisplayHelp(m_pHistory->at(m_pHistoryIndex), 0);
		// Highlight the entry in the tree
		ui.treeView->setCurrentIndex(FindIndexByName(m_pHistory->at(m_pHistoryIndex)));
    }
}

// Forward-Button
void HelpTreeDockWidget::navigateForwards()
{
    if (m_pHistoryIndex < m_pHistory->length()-1)
    {
        m_pHistoryIndex++;
        DisplayHelp(m_pHistory->at(m_pHistoryIndex), 0);
		// Highlight the entry in the tree
		ui.treeView->setCurrentIndex(FindIndexByName(m_pHistory->at(m_pHistoryIndex)));
    }
}

// Reload Database and clear search-edit
void HelpTreeDockWidget::reloadDB()
{
	ito::RetVal retval;
	QList<QString> sqlList;

	//Create and Display Mainmodel
	m_pMainModel->clear();
    ui.treeView->reset();
    //ui.textBrowser->setLineWrapMode(QTextEdit::NoWrap);
	QDirIterator it(m_dbPath, QStringList("*.db"), QDir::Files | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
    while(it.hasNext())
    {
		sqlList.clear();
		QString temp = it.next();
        retval += readSQL("", temp, sqlList);

		if (!retval.containsWarningOrError())
		{
			CreateItemRek(*m_pMainModel, *m_pMainModel->invisibleRootItem(), "", sqlList);
		}
    }
	m_pMainFilterModel->setSourceModel(m_pMainModel);
    ui.treeView->setModel(m_pMainFilterModel);
}

// Tree einblenden
void HelpTreeDockWidget::showTreeview()
{
	ui.horizontalLayout_3->setStretch(1,60);
	ui.horizontalLayout_3->setStretch(2,37);
}

// Tree ausblenden
void HelpTreeDockWidget::unshowTreeview()
{
	ui.horizontalLayout_3->setStretch(1,1);
	ui.horizontalLayout_3->setStretch(2,1000);
}
