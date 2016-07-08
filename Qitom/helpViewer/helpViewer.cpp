/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#include "helpViewer.h"

#ifdef ITOM_USEHELPVIEWER

#include <qurl.h>
#include <qwebengineview.h>
#include <qwebenginepage.h>
#include <qwebengineprofile.h>
#include <qwebengineurlschemehandler.h>
#include <qhelpengine.h>
#include <qdockwidget.h>
#include <qhelpcontentwidget.h>
#include <qhelpindexwidget.h>
#include <qhelpsearchengine.h>
#include <qhelpsearchquerywidget.h>
#include <qhelpsearchresultwidget.h>
#include <qdebug.h>
#include <qtimer.h>
#include "qtHelpUrlSchemeHandler.h"
#include <qlayout.h>
#include <qtoolbar.h>
#include <qmenu.h>
#include <qmenubar.h>
#include <QLineEdit.h>
#include <qlabel.h>
#include <qregexp.h>
#include <qlist.h>
#include <qapplication.h>
#include <qdebug.h>
#include <qicon.h>
#include <ui/widgetFindWord.h>


namespace ito {

//----------------------------------------------------------------------------------------
HelpViewer::HelpViewer(QWidget *parent /*= NULL*/) :
    QMainWindow(parent),
    m_pView(NULL),
	m_pFindWord(NULL),
    m_pHelpEngine(NULL),
	m_pSchemeHandler(NULL)
{
    m_pView = new QWebEngineView(this);
    //m_pView->load(QUrl("http://itom.bitbucket.org"));
	m_pFindWord = new WidgetFindWord(this);
	m_pFindWord->setVisible(false);
	m_pFindWord->setFindBarEnabled(true, true);
	
	connect(m_pFindWord, SIGNAL(findNext(QString, bool, bool, bool, bool, bool, bool)), this, SLOT(findNextWord(QString, bool, bool, bool, bool, bool, bool)));
	connect(m_pFindWord, SIGNAL(hideSearchBar()), this, SLOT(hideFindWordBar()));
	
	QVBoxLayout *layoutCentral = new QVBoxLayout(this);
	layoutCentral->addWidget(m_pView);
	layoutCentral->addWidget(m_pFindWord);
	QWidget *widgetCentral = new QWidget(this);
	widgetCentral->setLayout(layoutCentral);
	setCentralWidget(widgetCentral);

	// QWebEnginePage
	QWebEnginePage *page = m_pView->page();
	QWebEngineProfile *profile = page->profile();

	m_pDefaultZoomFactor = m_pView->zoomFactor();
	m_pZoomFactor = m_pDefaultZoomFactor;

	m_pHelpEngine = new QHelpEngine("", this);
	m_pSchemeHandler = new QtHelpUrlSchemeHandler(m_pHelpEngine, this);
	profile->installUrlSchemeHandler("qthelp", m_pSchemeHandler);
    
	//dockWidgetContent
    QHelpContentWidget *hcw = m_pHelpEngine->contentWidget();
    QDockWidget *dockWidgetContent = new QDockWidget(tr("content"), this);
	dockWidgetContent->setWidget(hcw);
	addDockWidget(Qt::LeftDockWidgetArea, dockWidgetContent);
	connect(hcw, SIGNAL(linkActivated(QUrl)), this, SLOT(linkActivated(QUrl)));
	connect(m_pView, SIGNAL(urlChanged(QUrl)), this, SLOT(urlChanged(QUrl)));	
	connect(m_pHelpEngine, SIGNAL(setupFinished()), this, SLOT(setupFinished()));
	QHelpContentModel *hcm = m_pHelpEngine->contentModel();
	connect(hcm, SIGNAL(contentsCreated()), this, SLOT(expandContent()));
	
	//dockWidgetIndex
	QVBoxLayout *layoutIndex = new QVBoxLayout(this);  
	QLineEdit *indexEdit = new QLineEdit(this); 
	connect(indexEdit, SIGNAL(textChanged(QString)), this, SLOT(textChanged(QString)));
	connect(indexEdit, SIGNAL(returnPressed()), this, SLOT(returnPressed()));
	QLabel *indexText = new QLabel(tr("Search for:"), this);
	QHelpIndexWidget *hiw = m_pHelpEngine->indexWidget();
	connect(hiw, SIGNAL(linkActivated(QUrl, QString)), this, SLOT(linkActivated(QUrl, QString)));
	layoutIndex->addWidget(indexText);
	layoutIndex->addWidget(indexEdit);
	layoutIndex->addWidget(hiw);
	QDockWidget *dockWidgetIndex = new QDockWidget(tr("index"), this);
	QWidget *indexContent = new QWidget(this);
	indexContent->setLayout(layoutIndex);
	dockWidgetIndex->setWidget(indexContent);
	addDockWidget(Qt::LeftDockWidgetArea, dockWidgetIndex);
	
	//dockWidgetSearch
	QVBoxLayout *layoutSearch = new QVBoxLayout(this);
	QHelpSearchEngine *searchEngine = m_pHelpEngine->searchEngine(); 
	QHelpSearchResultWidget *resultWidget = searchEngine->resultWidget();
	QHelpSearchQueryWidget *queryWidget = searchEngine->queryWidget();
	setFocusProxy(queryWidget);
	connect(queryWidget, SIGNAL(search()), this, SLOT(search()));
	connect(resultWidget, SIGNAL(requestShowLink(QUrl)), this, SLOT(requestShowLink(QUrl)));
	connect(searchEngine, SIGNAL(searchingStarted()), this, SLOT(searchingStarted()));
	connect(searchEngine, SIGNAL(searchingFinished(int)), this, SLOT(searchingFinished(int)));
	connect(searchEngine, SIGNAL(indexingStarted()), this, SLOT(indexingStarted()));
	connect(searchEngine, SIGNAL(indexingFinished()), this, SLOT(indexingFinished()));
	layoutSearch->addWidget(queryWidget);
	layoutSearch->addWidget(resultWidget);
	QDockWidget *dockWidgetSearch = new QDockWidget(tr("search"), this);
	QWidget *searchWidget = new QWidget(this);
	searchWidget->setLayout(layoutSearch);
	dockWidgetSearch->setWidget(searchWidget);
	addDockWidget(Qt::LeftDockWidgetArea, dockWidgetSearch);

	//tabs the 3 dockWidgets together and makes the dockWidgetContent on top
	tabifyDockWidget(dockWidgetContent, dockWidgetIndex);
	tabifyDockWidget(dockWidgetIndex, dockWidgetSearch);
	setTabPosition(Qt::LeftDockWidgetArea, QTabWidget::North);
	dockWidgetContent->raise();

	//toolbar
	QToolBar *toolbar = new QToolBar(tr("toolBar"), this);
	toolbar->addAction(m_pView->pageAction(QWebEnginePage::Back));
	toolbar->addAction(m_pView->pageAction(QWebEnginePage::Forward));
	toolbar->addAction(m_pView->pageAction(QWebEnginePage::Reload));

	toolbar->addSeparator();

	QAction *homeAction = new QAction(QIcon(":/itomDesignerPlugins/general/icons/home.png"), tr("home"), this);
	connect(homeAction, SIGNAL(triggered()), this, SLOT(setupFinished()));
	toolbar->addAction(homeAction);

	toolbar->addSeparator();

	QAction *zoomInAction = new QAction(QIcon(":/qt-project.org/dialogs/qprintpreviewdialog/images/zoom-in-24.png"), tr("zoom in"), this);
	connect(zoomInAction, SIGNAL(triggered()), this, SLOT(mnuZoomInWindow()));
	toolbar->addAction(zoomInAction);
	
	QAction *zoomOutAction = new QAction(QIcon(":/qt-project.org/dialogs/qprintpreviewdialog/images/zoom-out-24.png"), tr("zoom out"), this);
	connect(zoomOutAction, SIGNAL(triggered()), this, SLOT(mnuZoomOutWindow()));
	toolbar->addAction(zoomOutAction);

	QAction *defaultZoomAction = new QAction(QIcon(":/plots/icons/zoom-3.png"), tr("default zoom"), this);
	connect(defaultZoomAction, SIGNAL(triggered()), this, SLOT(mnuDefaultZoomWindow()));
	toolbar->addAction(defaultZoomAction);

	QAction *showFindWordBar = new QAction(QIcon(":/files/icons/fileModified.png"), tr("search text"), this);
	connect(showFindWordBar, SIGNAL(triggered()), this, SLOT(showFindWordBar()));
	toolbar->addAction(showFindWordBar);

	QAction *closeHelpAction = new QAction(QIcon(":/files/icons/close.png"), tr("Exit"), this);
	connect(closeHelpAction, SIGNAL(triggered()), this, SLOT(mnuCloseWindow()));
	toolbar->addAction(closeHelpAction);

	addToolBar(toolbar);
	
	//menubar
	QMenuBar *menuBar = new QMenuBar(this);

	//filemenu
	QMenu *fileMenu = menuBar->addMenu(tr("File"));	
	fileMenu->addAction(closeHelpAction);
	
	//editmenu
	QMenu *editMenu = menuBar->addMenu(tr("Edit"));	
	editMenu->addAction(showFindWordBar);

	//viewmenu
	QMenu *viewMenu = menuBar->addMenu(tr("View"));
	viewMenu->addAction(m_pView->pageAction(QWebEnginePage::Reload));
	viewMenu->addAction(homeAction);
	viewMenu->addAction(zoomInAction);
	viewMenu->addAction(zoomOutAction);
	viewMenu->addAction(defaultZoomAction);

	//gotomenu
	QMenu *goToMenu = menuBar->addMenu(tr("Go to"));
	goToMenu->addAction(homeAction);
	goToMenu->addAction(m_pView->pageAction(QWebEnginePage::Back));
	goToMenu->addAction(m_pView->pageAction(QWebEnginePage::Forward));
	
	setMenuWidget(menuBar);

	showMaximized();
}

//----------------------------------------------------------------------------------------
HelpViewer::~HelpViewer()
{
	DELETE_AND_SET_NULL(m_pHelpEngine);
    DELETE_AND_SET_NULL(m_pView);
	DELETE_AND_SET_NULL(m_pSchemeHandler);
	DELETE_AND_SET_NULL(m_pFindWord);
}

//----------------------------------------------------------------------------------------
void HelpViewer::setCollectionFile(const QString &collectionFile)
{
	QHelpContentWidget *hcw = m_pHelpEngine->contentWidget();
	m_pHelpEngine->setCollectionFile(collectionFile);
	m_pHelpEngine->setupData();
}

//----------------------------------------------------------------------------------------
void HelpViewer::search()
{
	QHelpSearchEngine *searchEngine = m_pHelpEngine->searchEngine();
	QHelpSearchQueryWidget *query = searchEngine->queryWidget();
	QList<QHelpSearchQuery> queryList = query->query();
	searchEngine->search(queryList);
}

//----------------------------------------------------------------------------------------
void HelpViewer::searchingStarted()
{
	QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
}

//----------------------------------------------------------------------------------------
void HelpViewer::searchingFinished(const int &hits)
{
	QApplication::restoreOverrideCursor();
}

//----------------------------------------------------------------------------------------
void HelpViewer::indexingStarted()
{
	QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
}

//----------------------------------------------------------------------------------------
void HelpViewer::indexingFinished()
{
	QApplication::restoreOverrideCursor();
}

//----------------------------------------------------------------------------------------
void HelpViewer::requestShowLink(const QUrl &url)
{
	linkActivated(url);
}

//----------------------------------------------------------------------------------------
void HelpViewer::setupFinished()
{
	QHelpContentWidget *hcw = m_pHelpEngine->contentWidget();
	QString itomVersion = QString("%1.%2.%3").arg(QString::number(ITOM_VERSION_MAJOR)).arg(QString::number(ITOM_VERSION_MINOR)).arg(QString::number(ITOM_VERSION_PATCH));
	QUrl mainPageUrl;
	mainPageUrl.setUrl(tr("qthelp://org.sphinx.itomdocumentation.%1/doc/index.html").arg(itomVersion));
	//QUrl pluginPageUrl = pluginPageUrl.setUrl("qthelp://org.sphinx.itomplugindoc/doc/index.html");
	linkActivated(mainPageUrl);
}

//----------------------------------------------------------------------------------------
void HelpViewer::expandContent()
{
	QHelpContentWidget *hcw = m_pHelpEngine->contentWidget();
	hcw->expandToDepth(0);
}

//----------------------------------------------------------------------------------------
void HelpViewer::returnPressed()
{
	QHelpIndexWidget *hiw = m_pHelpEngine->indexWidget();
	hiw->activateCurrentItem();
}

//----------------------------------------------------------------------------------------
void HelpViewer::textChanged(const QString &text)
{
	QHelpIndexWidget *hiw = m_pHelpEngine->indexWidget();
	if (text.contains('*'))
	{
		hiw->filterIndices(text, text);
	}
	else
	{
		hiw->filterIndices(text, QString());
	}
}

//----------------------------------------------------------------------------------------
void HelpViewer::mnuDefaultZoomWindow()
{
	m_pView->setZoomFactor(m_pDefaultZoomFactor);
	m_pZoomFactor = m_pDefaultZoomFactor;
}

//----------------------------------------------------------------------------------------
void HelpViewer::mnuZoomInWindow()
{
	qreal zoomFactor = m_pView->zoomFactor();
	m_pZoomFactor = zoomFactor + zoomFactor / 20;
	m_pView->setZoomFactor(m_pZoomFactor);
}

//----------------------------------------------------------------------------------------
void HelpViewer::mnuZoomOutWindow()
{
	qreal zoomFactor = m_pView->zoomFactor();
	m_pZoomFactor = zoomFactor - zoomFactor / 20;
	m_pView->setZoomFactor(m_pZoomFactor);
}

//----------------------------------------------------------------------------------------
void HelpViewer::mnuCloseWindow()
{
	this->hide();
}

//----------------------------------------------------------------------------------------
void HelpViewer::linkActivated(const QUrl &url)
{
	m_pView->setHtml(m_pHelpEngine->fileData(url), url);
	m_pView->setZoomFactor(m_pZoomFactor);
}

//----------------------------------------------------------------------------------------
void HelpViewer::linkActivated(const QUrl &url, const QString &text)
{
	m_pView->setHtml(m_pHelpEngine->fileData(url), url);
	m_pView->setZoomFactor(m_pZoomFactor);
}

//----------------------------------------------------------------------------------------
void HelpViewer::urlChanged(const QUrl &url)
{
	QHelpContentWidget *hcw = m_pHelpEngine->contentWidget();
	QModelIndex index = hcw->indexOf(url);
	hcw->setCurrentIndex(index);
	m_pView->setZoomFactor(m_pZoomFactor);
}

//----------------------------------------------------------------------------------------
void HelpViewer::findNextWord(QString expr, bool regExpr, bool caseSensitive, bool wholeWord, bool wrap, bool forward, bool isQuickSeach)
{
	if (forward)
	{
		m_pView->findText(expr, 0);
	}
	else
	{
		m_pView->findText(expr, QWebEnginePage::FindBackward);
	}
	
}

//----------------------------------------------------------------------------------------
void HelpViewer::hideFindWordBar()
{
	m_pFindWord->hide();
	m_pView->findText("", 0);
}

//----------------------------------------------------------------------------------------
void HelpViewer::showFindWordBar()
{
	m_pFindWord->show();
	m_pFindWord->setCursorToTextField();
}


//----------------------------------------------------------------------------------------
void HelpViewer::keyPressEvent(QKeyEvent *event)
{
	int key = event->key();
	//Qt::KeyboardModifiers modifiers = event->modifiers();

	if ((key == Qt::Key_F) && QApplication::keyboardModifiers() && Qt::ControlModifier &&!m_pFindWord->isVisible())
	{
		showFindWordBar();
	}
	else if ((key == Qt::Key_F) && QApplication::keyboardModifiers() && Qt::ControlModifier &&m_pFindWord->isVisible())
	{
		hideFindWordBar();
	}
}


} //end namespace ito

#endif