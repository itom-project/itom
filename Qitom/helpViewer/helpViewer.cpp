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

namespace ito {

//----------------------------------------------------------------------------------------
HelpViewer::HelpViewer(QWidget *parent /*= NULL*/) :
    QMainWindow(parent),
    m_pView(NULL),
    m_pHelpEngine(NULL),
	m_pSchemeHandler(NULL)
{
    m_pView = new QWebEngineView(this);
    //m_pView->load(QUrl("http://itom.bitbucket.org"));
    setCentralWidget(m_pView);

	QWebEnginePage *page = m_pView->page();
	QWebEngineProfile *profile = page->profile();

    m_pHelpEngine = new QHelpEngine("", this);
	m_pSchemeHandler = new QtHelpUrlSchemeHandler(m_pHelpEngine, this);
	profile->installUrlSchemeHandler("qthelp", m_pSchemeHandler);
    
    QHelpContentWidget *hcw = m_pHelpEngine->contentWidget();
    QDockWidget *dockWidgetContent = new QDockWidget("content", this);
	dockWidgetContent->setWidget(hcw);
	addDockWidget(Qt::LeftDockWidgetArea, dockWidgetContent);
	connect(hcw, SIGNAL(linkActivated(QUrl)), this, SLOT(showPage(QUrl)));
	connect(m_pView, SIGNAL(urlChanged(QUrl)), this, SLOT(changeIndex(QUrl)));	
	connect(m_pHelpEngine, SIGNAL(setupFinished()), this, SLOT(showStartPage()));
	QHelpContentModel *hcm = m_pHelpEngine->contentModel();
	connect(hcm, SIGNAL(contentsCreated()), this, SLOT(expandContent()));
	
    QHelpIndexWidget *hiw = m_pHelpEngine->indexWidget();
    QDockWidget *dockWidgetIndex = new QDockWidget("index", this);
	dockWidgetIndex->setWidget(hiw);
	addDockWidget(Qt::LeftDockWidgetArea, dockWidgetIndex);

	QVBoxLayout *layout = new QVBoxLayout(this);
	layout->addWidget(m_pHelpEngine->searchEngine()->queryWidget());
	layout->addWidget(m_pHelpEngine->searchEngine()->resultWidget());
	QDockWidget *dockWidgetSearch = new QDockWidget("search", this);
	dockWidgetSearch->setLayout(layout);
	addDockWidget(Qt::LeftDockWidgetArea, dockWidgetSearch);

	tabifyDockWidget(dockWidgetContent, dockWidgetIndex);
	tabifyDockWidget(dockWidgetIndex, dockWidgetSearch);
	setTabPosition(Qt::LeftDockWidgetArea, QTabWidget::North);
	dockWidgetContent->raise();

	QToolBar *toolbar = new QToolBar(this);
	toolbar->addAction(m_pView->pageAction(QWebEnginePage::Back));
	toolbar->addAction(m_pView->pageAction(QWebEnginePage::Forward));
	toolbar->addAction(m_pView->pageAction(QWebEnginePage::Reload));
	addToolBar(toolbar);
	
	QMenuBar *menuBar = new QMenuBar(this);
	QMenu *fileMenu = menuBar->addMenu(tr("File").toLatin1().data());
	QMenu *editMenu = menuBar->addMenu(tr("Edit").toLatin1().data());
	fileMenu->addAction(m_pView->pageAction(QWebEnginePage::Back));
	fileMenu->addAction(m_pView->pageAction(QWebEnginePage::RequestClose));
	setMenuWidget(menuBar);

	showMaximized();
}

//----------------------------------------------------------------------------------------
HelpViewer::~HelpViewer()
{
    DELETE_AND_SET_NULL(m_pHelpEngine);
    DELETE_AND_SET_NULL(m_pView);
	DELETE_AND_SET_NULL(m_pSchemeHandler);
}

//----------------------------------------------------------------------------------------
void HelpViewer::setCollectionFile(const QString &collectionFile)
{
	QHelpContentWidget *hcw = m_pHelpEngine->contentWidget();
    m_pHelpEngine->setCollectionFile(collectionFile);
    m_pHelpEngine->setupData();
}

//----------------------------------------------------------------------------------------
void HelpViewer::showStartPage()
{
	if (m_pView)
	{
		QHelpContentWidget *hcw = m_pHelpEngine->contentWidget();
		QString itomVersion = QString("%1.%2.%3").arg(QString::number(ITOM_VERSION_MAJOR)).arg(QString::number(ITOM_VERSION_MINOR)).arg(QString::number(ITOM_VERSION_PATCH));
		QUrl mainPageUrl, pluginPageUrl;
		mainPageUrl.setUrl(tr("qthelp://org.sphinx.itomdocumentation.%1/doc/index.html").arg(itomVersion));
		//pluginPageUrl.setUrl("qthelp://org.sphinx.itomplugindoc/doc/index.html");
		showPage(mainPageUrl);		
	}	
}

//----------------------------------------------------------------------------------------
void HelpViewer::expandContent()
{
	if (m_pView)
	{
		QHelpContentWidget *hcw = m_pHelpEngine->contentWidget();
		hcw->expandToDepth(0);
	}
}

//----------------------------------------------------------------------------------------
void HelpViewer::showPage(const QUrl &url)
{
    if (m_pView)
    {
        m_pView->setHtml(m_pHelpEngine->fileData(url), url);
    }
}

//----------------------------------------------------------------------------------------
void HelpViewer::changeIndex(const QUrl &url)
{
	if (m_pView)
	{
		QHelpContentWidget *hcw = m_pHelpEngine->contentWidget();
		QModelIndex index = hcw->indexOf(url);
		hcw->setCurrentIndex(index);
	}
}


} //end namespace ito

#endif

