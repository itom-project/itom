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
#include <qhelpengine.h>
#include <qdockwidget.h>
#include <qhelpcontentwidget.h>
#include <qhelpindexwidget.h>
#include <qdebug.h>

namespace ito {

//----------------------------------------------------------------------------------------
HelpViewer::HelpViewer(QWidget *parent /*= NULL*/) :
    QMainWindow(parent),
    m_pView(NULL),
    m_pHelpEngine(NULL)
{
    m_pView = new QWebEngineView(this);
    //m_pView->load(QUrl("http://itom.bitbucket.org"));
    setCentralWidget(m_pView);

    m_pHelpEngine = new QHelpEngine("", this);
    
    QHelpContentWidget *hcw = m_pHelpEngine->contentWidget();
    QDockWidget *dockWidget = new QDockWidget("content", this);
    dockWidget->setWidget(hcw);
    addDockWidget(Qt::LeftDockWidgetArea, dockWidget);
    connect(hcw, SIGNAL(linkActivated(QUrl)), this, SLOT(showPage(QUrl)));

    QHelpIndexWidget *hiw = m_pHelpEngine->indexWidget();
    dockWidget = new QDockWidget("index", this);
    dockWidget->setWidget(hiw);
    addDockWidget(Qt::RightDockWidgetArea, dockWidget);
}

//----------------------------------------------------------------------------------------
HelpViewer::~HelpViewer()
{
    DELETE_AND_SET_NULL(m_pHelpEngine);
    DELETE_AND_SET_NULL(m_pView);
}

//----------------------------------------------------------------------------------------
void HelpViewer::setCollectionFile(const QString &collectionFile)
{
    m_pHelpEngine->setCollectionFile(collectionFile);
    m_pHelpEngine->setupData();
}

//----------------------------------------------------------------------------------------
void HelpViewer::showPage(const QUrl &url)
{
    if (m_pView)
    {
        m_pView->setHtml(m_pHelpEngine->fileData(url), url);
    }
}


} //end namespace ito

#endif

