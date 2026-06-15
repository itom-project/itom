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

#ifndef HELPVIEWER_H
#define HELPVIEWER_H

#include "../global.h"

#ifdef ITOM_USEHELPVIEWER

#include <qmainwindow.h>

class QWebEngineView; //forward declaration
class QHelpEngine; //forward declaration
class QLineEdit;//forward declaration
class QUrl; //forward declaration
class QMainWindow; //forward declaration
class QMoveEvent; //forward declaration

namespace ito {

class WidgetFindWord; //forward declaration
class QtHelpUrlSchemeHandler; //forward declaration

class HelpViewer : public QMainWindow
{
    Q_OBJECT

public:
    HelpViewer(QWidget *parent = NULL);
    ~HelpViewer();

    void setCollectionFile(const QString &collectionFile);
    void getHelpViewer(const QWidget& helpViewer);
    void showUrl(const QString& url);

private:
    QWebEngineView *m_pView;
	WidgetFindWord *m_pFindWord;
    QString m_collectionFile;
	QHelpEngine *m_pHelpEngine;
	QtHelpUrlSchemeHandler *m_pSchemeHandler;
	qreal m_pDefaultZoomFactor;
	qreal m_pZoomFactor;
	QLineEdit *m_plineEditIndex;
	bool m_pSearched = false;

protected:
	void keyPressEvent(QKeyEvent *event);
	void mousePressEvent(QMouseEvent *event);
    void moveEvent(QMoveEvent* event);

private slots:
	void linkActivated(const QUrl &url);
	void linkActivated(const QUrl &url, const QString &text);
	void urlChanged(const QUrl &url);
	void setupFinished();
	void expandContent();
	void mnuCloseWindow();
	void mnuZoomInWindow();
	void mnuZoomOutWindow();
	void mnuDefaultZoomWindow();
	void textChanged(const QString &text);
	void returnPressed();
	void search();
	void requestShowLink(const QUrl &url);
	void searchingStarted();
	void searchingFinished(const int &hits);
	void indexingStarted();
	void indexingFinished();
	void clicked(const QModelIndex &index);
	void findNextWord(QString expr, bool regExpr, bool caseSensitive, bool wholeWord, bool wrap, bool forward, bool isQuickSeach);
	void hideFindWordBar();
	void showFindWordBar();
	void visibilityChangedIndexWidget(bool visible);
	void visibilityChangedSearchWidget(bool visible);
	void loadFinished(bool ok);

};

} //end namespace ito

#endif
#endif
