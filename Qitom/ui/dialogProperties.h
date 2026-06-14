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

#ifndef DIALOGPROPERTIES_H
#define DIALOGPROPERTIES_H

#include "abstractPropertyPageWidget.h"

#include <QtGui>
#include <qdialog.h>
#include <qmap.h>
#include <qtreewidget.h>
#include <qstringlist.h>
#include <qstackedwidget.h>
#include <qsplitter.h>
#include <qdialogbuttonbox.h>
#include <qlayout.h>
#include <qlabel.h>

namespace ito
{

class DialogProperties : public QDialog
{
    Q_OBJECT

public:

    struct PropertyPage
    {
        PropertyPage() : m_widget(NULL), m_icon(), m_visited(false) {}
        PropertyPage(QString name, QString title, QString fullname,
            AbstractPropertyPageWidget* widget, QIcon icon) : m_title(title),
            m_name(name), m_fullname(fullname), m_widget(widget),
            m_icon(icon), m_visited(false) {}
        QString m_title;
        QString m_name;
        QString m_fullname;
        AbstractPropertyPageWidget* m_widget;
        QIcon m_icon;
        bool m_visited;
    };

    DialogProperties(QWidget* parent = 0, Qt::WindowFlags f = Qt::WindowFlags());
    ~DialogProperties();

    bool selectTabByKey(QString &key, QTreeWidgetItem *parent = NULL);

protected:
    void initPages();

    void addPage(PropertyPage page, QTreeWidgetItem *parent, QStringList remainingPathes);

private:
    QStackedWidget *m_pStackedWidget;
    QSplitter *m_pSplitter;
    QTreeWidget *m_pCategories;
    QDialogButtonBox *m_pButtonBox;

    QHBoxLayout *m_pHorizontalLayout;
    QVBoxLayout *m_pVerticalLayout;

    QLabel *m_pPageTitle;
    QFrame *m_pLine;
    QWidget *m_pEmptyPage;

    QString m_CurrentPropertyKey;

    QMap<QString, PropertyPage> m_pages;

signals:
    void propertiesChanged();

public slots:

private slots:
    void categoryChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous);
    void accepted();
    void rejected();
    void apply();
};

} //end namespace ito

#endif
