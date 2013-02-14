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

#ifndef ICONBROWSERDOCKWIDGET_H
#define ICONBROWSERDOCKWIDGET_H

#include "qdialog.h"
#include <qtreewidget.h>

class IconBrowserDialog : public QDialog
{
    Q_OBJECT

public:
    IconBrowserDialog(QWidget *parent = NULL);
    ~IconBrowserDialog();

protected:

    class IconRescourcesTreeView : public QTreeWidget
    {
    public:
        IconRescourcesTreeView ( QWidget * parent = 0 ) : QTreeWidget(parent) {}
        ~IconRescourcesTreeView () {};

		
        QModelIndexList selectedIndexes() const
        { 
            return QTreeWidget::selectedIndexes();
        }
    };

private:

    IconRescourcesTreeView* m_pTreeWidget;

signals:

private slots:

public slots:
    void copyCurrentName();
};

#endif
