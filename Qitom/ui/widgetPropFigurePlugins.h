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

#ifndef WIDGETPROPFIGUREPLUGINS_H
#define WIDGETPROPFIGUREPLUGINS_H

#include "abstractPropertyPageWidget.h"

#include <QtGui>
#include <qstandarditemmodel.h>
#include <qitemdelegate.h>

#include "../organizer/designerWidgetOrganizer.h"

#include "ui_widgetPropFigurePlugins.h"

namespace ito
{

class FigurePluginDelegate; //forward declaration

class WidgetPropFigurePlugins: public AbstractPropertyPageWidget
{
    Q_OBJECT

public:
    WidgetPropFigurePlugins(QWidget *parent = NULL);
    ~WidgetPropFigurePlugins();

    void readSettings();
    void writeSettings();

protected:
    void init();

private:
    Ui::WidgetPropFigurePlugins ui;

    QStandardItemModel *m_loadedFiguresModel;
    QStandardItemModel *m_figureCategoryModel;
    FigurePluginDelegate *m_delegate;

signals:

public slots:
    void on_btnResetDefaultFigures_clicked();

private slots:

};


class FigurePluginDelegate : public QItemDelegate
{
    Q_OBJECT
public:
    FigurePluginDelegate(QObject *parent = 0);

    void append(int rowIndex, const FigureCategory &figureCategory);

    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;

     void setEditorData(QWidget *editor, const QModelIndex &index) const;
     void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;

     void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;

private:
    QMap<int, FigureCategory> m_figureCategories;
    QMap<int, QStringList> m_possibleClassNames;


};


} //end namespace ito

#endif
