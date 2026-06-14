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

#include "widgetPropFigurePlugins.h"

#include "../global.h"
#include "../AppManagement.h"
#include "../helper/guiHelper.h"

#include <qtableview.h>
#include <qcombobox.h>
#include <qheaderview.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropFigurePlugins::WidgetPropFigurePlugins(QWidget *parent) :
    AbstractPropertyPageWidget(parent),
    m_loadedFiguresModel(NULL),
    m_figureCategoryModel(NULL),
    m_delegate(NULL)
{
    ui.setupUi(this);

    init();
    ui.tableFigurePlugins->setModel(m_loadedFiguresModel);
    ui.tableFigureCategory->setModel(m_figureCategoryModel);
    ui.tableFigureCategory->setItemDelegate(m_delegate);

    ui.tableFigurePlugins->resizeColumnsToContents();
    ui.tableFigureCategory->resizeColumnsToContents();

    float dpiFactor = GuiHelper::screenDpiFactor(); //factor related to 96dpi (1.0)
    ui.tableFigurePlugins->verticalHeader()->setMinimumSectionSize(21 * dpiFactor);
    ui.tableFigureCategory->verticalHeader()->setMinimumSectionSize(21 * dpiFactor);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropFigurePlugins::~WidgetPropFigurePlugins()
{
    DELETE_AND_SET_NULL(m_loadedFiguresModel);
    DELETE_AND_SET_NULL(m_figureCategoryModel);
    DELETE_AND_SET_NULL(m_delegate);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropFigurePlugins::init()
{
    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    if (dwo)
    {
        QList<ito::FigurePlugin> plugins = dwo->getPossibleFigureClasses(0, 0, 0);
        QStandardItem *i;
        QStringList sl;
        QFileInfo fi;
        int row = 0;

        m_loadedFiguresModel = new QStandardItemModel(plugins.size(), 6, this);
        m_loadedFiguresModel->setHeaderData(0, Qt::Horizontal, tr("class name"), Qt::DisplayRole);
        m_loadedFiguresModel->setHeaderData(1, Qt::Horizontal, tr("data types"), Qt::DisplayRole);
        m_loadedFiguresModel->setHeaderData(2, Qt::Horizontal, tr("data formats"), Qt::DisplayRole);
        m_loadedFiguresModel->setHeaderData(3, Qt::Horizontal, tr("features"), Qt::DisplayRole);
        m_loadedFiguresModel->setHeaderData(4, Qt::Horizontal, tr("plot type"), Qt::DisplayRole);
        m_loadedFiguresModel->setHeaderData(5, Qt::Horizontal, tr("plugin file"), Qt::DisplayRole);

        foreach (const FigurePlugin &f, plugins)
        {
            i = new QStandardItem(QIcon(":/application/icons/itomicon/itomIcon32") /*f.icon*/, f.classname);
            m_loadedFiguresModel->setItem(row,0,i);

            sl = dwo->getPlotInputTypes(f.plotDataTypes);
            i = new QStandardItem(sl.join(", "));
            m_loadedFiguresModel->setItem(row, 1, i);

            sl.clear();
            sl = dwo->getPlotDataFormats(f.plotDataFormats);
            i = new QStandardItem(sl.join(", "));
            m_loadedFiguresModel->setItem(row, 2, i);

            sl.clear();
            sl = dwo->getPlotFeatures(f.plotFeatures);
            i = new QStandardItem(sl.join(", "));
            m_loadedFiguresModel->setItem(row, 3, i);

            sl.clear();
            sl = dwo->getPlotType(f.plotFeatures);
            i = new QStandardItem(sl.join(", "));
            m_loadedFiguresModel->setItem(row, 4, i);

            fi = QFileInfo(f.filename);
            i = new QStandardItem(fi.completeBaseName());
            i->setToolTip(f.filename);
            m_loadedFiguresModel->setItem(row, 5, i);
            row++;
        }

        const QMap<QString, FigureCategory> figureCategories = dwo->getFigureCategories();

        m_delegate = new FigurePluginDelegate(this);

        m_figureCategoryModel = new QStandardItemModel(figureCategories.size(), 3);
        m_figureCategoryModel->setHeaderData(0, Qt::Horizontal, tr("category"), Qt::DisplayRole);
        m_figureCategoryModel->setHeaderData(1, Qt::Horizontal, tr("description"), Qt::DisplayRole);
        m_figureCategoryModel->setHeaderData(2, Qt::Horizontal, tr("default figure plot"), Qt::DisplayRole);

        QMap<QString, FigureCategory>::const_iterator j = figureCategories.constBegin();
        QString defaultClassName = "";
        row = 0;
        ito::RetVal retVal;

        while (j != figureCategories.constEnd())
        {
            i = new QStandardItem(j.key());
            i->setEditable(false);
            m_figureCategoryModel->setItem(row,0,i);

            i = new QStandardItem(j->m_description);
            i->setEditable(false);
            m_figureCategoryModel->setItem(row,1,i);

            defaultClassName = "";
            defaultClassName = dwo->getFigureClass(j.key(), defaultClassName, retVal);
            i = new QStandardItem(defaultClassName);
            m_figureCategoryModel->setItem(row,2,i);

            m_delegate->append(row, j.value());
            ++j;
            ++row;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropFigurePlugins::readSettings()
{
    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    if (dwo)
    {
        const QMap<QString, FigureCategory> figureCategories = dwo->getFigureCategories();
        QMap<QString, FigureCategory>::const_iterator j = figureCategories.constBegin();
        QStandardItem *item;
        QString defaultClassName = "";
        int row = 0;
        ito::RetVal retVal;

        while (j != figureCategories.constEnd())
        {
            defaultClassName = "";
            defaultClassName = dwo->getFigureClass(j.key(), defaultClassName, retVal);

            item = m_figureCategoryModel->item(row,2);
            item->setData(defaultClassName, Qt::DisplayRole);
            ++j;
            ++row;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropFigurePlugins::writeSettings()
{
    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    if (dwo)
    {
        const QMap<QString, FigureCategory> figureCategories = dwo->getFigureCategories();
        QMap<QString, FigureCategory>::const_iterator j = figureCategories.constBegin();
        QStandardItem *item;
        int row = 0;

        while (j != figureCategories.constEnd())
        {
            item = m_figureCategoryModel->item(row, 2);
            dwo->setFigureDefaultClass(j.key(), item->data(Qt::DisplayRole).toString());
            ++j;
            ++row;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropFigurePlugins::on_btnResetDefaultFigures_clicked()
{
    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    if (dwo)
    {
        const QMap<QString, FigureCategory> figureCategories = dwo->getFigureCategories();
        QMap<QString, FigureCategory>::const_iterator j = figureCategories.constBegin();
        QStandardItem *item;
        QString defaultClassName = "";
        int row = 0;
        ito::RetVal retVal;

        while (j != figureCategories.constEnd())
        {
            defaultClassName = j->m_defaultClassName;
            item = m_figureCategoryModel->item(row, 2);
            item->setData(defaultClassName, Qt::DisplayRole);
            ++j;
            ++row;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
FigurePluginDelegate::FigurePluginDelegate(QObject *parent /*= 0*/) :
    QItemDelegate(parent)
{

}

//----------------------------------------------------------------------------------------------------------------------------------
QWidget *FigurePluginDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    if (index.column() == 2)
    {
        QComboBox *editor = new QComboBox(parent);

        if (m_possibleClassNames.contains(index.row()))
        {
            editor->addItems(m_possibleClassNames[index.row()]);
        }

        return editor;
    }
    return QItemDelegate::createEditor(parent, option, index);
}

//----------------------------------------------------------------------------------------------------------------------------------
void FigurePluginDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
    if (index.column() == 2)
    {
        QString figurePluginClassName = index.model()->data(index, Qt::EditRole).toString();
        QComboBox *combo = qobject_cast<QComboBox*>(editor);

        int idx;
        if (m_possibleClassNames.contains(index.row()) == false)
        {
            idx = -1;
        }
        else
        {
            idx = m_possibleClassNames[index.row()].indexOf(figurePluginClassName);
        }

        combo->setCurrentIndex(idx);
    }
    else
    {
        QItemDelegate::setEditorData(editor, index);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FigurePluginDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
    if (index.column() == 2)
    {
        QComboBox *combo = qobject_cast<QComboBox*>(editor);
        int idx = combo->currentIndex();

        if (m_possibleClassNames.contains(index.row()) == false)
        {
            idx = -1;
        }
        else if (idx < 0 || idx >= m_possibleClassNames[index.row()].size())
        {
            idx = -1;
        }

        if (idx == -1)
        {
            model->setData(index, "", Qt::EditRole);
        }
        else
        {
            model->setData(index, m_possibleClassNames[index.row()][idx], Qt::EditRole);
        }
    }
    else
    {
        QItemDelegate::setModelData(editor,model,index);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FigurePluginDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    if (index.column() == 2)
    {
        editor->setGeometry(option.rect);
    }
    else
    {
        QItemDelegate::updateEditorGeometry(editor, option, index);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FigurePluginDelegate::append(int rowIndex, const FigureCategory &figureCategory)
{
    m_figureCategories[rowIndex] = figureCategory;
    QStringList possibleClassNames;

    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    if (dwo)
    {
        QList<FigurePlugin> figureClasses = dwo->getPossibleFigureClasses(figureCategory);
        foreach(const FigurePlugin &p, figureClasses)
        {
            possibleClassNames.append(p.classname);
        }
    }

    m_possibleClassNames[rowIndex] = possibleClassNames;
}

} //end namespace ito
