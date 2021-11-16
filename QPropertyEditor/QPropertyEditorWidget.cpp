// *************************************************************************************************
//
// QPropertyEditor v 0.3
//
// --------------------------------------
// Copyright (C) 2007 Volker Wiendl
// Acknowledgements to Roman alias banal from qt-apps.org for the Enum enhancement
//
//
// The QPropertyEditor Library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by the Free Software
// Foundation; either version 2 of the License, or (at your option) any later
// version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with
// this program; if not, write to the Free Software Foundation, Inc., 59 Temple
// Place - Suite 330, Boston, MA 02111-1307, USA, or go to
// http://www.gnu.org/copyleft/lesser.txt.
//
// *************************************************************************************************

#include "QPropertyEditorWidget.h"
#include "Property.h"
#include "QPropertyModel.h"
#include "QVariantDelegate.h"
#include <qaction.h>
#include <qevent.h>
#include <qheaderview.h>
#include <qsortfilterproxymodel.h>

#include "itomCustomTypes.h"


class PropertyEditorSortFilterProxyModel : public QSortFilterProxyModel
{
public:
    PropertyEditorSortFilterProxyModel(QWidget *parent = nullptr) :
        QSortFilterProxyModel(parent)
    {

    }

protected:
    bool filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const
    {
        const QPropertyModel* propModel = qobject_cast<QPropertyModel*>(sourceModel());
        QModelIndex srcIndex = propModel->index(sourceRow, 0, sourceParent);
        int lvl = 0;

        while (srcIndex.parent().isValid())
        {
            lvl++;
            srcIndex = propModel->parent(srcIndex);
        }

        if (propModel->groupByInheritance())
        {
            if (lvl == 1)
            {
                return QSortFilterProxyModel::filterAcceptsRow(sourceRow, sourceParent);
            }
        }
        else
        {
            if (lvl == 0)
            {
                return QSortFilterProxyModel::filterAcceptsRow(sourceRow, sourceParent);
            }
        }

        return true;
    }
};

//-------------------------------------------------------------------------------------
QPropertyEditorWidget::QPropertyEditorWidget(QWidget* parent /*= 0*/) : QTreeView(parent)
{
    m_model = new QPropertyModel(this);

    QSortFilterProxyModel *sortModel = new PropertyEditorSortFilterProxyModel(this);
    sortModel->setSourceModel(m_model);
    sortModel->setFilterWildcard("*");
    sortModel->setFilterKeyColumn(0);
    sortModel->setFilterCaseSensitivity(Qt::CaseInsensitive);

    setModel(sortModel);

    setItemDelegate(new QVariantDelegate(this));
    // setEditTriggers( QAbstractItemView::SelectedClicked | QAbstractItemView::EditKeyPressed |
    // QAbstractItemView::AnyKeyPressed /*QAbstractItemView::AllEditTriggers*/ );

    // triggers are handled by mousepress and keypress event below (is better than original)
    setEditTriggers(QAbstractItemView::EditKeyPressed);
    setSelectionBehavior(QAbstractItemView::SelectRows);
    setAlternatingRowColors(true);

    ito::itomCustomTypes::registerTypes();
    registerCustomPropertyCB(ito::itomCustomTypes::createCustomProperty);

    QAction* action = new QAction(QIcon(":/classNavigator/icons/sortAZAsc.png"), tr("Enable sorting"), this);
    action->setCheckable(true);
    action->setChecked(isSortingEnabled());
    connect(action, &QAction::triggered, this, &QPropertyEditorWidget::sortedAction);
    addAction(action);

    action = new QAction(QIcon(":/files/icons/browser.png"), tr("Group by inheritance"), this);
    action->setCheckable(true);
    action->setChecked(m_model->groupByInheritance());
    connect(action, &QAction::triggered, this, &QPropertyEditorWidget::setGroupByInheritance);
    addAction(action);

    setContextMenuPolicy(Qt::ActionsContextMenu);
}

//-------------------------------------------------------------------------------------
QPropertyEditorWidget::~QPropertyEditorWidget()
{
}

//-------------------------------------------------------------------------------------
void QPropertyEditorWidget::addObject(QObject* propertyObject)
{
    m_model->addItem(propertyObject);

    if (m_model->groupByInheritance())
    {
        expandToDepth(0);
    }
}

//-------------------------------------------------------------------------------------
void QPropertyEditorWidget::setObject(QObject* propertyObject)
{
    m_model->clear();
    if (propertyObject)
    {
        addObject(propertyObject);
    }
}

//-------------------------------------------------------------------------------------
void QPropertyEditorWidget::updateObject(QObject* propertyObject)
{
    if (propertyObject)
    {
        m_model->updateItem(propertyObject);
    }
}

//-------------------------------------------------------------------------------------
void QPropertyEditorWidget::registerCustomPropertyCB(UserTypeCB callback)
{
    m_model->registerCustomPropertyCB(callback);
}

//-------------------------------------------------------------------------------------
void QPropertyEditorWidget::unregisterCustomPropertyCB(UserTypeCB callback)
{
    m_model->unregisterCustomPropertyCB(callback);
}

//-------------------------------------------------------------------------------------
void QPropertyEditorWidget::mousePressEvent(QMouseEvent* event)
{
    QTreeView::mousePressEvent(event);
    QModelIndex index = indexAt(event->pos());

    if (index.isValid())
    {
        if (/*(item != m_editorPrivate->editedItem()) && */ (event->button() == Qt::LeftButton) &&
            (header()->logicalIndexAt(event->pos().x()) == 1) &&
            ((model()->flags(index) & (Qt::ItemIsEditable | Qt::ItemIsEnabled)) ==
             (Qt::ItemIsEditable | Qt::ItemIsEnabled)))
        {
            // editItem(item, 1);
            try
            {
                edit(index);
            }
            catch (...)
            {
                int i = 1;
            }
        }
        /*else if (!m_editorPrivate->hasValue(item) && m_editorPrivate->markPropertiesWithoutValue()
        && !rootIsDecorated())
        {
            if (event->pos().x() + header()->offset() < 20)
                item->setExpanded(!item->isExpanded());
        }*/
    }
}

//-------------------------------------------------------------------------------------
void QPropertyEditorWidget::keyPressEvent(QKeyEvent* event)
{
    switch (event->key())
    {
    case Qt::Key_Return:
    case Qt::Key_Enter:
    case Qt::Key_Space: // Trigger Edit
        // if (!m_editorPrivate->editedItem())
        {
            QModelIndex index = currentIndex();

            if (index.isValid())
            {
                if (model()->columnCount(index) >= 2 &&
                    ((model()->flags(index) & (Qt::ItemIsEditable | Qt::ItemIsEnabled)) ==
                     (Qt::ItemIsEditable | Qt::ItemIsEnabled)))
                {
                    event->accept();
                    // If the current position is at column 0, move to 1.
                    if (index.column() == 0)
                    {
                        index = index.sibling(index.row(), 1);
                        setCurrentIndex(index);
                    }
                    edit(index);
                    return;
                }
            }
        }
        break;
    default:
        break;
    }
    QTreeView::keyPressEvent(event);
}

//-------------------------------------------------------------------------------------
//!< todo: deprecated
void QPropertyEditorWidget::setSorted(bool enabled)
{
}

//-------------------------------------------------------------------------------------
//!< todo: deprecated
bool QPropertyEditorWidget::sorted() const
{

    return isSortingEnabled();
}

//-------------------------------------------------------------------------------------
void QPropertyEditorWidget::setGroupByInheritance(bool enabled)
{
    m_model->setGroupByInheritance(enabled);

    // first action corresponds to sortingEnabled
    QAction *action = actions()[1];
    action->blockSignals(true);
    action->setChecked(enabled);
    action->blockSignals(false);

    if (!enabled)
    {
        expandToDepth(0);
    }
    else
    {
        expandToDepth(1);
    }
}

//-------------------------------------------------------------------------------------
bool QPropertyEditorWidget::groupByInheritance() const
{
    return m_model->groupByInheritance();
}

//-------------------------------------------------------------------------------------
QString QPropertyEditorWidget::nameFilterPattern() const
{
    const auto proxyModel = qobject_cast<QSortFilterProxyModel*>(model());
    
    if (proxyModel)
    {
        return proxyModel->filterRegularExpression().pattern();
    }

    return "";
}

//-------------------------------------------------------------------------------------
void QPropertyEditorWidget::setNameFilterPattern(const QString &wildcardPattern)
{
    const auto proxyModel = qobject_cast<QSortFilterProxyModel*>(model());

    if (proxyModel)
    {
        proxyModel->setFilterWildcard(wildcardPattern);
    }
}

//-------------------------------------------------------------------------------------
void QPropertyEditorWidget::sortedAction(bool checked)
{
    setSortingEnabled(checked);

    if (!checked)
    {
        sortByColumn(-1);
    }
    
    QAction *action = actions()[0];
    action->blockSignals(true);
    action->setChecked(checked);
    action->blockSignals(false);
}
