/* ********************************************************************
itom measurement system
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2017, Institut fuer Technische Optik (ITO),
Universitaet Stuttgart, Germany

This file is part of itom.

itom is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

itom is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "paramEditorWidget.h"
#include "paramEditorModel.h"
#include "QVariantDelegate.h"
#include <qevent.h>
#include <qheaderview.h>
#include <qaction.h>

class ParamEditorWidgetPrivate
{
	Q_DECLARE_PUBLIC(ParamEditorWidget);

public:
	QScopedPointer<ParamEditorModel> m_model;
};


ParamEditorWidget::ParamEditorWidget(QWidget* parent /*= 0*/) : 
	QTreeView(parent),
	d_ptr(new ParamEditorWidgetPrivate())
{
	Q_D(ParamEditorWidget);

	d->m_model = new ParamEditorModel(this);

    m_model = new QPropertyModel(this);    

    setModel(m_model);

    setItemDelegate(new QVariantDelegate(this));
    //setEditTriggers( QAbstractItemView::SelectedClicked | QAbstractItemView::EditKeyPressed | QAbstractItemView::AnyKeyPressed /*QAbstractItemView::AllEditTriggers*/ );
    setEditTriggers( QAbstractItemView::EditKeyPressed ); //triggers are handled by mousepress and keypress event below (is better than original)
    setSelectionBehavior( QAbstractItemView::SelectRows );
    setAlternatingRowColors(true);

    QAction *action = new QAction(tr("sort entries"), this);
    action->setCheckable(true);
    action->setChecked(sorted());
    addAction(action);
    connect(action, SIGNAL(triggered(bool)), this, SLOT(sortedAction(bool)));

    setContextMenuPolicy(Qt::ActionsContextMenu);

    ito::itomCustomTypes::registerTypes();
    registerCustomPropertyCB(ito::itomCustomTypes::createCustomProperty);
}


QPropertyEditorWidget::~QPropertyEditorWidget()
{
}

void QPropertyEditorWidget::addObject(QObject* propertyObject)
{
    m_model->addItem(propertyObject);
    if(!m_model->sorted())
    {
        expandToDepth(0);
    }
}

void QPropertyEditorWidget::setObject(QObject* propertyObject)
{
    m_model->clear();
    if (propertyObject)
    {
        addObject(propertyObject);
    }
}

void QPropertyEditorWidget::updateObject(QObject* propertyObject)
{
    if (propertyObject)
        m_model->updateItem(propertyObject);    
}

void QPropertyEditorWidget::registerCustomPropertyCB(UserTypeCB callback)
{
    m_model->registerCustomPropertyCB(callback);
}

void QPropertyEditorWidget::unregisterCustomPropertyCB(UserTypeCB callback)
{
    m_model->unregisterCustomPropertyCB(callback);
}


void QPropertyEditorWidget::mousePressEvent(QMouseEvent *event)
{
    QTreeView::mousePressEvent(event);
    QModelIndex index = indexAt( event->pos() );

    if (index.isValid()) 
    {
        if (/*(item != m_editorPrivate->editedItem()) && */(event->button() == Qt::LeftButton) 
                && (header()->logicalIndexAt(event->pos().x()) == 1)
                && ((m_model->flags(index) & (Qt::ItemIsEditable | Qt::ItemIsEnabled)) == (Qt::ItemIsEditable | Qt::ItemIsEnabled))) 
        {
            //editItem(item, 1);
            edit(index);
        } 
        /*else if (!m_editorPrivate->hasValue(item) && m_editorPrivate->markPropertiesWithoutValue() && !rootIsDecorated()) 
        {
            if (event->pos().x() + header()->offset() < 20)
                item->setExpanded(!item->isExpanded());
        }*/
    }
}

void QPropertyEditorWidget::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) 
    {
    case Qt::Key_Return:
    case Qt::Key_Enter:
    case Qt::Key_Space: // Trigger Edit
        //if (!m_editorPrivate->editedItem())
        {
            QModelIndex index = currentIndex();

            if (index.isValid() )
            {
                if (m_model->columnCount(index) >= 2 && ((m_model->flags(index) & (Qt::ItemIsEditable | Qt::ItemIsEnabled)) == (Qt::ItemIsEditable | Qt::ItemIsEnabled))) 
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

void QPropertyEditorWidget::setSorted(bool value)
{
    m_model->setSorted(value);
    
    m_sorted = value;

    //first action corresponds to sorted
    if (actions().size() > 0)
    {
        actions()[0]->setChecked(value);
    }

    if(m_sorted)
    {
    }
    else
    {
        expandToDepth(0);
    }
}

bool QPropertyEditorWidget::sorted() const
{
    return m_model->sorted();
}

void QPropertyEditorWidget::sortedAction(bool checked)
{
    setSorted(checked);
}
