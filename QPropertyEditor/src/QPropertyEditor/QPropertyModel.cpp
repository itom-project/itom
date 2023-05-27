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

#include "QPropertyModel.h"

#include "Property.h"
#include "EnumProperty.h"

#include <qapplication.h>
#include <qmetaobject.h>
#include <qitemeditorfactory.h>

struct PropertyPair
{
    PropertyPair(const QMetaObject* obj, QMetaProperty property) : Property(property), Object(obj) {}

    QMetaProperty       Property;
    const QMetaObject* Object;

    bool operator==(const PropertyPair& other) const {return QString(other.Property.name()) == QString(Property.name());}
};


QPropertyModel::QPropertyModel(QObject* parent /*= 0*/) : QAbstractItemModel(parent), m_sorted(true)
{
    m_rootItem = new Property("Root",0, this);
}


QPropertyModel::~QPropertyModel()
{
}

QModelIndex QPropertyModel::index ( int row, int column, const QModelIndex & parent /*= QModelIndex()*/ ) const
{
    Property *parentItem = m_rootItem;

    if(m_sorted && parent.isValid() == false)
    {
        int newRow = row;

        foreach(const QObject *obj, parentItem->children()) //iterate through all metaObjects
        {
            if(newRow >= 0 && newRow < obj->children().size())
            {
                return createIndex(row, column, obj->children().at(newRow));
            }
            newRow -= obj->children().size();
        }
        return QModelIndex();
    }
    else
    {
        if (parent.isValid())
            parentItem = static_cast<Property*>(parent.internalPointer());
        if (row >= parentItem->children().size() || row < 0)
            return QModelIndex();
        return createIndex(row, column, parentItem->children().at(row));
    }

}

QModelIndex QPropertyModel::parent ( const QModelIndex & index ) const
{
    if (!index.isValid())
        return QModelIndex();

    Property *childItem = static_cast<Property*>(index.internalPointer());
    Property *parentItem = qobject_cast<Property*>(childItem->parent());

    if (!parentItem || parentItem == m_rootItem)
    {
        return QModelIndex();
    }

    if (m_sorted && parentItem->isRoot())
    {
        return QModelIndex();
    }

    return createIndex(parentItem->row(), 0, parentItem);
}

int QPropertyModel::rowCount ( const QModelIndex & parent /*= QModelIndex()*/ ) const
{
    Property *parentItem = m_rootItem;

    if (m_sorted && parent.isValid() == false)
    {
        int rows = 0;
        foreach(const QObject *obj, parentItem->children()) //iterate through all metaObjects
        {
            rows += obj->children().size();
        }

        return rows;
    }
    else
    {
        if (parent.isValid())
        {
            parentItem = static_cast<Property*>(parent.internalPointer());
        }
        return parentItem->children().size();
    }
}

int QPropertyModel::columnCount ( const QModelIndex & /*parent = QModelIndex()*/ ) const
{
    return 2;
}

QVariant QPropertyModel::data ( const QModelIndex & index, int role /*= Qt::DisplayRole*/ ) const
{
    if (!index.isValid())
        return QVariant();

    Property *item = static_cast<Property*>(index.internalPointer());
    switch(role)
    {
    case Qt::ToolTipRole:
    case Qt::DecorationRole:
    case Qt::DisplayRole:
    case Qt::EditRole:
        if (index.column() == 0)
            return item->objectName().replace('_', ' ');
        if (index.column() == 1)
            return item->value(role);
    case Qt::BackgroundRole:
        if (item->isRoot())    return QApplication::palette("QTreeView").brush(QPalette::Normal, QPalette::Button).color();
        break;
    };
    return QVariant();
}

// edit methods
bool QPropertyModel::setData ( const QModelIndex & index, const QVariant & value, int role /*= Qt::EditRole*/ )
{
    if (index.isValid() && role == Qt::EditRole)
    {
        Property *item = static_cast<Property*>(index.internalPointer());
        item->setValue(value);
        emit dataChanged(index, index);
        return true;
    }
    return false;
}

Qt::ItemFlags QPropertyModel::flags ( const QModelIndex & index ) const
{
    if (!index.isValid())
        return Qt::ItemIsEnabled;
    Property *item = static_cast<Property*>(index.internalPointer());
    // only allow change of value attribute
    if (item->isRoot())
        return Qt::ItemIsEnabled;
    else if (item->isReadOnly())
        return Qt::ItemIsDragEnabled | Qt::ItemIsSelectable;
    else
        return Qt::ItemIsDragEnabled | Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsEditable;
}


QVariant QPropertyModel::headerData ( int section, Qt::Orientation orientation, int role /*= Qt::DisplayRole*/ ) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
    {
        switch (section)
        {
             case 0:
                 return tr("Name");
             case 1:
                 return tr("Value");
        }
    }
    return QVariant();
}

QModelIndex QPropertyModel::buddy ( const QModelIndex & index ) const
{
    if (index.isValid() && index.column() == 0)
        return createIndex(index.row(), 1, index.internalPointer());
    return index;
}

void QPropertyModel::addItem(QObject *propertyObject)
{
    // first create property <-> class hierarchy
    QList<PropertyPair> propertyMap;
    QList<const QMetaObject*> classList;
    const QMetaObject* metaObject = propertyObject->metaObject();
    do
    {
        int count = metaObject->propertyCount();
        for (int i=0; i<count; ++i)
        {
            QMetaProperty property = metaObject->property(i);
            if( property.isUser() ) // Hide Qt specific properties
            {
                PropertyPair pair(metaObject, property);
                int index = propertyMap.indexOf(pair);
                if (index != -1)
                    propertyMap[index] = pair;
                else
                    propertyMap.push_back(pair);
            }
        }
        classList.push_front(metaObject);
    }
    while ((metaObject = metaObject->superClass())!=0);

    QList<const QMetaObject*> finalClassList;
    // remove empty classes from hierarchy list
    foreach(const QMetaObject* obj, classList)
    {
        bool keep = false;
        foreach(PropertyPair pair, propertyMap)
        {
            if (pair.Object == obj)
            {
                keep = true;
                break;
            }
        }
        if (keep)
            finalClassList.push_back(obj);
    }

    // finally insert properties for classes containing them
    int i=rowCount();
    Property* propertyItem = 0;
    beginInsertRows( QModelIndex(), i, i + finalClassList.count() );
    foreach(const QMetaObject* metaObject, finalClassList)
    {
        // Set default name of the hierarchy property to the class name
        QString name = metaObject->className();
        // Check if there is a special name for the class
        int index = metaObject->indexOfClassInfo(qPrintable(name));
        if (index != -1)
            name = metaObject->classInfo(index).value();
        // Create Property Item for class node
        propertyItem = new Property(name, 0, m_rootItem);
        foreach(PropertyPair pair, propertyMap)
        {
            // Check if the property is associated with the current class from the finalClassList
            if (pair.Object == metaObject)
            {
                QMetaProperty property(pair.Property);
                Property* p = 0;
                if (property.type() == QVariant::UserType && !m_userCallbacks.isEmpty())
                {
                    QList<QPropertyEditorWidget::UserTypeCB>::iterator iter = m_userCallbacks.begin();
                    while( p == 0 && iter != m_userCallbacks.end() )
                    {
                        p = (*iter)(property.name(), propertyObject, propertyItem);
                        ++iter;
                    }
                }
                if( p == 0){
                    if(property.isEnumType()){
                        p = new EnumProperty(property.name(), propertyObject, propertyItem);
                    } else {
                        p = new Property(property.name(), propertyObject, propertyItem);
                    }
                }
                int index = metaObject->indexOfClassInfo(property.name());
                if (index != -1)
                    p->setEditorHints(metaObject->classInfo(index).value());
            }
        }
    }
    endInsertRows();
    if( propertyItem )    addDynamicProperties( propertyItem, propertyObject );
}

void QPropertyModel::updateItem ( QObject* propertyObject, const QModelIndex& parent /*= QModelIndex() */ )
{
    Property *parentItem = m_rootItem;
    if (parent.isValid())
        parentItem = static_cast<Property*>(parent.internalPointer());
    if (parentItem->propertyObject() != propertyObject)
        parentItem = parentItem->findPropertyObject(propertyObject);
    if (parentItem) // Indicate view that the data for the indices have changed
    {
        QModelIndex itemIndex = createIndex(parentItem->row(), 0, static_cast<Property*>(parentItem));
        dataChanged(itemIndex, createIndex(parentItem->row(), 1, static_cast<Property*>(parentItem)));
        QList<QByteArray> dynamicProperties = propertyObject->dynamicPropertyNames();
        QList<QObject*> childs = parentItem->parent()->children();
        int removed = 0;
        for(int i = 0; i < childs.count(); ++i )
        {
            QObject* obj = childs[i];
            if( !obj->property("__Dynamic").toBool() || dynamicProperties.contains( obj->objectName().toLocal8Bit() ) )
                continue;
            beginRemoveRows(itemIndex.parent(), i - removed, i - removed);
            ++removed;
            delete obj;
            endRemoveRows();
        }
        addDynamicProperties(static_cast<Property*>(parentItem->parent()), propertyObject);
    }
}

void QPropertyModel::addDynamicProperties( Property* parent, QObject* propertyObject )
{
    // Get dynamic property names
    QList<QByteArray> dynamicProperties = propertyObject->dynamicPropertyNames();

    QList<QObject*> childs = parent->children();

    // Remove already existing properties from list
    for(int i = 0; i < childs.count(); ++i )
    {
        if( !childs[i]->property("__Dynamic").toBool() ) continue;

        int index = dynamicProperties.indexOf( childs[i]->objectName().toLocal8Bit() );
        if( index != -1)
        {
            dynamicProperties.removeAt(index);
            continue;
        }
    }

    // Remove invalid properites and those we don't want to add
    for(int i = 0; i < dynamicProperties.size(); ++i )
    {
        QString dynProp = dynamicProperties[i];
        // Skip properties starting with _ (because there may be dynamic properties from Qt with _q_ and we may
        // have user defined hidden properties starting with _ too
        if( dynProp.startsWith("_") || !propertyObject->property( qPrintable(dynProp) ).isValid() )
        {
            dynamicProperties.removeAt(i);
            --i;
        }
    }

    if( dynamicProperties.empty() ) return;

    QModelIndex parentIndex = createIndex(parent->row(), 0, static_cast<Property*>(parent));
    int rows = rowCount(parentIndex);
    beginInsertRows(parentIndex, rows, rows + dynamicProperties.count() - 1 );
    // Add properties left in the list
    foreach(QByteArray dynProp, dynamicProperties )
    {
        QVariant v = propertyObject->property(dynProp);
        Property* p = 0;
        if( v.type() == QVariant::UserType && !m_userCallbacks.isEmpty() )
        {
            QList<QPropertyEditorWidget::UserTypeCB>::iterator iter = m_userCallbacks.begin();
            while( p == 0 && iter != m_userCallbacks.end() )
            {
                p = (*iter)(dynProp, propertyObject, parent);
                ++iter;
            }
        }
        if( p == 0 ) p = new Property(dynProp, propertyObject, parent);
        p->setProperty("__Dynamic", true);
    }
    endInsertRows();
}

void QPropertyModel::clear()
{
    beginRemoveRows(QModelIndex(), 0, rowCount());
    delete m_rootItem;
    m_rootItem = new Property("Root",0, this);
    endRemoveRows();
}

void QPropertyModel::registerCustomPropertyCB(QPropertyEditorWidget::UserTypeCB callback)
{
    if ( !m_userCallbacks.contains(callback) )
        m_userCallbacks.push_back(callback);
}

void QPropertyModel::unregisterCustomPropertyCB(QPropertyEditorWidget::UserTypeCB callback)
{
    int index = m_userCallbacks.indexOf(callback);
    if( index != -1 )
        m_userCallbacks.removeAt(index);
}

void QPropertyModel::setSorted(bool value)
{
    if(value != m_sorted)
    {
        beginResetModel();
        m_sorted = value;
        endResetModel();
    }
}
