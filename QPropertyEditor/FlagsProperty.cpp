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
#include "FlagsProperty.h"

#include "../itomWidgets/checkableComboBox.h"
#include <qdebug.h>


/////////////////////////////////////////////////////////////////////////////////////////////
// Constructor
/////////////////////////////////////////////////////////////////////////////////////////////
FlagsProperty::FlagsProperty(
    const QString& name /* = QString()*/,
    QObject* propertyObject /* = 0*/,
    QObject* parent /* = 0*/) :
    Property(name, propertyObject, parent),
    m_comboBox(NULL), m_inModification(false)
{
    // get the meta property object
    const QMetaObject* meta = propertyObject->metaObject();
    QMetaProperty prop = meta->property(meta->indexOfProperty(qPrintable(name)));

    // if it is indeed an enum type, fill the QStringList member with the keys
    if (prop.isEnumType())
    {
        QMetaEnum qenum = prop.enumerator();
        for (int i = 0; i < qenum.keyCount(); i++)
        {
            m_enum << qenum.key(i);
            m_enumIndices << qenum.value(i);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// value
/////////////////////////////////////////////////////////////////////////////////////////////
QVariant FlagsProperty::value(int role /* = Qt::UserRole */) const
{
    if (role == Qt::DisplayRole)
    {
        if (m_propertyObject)
        {
            // resolve the value to the corresponding enum key
            int index = m_propertyObject->property(qPrintable(objectName())).toInt();

            const QMetaObject* meta = m_propertyObject->metaObject();
            QMetaProperty prop = meta->property(meta->indexOfProperty(qPrintable(objectName())));
            QMetaEnum propenum = prop.enumerator();
            QStringList list;

            for (int idx = 0; idx < propenum.keyCount(); ++idx)
            {
                if (propenum.value(idx) & index)
                {
                    list.append(propenum.key(idx));
                }
            }
            return list.join(";");
        }
        else
        {
            return QVariant();
        }
    }
    else
    {
        return Property::value(role);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// createEditor
/////////////////////////////////////////////////////////////////////////////////////////////
QWidget* FlagsProperty::createEditor(QWidget* parent, const QStyleOptionViewItem& option)
{
    // create a CheckableComboBox and fill it with the QStringList values
    CheckableComboBox* editor = new CheckableComboBox(parent);
    editor->addItems(m_enum);

    connect(editor, SIGNAL(checkedIndexesChanged()), this, SLOT(checkedIndexesChanged()));
    m_comboBox = editor;
    return editor;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// setEditorData
/////////////////////////////////////////////////////////////////////////////////////////////
bool FlagsProperty::setEditorData(QWidget* editor, const QVariant& data)
{
    if (!m_inModification)
    {
        CheckableComboBox* combo = 0;
        // TODO: maybe malformed if statment or put brackets to make gcc happy
        if (combo = qobject_cast<CheckableComboBox*>(editor))
        {
            int value = data.toInt();
            QAbstractItemModel* aim = combo->checkableModel();
            QModelIndex index;
            Qt::CheckState checkState;

            m_inModification = true;

            for (int i = 0; i < m_enumIndices.size(); ++i)
            {
                index = aim->index(i, 0);
                checkState = (value & m_enumIndices[i]) ? Qt::Checked : Qt::Unchecked;
                if (combo->checkState(index) != checkState)
                    combo->setCheckState(aim->index(i, 0), checkState);
            }

            m_inModification = false;
        }
        else
        {
            return false;
        }
    }

    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// editorData
/////////////////////////////////////////////////////////////////////////////////////////////
QVariant FlagsProperty::editorData(QWidget* editor)
{
    CheckableComboBox* combo = 0;
    // TODO: maybe malformed if statment or put brackets to make gcc happy
    if (combo = qobject_cast<CheckableComboBox*>(editor))
    {
        int result = 0;
        QModelIndexList indexList = combo->checkedIndexes();
        foreach (const QModelIndex& idx, indexList)
        {
            result |= m_enumIndices[idx.row()];
        }

        return QVariant(result);
    }
    else
    {
        return QVariant();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// valueChanged
/////////////////////////////////////////////////////////////////////////////////////////////
void FlagsProperty::checkedIndexesChanged()
{
    CheckableComboBox* comboBox = qobject_cast<CheckableComboBox*>(m_comboBox);
    if (comboBox && !m_inModification)
    {
        setValue(editorData(comboBox));
    }
}
