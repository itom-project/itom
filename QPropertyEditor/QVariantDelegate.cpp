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

#include "QVariantDelegate.h"

#include "Property.h"

#include <qabstractitemview.h>
#include <qsignalmapper.h>
#include <qsortfilterproxymodel.h>
#include <qregularexpression.h>

//-------------------------------------------------------------------------------------
QVariantDelegate::QVariantDelegate(QObject* parent) : QItemDelegate(parent), m_finishedMapper(nullptr)
{
    m_finishedMapper = new QSignalMapper(this);

#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    connect(m_finishedMapper, &QSignalMapper::mappedObject, this,
        [=](QObject* obj)
        {
            QWidget* wid = qobject_cast<QWidget*>(obj);
            if (wid)
            {
                commitData(wid);
                closeEditor(wid);
            }
        }
    );
#else
    connect(m_finishedMapper, SIGNAL(mapped(QWidget*)), this, SIGNAL(commitData(QWidget*)));
    connect(m_finishedMapper, SIGNAL(mapped(QWidget*)), this, SIGNAL(closeEditor(QWidget*)));
#endif
}

//-------------------------------------------------------------------------------------
QVariantDelegate::~QVariantDelegate()
{
}

//-------------------------------------------------------------------------------------
Property* QVariantDelegate::propertyFromModel(const QModelIndex &index) const
{
    // if the index is based on QSortFilterProxyModel, its internalPointer
    // is used differently. Therefore, this method is required.
    const QSortFilterProxyModel *proxyModel = qobject_cast<const QSortFilterProxyModel*>(index.model());

    if (proxyModel)
    {
        auto srcIndex = proxyModel->mapToSource(index);
        return static_cast<Property*>(srcIndex.internalPointer());
    }

    return static_cast<Property*>(index.internalPointer());
}

//-------------------------------------------------------------------------------------
QWidget* QVariantDelegate::createEditor(
    QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    QWidget* editor = 0;
    Property* p = propertyFromModel(index);

    switch (p->value().type())
    {
    case QVariant::Bool:
    case QVariant::Color:
    case QVariant::Font:
    case QVariant::StringList:
    case QVariant::Int:
    case QMetaType::Float:
    case QVariant::Double:
    case QVariant::UserType:
        editor = p->createEditor(parent, option);
        if (editor)
        {
            if (editor->metaObject()->indexOfSignal("editFinished()") != -1)
            {
                //connect(editor &QWidget::edit)
                connect(editor, SIGNAL(editFinished()), m_finishedMapper, SLOT(map()));
                m_finishedMapper->setMapping(editor, editor);
            }
            break; // if no editor could be created take default case
        }
    default:
        editor = QItemDelegate::createEditor(parent, option, index);
    }
    parseEditorHints(editor, p->editorHints());
    return editor;
}


//-------------------------------------------------------------------------------------
void QVariantDelegate::setEditorData(QWidget* editor, const QModelIndex& index) const
{
    m_finishedMapper->blockSignals(true);
    QVariant data = index.model()->data(index, Qt::EditRole);

    switch (data.type())
    {
    case QVariant::Bool:
    case QVariant::Color:
    case QVariant::Font:
    case QVariant::StringList:
    case QMetaType::Double:
    case QMetaType::Float:
    case QVariant::UserType:
    case QVariant::Int:
        if (propertyFromModel(index)->setEditorData(editor, data))
        {
            break;
        }
    default:
        // if editor couldn't be recognized use default
        QItemDelegate::setEditorData(editor, index);
        break;
    }
    m_finishedMapper->blockSignals(false);
}

//-------------------------------------------------------------------------------------
void QVariantDelegate::setModelData(
    QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const
{
    QVariant data = index.model()->data(index, Qt::EditRole);
    switch (data.type())
    {
    case QVariant::Bool:
    case QVariant::Color:
    case QVariant::Font:
    case QVariant::StringList:
    case QMetaType::Double:
    case QMetaType::Float:
    case QVariant::UserType:
    case QVariant::Int: {
        QVariant data = propertyFromModel(index)->editorData(editor);

        if (data.isValid())
        {
            model->setData(index, data, Qt::EditRole);
            break;
        }
    }
    default:
        QItemDelegate::setModelData(editor, model, index);
        break;
    }
}

//-------------------------------------------------------------------------------------
void QVariantDelegate::updateEditorGeometry(
    QWidget* editor, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    return QItemDelegate::updateEditorGeometry(editor, option, index);
}

//-------------------------------------------------------------------------------------
QSize QVariantDelegate::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    // height control of each row. \Todo: it would be nice, if the minimum height of 18px would be
    // extracted from the os-dependent height of a widget (like lineEdit, comboBox...). Until now,
    // the minimum height is fixed to 18px.
    QSize sizeHint(QItemDelegate::sizeHint(option, index));
    sizeHint.rheight() = qMax(sizeHint.rheight(), 18);
    return sizeHint;
}

//-------------------------------------------------------------------------------------
void QVariantDelegate::parseEditorHints(QWidget* editor, const QString& editorHints) const
{
    if (editor && !editorHints.isEmpty())
    {
        editor->blockSignals(true);

        // Parse for property values
        QStringList hintList = editorHints.split(";");
        QString hintTrimmed;
        QString pattern = QString("^(.*)(=\\s*)(.*)$");
        QRegularExpression rx(pattern);
        QRegularExpressionMatch match;
        QString name, value;

        foreach(const QString &hint, hintList)
        {
            hintTrimmed = hint.trimmed();

            if (hintTrimmed != "")
            {
                if ((match = rx.match(hintTrimmed)).hasMatch())
                {
                    name = match.captured(1).trimmed();
                    value = match.captured(4).trimmed();
                    editor->setProperty(qPrintable(name), value);
                }
            }
        }

        editor->blockSignals(false);
    }
}
