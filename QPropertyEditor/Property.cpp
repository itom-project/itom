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
#include "Property.h"
#include "BooleanCombo.h"
#include "ColorCombo.h"
#include "FontEditor.h"
#include "stringListEditor.h"

#include <qmetaobject.h>
#include <qspinbox.h>

//-------------------------------------------------------------------------------------
Property::Property(
    const QString& name /*= QString()*/, QObject* propertyObject /*= 0*/, QObject* parent /*= 0*/) :
    QObject(parent),
    m_propertyObject(propertyObject), m_enabled(true)
{
    setObjectName(name);
}

//-------------------------------------------------------------------------------------
void Property::setEnabled(bool enabled)
{
    m_enabled = enabled;
}

//-------------------------------------------------------------------------------------
QVariant Property::value(int /*role = Qt::UserRole*/) const
{
    if (m_propertyObject)
        return m_propertyObject->property(qPrintable(objectName()));
    else
        return QVariant();
}

//-------------------------------------------------------------------------------------
QString Property::displayValue(int role /*= Qt::UserRole*/) const
{
    QVariant v = value(role);
    if (v.type() == QVariant::StringList)
    {
        QStringList stringlist = v.toStringList();

        switch (stringlist.count())
        {
        case 0:
            return "[]";
        case 1:
            return QString("[%1]").arg(stringlist[0]);
        case 2:
            return QString("[%1; %2]").arg(stringlist[0]).arg(stringlist[1]);
        case 3:
            return QString("[%1; %2; %3]").arg(stringlist[0]).arg(stringlist[1]).arg(stringlist[2]);
        default:
            return QString("[%1; %2; %3; ...]")
                .arg(stringlist[0])
                .arg(stringlist[1])
                .arg(stringlist[2]);
        }
    }
    else
    {
        switch (v.type())
        {
        case QMetaType::Float: {
            QLocale defaultLocale;
            return defaultLocale.toString(v.value<float>());
        }
        case QMetaType::Double: {
            QLocale defaultLocale;
            return defaultLocale.toString(v.value<double>());
        }
        default:
            return v.toString();
        }
    }
}

//-------------------------------------------------------------------------------------
void Property::setValue(const QVariant& value)
{
    if (m_propertyObject)
        m_propertyObject->setProperty(qPrintable(objectName()), value);
}

//-------------------------------------------------------------------------------------
bool Property::isReadOnly()
{
    if (!m_enabled)
    {
        return true;
    }
    else if (m_propertyObject->dynamicPropertyNames().contains(objectName().toLocal8Bit()))
    {
        return false;
    }
    if (m_propertyObject &&
        m_propertyObject->metaObject()
            ->property(m_propertyObject->metaObject()->indexOfProperty(qPrintable(objectName())))
            .isWritable())
    {
        return false;
    }
    else
    {
        return true;
    }
}

//-------------------------------------------------------------------------------------
QWidget* Property::createEditor(QWidget* parent, const QStyleOptionViewItem& /*option*/)
{
    // TODO: QMetaType::Float not in enumerated type
    QWidget* editor = nullptr;

    switch (value().type())
    {
    case QVariant::Bool:
        editor = new BooleanCombo(parent);
        connect(editor, SIGNAL(boolChanged(bool)), this, SLOT(setValue(bool)));
        break;
    case QVariant::Color:
        editor = new ColorCombo(parent);
        connect(editor, SIGNAL(colorChanged(const QColor)), this, SLOT(setValue(const QColor)));
        break;
    case QVariant::Int:
        editor = new QSpinBox(parent);
        editor->setProperty("minimum", -INT_MAX);
        editor->setProperty("maximum", INT_MAX);
        ((QAbstractSpinBox*)editor)->setKeyboardTracking(false);
        connect(editor, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));
        break;
    case QMetaType::Float:
    case QVariant::Double:
        editor = new QDoubleSpinBox(parent);
        editor->setProperty("minimum", -INT_MAX);
        editor->setProperty("maximum", INT_MAX);
        editor->setProperty("decimals", 4);
        ((QAbstractSpinBox*)editor)->setKeyboardTracking(false);
        connect(editor, SIGNAL(valueChanged(double)), this, SLOT(setValue(double)));
        break;
    case QVariant::Font:
        editor = new FontEditor(parent);
        connect(editor, SIGNAL(fontChanged(QFont)), this, SLOT(setValue(QFont)));
        break;
    case QVariant::StringList:
        editor = new StringListEditor(parent);
        connect(editor, SIGNAL(stringListChanged(QStringList)), this, SLOT(setValue(QStringList)));
        break;
    default:
        return editor;
    }
    return editor;
}

//-------------------------------------------------------------------------------------
bool Property::setEditorData(QWidget* editor, const QVariant& data)
{
    // TODO: QMetaType::Float not in enumerated type
    switch (value().type())
    {
    case QVariant::Color:
        static_cast<ColorCombo*>(editor)->setColor(data.value<QColor>());
        return true;
    case QVariant::Bool:
        static_cast<BooleanCombo*>(editor)->setValue(data.toBool());
        return true;
    case QVariant::Int:
        editor->blockSignals(true);
        static_cast<QSpinBox*>(editor)->setValue(data.toInt());
        editor->blockSignals(false);
        return true;
    case QMetaType::Float:
    case QVariant::Double:
        editor->blockSignals(true);
        static_cast<QDoubleSpinBox*>(editor)->setValue(data.toDouble());
        editor->blockSignals(false);
        return true;
    case QVariant::Font:
        static_cast<FontEditor*>(editor)->setValue(data.value<QFont>());
        return true;
    case QVariant::StringList:
        static_cast<StringListEditor*>(editor)->setValue(data.toStringList());
        return true;
    default:
        return false;
    }
    return false;
}

//-------------------------------------------------------------------------------------
QVariant Property::editorData(QWidget* editor)
{
    // TODO: QMetaType::Float not in enumerated type
    switch (value().type())
    {
    case QVariant::Color:
        return QVariant::fromValue(static_cast<ColorCombo*>(editor)->color());
    case QVariant::Bool:
        return QVariant(static_cast<BooleanCombo*>(editor)->value());
    case QVariant::Int:
        return QVariant(static_cast<QSpinBox*>(editor)->value());
    case QMetaType::Float:
    case QVariant::Double:
        return QVariant(static_cast<QDoubleSpinBox*>(editor)->value());
    case QVariant::Font:
        return QVariant(static_cast<FontEditor*>(editor)->value());
    case QVariant::StringList:
        return QVariant(static_cast<StringListEditor*>(editor)->value());
    default:
        return QVariant();
    }
}

//-------------------------------------------------------------------------------------
Property* Property::findPropertyObject(QObject* propertyObject)
{
    if (m_propertyObject == propertyObject)
    {
        return this;
    }

    for (int i = 0; i < children().size(); ++i)
    {
        Property* child = static_cast<Property*>(children()[i])->findPropertyObject(propertyObject);
        if (child)
        {
            return child;
        }
    }
    return 0;
}

//-------------------------------------------------------------------------------------
void Property::setValue(double value)
{
    setValue(QVariant(value));
}

//-------------------------------------------------------------------------------------
void Property::setValue(int value)
{
    setValue(QVariant(value));
}

//-------------------------------------------------------------------------------------
void Property::setValue(QColor value)
{
    setValue(QVariant(value));
}

//-------------------------------------------------------------------------------------
void Property::setValue(bool value)
{
    setValue(QVariant(value));
}

//-------------------------------------------------------------------------------------
void Property::setValue(QFont value)
{
    setValue(QVariant(value));
}

//-------------------------------------------------------------------------------------
void Property::setValue(QStringList value)
{
    setValue(QVariant(value));
}
