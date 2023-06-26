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
#ifndef __ENUMPROPERTY_H__
#define __ENUMPROPERTY_H__

#include <qstringlist.h>
#include <qmetaobject.h>
#include <qmetatype.h>
#include <qcombobox.h>
#include "Property.h"

/**
    The Enum Property Class extends a Property to add enum functionality to the
    QPropertyEditor.
    Enum Properties are automatically created in the QPropertyModel for objects that
    have an enum as property value.

    @author Roman Schmid
*/
class EnumProperty : public Property
{
    Q_OBJECT

public:
    EnumProperty(const QString& name = QString(), QObject* propertyObject = 0, QObject* parent = 0);

    /** @see Property::value */
    virtual QVariant value(int role = Qt::UserRole) const;
    /** @see Property::createEditor */
    virtual QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem& option);
    /** @see Property::setEditorData */
    virtual bool setEditorData(QWidget *editor, const QVariant& data);
    /** @see Property::editorData */
    virtual QVariant editorData(QWidget *editor);

private slots:
    /** slot that is being called by the editor widget */
    void valueChanged(const QString item);

private:
    /** QStringList with possible enum values */
    QStringList m_enum;
};
#endif
