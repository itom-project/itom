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

#include "ColorCombo.h"

#include <qcolordialog.h>
#include <qdebug.h>

ColorCombo::ColorCombo(QWidget* parent /*= 0*/) : QComboBox(parent)
{
    QStringList colorNames = QColor::colorNames();
    for (int i = 0; i < colorNames.size(); ++i) {
        QColor color(colorNames[i]);
        insertItem(i, colorNames[i]);
        setItemData(i, color, Qt::DecorationRole);
    }
    addItem(tr("Custom"), QVariant((int)QVariant::UserType));
    connect(this, SIGNAL(currentIndexChanged(int)), this, SLOT(currentChanged(int)));
}


ColorCombo::~ColorCombo()
{
}


QColor ColorCombo::color() const
{
    return qVariantValue<QColor>(itemData(currentIndex(), Qt::DecorationRole));
}

void ColorCombo::setColor(QColor color)
{
    m_init = color;
    setCurrentIndex(findData(color, int(Qt::DecorationRole)));
    if (currentIndex() == -1)
    {
        addItem(color.name());
        setItemData(count()-1, color, Qt::DecorationRole);
        setCurrentIndex(count()-1);
    }
}

void ColorCombo::currentChanged(int index)
{
    qDebug() << "currentChanged: " << index;
    if (itemData(index).isValid() && itemData(index) == QVariant((int)QVariant::UserType))
    {
        QColor color = QColorDialog::getColor(m_init, this);
        if (color.isValid())
        {
            if (findData(color, int(Qt::DecorationRole)) == -1)
            {
                addItem(color.name());
                setItemData(count()-1, color, Qt::DecorationRole);
            }
            setCurrentIndex(findData(color, int(Qt::DecorationRole)));
            emit colorChanged(color);
        }
        else
        {
            setCurrentIndex(findData(m_init));
            emit colorChanged(m_init);
        }
    }
    else
    {
        emit colorChanged( qVariantValue<QColor>(itemData(currentIndex(), Qt::DecorationRole)) );
    }
}
