/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */


#include "qVector2DProperty.h"

#include <qvector2d.h>
#include <qregexp.h>

namespace ito
{
    QVector2DProperty::QVector2DProperty(const QString& name /*= QString()*/, QObject* propertyObject /*= 0*/, QObject* parent /*= 0*/) : Property(name, propertyObject, parent)
    {
        m_x = new Property("x", this, this);
        m_y = new Property("y", this, this);
        setEditorHints("minimumX=-2147483647;maximumX=2147483647;minimumY=-2147483647;maximumY=2147483647");
    }

    QVariant QVector2DProperty::value(int role) const
    {
        QVariant data = Property::value();
        if (data.isValid() && role != Qt::UserRole)
        {
            switch (role)
            {
            case Qt::DisplayRole:
                return tr("[%1, %2]").arg(data.value<QVector2D>().x()).arg(data.value<QVector2D>().y());
            case Qt::EditRole:
                return tr("%1, %2").arg(data.value<QVector2D>().x()).arg(data.value<QVector2D>().y());
            };
        }
        return data;
    }

    void QVector2DProperty::setValue(const QVariant& value)
    {
        if (value.type() == QVariant::String)
        {
            QString v = value.toString();
            QRegExp rx("([+-]?([0-9]*[\\.,])?[0-9]+(e[+-]?[0-9]+)?)");
            rx.setCaseSensitivity(Qt::CaseInsensitive);
            int count = 0;
            int pos = 0;
            float x = 0.0f, y = 0.0f;
            while ((pos = rx.indexIn(v, pos)) != -1)
            {
                if (count == 0)
                    x = rx.cap(1).toDouble();
                else if (count == 1)
                    y = rx.cap(1).toDouble();
                else if (count > 1)
                    break;
                ++count;
                pos += rx.matchedLength();
            }
            m_x->setProperty("x", x);
            m_y->setProperty("y", y);
            Property::setValue(QVariant::fromValue(QVector2D(x, y)));
        }
        else
            Property::setValue(value);
    }

    void QVector2DProperty::setEditorHints(const QString& hints)
    {
        m_x->setEditorHints(parseHints(hints, 'X'));
        m_y->setEditorHints(parseHints(hints, 'Y'));
    }

    float QVector2DProperty::x() const
    {
        return value().value<QVector2D>().x();
    }

    void QVector2DProperty::setX(float x)
    {
        Property::setValue(QVariant::fromValue(QVector2D(x, y())));
    }

    float QVector2DProperty::y() const
    {
        return value().value<QVector2D>().y();
    }

    void QVector2DProperty::setY(float y)
    {
        Property::setValue(QVariant::fromValue(QVector2D(x(), y)));
    }

    QString QVector2DProperty::parseHints(const QString& hints, const QChar component)
    {
        QRegExp rx(QString("(.*)(") + component + QString("{1})(=\\s*)(.*)(;{1})"));
        rx.setMinimal(true);
        int pos = 0;
        QString componentHints;
        while ((pos = rx.indexIn(hints, pos)) != -1)
        {
            // cut off additional front settings (TODO create correct RegExp for that)
            if (rx.cap(1).lastIndexOf(';') != -1)
                componentHints += QString("%1=%2;").arg(rx.cap(1).remove(0, rx.cap(1).lastIndexOf(';') + 1)).arg(rx.cap(4));
            else
                componentHints += QString("%1=%2;").arg(rx.cap(1)).arg(rx.cap(4));
            pos += rx.matchedLength();
        }
        return componentHints;
    }

} //end namespace ito