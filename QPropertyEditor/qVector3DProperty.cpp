/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2021, Institut fuer Technische Optik (ITO),
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

#include "qVector3DProperty.h"

#include <qregularexpression.h>
#include <qvector3d.h>
#include <qlocale.h>

namespace ito {

//-------------------------------------------------------------------------------------
QVector3DProperty::QVector3DProperty(
    const QString& name /*= QString()*/, QObject* propertyObject /*= 0*/, QObject* parent /*= 0*/) :
    Property(name, propertyObject, parent)
{
    m_x = new Property("x", this, this);
    m_y = new Property("y", this, this);
    m_z = new Property("z", this, this);
    setEditorHints("minimumX=-2147483647;maximumX=2147483647;minimumY=-2147483647;maximumY="
                   "2147483647;minimumZ=-2147483647;maximumZ=2147483647;");
}

//-------------------------------------------------------------------------------------
QVariant QVector3DProperty::value(int role) const
{
    QVariant data = Property::value();
    QLocale defaultLocale;
    QString xStr = defaultLocale.toString(data.value<QVector3D>().x());
    QString yStr = defaultLocale.toString(data.value<QVector3D>().y());
    QString zStr = defaultLocale.toString(data.value<QVector3D>().z());

    if (data.isValid() && role != Qt::UserRole)
    {
        switch (role)
        {
        case Qt::DisplayRole:
            return tr("[%1, %2, %3]").arg(xStr, yStr, zStr);
        case Qt::EditRole:
            return tr("%1, %2, %3").arg(xStr, yStr, zStr);
        };
    }
    return data;
}

//-------------------------------------------------------------------------------------
void QVector3DProperty::setValue(const QVariant& value)
{
    if (value.type() == QVariant::String)
    {
        QString v = value.toString();
        QRegularExpression rx("([+-]?([0-9]*[\\.,])?[0-9]+(e[+-]?[0-9]+)?)", QRegularExpression::CaseInsensitiveOption);
        QLocale defaultLocale;
        int count = 0;
        int pos = 0;
        float x = 0.0f, y = 0.0f, z = 0.0f;
        QRegularExpressionMatch match;

        while ((match = rx.match(v, pos)).hasMatch())
        {
            if (count == 0)
            {
                x = match.captured(1).toDouble();
            }
            else if (count == 1)
            {
                y = match.captured(1).toDouble();
            }
            else if (count == 2)
            {
                z = match.captured(1).toDouble();
            }
            else if (count > 2)
            {
                break;
            }

            ++count;
            pos = match.capturedEnd();
        }

        m_x->setProperty("x", x);
        m_y->setProperty("y", y);
        m_z->setProperty("z", z);
        Property::setValue(QVariant::fromValue(QVector3D(x, y, z)));
    }
    else
    {
        Property::setValue(value);
    }
}

//-------------------------------------------------------------------------------------
void QVector3DProperty::setEditorHints(const QString& hints)
{
    m_x->setEditorHints(parseHints(hints, 'X'));
    m_y->setEditorHints(parseHints(hints, 'Y'));
    m_z->setEditorHints(parseHints(hints, 'Z'));
}

//-------------------------------------------------------------------------------------
float QVector3DProperty::x() const
{
    return value().value<QVector3D>().x();
}

//-------------------------------------------------------------------------------------
void QVector3DProperty::setX(float x)
{
    Property::setValue(QVariant::fromValue(QVector3D(x, y(), z())));
}

//-------------------------------------------------------------------------------------
float QVector3DProperty::y() const
{
    return value().value<QVector3D>().y();
}

//-------------------------------------------------------------------------------------
void QVector3DProperty::setY(float y)
{
    Property::setValue(QVariant::fromValue(QVector3D(x(), y, z())));
}

//-------------------------------------------------------------------------------------
float QVector3DProperty::z() const
{
    return value().value<QVector3D>().z();
}

//-------------------------------------------------------------------------------------
void QVector3DProperty::setZ(float z)
{
    Property::setValue(QVariant::fromValue(QVector3D(x(), y(), z)));
}

//-------------------------------------------------------------------------------------
QString QVector3DProperty::parseHints(const QString& hints, const QChar component)
{
    QStringList hintList = hints.split(";");
    QString hintTrimmed;
    QString pattern = QString("^(.*)(%1)(=\\s*)(.*)$").arg(component);
    QRegularExpression rx(pattern);
    QRegularExpressionMatch match;
    QStringList componentHints;
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
                componentHints += QString("%1=%2").arg(name).arg(value);
            }
        }
    }

    return componentHints.join(";");
}

} // end namespace ito
