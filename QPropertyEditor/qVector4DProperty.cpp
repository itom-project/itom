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


#include "qVector4DProperty.h"

#include <qregularexpression.h>
#include <qlocale.h>

namespace ito {

//-------------------------------------------------------------------------------------
QVector4DProperty::QVector4DProperty(
    const QString& name /*= QString()*/, QObject* propertyObject /*= 0*/, QObject* parent /*= 0*/) :
    Property(name, propertyObject, parent)
{
    m_x = new Property("x", this, this);
    m_y = new Property("y", this, this);
    m_z = new Property("z", this, this);
    m_w = new Property("w", this, this);
    setEditorHints(
        "minimumX=-2147483647;maximumX=2147483647;minimumY=-2147483647;maximumY=2147483647;"
        "minimumZ=-2147483647;maximumZ=2147483647;minimumW=-2147483647;maximumW=2147483647;");
}

//-------------------------------------------------------------------------------------
QVariant QVector4DProperty::value(int role) const
{
    QVariant data = Property::value();
    QLocale defaultLocale;
    QString xStr = defaultLocale.toString(data.value<QVector4D>().x());
    QString yStr = defaultLocale.toString(data.value<QVector4D>().y());
    QString zStr = defaultLocale.toString(data.value<QVector4D>().z());
    QString wStr = defaultLocale.toString(data.value<QVector4D>().w());

    if (data.isValid() && role != Qt::UserRole)
    {
        switch (role)
        {
        case Qt::DisplayRole:
            return tr("[%1, %2, %3, %4]").arg(xStr, yStr, zStr, wStr);
        case Qt::EditRole:
            return tr("%1, %2, %3, %4").arg(xStr, yStr, zStr, wStr);
        };
    }
    return data;
}

//-------------------------------------------------------------------------------------
void QVector4DProperty::setValue(const QVariant& value)
{
    if (value.type() == QVariant::String)
    {
        QString v = value.toString();
        QRegularExpression rx("([+-]?([0-9]*[\\.,])?[0-9]+(e[+-]?[0-9]+)?)", QRegularExpression::CaseInsensitiveOption);
        QLocale defaultLocale;
        int count = 0;
        int pos = 0;
        float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;
        QRegularExpressionMatch match;

        while ((match = rx.match(v, pos)).hasMatch())
        {
            if (count == 0)
            {
                x = defaultLocale.toDouble(match.captured(1));
            }
            else if (count == 1)
            {
                y = defaultLocale.toDouble(match.captured(1));
            }
            else if (count == 2)
            {
                z = defaultLocale.toDouble(match.captured(1));
            }
            else if (count == 3)
            {
                w = defaultLocale.toDouble(match.captured(1));
            }
            else if (count > 3)
            {
                break;
            }

            ++count;
            pos = match.capturedEnd();
        }

        m_x->setProperty("x", x);
        m_y->setProperty("y", y);
        m_z->setProperty("z", z);
        m_w->setProperty("w", w);
        Property::setValue(QVariant::fromValue(QVector4D(x, y, z, w)));
    }
    else
    {
        Property::setValue(value);
    }
}

//-------------------------------------------------------------------------------------
void QVector4DProperty::setEditorHints(const QString& hints)
{
    m_x->setEditorHints(parseHints(hints, 'X'));
    m_y->setEditorHints(parseHints(hints, 'Y'));
    m_z->setEditorHints(parseHints(hints, 'Z'));
    m_w->setEditorHints(parseHints(hints, 'W'));
}

//-------------------------------------------------------------------------------------
float QVector4DProperty::x() const
{
    return value().value<QVector4D>().x();
}

//-------------------------------------------------------------------------------------
void QVector4DProperty::setX(float x)
{
    Property::setValue(QVariant::fromValue(QVector4D(x, y(), z(), w())));
}

//-------------------------------------------------------------------------------------
float QVector4DProperty::y() const
{
    return value().value<QVector4D>().y();
}

//-------------------------------------------------------------------------------------
void QVector4DProperty::setY(float y)
{
    Property::setValue(QVariant::fromValue(QVector4D(x(), y, z(), w())));
}

//-------------------------------------------------------------------------------------
float QVector4DProperty::z() const
{
    return value().value<QVector4D>().z();
}

//-------------------------------------------------------------------------------------
void QVector4DProperty::setZ(float z)
{
    Property::setValue(QVariant::fromValue(QVector4D(x(), y(), z, w())));
}

//-------------------------------------------------------------------------------------
float QVector4DProperty::w() const
{
    return value().value<QVector4D>().w();
}

//-------------------------------------------------------------------------------------
void QVector4DProperty::setW(float w)
{
    Property::setValue(QVariant::fromValue(QVector4D(x(), y(), z(), w)));
}

//-------------------------------------------------------------------------------------
QString QVector4DProperty::parseHints(const QString& hints, const QChar component)
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
