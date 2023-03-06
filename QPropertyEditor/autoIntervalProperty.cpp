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

#include "autoIntervalProperty.h"

#include "../common/interval.h"

#include <qmetatype.h>
#include <qregularexpression.h>
#include <qlocale.h>

Q_DECLARE_METATYPE(ito::AutoInterval)

namespace ito {

//-------------------------------------------------------------------------------------
AutoIntervalProperty::AutoIntervalProperty(
    const QString& name /*= QString()*/, QObject* propertyObject /*= 0*/, QObject* parent /*= 0*/) :
    Property(name, propertyObject, parent)
{
    m_minimum = new Property("minimum", this, this);
    m_maximum = new Property("maximum", this, this);
    m_autoScaling = new Property("autoScaling", this, this);
    // m_autoScaling->setInfo("auto Scaling");

    ito::AutoInterval ai =
        propertyObject->property(name.toLatin1().data()).value<ito::AutoInterval>();
    m_minimum->setEnabled(!ai.isAuto());
    m_maximum->setEnabled(!ai.isAuto());
    // setEditorHints("minimum=-2147483647;maximumX=2147483647;minimumY=-2147483647;maximumY=2147483647;minimumZ=-2147483647;maximumZ=2147483647;");
}

//-------------------------------------------------------------------------------------
QVariant AutoIntervalProperty::value(int role) const
{
    QVariant data = Property::value();
    QLocale defaultLocale;

    if (data.isValid() && role != Qt::UserRole)
    {
        ito::AutoInterval ai = qvariant_cast<ito::AutoInterval>(data);
        QString minStr = defaultLocale.toString(ai.minimum());
        QString maxStr = defaultLocale.toString(ai.maximum());

        switch (role)
        {
        case Qt::DisplayRole:
            if (ai.isAuto())
            {
                return tr("auto");
            }
            else
            {
                return tr("[%1; %2]").arg(minStr, maxStr);
            }
        case Qt::EditRole:
            if (ai.isAuto())
            {
                return tr("auto");
            }
            else
            {
                return tr("%1; %2").arg(minStr, maxStr);
            }
        };
    }
    return data;
}

//-------------------------------------------------------------------------------------
void AutoIntervalProperty::setValue(const QVariant& value)
{
    if (value.type() == QVariant::String)
    {
        QString v = value.toString();
        bool autoScaling;
        double min = minimum();
        double max = maximum();
        QLocale defaultLocale;

        if (QString::compare(v, "auto", Qt::CaseInsensitive) == 0 ||
            QString::compare(v, "<auto>", Qt::CaseInsensitive) == 0)
        {
            autoScaling = true;
        }
        else
        {
            autoScaling = false;
            QRegularExpression rx("([+-]?([0-9]*[\\.,])?[0-9]+(e[+-]?[0-9]+)?)", QRegularExpression::CaseInsensitiveOption);
            int count = 0;
            int pos = 0;
            QRegularExpressionMatch match;

            while ((match = rx.match(v, pos)).hasMatch())
            {
                if (count == 0)
                {
                    min = defaultLocale.toDouble(match.captured(1));
                }
                else if (count == 1)
                {
                    max = defaultLocale.toDouble(match.captured(1));
                }
                else if (count > 1)
                {
                    break;
                }

                ++count;
                pos = match.capturedEnd();
            }
        }

        m_minimum->setProperty("minimum", min);
        m_maximum->setProperty("maximum", max);
        m_autoScaling->setProperty("autoScaling", autoScaling);
        m_minimum->setEnabled(!autoScaling);
        m_maximum->setEnabled(!autoScaling);
        Property::setValue(QVariant::fromValue(ito::AutoInterval(min, max, autoScaling)));
    }
    else if (value.userType() == QMetaType::type("ito::AutoInterval"))
    {
        ito::AutoInterval ai = value.value<ito::AutoInterval>();
        m_minimum->setProperty("minimum", ai.minimum());
        m_maximum->setProperty("maximum", ai.maximum());
        m_minimum->setEnabled(!ai.isAuto());
        m_maximum->setEnabled(!ai.isAuto());
        m_autoScaling->setProperty("autoScaling", ai.isAuto());
        Property::setValue(value);
    }
    else
    {
        Property::setValue(value);
    }
}

//-------------------------------------------------------------------------------------
void AutoIntervalProperty::setEditorHints(const QString& hints)
{
    m_minimum->setEditorHints(""); // parseHints(hints, 'X'));
    m_maximum->setEditorHints(""); // parseHints(hints, 'Y'));
    m_autoScaling->setEditorHints(""); // parseHints(hints, 'Z'));
}

//-------------------------------------------------------------------------------------
double AutoIntervalProperty::minimum() const
{
    return value().value<ito::AutoInterval>().minimum();
}

//-------------------------------------------------------------------------------------
void AutoIntervalProperty::setMinimum(double minimum)
{
    AutoIntervalProperty::setValue(
        QVariant::fromValue(ito::AutoInterval(minimum, maximum(), autoScaling())));
}

//-------------------------------------------------------------------------------------
double AutoIntervalProperty::maximum() const
{
    return value().value<ito::AutoInterval>().maximum();
}

//-------------------------------------------------------------------------------------
void AutoIntervalProperty::setMaximum(double maximum)
{
    AutoIntervalProperty::setValue(
        QVariant::fromValue(ito::AutoInterval(minimum(), maximum, autoScaling())));
}

//-------------------------------------------------------------------------------------
bool AutoIntervalProperty::autoScaling() const
{
    return value().value<ito::AutoInterval>().isAuto();
}

//-------------------------------------------------------------------------------------
void AutoIntervalProperty::setAutoScaling(bool autoScaling)
{
    AutoIntervalProperty::setValue(
        QVariant::fromValue(ito::AutoInterval(minimum(), maximum(), autoScaling)));
}

//-------------------------------------------------------------------------------------
QString AutoIntervalProperty::parseHints(const QString& hints, const QChar component)
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
                name = match.captured(1);
                value = match.captured(4);
                componentHints += QString("%1=%2").arg(name).arg(value);
            }
        }
    }

    return componentHints.join(";");
}

} // namespace ito
