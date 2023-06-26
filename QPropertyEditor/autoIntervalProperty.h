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

#ifndef AUTOINTERVALPROPERTY_H
#define AUTOINTERVALPROPERTY_H

#include <qvariant.h>
#include "Property.h"

class Property;
class QObject;


namespace ito
{
    class QPROPERTYEDITOR_TEST_EXPORT AutoIntervalProperty : public Property
    {
        Q_OBJECT
        Q_PROPERTY(double minimum READ minimum WRITE setMinimum DESIGNABLE true USER true)
        Q_PROPERTY(double maximum READ maximum WRITE setMaximum DESIGNABLE true USER true)
        Q_PROPERTY(bool autoScaling READ autoScaling WRITE setAutoScaling DESIGNABLE true USER true)

    public:
        AutoIntervalProperty(const QString& name = QString(), QObject* propertyObject = 0, QObject* parent = 0);

        QVariant value(int role = Qt::UserRole) const;
        virtual void setValue(const QVariant& value);

        void setEditorHints(const QString& hints);

        double minimum() const;
        void setMinimum(double minimum);

        double maximum() const;
        void setMaximum(double maximum);

        bool autoScaling() const;
        void setAutoScaling(bool autoScaling);

    private:
        QString parseHints(const QString& hints, const QChar component);

        Property*    m_minimum;
        Property*    m_maximum;
        Property*    m_autoScaling;
    };

}
#endif //AUTOINTERVALPROPERTY_H
