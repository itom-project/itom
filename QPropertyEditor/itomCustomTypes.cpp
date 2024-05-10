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

#include "itomCustomTypes.h"

#include "../common/interval.h"

#include "autoIntervalProperty.h"
#include "qVector2DProperty.h"
#include "qVector3DProperty.h"
#include "qVector4DProperty.h"

#include "Property.h"

#include <qmetatype.h>


namespace ito {
namespace itomCustomTypes {

//-------------------------------------------------------------------------------------
void registerTypes()
{
    static bool registered = false;
    if (!registered)
    {
        qRegisterMetaType<ito::AutoInterval>("ito::AutoInterval");
        qRegisterMetaType<QVector2D>("QVector2D");
        qRegisterMetaType<QVector3D>("QVector3D");
        qRegisterMetaType<QVector4D>("QVector4D");
        registered = true;
    }
}

//-------------------------------------------------------------------------------------
Property* createCustomProperty(const QString& name, QObject* propertyObject, Property* parent)
{
    int userType = propertyObject->property(qPrintable(name)).userType();
    if (userType == QMetaType::type("ito::AutoInterval"))
    {
        return new AutoIntervalProperty(name, propertyObject, parent);
    }
    else if (userType == QMetaType::type("QVector2D"))
    {
        return new QVector2DProperty(name, propertyObject, parent);
    }
    else if (userType == QMetaType::type("QVector3D"))
    {
        return new QVector3DProperty(name, propertyObject, parent);
    }
    else if (userType == QMetaType::type("QVector4D"))
    {
        return new QVector4DProperty(name, propertyObject, parent);
    }
    else
    {
        return nullptr;
    }
}

} // end namespace itomCustomTypes
} // namespace ito
