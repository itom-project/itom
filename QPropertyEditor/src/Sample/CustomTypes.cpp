// *************************************************************************************************
//
// This code is part of the Sample Application to demonstrate the use of the QPropertyEditor library.
// It is distributed as public domain and can be modified and used by you without any limitations.
//
// Your feedback is welcome!
//
// Author: Volker Wiendl
// Enum enhancement by Roman alias banal from qt-apps.org
// *************************************************************************************************


#include "CustomTypes.h"

#include "Vec3fProperty.h"

#include <QPropertyEditor/Property.h>

namespace CustomTypes
{
    void registerTypes()
    {
        static bool registered = false;
        if (!registered)
        {
            qRegisterMetaType<Vec3f>("Vec3f");
            registered = true;
        }
    }

    Property* createCustomProperty(const QString& name, QObject* propertyObject, Property* parent)
    {
        int userType = propertyObject->property(qPrintable(name)).userType();
        if (userType == QMetaType::type("Vec3f"))
            return new Vec3fProperty(name, propertyObject, parent);
        else
            return new Property(name, propertyObject, parent);
    }
}
