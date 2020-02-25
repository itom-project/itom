/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom.
  
    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef WIDGETWRAPPER_H
#define WIDGETWRAPPER_H

#include "../python/pythonItomMetaObject.h"

#include <qobject.h>
#include <qhash.h>
#include <qmetaobject.h>
#include "../global.h"

namespace ito
{

class WidgetWrapper
{
public:
    WidgetWrapper(); //constructor
    ~WidgetWrapper(); //destructor

    MethodDescriptionList getMethodList(QObject *object);
    ito::RetVal call(QObject *object, int methodIndex, void **_a);

    QMetaProperty fakeProperty(const QObject *baseObject, const QString &fakePropertyName, QObject **destinationObject);

private:
    void initMethodHash();
    MethodDescription buildMethodDescription(QByteArray signature, QString retType, int methodIndex, bool &ok);

    QHash<QString, MethodDescriptionList> methodHash; /*!< Hash-table containing a list of method description for all public methods of a class derived from QObject which should be accessed by the call method of WidgetWrapper at runtime. */
    bool initialized; /*!< member indicating whether the initMethodHash method already has been executed, which is done in the constructor of WidgetWrapper. */
};

} //end namespace ito

#endif

