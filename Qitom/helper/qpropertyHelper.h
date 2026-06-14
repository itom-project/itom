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

#ifndef QPROPERTYHELPER
#define QPROPERTYHELPER

#include "../global.h"
#include "../common/retVal.h"
#include "../common/addInInterface.h"
#include "../common/interval.h"
#include "../common/shape.h"
#include "../DataObject/dataobj.h"
#include "../common/qtMetaTypeDeclarations.h"
#include "common/itomPlotHandle.h"

#if ITOM_POINTCLOUDLIBRARY > 0
    #include "PointCloud/pclStructures.h"
#endif

#include <qvariant.h>
#include <qmetaobject.h>
#include <qpointer.h>
#include <qsharedpointer.h>

#if ITOM_POINTCLOUDLIBRARY > 0
    Q_DECLARE_METATYPE(ito::PCLPointCloud)
    Q_DECLARE_METATYPE(QSharedPointer<ito::PCLPointCloud>)
    Q_DECLARE_METATYPE(QSharedPointer<ito::PCLPolygonMesh>)
    Q_DECLARE_METATYPE(ito::PCLPoint)
    Q_DECLARE_METATYPE(ito::PCLPolygonMesh)
#endif

Q_DECLARE_METATYPE(QSharedPointer<ito::DataObject>)
Q_DECLARE_METATYPE(QPointer<ito::AddInBase>)
Q_DECLARE_METATYPE(QPointer<ito::AddInDataIO>)
Q_DECLARE_METATYPE(QPointer<ito::AddInActuator>)
Q_DECLARE_METATYPE(ito::Shape)
Q_DECLARE_METATYPE(QVector<ito::Shape>)
Q_DECLARE_METATYPE(ito::ItomPlotHandle)

namespace ito
{

    class QPropertyHelper
    {
    public:

        static QVariant QVariantCast(const QVariant &item, int QVariantCast, ito::RetVal &retval);
        static QVariant QVariantToEnumCast(const QVariant &item, const QMetaEnum &enumerator, ito::RetVal &retval);
        static RetVal readProperty(const QObject *object, const char* propName, QVariant &value);
        static RetVal writeProperty(QObject *object, const char* propName, const QVariant &value);
    };

} //end namespace ito

#endif //QPROPERTYHELPER
