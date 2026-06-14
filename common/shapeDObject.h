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

#ifndef SHAPEDOBJECT_H
#define SHAPEDOBJECT_H

#include "../common/shape.h"
#include "../shape/shapeCommon.h"
#include "typeDefs.h"

#include "../DataObject/dataobj.h"

#if !defined(Q_MOC_RUN) ||                                                                         \
    defined(ITOMSHAPE_MOC) // only moc this file in itomShapeLib but not in other libraries or
                           // executables linking against this itomCommonQtLib

namespace ito {
class ITOMSHAPE_EXPORT ShapeDObject
{
public:
    static ito::DataObject maskFromMultipleShapes(
        const ito::DataObject& dataObject, const QVector<ito::Shape>& shapes, bool inverse = false);
    static ito::DataObject mask(
        const ito::DataObject& dataObject, const ito::Shape& shape, bool inverse = false);

protected:
    static void maskHelper(
        const ito::DataObject& dataObject,
        ito::DataObject& mask,
        const ito::Shape& shape,
        bool inverse = false);

};

} // namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif // SHAPEDOBJECT_H
