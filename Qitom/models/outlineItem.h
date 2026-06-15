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

#pragma once

#include <qlist.h>
#include <qstring.h>
#include <qicon.h>
#include <qsharedpointer.h>

namespace ito {

class OutlineItem;

class OutlineItem
{
public:
    enum Type
    {
        typeRoot,
        typeClass, //!< class method
        typeFunction, //!< unbound function
        typeMethod, //!< bound method of a class (first arg is self)
        typePropertyGet,
        typePropertySet,
        typeStaticMethod,
        typeClassMethod
    };

    explicit OutlineItem(Type type);
    ~OutlineItem();

    QIcon icon() const;

    Type m_type;
    QString m_name;
    QString m_args;
    QString m_returnType;
    int m_startLineIdx; //!< the first line where the block starts
    int m_endLineIdx; //!< the last line where the block ends
    bool m_private;
    bool m_async;
    QWeakPointer<OutlineItem> m_parent;

    QList<QSharedPointer<OutlineItem>> m_childs;
};

} //end namespace ito

Q_DECLARE_METATYPE(QSharedPointer<ito::OutlineItem>) //must be outside of namespace
