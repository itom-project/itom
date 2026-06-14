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

#include "outlineItem.h"
#include <qicon.h>
#include <qobject.h>


namespace ito
{

//-------------------------------------------------------------------------------------
OutlineItem::OutlineItem(Type type) :
    m_type(type),
    m_async(false),
    m_private(false),
    m_startLineIdx(-1),
    m_endLineIdx(-1)
{
    if (type == typeRoot)
    {
        m_name = QObject::tr("{Global Scope}");
    }

}

//-------------------------------------------------------------------------------------
OutlineItem::~OutlineItem()
{
    m_childs.clear();
    m_parent.clear();
}

//-------------------------------------------------------------------------------------
QIcon OutlineItem::icon() const
{
    switch (m_type)
    {
    case typeRoot:
        return QIcon(":/classNavigator/icons/global.png");
    case typeClass:
        return QIcon(":/classNavigator/icons/class.png");
    case typePropertyGet:
        return QIcon(":/classNavigator/icons/property_get.png");
    case typePropertySet:
        return QIcon(":/classNavigator/icons/property_set.png");
    case typeClassMethod:
        if (m_private)
        {
            return QIcon(":/classNavigator/icons/method_clmethod_private.png");
        }
        else
        {
            return QIcon(":/classNavigator/icons/method_clmethod.png");
        }
    case typeStaticMethod:
        if (m_private)
        {
            return QIcon(":/classNavigator/icons/method_static_private.png");
        }
        else
        {
            return QIcon(":/classNavigator/icons/method_static.png");
        }
    case typeFunction:
    case typeMethod:
        if (m_private)
        {
            return QIcon(":/classNavigator/icons/method_private.png");
        }
        else
        {
            return QIcon(":/classNavigator/icons/method.png");
        }
    default:
        return QIcon();
    }
}


} //end namespace ito
