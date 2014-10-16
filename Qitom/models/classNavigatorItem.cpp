/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#include "classNavigatorItem.h"
#include <qicon.h>



namespace ito
{
    
//----------------------------------------------------------------------------------------------------------------------------------
// Constructor
ClassNavigatorItem::ClassNavigatorItem()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
// Destructor
ClassNavigatorItem::~ClassNavigatorItem()
{
    for(int i = 0; i < m_member.length(); ++i)
    {
        delete m_member[i];
    }

    m_member.clear();
}

//----------------------------------------------------------------------------------------------------------------------------------
void ClassNavigatorItem::setInternalType(t_type t)
{// enum t_type {typePyRoot, typePyGlobal, typePyClass, typePyDef, typePyStaticDef, typePyClMethDef};
    this->m_internalType = t;
    setIcon(t);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ClassNavigatorItem::setIcon(t_type t)
{
    switch(t)
    {
        // TODO ICONS dynamisch erstellen und code anpassen (paint priv. Lock on top)
        case typePyRoot:
            {this->m_icon = QIcon(":/classNavigator/icons/global.png"); break;}
        case typePyGlobal: //TODO: Maybe change Icon of global methods to normal method-icon
            {this->m_icon = QIcon(":/classNavigator/icons/global.png"); break;}
        case typePyClass:
            {this->m_icon = QIcon(":/classNavigator/icons/class.png"); break;}
        case typePyDef:
        {
            if (this->m_priv)
            {
                {this->m_icon = QIcon(":/classNavigator/icons/method_private.png"); break;}
            }
            else
            {
                {this->m_icon = QIcon(":/classNavigator/icons/method.png"); break;}
            }
        }
        case typePyStaticDef:
        {
            if (this->m_priv)
            {
                {this->m_icon = QIcon(":/classNavigator/icons/method_static_private.png"); break;}
            }
            else
            {
                {this->m_icon = QIcon(":/classNavigator/icons/method_static.png"); break;}
            }
        }
        case typePyClMethDef:
        {
            if (this->m_priv)
            {
                {this->m_icon = QIcon(":/classNavigator/icons/method_clmethod_private.png"); break;}
            }
            else
            {
                {this->m_icon = QIcon(":/classNavigator/icons/method_clmethod.png"); break;}
            }
        }
        default:
        {
            this->m_icon = QIcon();
        }
    }
}


} //end namespace ito