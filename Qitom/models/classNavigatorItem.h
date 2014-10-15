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

#ifndef CLASSNAVIGATORITEM_H
#define CLASSNAVIGATORITEM_H

#include <qlist.h>
#include <qmap.h>
#include <qstring.h>
#include <qstringlist.h>
#include <qicon.h>

namespace ito {

class ClassNavigatorItem;
class t_type;

class ClassNavigatorItem
{

public:
    // Enumeration
    enum t_type {typePyRoot, typePyGlobal, typePyClass, typePyDef, typePyStaticDef, typePyClMethDef};
    
    ClassNavigatorItem();
    ~ClassNavigatorItem();

    // Methods
    void setInternalType(t_type t);
    void setIcon(t_type t);
   
    // Variables
    int m_lineno;
    QString m_name;
    QString m_args;
    QIcon m_icon;
    QList<const ClassNavigatorItem*> m_member;
    t_type m_internalType;
    bool m_priv;
    
};

} //end namespace ito

#endif

