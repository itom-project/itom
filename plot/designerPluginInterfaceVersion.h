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

#ifndef DESIGNERPLUGININTERFACEVERSION_H
#define DESIGNERPLUGININTERFACEVERSION_H

#include "../common/typeDefs.h"

#include <qstring.h>

//these defines exist since itom 1.3.1
#define CREATE_DESIGNERPLUGININTERFACE_VERSION_STR(major,minor,patch) "ito.DesignerPlugin.Interface/"#major"."#minor"."#patch


#define ITOM_DESIGNERPLUGININTERFACE_MAJOR 2 //change this number below as well (2x)
#define ITOM_DESIGNERPLUGININTERFACE_MINOR 1 //change this number below as well (1x)
#define ITOM_DESIGNERPLUGININTERFACE_PATCH 1 //change this number below as well (1x)
#define ITOM_DESIGNERPLUGININTERFACE_VERSION CREATEVERSION(ITOM_DESIGNERPLUGININTERFACE_MAJOR,ITOM_DESIGNERPLUGININTERFACE_MINOR,ITOM_DESIGNERPLUGININTERFACE_PATCH)
#define ITOM_DESIGNERPLUGININTERFACE_VERSION_STR CREATE_DESIGNERPLUGININTERFACE_VERSION_STR(2,1,1)

#endif
