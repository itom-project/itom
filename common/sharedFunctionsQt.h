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

#ifndef SHAREDFUNCTIONSQT_H
#define SHAREDFUNCTIONSQT_H

#include "typeDefs.h"
#include "../DataObject/dataobj.h"
#include "sharedStructures.h"

#include <qfile.h>
#include <qmap.h>
#include <qstring.h>

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{
    //!< Function to convert double values with unit to scaled values with scaled units (0.01m -> 10mm)
    ITOMCOMMONQT_EXPORT ito::RetVal formatDoubleWithUnit(QStringList scaleThisUnitsOnly, QString unitIn, double dVal, double &dValOut, QString &unitOut);
    //!< Function generates the auto save filename from the plugin name and the dll-folder
    ITOMCOMMONQT_EXPORT ito::RetVal generateAutoSaveParamFile(QString plugInName, QFile &paramFile);
    //!< loadXML2QLIST loads parameters from an XML-File and saves them to paramList
    ITOMCOMMONQT_EXPORT ito::RetVal loadXML2QLIST(QMap<QString, ito::Param> *paramList , QString id, QFile &paramFile);
    //!< saveQLIST2XML writes parameters from paramList to an XML-File
    ITOMCOMMONQT_EXPORT ito::RetVal saveQLIST2XML(QMap<QString, ito::Param> *paramList , QString id, QFile &paramFile);
    //!< mergeQLists copies parameters from newList to oldList and performs some checks
    ITOMCOMMONQT_EXPORT ito::RetVal mergeQLists(QMap<QString, ito::Param> *oldList, QMap<QString, ito::Param> *newList, bool checkAutoSave, bool deleteUnchangedParams = false);

    //!< Save a dataObject to harddrive in a readable ITO-XML-Format (.ido or .idh)
    ITOMCOMMONQT_EXPORT ito::RetVal saveDOBJ2XML(ito::DataObject *dObjOut, QString folderFileName, bool onlyHeaderObjectFile = false, bool doubleAsBinary = false);

    //!< Import a dataObject from harddrive, saved in the ITO-XML-Format (.ido or .idh)
    ITOMCOMMONQT_EXPORT ito::RetVal loadXML2DOBJ(ito::DataObject *dObjIn, QString folderFileName, bool onlyHeaderObjectFile = false, bool appendEnding = true);

}   // end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif
