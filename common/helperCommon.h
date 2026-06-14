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

#pragma once

#include "param.h"
#include "retVal.h"
#include "typeDefs.h"

#include <qobject.h>
#include <qstring.h>
#include <qvector.h>

// only moc this file in itomCommonQtLib but not in other libraries or executables linking against
// this itomCommonQtLib
#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

namespace ito {

//!< checks param vector to be not a nullptr.
ITOMCOMMONQT_EXPORT ito::RetVal checkParamVector(const QVector<ito::Param>* params);

//!< checks the relevant parameter vectors for algorithm calls to be not nullptrs and clears all
//!< vectors.
ITOMCOMMONQT_EXPORT ito::RetVal checkParamVectors(
    QVector<ito::Param>* paramsMand,
    QVector<ito::Param>* paramsOpt,
    QVector<ito::Param>* paramsOut);

//!< searches for a specific parameter in the vector and returns it.
ITOMCOMMONQT_EXPORT ito::Param* getParamByName(
    QVector<ito::Param>* paramVec, const char* name, ito::RetVal* retval = nullptr);

//!< searches for a specific base parameter in the vector and returns it.
ITOMCOMMONQT_EXPORT ito::ParamBase* getParamByName(
    QVector<ito::ParamBase>* paramVec, const char* name, ito::RetVal* retval = nullptr);

//!< Check if the numeric value is within the min/max range of the meta info of param.
ITOMCOMMONQT_EXPORT bool checkNumericParamRange(
    const ito::Param& param, double& value, bool* ok = nullptr);

//!< parses a parameter name key and extracts the real name, an optional index and / or suffix tag.
ITOMCOMMONQT_EXPORT ito::RetVal parseParamName(
    const QString& key, QString& paramName, bool& hasIndex, int& index, QString& additionalTag);

//!< searches and returns a parameter from a map based on its full key.
ITOMCOMMONQT_EXPORT ito::RetVal getParamValue(
    const QMap<QString, Param>* params,
    const QString& key,
    ito::Param& value,
    QString& name,
    int& index);

//!< searches a parameter from a map based on its full key and sets its value.
ITOMCOMMONQT_EXPORT ito::RetVal setParamValue(
    QMap<QString, Param>* params,
    const QString& key,
    const ito::ParamBase& value,
    QString& name,
    int& index);

//!< parses the type and meta information of param and returns a readable
//!< string of the meta information and default value (if possible) as well
//!< as a readable type information in a python-like representation.
ITOMCOMMONQT_EXPORT QString
getMetaDocstringFromParam(const Param& param, bool translate, QString& pythonLikeTypename);

}; // end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)
