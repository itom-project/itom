/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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

#ifndef HELPERCOMMON_H
#define HELPERCOMMON_H

#include "typeDefs.h"
#include "sharedStructures.h"

#include <qstring.h>
#include <qvector.h>
#include <qobject.h>
#include <qhash.h>

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{
    ito::RetVal ITOMCOMMONQT_EXPORT checkParamVector(QVector<ito::Param> *params);
    ito::RetVal ITOMCOMMONQT_EXPORT checkParamVectors(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut);

    ITOMCOMMONQT_EXPORT ito::Param* getParamByName(QVector<ito::Param> *paramVec, const char* name, ito::RetVal *retval = NULL);
    ITOMCOMMONQT_EXPORT ito::ParamBase* getParamByName(QVector<ito::ParamBase> *paramVec, const char* name, ito::RetVal *retval = NULL);
    ITOMCOMMONQT_EXPORT QHash<QString, ito::Param*> createParamHashTable(QVector<ito::Param> *paramVec);

    bool ITOMCOMMONQT_EXPORT checkNumericParamRange(const ito::Param &param, double &value, bool *ok = NULL);

    ito::RetVal ITOMCOMMONQT_EXPORT parseParamName(const QString &name, QString &paramName, bool &hasIndex, int &index, QString &additionalTag);

    ito::RetVal ITOMCOMMONQT_EXPORT getParamValue(const QMap<QString, Param> *m_params, const QString &key, ito::Param &value, QString &pkey, int &index);
    ito::RetVal ITOMCOMMONQT_EXPORT setParamValue(const QMap<QString, Param> *m_params, const QString &key, const ito::ParamBase &value, QString &pkey, int &index);
};   // end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif
