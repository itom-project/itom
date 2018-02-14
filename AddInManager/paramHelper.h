/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#ifndef PARAMHELPER_H
#define PARAMHELPER_H

#include "addInMgrDefines.h"
#include "../common/sharedStructures.h"

#include <qmap.h>
#include <qsharedpointer.h>

namespace ito 
{
    class AddInBase; //forward declaration

    class ADDINMGR_EXPORT ParamHelper
    {
    public: 

        static tCompareResult compareParam(const ito::Param &paramTemplate, const ito::Param &param, ito::RetVal &ret);
        static tCompareResult compareMetaParam(const ito::ParamMeta *metaTemplate, const ito::ParamMeta *meta, const char* nameTemplate, const char *name, ito::RetVal &ret);

        static ito::RetVal validateStringMeta(const ito::StringMeta *meta, const char* value, bool mandatory = false, const char* name = NULL);
        static ito::RetVal validateDoubleMeta(const ito::DoubleMeta *meta, double value, const char* name = NULL);
        static ito::RetVal validateDoubleMetaAndRoundToStepSize(const ito::DoubleMeta *meta, ito::ParamBase &doubleParam, bool allowRounding = true, const char* name = NULL);
        static ito::RetVal validateDoubleMetaAndRoundToStepSize(const ito::DoubleMeta *meta, double &value, bool allowRounding = true, const char* name = NULL);
        static ito::RetVal validateIntMeta(const ito::IntMeta *meta, int value, const char* name = NULL);
        static ito::RetVal validateCharMeta(const ito::CharMeta *meta, char value, const char* name = NULL);
        static ito::RetVal validateCharArrayMeta(const ito::ParamMeta *meta, const char* values, size_t len, const char* name = NULL);
        static ito::RetVal validateIntArrayMeta(const ito::ParamMeta *meta, const int* values, size_t len, const char* name = NULL);
        static ito::RetVal validateDoubleArrayMeta(const ito::ParamMeta *meta, const double* values, size_t len, const char* name = NULL);
        static ito::RetVal validateHWMeta(const ito::HWMeta *meta, ito::AddInBase *value, bool mandatory = false, const char* name = NULL);
        static ito::RetVal validateParam(const ito::Param &templateParam, const ito::ParamBase &param, bool strict = true, bool mandatory = false);
        static ito::RetVal validateAndCastParam(const ito::Param &templateParam, ito::ParamBase &param, bool strict = true, bool mandatory = false, bool roundToSteps = false);
        static ito::ParamBase convertParam(const ito::ParamBase &source, int destType, bool *ok = NULL);
        static ito::RetVal getParamFromMapByKey( QMap<QString,ito::Param> &paramMap, const QString &key, QMap<QString,ito::Param>::iterator &found, bool errorIfReadOnly);
        static ito::RetVal parseParamName(const QString &name, QString &paramName, bool &hasIndex, int &index, QString &additionalTag);

        static ito::RetVal getItemFromArray(const ito::Param &arrayParam, const int index, ito::Param &itemParam);
        static ito::Param getParam(const ito::Param &param, const bool hasIndex, const int index, ito::RetVal &ret);

        static ito::RetVal updateParameters(QMap<QString, ito::Param> &paramMap, const QVector<QSharedPointer<ito::ParamBase> > &values);

    private:
        static bool fitToDoubleStepSize(double min, double step, double val);
        static QString parseNamePrefix(const char *name);

        ParamHelper(){};
        ~ParamHelper(){};
    };
} //end namespace ito

#endif

