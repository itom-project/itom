/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2022, Institut fuer Technische Optik (ITO),
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

#include "addInMgrDefines.h"
#include "../common/sharedStructures.h"
#include "../common/param.h"
#include "../common/paramMeta.h"

#include <qmap.h>
#include <qsharedpointer.h>

namespace ito
{
    class AddInBase; // forward declaration
    class DataObject; // forward declaration

    class ADDINMGR_EXPORT ParamHelper
    {
    public:

        static tCompareResult compareParam(const ito::Param &paramTemplate, const ito::Param &param, ito::RetVal &ret);
        static tCompareResult compareMetaParam(const ito::ParamMeta *metaTemplate, const ito::ParamMeta *meta, const char* nameTemplate, const char *name, ito::RetVal &ret);

        //!< validates if the given string value matches a string meta object
        /*
        \param meta is the meta information or nullptr if no meta restrictions
            are given (every value is valid).
        \param value is the value to be checked
        \param mandatory must be true, if an error is also returned in value is nullptr
        \param name is the name of the parameter.
        \return ito::retOk if the value fits to meta or ito::retError if the
            value does either not fit to the meta information or if its not
            given, but mandatory.
        \sa validateParam, validateAndCastParam
        */
        static ito::RetVal validateStringMeta(const ito::StringMeta *meta, const char* value, bool mandatory = false, const char* name = nullptr);

        //!< validates if the given double value matches a double meta object
        /*
        \param meta is the meta information or nullptr if no meta restrictions
            are given (every value is valid).
        \param value is the value to be checked
        \param name is the name of the parameter.
        \return ito::retOk if the value fits to meta, else ito::retError
        \sa validateParam, validateAndCastParam
        */
        static ito::RetVal validateDoubleMeta(const ito::DoubleMeta *meta, double value, const char* name = nullptr);

        //!< validates if the given double value matches a double meta object
        /*
        This method also verifies, if the value fits to an optionally given
        step size of the meta information and rounds the wrapped value to the
        next valid value (if allowRounding is true).

        \param meta is the meta information or nullptr if no meta restrictions
            are given (every value is valid).
        \param value is the value to be checked
        \param allowRounding is true, if an optional rounding according to the
            step size of the meta information should be done on the fly.
        \param name is the name of the parameter.
        \return ito::retOk if the value fits to meta, else ito::retError
        \sa validateParam, validateAndCastParam
        */
        static ito::RetVal validateDoubleMetaAndRoundToStepSize(const ito::DoubleMeta *meta, ito::ParamBase &doubleParam, bool allowRounding = true, const char* name = nullptr);

        //!< validates if the given double value matches a double meta object
        /*
        This method also verifies, if the value fits to an optionally given
        step size of the meta information and rounds the wrapped value to the
        next valid value (if allowRounding is true).

        \param meta is the meta information or nullptr if no meta restrictions
            are given (every value is valid).
        \param value is the value to be checked
        \param allowRounding is true, if an optional rounding according to the
            step size of the meta information should be done on the fly.
        \param name is the name of the parameter.
        \return ito::retOk if the value fits to meta, else ito::retError
        \sa validateParam, validateAndCastParam
        */
        static ito::RetVal validateDoubleMetaAndRoundToStepSize(const ito::DoubleMeta *meta, double &value, bool allowRounding = true, const char* name = nullptr);
        static ito::RetVal validateIntMeta(const ito::IntMeta *meta, int value, const char* name = nullptr);
        static ito::RetVal validateCharMeta(const ito::CharMeta *meta, char value, const char* name = nullptr);
        static ito::RetVal validateDObjMeta(const ito::DObjMeta *meta, const ito::DataObject* value, bool mandatory = false, const char* name = nullptr);
        static ito::RetVal validateCharArrayMeta(const ito::ParamMeta *meta, const char* values, size_t len, const char* name = nullptr);
        static ito::RetVal validateIntArrayMeta(const ito::ParamMeta *meta, const int* values, size_t len, const char* name = nullptr);
        static ito::RetVal validateDoubleArrayMeta(const ito::ParamMeta *meta, const double* values, size_t len, const char* name = nullptr);
        static ito::RetVal validateStringListMeta(const ito::ParamMeta *meta, const ito::ByteArray* values, size_t len, const char* name = nullptr);
        static ito::RetVal validateHWMeta(const ito::HWMeta *meta, ito::AddInBase *value, bool mandatory = false, const char* name = nullptr);

        static ito::RetVal validateParam(const ito::Param &templateParam, const ito::ParamBase &param, bool strict = true, bool mandatory = false);
        static ito::RetVal validateAndCastParam(const ito::Param &templateParam, ito::ParamBase &param, bool strict = true, bool mandatory = false, bool roundToSteps = false);
        static ito::ParamBase convertParam(const ito::ParamBase &source, int destType, bool *ok = nullptr);

        static ito::RetVal getParamFromMapByKey( QMap<QString,ito::Param> &paramMap, const QString &key, QMap<QString,ito::Param>::iterator &found, bool errorIfReadOnly);

        //!< parses a full parameter name, verifies it and splits it in case of success into its components
        /*
        Valid parameter keys are:

        * name
        * name[index]
        * name:suffix
        * name[index]:suffix

        where index must be a number.

        \param key is the full parameter name with optional suffix, index etc.
        \param paramName is the name of the parameter, used in a possible error return value
        \param hasIndex is true, if an index is set
        \param index is the parsed index or -1, if no index is given
        \param additionalTag is the optional suffix, or an empty string
        \return ito::retOk if key could be properly parsed, else ito::retError.
        */
        static ito::RetVal parseParamName(const QString &key, QString &paramName, bool &hasIndex, int &index, QString &additionalTag);

        static ito::RetVal getItemFromArray(const ito::Param &arrayParam, const int index, ito::Param &itemParam);
        static ito::Param getParam(const ito::Param &param, const bool hasIndex, const int index, ito::RetVal &ret);

        static ito::RetVal updateParameters(QMap<QString, ito::Param> &paramMap, const QVector<QSharedPointer<ito::ParamBase> > &values);

    private:
        static bool fitToDoubleStepSize(double min, double step, double val);
        static QString parseNamePrefix(const char *name);

        ParamHelper() = delete;
        ~ParamHelper() = delete;
    };

} //end namespace ito
