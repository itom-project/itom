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

#include "paramHelper.h"

#include "../common/addInInterface.h"
#include "addInManagerPrivate.h"
#include "../DataObject/dataobj.h"

#include <qregularexpression.h>


namespace ito {



//----------------------------------------------------------------------------------------------------------------------------------
tCompareResult ParamHelper::compareParam(
    const ito::Param& paramTemplate, const ito::Param& param, ito::RetVal& ret)
{
    // check whether type is equal
    if (paramTemplate.getType() != param.getType())
    {
        ret += ito::RetVal::format(
            ito::retError,
            0,
            QObject::tr(
                "Types of parameter '%s' is unequal to required type of interface parameter '%s'")
                .toLatin1()
                .data(),
            param.getName(),
            paramTemplate.getName());
        return tCmpFailed;
    }

    // check in/out flags
    int inOutFlags = ito::ParamBase::In | ito::ParamBase::Out;
    if ((paramTemplate.getFlags() & inOutFlags) != (param.getFlags() & inOutFlags))
    {
        ret += ito::RetVal::format(
            ito::retError,
            0,
            QObject::tr("In/Out flags of parameter '%s' are unequal to required flags of interface "
                        "parameter '%s'")
                .toLatin1()
                .data(),
            param.getName(),
            paramTemplate.getName());
        return tCmpFailed;
    }

    // check meta information
    const ito::ParamMeta* metaTemplate = paramTemplate.getMeta();
    const ito::ParamMeta* meta = param.getMeta();

    return compareMetaParam(metaTemplate, meta, paramTemplate.getName(), param.getName(), ret);
}

//-------------------------------------------------------------------------------------
//!< verifies the meta informaiton of two different parameters for compatibility
/*
The method verifies if meta information of a given parameter (meta, name) is
compatible to the meta information of a template parameter.

Both meta information are equal, if they are both either nullptr or if both
have the same type and configurations. If their type is different, the compatiblity
failed. A compatibility is returned, if the meta information is of the same type
than the template but less restrictive than the template. If the meta information
would be more restrictive than the template, the compatibility failed, too.

\param metaTemplate is the meta information of a parameter ``nameTemplate``
\param meta is the meta information of another parameter ``name``
\param nameTemplate is only the name (for error messages) of the template parameter
\param name is the name of the parameter to be checked
\param ret contains a detailed error message in case of an incompatibility

\returns tCompareResult to signal the comparison result: equal, compatible, failed.
*/
tCompareResult ParamHelper::compareMetaParam(
    const ito::ParamMeta* metaTemplate,
    const ito::ParamMeta* meta,
    const char* nameTemplate,
    const char* name,
    ito::RetVal& ret)
{
    if (metaTemplate == nullptr && meta == nullptr)
    {
        return tCmpEqual;
    }
    else if (meta == nullptr)
    {
        // param is compatible to paramTemplate, since it has no meta block
        // defined, but paramTemplate has. A meta bock always is more
        // restrictive than none.
        return tCmpCompatible;
    }
    else if (metaTemplate == nullptr)
    {
        ret += ito::RetVal::format(
            ito::retError,
            0,
            QObject::tr("The parameter '%s' is restricted by meta information while the interface "
                        "parameter '%s' is not.")
                .toLatin1()
                .data(),
            name,
            nameTemplate);
        return tCmpFailed;
    }

    if (metaTemplate->getType() != meta->getType())
    {
        ret += ito::RetVal::format(
            ito::retError,
            0,
            QObject::tr("The type of the meta information of parameter '%s' is unequal to this of "
                        "the interface parameter '%s'.")
                .toLatin1()
                .data(),
            name,
            nameTemplate);
        return tCmpFailed;
    }

    switch (metaTemplate->getType())
    {
    case ito::ParamMeta::rttiCharMeta: {
        const ito::CharMeta* mT = static_cast<const ito::CharMeta*>(metaTemplate);
        const ito::CharMeta* m = static_cast<const ito::CharMeta*>(meta);

        if (!mT || !m)
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The type of the meta information of parameter '%s' is unequal to this "
                            "of the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
        if (m->getMax() == mT->getMax() && m->getMin() == mT->getMin())
        {
            return tCmpEqual;
        }
        else if (m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin())
        {
            return tCmpCompatible;
        }
        else
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The allowed char range of parameter '%s' is smaller than the "
                            "requested range from interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
    }
    break;

    case ito::ParamMeta::rttiCharArrayMeta: {
        const ito::CharArrayMeta* mT = static_cast<const ito::CharArrayMeta*>(metaTemplate);
        const ito::CharArrayMeta* m = static_cast<const ito::CharArrayMeta*>(meta);
        if (!mT || !m)
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The type of the meta information of parameter '%s' is unequal to this "
                            "of the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
        if (m->getMax() == mT->getMax() && m->getMin() == mT->getMin() &&
            m->getNumMin() == mT->getNumMin() && m->getNumMax() == mT->getNumMax())
        {
            return tCmpEqual;
        }
        else if (
            m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin() &&
            m->getNumMax() >= mT->getNumMax() && m->getNumMin() <= mT->getNumMin())
        {
            return tCmpCompatible;
        }
        else
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The allowed char range or the allowed range of numbers of elements of "
                            "parameter '%s' is smaller than the requested range from interface "
                            "parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
    }
    break;

    case ito::ParamMeta::rttiIntMeta: {
        const ito::IntMeta* mT = static_cast<const ito::IntMeta*>(metaTemplate);
        const ito::IntMeta* m = static_cast<const ito::IntMeta*>(meta);
        if (!mT || !m)
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The type of the meta information of parameter '%s' is unequal to this "
                            "of the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
        if (m->getMax() == mT->getMax() && m->getMin() == mT->getMin())
        {
            return tCmpEqual;
        }
        else if (m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin())
        {
            return tCmpCompatible;
        }
        else
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The allowed integer range of parameter '%s' is smaller than the "
                            "requested range from interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
    }
    break;

    case ito::ParamMeta::rttiIntArrayMeta: {
        const ito::IntArrayMeta* mT = static_cast<const ito::IntArrayMeta*>(metaTemplate);
        const ito::IntArrayMeta* m = static_cast<const ito::IntArrayMeta*>(meta);
        if (!mT || !m)
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The type of the meta information of parameter '%s' is unequal to this "
                            "of the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
        if (m->getMax() == mT->getMax() && m->getMin() == mT->getMin() &&
            m->getNumMin() == mT->getNumMin() && m->getNumMax() == mT->getNumMax())
        {
            return tCmpEqual;
        }
        else if (
            m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin() &&
            m->getNumMax() >= mT->getNumMax() && m->getNumMin() <= mT->getNumMin())
        {
            return tCmpCompatible;
        }
        else
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The allowed integer range or the allowed range of numbers of elements "
                            "of parameter '%s' is smaller than the requested range from interface "
                            "parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
    }
    break;

    case ito::ParamMeta::rttiIntervalMeta:
    case ito::ParamMeta::rttiRangeMeta: {
        const ito::IntervalMeta* mT = static_cast<const ito::IntervalMeta*>(metaTemplate);
        const ito::IntervalMeta* m = static_cast<const ito::IntervalMeta*>(meta);
        if (!mT || !m)
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The type of the meta information of parameter '%s' is unequal to this "
                            "of the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
        if (m->getMax() == mT->getMax() && m->getMin() == mT->getMin() &&
            m->getSizeMin() == mT->getSizeMin() && m->getSizeMax() == mT->getSizeMax())
        {
            return tCmpEqual;
        }
        else if (
            m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin() &&
            m->getSizeMax() >= mT->getSizeMax() && m->getSizeMin() <= mT->getSizeMin())
        {
            return tCmpCompatible;
        }
        else
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr(
                    "The allowed value range or the allowed interval/range of parameter '%s' is "
                    "smaller than the requested range from interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
    }
    break;

    case ito::ParamMeta::rttiRectMeta: {
        const ito::RectMeta* mT = static_cast<const ito::RectMeta*>(metaTemplate);
        const ito::RectMeta* m = static_cast<const ito::RectMeta*>(meta);
        if (!mT || !m)
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The type of the meta information of parameter '%s' is unequal to this "
                            "of the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
        ito::tCompareResult res1 = compareMetaParam(
            &(mT->getHeightRangeMeta()), &m->getHeightRangeMeta(), nameTemplate, name, ret);
        ito::tCompareResult res2 = compareMetaParam(
            &(mT->getWidthRangeMeta()), &m->getWidthRangeMeta(), nameTemplate, name, ret);

        if (res1 == res2)
        {
            return res1;
        }
        else if (res1 == tCmpFailed || res2 == tCmpFailed)
        {
            return tCmpFailed;
        }
        else if (res1 == tCmpCompatible || res2 == tCmpCompatible)
        {
            return tCmpCompatible;
        }
        else
        {
            return tCmpEqual;
        }
    }
    break;

    case ito::ParamMeta::rttiDoubleMeta: {
        const ito::DoubleMeta* mT = static_cast<const ito::DoubleMeta*>(metaTemplate);
        const ito::DoubleMeta* m = static_cast<const ito::DoubleMeta*>(meta);
        if (!mT || !m)
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The type of the meta information of parameter '%s' is unequal to this "
                            "of the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
        if (m->getMax() == mT->getMax() && m->getMin() == mT->getMin())
        {
            return tCmpEqual;
        }
        else if (m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin())
        {
            return tCmpCompatible;
        }
        else
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The allowed double range of parameter '%s' is smaller than the "
                            "requested range from interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
    }
    break;

    case ito::ParamMeta::rttiDoubleArrayMeta: {
        const ito::DoubleArrayMeta* mT = static_cast<const ito::DoubleArrayMeta*>(metaTemplate);
        const ito::DoubleArrayMeta* m = static_cast<const ito::DoubleArrayMeta*>(meta);
        if (!mT || !m)
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The type of the meta information of parameter '%s' is unequal to this "
                            "of the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
        if (m->getMax() == mT->getMax() && m->getMin() == mT->getMin() &&
            m->getNumMin() == mT->getNumMin() && m->getNumMax() == mT->getNumMax())
        {
            return tCmpEqual;
        }
        else if (
            m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin() &&
            m->getNumMax() >= mT->getNumMax() && m->getNumMin() <= mT->getNumMin())
        {
            return tCmpCompatible;
        }
        else
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The allowed double range or the allowed range of numbers of elements "
                            "of parameter '%s' is smaller than the requested range from interface "
                            "parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
    }
    break;

    case ito::ParamMeta::rttiDoubleIntervalMeta: {
        const ito::DoubleIntervalMeta* mT =
            static_cast<const ito::DoubleIntervalMeta*>(metaTemplate);
        const ito::DoubleIntervalMeta* m = static_cast<const ito::DoubleIntervalMeta*>(meta);
        if (!mT || !m)
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The type of the meta information of parameter '%s' is unequal to this "
                            "of the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
        if (m->getMax() == mT->getMax() && m->getMin() == mT->getMin() &&
            m->getSizeMin() == mT->getSizeMin() && m->getSizeMax() == mT->getSizeMax())
        {
            return tCmpEqual;
        }
        else if (
            m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin() &&
            m->getSizeMax() >= mT->getSizeMax() && m->getSizeMin() <= mT->getSizeMin())
        {
            return tCmpCompatible;
        }
        else
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The allowed value range or the allowed range of parameter '%s' is "
                            "smaller than the requested range from interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }
    }
    break;

    case ito::ParamMeta::rttiStringMeta: {
        const ito::StringMeta* mT = static_cast<const ito::StringMeta*>(metaTemplate);
        const ito::StringMeta* m = static_cast<const ito::StringMeta*>(meta);
        if (!mT || !m)
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The type of the meta information of parameter '%s' is unequal to this "
                            "of the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }

        if (m->getStringType() != mT->getStringType())
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The string type of the meta information of parameter '%s' is unequal "
                            "to this of the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }

        const char* sT = nullptr;
        bool found = false;

        for (int i = 0; i < mT->getLen(); i++)
        {
            sT = mT->getString(i);
            found = false;

            for (int j = 0; j < m->getLen(); j++)
            {
                if (strcmp(sT, m->getString(j)) == 0)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                ret += ito::RetVal::format(
                    ito::retError,
                    0,
                    QObject::tr("String '%s', requested by meta data of interface parameter '%s' "
                                "could not be found in meta data of parameter '%s'.")
                        .toLatin1()
                        .data(),
                    sT,
                    nameTemplate,
                    name);
                return tCmpFailed;
            }
        }

        if (m->getLen() == mT->getLen())
        {
            return tCmpEqual;
        }
        else
        {
            return tCmpCompatible;
        }
    }
    break;

    case ito::ParamMeta::rttiHWMeta: {
        const ito::HWMeta* mT = static_cast<const ito::HWMeta*>(metaTemplate);
        const ito::HWMeta* m = static_cast<const ito::HWMeta*>(meta);
        if (!mT || !m)
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The type of the meta information of parameter '%s' is unequal to this "
                            "of the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }

        if (mT->getHWAddInName() != nullptr)
        {
            if (m->getHWAddInName() != nullptr)
            {
                ret += ito::RetVal::format(
                    ito::retError,
                    0,
                    QObject::tr("The meta data of the interface parameter '%s' requires a plugin "
                                "with name '%s', but parameter '%s' does it not.")
                        .toLatin1()
                        .data(),
                    nameTemplate,
                    mT->getHWAddInName().data(),
                    name);
                return tCmpFailed;
            }
            else if (mT->getHWAddInName() != m->getHWAddInName())
            {
                ret += ito::RetVal::format(
                    ito::retError,
                    0,
                    QObject::tr("Both parameter '%s' and interface parameter '%s' require "
                                "different plugins.")
                        .toLatin1()
                        .data(),
                    name,
                    nameTemplate);
                return tCmpFailed;
            }
        }

        // every bit set in m->getMinType() must be set in mT->getMinType()
        if (mT->getMinType() == m->getMinType())
        {
            return tCmpEqual;
        }
        else if ((mT->getMinType() & m->getMinType()) == m->getMinType())
        {
            return tCmpCompatible;
        }

        ret += ito::RetVal::format(
            ito::retError,
            0,
            QObject::tr("The minimum plugin type bit mask of parameter '%s' is more restrictive "
                        "than this of the interface parameter '%s'.")
                .toLatin1()
                .data(),
            name,
            nameTemplate);
        return tCmpFailed;
    }
    break;

    case ito::ParamMeta::rttiDObjMeta: {
        const ito::DObjMeta* mT = static_cast<const ito::DObjMeta*>(metaTemplate);
        const ito::DObjMeta* m = static_cast<const ito::DObjMeta*>(meta);

        if (!mT || !m)
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The type of the meta information of parameter '%s' is unequal to this "
                            "of the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                name,
                nameTemplate);
            return tCmpFailed;
        }

        // all bits in allowedTypes of mT must be set in m, too
        if (m->getNumAllowedDataTypes() > 0)
        {
            if (mT->getNumAllowedDataTypes() == 0)
            {
                // the template allows all types, m only few types and is therefore more restrictive
                ret += ito::RetVal::format(
                    ito::retError,
                    0,
                    QObject::tr("The allowed data object types of parameter '%s' are more restrictive "
                        "than these required by the interface parameter '%s'.")
                    .toLatin1()
                    .data(),
                    name,
                    nameTemplate);
                return tCmpFailed;
            }
            else
            {
                for (int i = 0; i < mT->getNumAllowedDataTypes(); ++i)
                {
                    if (!m->isDataTypeAllowed(mT->getAllowedDataType(i)))
                    {
                        ret += ito::RetVal::format(
                            ito::retError,
                            0,
                            QObject::tr("The allowed data object types of parameter '%s' are more restrictive "
                                "than these required by the interface parameter '%s'.")
                            .toLatin1()
                            .data(),
                            name,
                            nameTemplate);
                        return tCmpFailed;
                    }
                }
            }
        }

        ito::DObjMeta m2 = *m;
        m2.setMinDim(mT->getMinDim());
        m2.setMaxDim(mT->getMaxDim());

        // check if the allowed types of m and mT are exactly equal... therefore compare m2 and mT
        if (m2 == *mT)
        {
            if (m->getMinDim() == mT->getMinDim() && m->getMaxDim() == mT->getMaxDim())
            {
                return tCmpEqual;
            }
            else if (m->getMinDim() <= mT->getMinDim() && m->getMaxDim() >= mT->getMaxDim())
            {
                return tCmpCompatible;
            }
            else
            {
                ret += ito::RetVal::format(
                    ito::retError,
                    0,
                    QObject::tr(
                        "The minimum and maximum dimensions of the data object of parameter '%s' "
                        "are more restrictive than these required by the interface parameter '%s'.")
                        .toLatin1()
                        .data(),
                    name,
                    nameTemplate);
                return tCmpFailed;
            }
        }
        else
        {
            if (m->getMinDim() <= mT->getMinDim() && m->getMaxDim() >= mT->getMaxDim())
            {
                return tCmpCompatible;
            }
            else
            {
                ret += ito::RetVal::format(
                    ito::retError,
                    0,
                    QObject::tr(
                        "The minimum and maximum dimensions of the data object of parameter '%s' "
                        "are more restrictive than these required by the interface parameter '%s'.")
                        .toLatin1()
                        .data(),
                    name,
                    nameTemplate);
                return tCmpFailed;
            }
        }
    }
    break;


    default: {
        ret += ito::RetVal::format(
            ito::retError,
            0,
            QObject::tr("meta data of interface parameter '%s' is unknown.").toLatin1().data(),
            nameTemplate);
        return tCmpFailed;
    }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateStringMeta(
    const ito::StringMeta* meta,
    const char* value,
    bool mandatory /*= false*/,
    const char* name /*= nullptr*/)
{
    QLatin1String value_(value);


    if (meta && meta->getLen() > 0 && value)
    {
        bool found = false;
        QRegularExpression reg;

        switch (meta->getStringType())
        {
        case ito::StringMeta::String:
            for (int i = 0; i < meta->getLen(); i++)
            {
                if (QLatin1String(meta->getString(i)) == value_)
                {
                    found = true;
                    break;
                }
            }
            break;
        case ito::StringMeta::Wildcard:

            for (int i = 0; i < meta->getLen(); i++)
            {
                reg.setPattern(AddInManagerPrivate::wildcardToRegularExpression(meta->getString(i)));

                if (reg.match(AddInManagerPrivate::regExpAnchoredPattern(value_)).hasMatch())
                {
                    found = true;
                    break;
                }
            }
            break;
        case ito::StringMeta::RegExp:

            for (int i = 0; i < meta->getLen(); i++)
            {
                reg.setPattern(QLatin1String(meta->getString(i)));

                if (reg.match(value_).hasMatch())
                {
                    found = true;
                    break;
                }
            }
            break;
        }

        if (!found)
        {
            QString constraints;
            QStringList items;

            for (int i = 0; i < meta->getLen(); i++)
            {
                items << QLatin1String(meta->getString(i));
            }

            switch (meta->getStringType())
            {
            case ito::StringMeta::String:
                constraints = QObject::tr("Exact string match: (%1)").arg(items.join(","));
                break;
            case ito::StringMeta::Wildcard:
                constraints = QObject::tr("Wildcard match: (%1)").arg(items.join(","));
                break;
            case ito::StringMeta::RegExp:
                constraints = QObject::tr("RegularExpression match: (%1)").arg(items.join(","));
                break;
            }

            return ito::RetVal::format(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr("String '%s' does not fit to given string-constraints. %1")
                     .arg(constraints))
                    .toLatin1()
                    .data(),
                value);
        }
    }

    if (mandatory && (value == nullptr))
    {
        return ito::RetVal(
            ito::retError,
            0,
            (parseNamePrefix(name) + QObject::tr("Mandatory string value is not given."))
                .toLatin1()
                .data());
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateDoubleMeta(
    const ito::DoubleMeta* meta, double value, const char* name /*= nullptr*/)
{
    if (meta)
    {
        double minVal = meta->getMin();
        double maxVal = meta->getMax();
        double eps = std::numeric_limits<double>::epsilon();

        if (value < (minVal - eps) || value > (maxVal + eps))
        {
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr("value out of range [%1, %2]").arg(minVal).arg(maxVal))
                    .toLatin1()
                    .data());
        }

        double step = meta->getStepSize();
        if (!fitToDoubleStepSize(minVal, step, value))
        {
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr("value does not fit to given step size [%1:%2:%3]")
                     .arg(minVal)
                     .arg(step)
                     .arg(maxVal))
                    .toLatin1()
                    .data());
        }
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateDoubleMetaAndRoundToStepSize(
    const ito::DoubleMeta* meta,
    ito::ParamBase& doubleParam,
    bool allowRounding /*= true*/,
    const char* name /*= nullptr*/)
{
    if (meta)
    {
        double value = doubleParam.getVal<double>();
        double minVal = meta->getMin();
        double maxVal = meta->getMax();
        double eps = std::numeric_limits<double>::epsilon();
        double step = meta->getStepSize();

        if (value < (minVal - eps) || value > (maxVal + eps))
        {
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr("value out of range [%1, %2]").arg(minVal).arg(maxVal))
                    .toLatin1()
                    .data());
        }

        if (!allowRounding || step < eps)
        {
            if (!fitToDoubleStepSize(minVal, step, value))
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("value does not fit to given step size [%1:%2:%3]")
                         .arg(minVal)
                         .arg(step)
                         .arg(maxVal))
                        .toLatin1()
                        .data());
            }
        }
        else
        {
            double step = meta->getStepSize();
            int multiple = qRound((value - minVal) / step);
            value = minVal + multiple * step;
            value = qBound(minVal, value, maxVal);
            doubleParam.setVal<double>(value);
        }
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateDoubleMetaAndRoundToStepSize(
    const ito::DoubleMeta* meta,
    double& value,
    bool allowRounding /*= true*/,
    const char* name /*= nullptr*/)
{
    if (meta)
    {
        double minVal = meta->getMin();
        double maxVal = meta->getMax();
        double eps = std::numeric_limits<double>::epsilon();
        double step = meta->getStepSize();

        if (value < (minVal - eps) || value > (maxVal + eps))
        {
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr("value out of range [%1, %2]").arg(minVal).arg(maxVal))
                    .toLatin1()
                    .data());
        }

        if (!allowRounding || step < eps)
        {
            if (!fitToDoubleStepSize(minVal, step, value))
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("value does not fit to given step size [%1:%2:%3]")
                         .arg(minVal)
                         .arg(step)
                         .arg(maxVal))
                        .toLatin1()
                        .data());
            }
        }
        else
        {
            double step = meta->getStepSize();
            int multiple = qRound((value - minVal) / step);
            value = minVal + multiple * step;
            value = qBound(minVal, value, maxVal);
        }
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateIntMeta(
    const ito::IntMeta* meta, int value, const char* name /*= nullptr*/)
{
    if (meta)
    {
        int minVal = meta->getMin();
        int maxVal = meta->getMax();

        if (value < minVal || value > maxVal)
        {
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr("value out of range [%1, %2]").arg(minVal).arg(maxVal))
                    .toLatin1()
                    .data());
        }

        int step = meta->getStepSize();
        if (step > 1 && ((value - minVal) % step) != 0)
        {
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr("value does not fit to given step size [%1:%2:%3]")
                     .arg(minVal)
                     .arg(step)
                     .arg(maxVal))
                    .toLatin1()
                    .data());
        }
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateCharMeta(
    const ito::CharMeta* meta, char value, const char* name /*= nullptr*/)
{
    if (meta)
    {
        char minVal = meta->getMin();
        char maxVal = meta->getMax();

        if (value < minVal || value > maxVal)
        {
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr("value out of range [%1, %2]").arg(minVal).arg(maxVal))
                    .toLatin1()
                    .data());
        }

        char step = meta->getStepSize();
        if (step > 1 && ((value - minVal) % step) != 0)
        {
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr("value does not fit to given step size [%1:%2:%3]")
                     .arg(minVal)
                     .arg(step)
                     .arg(maxVal))
                    .toLatin1()
                    .data());
        }
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateDObjMeta(const ito::DObjMeta *meta, const ito::DataObject* value, bool mandatory /*= false*/, const char* name /*= nullptr*/)
{
    if (meta && value)
    {
        const int dims = value->getDims();

        if (dims < meta->getMinDim() || dims > meta->getMaxDim())
        {
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                    QObject::tr("number of dimensions out of range."))
                .toLatin1()
                .data());
        }

        if (!meta->isDataTypeAllowed((ito::tDataType)(value->getType())))
        {
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                    QObject::tr("unallowed data type."))
                .toLatin1()
                .data());
        }
    }
    else if (mandatory && value == nullptr)
    {
        return ito::RetVal(
            ito::retError,
            0,
            (parseNamePrefix(name) + QObject::tr("DataObject must not be nullptr")).toLatin1().data());
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateHWMeta(
    const ito::HWMeta* meta,
    ito::AddInBase* value,
    bool mandatory /*= false*/,
    const char* name /*= nullptr*/)
{
    if (meta && value)
    {
        int minType = meta->getMinType();
        if ((minType & value->getBasePlugin()->getType()) != minType)
        {
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr("AddIn does not fit to minimum required type(s)."))
                    .toLatin1()
                    .data());
        }
        if (!(meta->getHWAddInName().empty()) &&
            QString::compare(
                QLatin1String(meta->getHWAddInName().data()),
                value->getBasePlugin()->objectName(),
                Qt::CaseInsensitive) != 0)
        {
            return ito::RetVal::format(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr("AddIn must be of the following plugin: '%s'."))
                    .toLatin1()
                    .data(),
                meta->getHWAddInName().data());
        }
    }
    else if (mandatory && value == nullptr)
    {
        return ito::RetVal(
            ito::retError,
            0,
            (parseNamePrefix(name) + QObject::tr("AddIn must not be nullptr")).toLatin1().data());
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateCharArrayMeta(
    const ito::ParamMeta* meta, const char* values, size_t len, const char* name /*= nullptr*/)
{
    if (meta)
    {
        switch (meta->getType())
        {
        case ito::ParamMeta::rttiCharArrayMeta: {
            const ito::CharArrayMeta* cam = (const ito::CharArrayMeta*)meta;
            size_t steps = cam->getNumStepSize();

            if (len < cam->getNumMin() || len > cam->getNumMax())
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("length of char array out of range [%1, %2]")
                         .arg(cam->getNumMin())
                         .arg(cam->getNumMax()))
                        .toLatin1()
                        .data());
            }
            else if (steps > 1 && ((len - cam->getNumMin()) % steps) != 0)
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("length of char array does not fit to given step size [%1:%2:%3]")
                         .arg(cam->getNumMin())
                         .arg(cam->getNumStepSize())
                         .arg(cam->getNumMax()))
                        .toLatin1()
                        .data());
            }
            else
            {
                ito::RetVal ret;

                // this cast is allowed since CharArrayMeta is derived from CharMeta
                const ito::CharMeta* cm = (const ito::CharMeta*)meta;

                for (size_t i = 0; i < len; ++i)
                {
                    ret += validateCharMeta(cm, values[i]);
                }

                return ret;
            }
        }
        break;
        default:
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr(
                     "the given meta information does not fit an array of character values"))
                    .toLatin1()
                    .data());
        }
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateStringListMeta(
    const ito::ParamMeta* meta,
    const ito::ByteArray* values,
    size_t len,
    const char* name /*= nullptr*/)
{
    if (meta)
    {
        switch (meta->getType())
        {
        case ito::ParamMeta::rttiStringListMeta: {
            const ito::StringListMeta* slm = (const ito::StringListMeta*)meta;
            size_t steps = slm->getNumStepSize();

            if (len < slm->getNumMin() || len > slm->getNumMax())
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("length of string list out of range [%1, %2]")
                         .arg(slm->getNumMin())
                         .arg(slm->getNumMax()))
                        .toLatin1()
                        .data());
            }
            else if (steps > 1 && ((len - slm->getNumMin()) % steps) != 0)
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("length of string list does not fit to given step size [%1:%2:%3]")
                         .arg(slm->getNumMin())
                         .arg(slm->getNumStepSize())
                         .arg(slm->getNumMax()))
                        .toLatin1()
                        .data());
            }
            else
            {
                ito::RetVal ret;

                // this cast is allowed since CharArrayMeta is derived from CharMeta
                const ito::StringMeta* sm = (const ito::StringMeta*)meta;

                for (size_t i = 0; i < len; ++i)
                {
                    ret += validateStringMeta(sm, values[i].data());
                }

                return ret;
            }
        }
        break;
        default:
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr("the given meta information does not fit an array of string values"))
                    .toLatin1()
                    .data());
        }
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateIntArrayMeta(
    const ito::ParamMeta* meta, const int* values, size_t len, const char* name /*= nullptr*/)
{
    if (meta)
    {
        switch (meta->getType())
        {
        case ito::ParamMeta::rttiIntArrayMeta: {
            const ito::IntArrayMeta* cam = (const ito::IntArrayMeta*)meta;
            size_t steps = cam->getNumStepSize();

            if (len < cam->getNumMin() || len > cam->getNumMax())
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("length of integer array out of range [%1, %2]")
                         .arg(cam->getNumMin())
                         .arg(cam->getNumMax()))
                        .toLatin1()
                        .data());
            }
            else if (steps > 1 && ((len - cam->getNumMin()) % steps) != 0)
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr(
                         "length of integer array does not fit to given step size [%1:%2:%3]")
                         .arg(cam->getNumMin())
                         .arg(cam->getNumStepSize())
                         .arg(cam->getNumMax()))
                        .toLatin1()
                        .data());
            }
            else
            {
                ito::RetVal ret;

                // this cast is allowed since CharArrayMeta is derived from CharMeta
                const ito::IntMeta* im = (const ito::IntMeta*)meta;

                for (size_t i = 0; i < len; ++i)
                {
                    ret += validateIntMeta(im, values[i]);
                }

                return ret;
            }
        }
        break;
        case ito::ParamMeta::rttiIntervalMeta:
        case ito::ParamMeta::rttiRangeMeta: {
            const ito::IntervalMeta* drm = (const ito::IntervalMeta*)meta;
            if (len != 2)
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) + QObject::tr("length of integer array must be 2."))
                        .toLatin1()
                        .data());
            }
            int min = drm->getMin();
            int max = drm->getMax();
            int offset = drm->isIntervalNotRange() ? 0 : 1;

            // this is the difference between interval and range
            int range = offset + values[1] - values[0];

            int ivalStep = drm->getSizeStepSize();
            int step = drm->getStepSize();

            if ((values[0] < min) || (values[0] > values[1]) || (values[1] > max))
            {
                if (drm->isIntervalNotRange())
                {
                    if (values[0] > values[1])
                    {
                        return ito::RetVal(
                            ito::retError,
                            0,
                            (parseNamePrefix(name) +
                             QObject::tr(
                                 "The given integer array [%1,%2] is considered to be an interval "
                                 "but the first value is bigger than the second one")
                                 .arg(values[0])
                                 .arg(values[1]))
                                .toLatin1()
                                .data());
                    }
                    else
                    {
                        return ito::RetVal(
                            ito::retError,
                            0,
                            (parseNamePrefix(name) +
                             QObject::tr("The given integer array [%1,%2] is considered to be an "
                                         "interval but does not fit to the limits [%3,%4]")
                                 .arg(values[0])
                                 .arg(values[1])
                                 .arg(min)
                                 .arg(max))
                                .toLatin1()
                                .data());
                    }
                }
                else
                {
                    if (values[0] > values[1])
                    {
                        return ito::RetVal(
                            ito::retError,
                            0,
                            (parseNamePrefix(name) +
                             QObject::tr("The given integer array [%1,%2] is considered to be a "
                                         "range but the first value is bigger than the second one")
                                 .arg(values[0])
                                 .arg(values[1]))
                                .toLatin1()
                                .data());
                    }
                    else
                    {
                        return ito::RetVal(
                            ito::retError,
                            0,
                            (parseNamePrefix(name) +
                             QObject::tr("The given integer array [%1,%2] is considered to be a "
                                         "range but does not fit to the limits [%3,%4]")
                                 .arg(values[0])
                                 .arg(values[1])
                                 .arg(min)
                                 .arg(max))
                                .toLatin1()
                                .data());
                    }
                }
            }

            if (step > 1 && (((values[0] - min) % step) != 0))
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("The 1st value %1 does not fit to given step size [%2:%3:%4]")
                         .arg(values[0])
                         .arg(min)
                         .arg(step)
                         .arg(max))
                        .toLatin1()
                        .data());
            }
            else if (step > 1 && (((offset + values[1] - min) % step) != 0))
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("The 2nd value %1 does not fit to given step size [%2:%3:%4]")
                         .arg(values[1])
                         .arg(min)
                         .arg(step)
                         .arg(max))
                        .toLatin1()
                        .data());
            }

            if (range < drm->getSizeMin() || range > drm->getSizeMax())
            {
                if (drm->isIntervalNotRange())
                {
                    return ito::RetVal(
                        ito::retError,
                        0,
                        (parseNamePrefix(name) +
                         QObject::tr(
                             "The given integer array [v1,v2] is considered to be an interval but "
                             "the size of the interval (v2-v1) is out of bounds [%1,%2]")
                             .arg(drm->getSizeMin())
                             .arg(drm->getSizeMax()))
                            .toLatin1()
                            .data());
                }
                else
                {
                    return ito::RetVal(
                        ito::retError,
                        0,
                        (parseNamePrefix(name) +
                         QObject::tr("The given integer array [v1,v2] is considered to be a range "
                                     "but the size of the range (1+v2-v1) is out of bounds [%1,%2]")
                             .arg(drm->getSizeMin())
                             .arg(drm->getSizeMax()))
                            .toLatin1()
                            .data());
                }
            }

            if (ivalStep > 1 && ((range - drm->getSizeMin()) % ivalStep) != 0)
            {
                if (drm->isIntervalNotRange())
                {
                    return ito::RetVal(
                        ito::retError,
                        0,
                        (parseNamePrefix(name) +
                         QObject::tr("The size of the interval (bound2-bound1) does not fit to "
                                     "given step size [%1:%2:%3]")
                             .arg(drm->getSizeMin())
                             .arg(ivalStep)
                             .arg(drm->getSizeMax()))
                            .toLatin1()
                            .data());
                }
                else
                {
                    return ito::RetVal(
                        ito::retError,
                        0,
                        (parseNamePrefix(name) +
                         QObject::tr("The size of the range (1+bound2-bound1) does not fit to "
                                     "given step size [%1:%2:%3]")
                             .arg(drm->getSizeMin())
                             .arg(ivalStep)
                             .arg(drm->getSizeMax()))
                            .toLatin1()
                            .data());
                }
            }
        }
        break;
        case ito::ParamMeta::rttiRectMeta: {
            const ito::RectMeta* rm = (const ito::RectMeta*)meta;
            const ito::RangeMeta& widthMeta = rm->getWidthRangeMeta();
            const ito::RangeMeta& heightMeta = rm->getHeightRangeMeta();

            // check for width
            int min = widthMeta.getMin();
            int max = widthMeta.getMax();
            int ivalStep = widthMeta.getSizeStepSize();
            int step = widthMeta.getStepSize();

            // the step of the interval must always be bigger or equal
            // than the step size of the left and right border
            ivalStep = std::max(ivalStep, step);

            int widthMax = std::min(widthMeta.getSizeMax(), max - min + 1);

            if (values[2] < widthMeta.getSizeMin() || values[2] > widthMax)
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("roi[2] (width) is out of range [%1,%2]")
                         .arg(widthMeta.getSizeMin())
                         .arg(widthMax))
                        .toLatin1()
                        .data());
            }

            if (ivalStep > 1 && ((values[2] - widthMeta.getSizeMin()) % ivalStep) != 0)
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("roi[2] (width) does not fit to given step size [%1:%2:%3]")
                         .arg(widthMeta.getSizeMin())
                         .arg(ivalStep)
                         .arg(widthMax))
                        .toLatin1()
                        .data());
            }

            if ((values[0] < min) || (values[0] > max))
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("roi[0] (x0) is out of range [%1,%2]").arg(min).arg(max))
                        .toLatin1()
                        .data());
            }

            if ((values[0] + values[2] - 1) > max)
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr(
                         "right side of roi exceeds the maximal limit of %1 (reduce x0 or width)")
                         .arg(max))
                        .toLatin1()
                        .data());
            }

            if (step > 1 && (((values[0] - min) % step) != 0))
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("roi[0] (x0) does not fit to given step size [%1:%2:%3]")
                         .arg(min)
                         .arg(step)
                         .arg(max))
                        .toLatin1()
                        .data());
            }

            // check for height
            min = heightMeta.getMin();
            max = heightMeta.getMax();
            ivalStep = heightMeta.getSizeStepSize();
            step = heightMeta.getStepSize();

            // the step of the interval must always be bigger or equal
            // than the step size of the left and right border
            ivalStep = std::max(ivalStep, step);

            int heightMax = std::min(heightMeta.getSizeMax(), max - min + 1);

            if (values[3] < heightMeta.getSizeMin() || values[3] > heightMax)
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("roi[3] (height) is out of range [%1,%2]")
                         .arg(heightMeta.getSizeMin())
                         .arg(heightMax))
                        .toLatin1()
                        .data());
            }

            if (ivalStep > 1 && ((values[3] - heightMeta.getSizeMin()) % ivalStep) != 0)
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("roi[3] (height) does not fit to given step size [%1:%2:%3]")
                         .arg(heightMeta.getSizeMin())
                         .arg(ivalStep)
                         .arg(heightMax))
                        .toLatin1()
                        .data());
            }

            if ((values[1] < min) || (values[1] > max))
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("roi[1] (y0) is out of range [%1,%2]").arg(min).arg(max))
                        .toLatin1()
                        .data());
            }

            if ((values[1] + values[3] - 1) > max)
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr(
                         "bottom side of roi exceeds maximal limit of %1 (reduce y0 or height)")
                         .arg(max))
                        .toLatin1()
                        .data());
            }

            if (step > 1 && (((values[1] - min) % step) != 0))
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("roi[1] (y0) does not fit to given step size [%1:%2:%3]")
                         .arg(min)
                         .arg(step)
                         .arg(max))
                        .toLatin1()
                        .data());
            }
        }
        break;
        default:
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr("the given meta information does not fit an array of integer values"))
                    .toLatin1()
                    .data());
        }
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateDoubleArrayMeta(
    const ito::ParamMeta* meta, const double* values, size_t len, const char* name /*= nullptr*/)
{
    if (meta)
    {
        switch (meta->getType())
        {
        case ito::ParamMeta::rttiDoubleArrayMeta: {
            const ito::DoubleArrayMeta* cam = (const ito::DoubleArrayMeta*)meta;
            size_t steps = cam->getNumStepSize();

            if (len < cam->getNumMin() || len > cam->getNumMax())
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("length of double array out of range [%1, %2]")
                         .arg(cam->getNumMin())
                         .arg(cam->getNumMax()))
                        .toLatin1()
                        .data());
            }
            else if (steps > 1 && ((len - cam->getNumMin()) % steps) != 0)
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr(
                         "length of double array does not fit to given step size [%1:%2:%3]")
                         .arg(cam->getNumMin())
                         .arg(cam->getNumStepSize())
                         .arg(cam->getNumMax()))
                        .toLatin1()
                        .data());
            }
            else
            {
                ito::RetVal ret;
                const ito::DoubleMeta* cm = (const ito::DoubleMeta*)
                    meta; // this cast is allowed since CharArrayMeta is derived from CharMeta
                for (size_t i = 0; i < len; ++i)
                {
                    ret += validateDoubleMeta(cm, values[i]);
                }
                return ret;
            }
        }
        break;
        case ito::ParamMeta::rttiDoubleIntervalMeta: {
            const ito::DoubleIntervalMeta* drm = (const ito::DoubleIntervalMeta*)meta;
            if (len != 2)
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) + QObject::tr("length of double array must be 2."))
                        .toLatin1()
                        .data());
            }
            double min = drm->getMin();
            double max = drm->getMax();
            double range = values[1] - values[0];
            double rangeStep = drm->getSizeStepSize();
            double step = drm->getStepSize();

            if (values[0] > values[1])
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("The first value of the given double interval [%1,%2] is bigger "
                                 "than the second value.")
                         .arg(values[0])
                         .arg(values[1]))
                        .toLatin1()
                        .data());
            }
            else if (values[0] < min || values[1] > max)
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("The given double array [v1=%1,v2=%2] is considered to be an "
                                 "interval but does not fit to v1=[%3,v2], v2=[v1,%4]")
                         .arg(values[0])
                         .arg(values[1])
                         .arg(min)
                         .arg(max))
                        .toLatin1()
                        .data());
            }

            if (!fitToDoubleStepSize(min, step, values[0]))
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("The 1st value %1 does not fit to given step size [%2:%3:%4]")
                         .arg(values[0])
                         .arg(min)
                         .arg(step)
                         .arg(max))
                        .toLatin1()
                        .data());
            }
            else if (!fitToDoubleStepSize(min, step, values[1]))
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("The 2nd value %1 does not fit to given step size [%2:%3:%4]")
                         .arg(values[1])
                         .arg(min)
                         .arg(step)
                         .arg(max))
                        .toLatin1()
                        .data());
            }

            if (range < drm->getSizeMin() || range > drm->getSizeMax())
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("The given double array [v1,v2] is considered to be an interval "
                                 "but the size of the interval (v2-v1) is out of bounds [%1,%2]")
                         .arg(drm->getSizeMin())
                         .arg(drm->getSizeMax()))
                        .toLatin1()
                        .data());
            }

            if (!fitToDoubleStepSize(drm->getSizeMin(), rangeStep, range))
            {
                return ito::RetVal(
                    ito::retError,
                    0,
                    (parseNamePrefix(name) +
                     QObject::tr("The size of the interval (bound2-bound1) does not fit to given "
                                 "step size [%1:%2:%3]")
                         .arg(drm->getSizeMin())
                         .arg(rangeStep)
                         .arg(drm->getSizeMax()))
                        .toLatin1()
                        .data());
            }
        }
        break;
        default:
            return ito::RetVal(
                ito::retError,
                0,
                (parseNamePrefix(name) +
                 QObject::tr(
                     "the given meta information does not fit an array of character values"))
                    .toLatin1()
                    .data());
        }
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
QString ParamHelper::parseNamePrefix(const char* name)
{
    if (name == nullptr || strcmp(name, "") == 0)
    {
        return QString();
    }
    else
    {
        return QObject::tr("Parameter %1: ").arg(name);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateParam(
    const ito::Param& templateParam,
    const ito::ParamBase& param,
    bool strict /*= true*/,
    bool mandatory /*= false*/)
{

    bool hasIndex = false;
    int index;
    QString paramName;
    QString additionalName;
    ito::RetVal retVal = parseParamName(param.getName(), paramName, hasIndex, index, additionalName);

    if (retVal.containsError())
    {
        return retVal;
    }

    const char* name = param.getName();

    if (!hasIndex && (templateParam.getType() == param.getType()))
    {
        switch (templateParam.getType())
        {
        case ito::ParamBase::Char: {
            retVal += validateCharMeta(
                dynamic_cast<const ito::CharMeta*>(templateParam.getMeta()),
                param.getVal<char>(),
                name);
        }
        break;
        case ito::ParamBase::Int: {
            retVal += validateIntMeta(
                dynamic_cast<const ito::IntMeta*>(templateParam.getMeta()),
                param.getVal<int>(),
                name);
        }
        break;
        case ito::ParamBase::Double: {
            retVal += validateDoubleMeta(
                dynamic_cast<const ito::DoubleMeta*>(templateParam.getMeta()),
                param.getVal<double>(),
                name);
        }
        break;
        case ito::ParamBase::CharArray: {
            retVal += validateCharArrayMeta(
                templateParam.getMeta(), param.getVal<const char*>(), param.getLen(), name);
        }
        break;
        case ito::ParamBase::IntArray: {
            // check intArray, range, interval and rect
            retVal += validateIntArrayMeta(
                templateParam.getMeta(), param.getVal<const int*>(), param.getLen(), name);
        }
        break;
        case ito::ParamBase::DoubleArray: {
            retVal += validateDoubleArrayMeta(
                templateParam.getMeta(), param.getVal<const double*>(), param.getLen(), name);
        }
        break;
        case ito::ParamBase::StringList: {
            retVal += validateStringListMeta(
                templateParam.getMeta(),
                param.getVal<const ito::ByteArray*>(),
                param.getLen(),
                name);
        }
        break;
        case ito::ParamBase::String: {
            retVal += validateStringMeta(
                dynamic_cast<const ito::StringMeta*>(templateParam.getMeta()),
                param.getVal<const char*>(),
                mandatory,
                name);
        }
        break;
        case ito::ParamBase::HWRef & ito::paramTypeMask: {
            retVal += validateHWMeta(
                dynamic_cast<const ito::HWMeta*>(templateParam.getMeta()),
                (ito::AddInBase*)param.getVal<void*>(),
                mandatory,
                name);
        }
        case ito::ParamBase::DObjPtr & ito::paramTypeMask: {
            retVal += validateDObjMeta(
                dynamic_cast<const ito::DObjMeta*>(templateParam.getMeta()),
                param.getVal<const ito::DataObject*>(),
                mandatory,
                name);
        }
        break;
        }
    }
    else if (
        hasIndex && (templateParam.getType() & ito::ParamBase::Pointer) &&
        ((templateParam.getType() == (param.getType() ^ ito::ParamBase::Pointer)) ||
         (templateParam.getType() == ito::ParamBase::StringList &&
          param.getType() == ito::ParamBase::String)))
    {
        if (index < 0 || index >= templateParam.getLen())
        {
            retVal += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("Index value is out of range [0, %i]").toLatin1().data(),
                templateParam.getLen() - 1);
        }
        else
        {
            const ito::ParamMeta* tmplMeta = templateParam.getMeta();

            if (tmplMeta)
            {
                switch (templateParam.getType())
                {
                case ito::ParamBase::CharArray: {
                    // for charArray there is only CharArrayMeta possible, this allows only checking
                    // single values
                    retVal += validateCharMeta(
                        dynamic_cast<const ito::CharMeta*>(tmplMeta), param.getVal<char>(), name);
                }
                break;
                case ito::ParamBase::StringList: {
                    // for stringList there is only StringListMeta possible, this allows only
                    // checking single values
                    retVal += validateStringMeta(
                        dynamic_cast<const ito::StringMeta*>(tmplMeta),
                        param.getVal<const char*>(),
                        name);
                }
                break;
                case ito::ParamBase::IntArray: {
                    if (tmplMeta->getType() == ito::ParamMeta::rttiIntArrayMeta)
                    {
                        retVal += validateIntMeta(
                            dynamic_cast<const ito::IntMeta*>(tmplMeta), param.getVal<int>(), name);
                    }
                    else if (
                        tmplMeta->getType() == ito::ParamMeta::rttiRangeMeta ||
                        tmplMeta->getType() == ito::ParamMeta::rttiRectMeta ||
                        tmplMeta->getType() == ito::ParamMeta::rttiIntervalMeta)
                    {
                        // create a temporary set of the list (up to 4 values, more are not allowed
                        // for range, interval, rect)
                        int vals[4];
                        memcpy(
                            vals,
                            templateParam.getVal<int*>(),
                            sizeof(int) * templateParam.getLen());
                        vals[index] = param.getVal<int>();
                        retVal +=
                            validateIntArrayMeta(tmplMeta, vals, templateParam.getLen(), name);
                    }
                    else
                    {
                        retVal += ito::RetVal(
                            ito::retWarning,
                            0,
                            QObject::tr("index-based parameter cannot be validated since non-index "
                                        "based parameter has an unhandled type")
                                .toLatin1()
                                .data());
                    }
                }
                break;
                case ito::ParamBase::DoubleArray: {
                    if (tmplMeta->getType() == ito::ParamMeta::rttiDoubleArrayMeta)
                    {
                        retVal += validateDoubleMeta(
                            dynamic_cast<const ito::DoubleMeta*>(tmplMeta), param.getVal<double>());
                    }
                    else if (tmplMeta->getType() == ito::ParamMeta::rttiDoubleIntervalMeta)
                    {
                        // create a temporary set of the list (up to 2 values, more are not allowed
                        // for interval)
                        double vals[2];
                        memcpy(
                            vals,
                            templateParam.getVal<int*>(),
                            sizeof(double) * templateParam.getLen());
                        vals[index] = param.getVal<double>();
                        retVal +=
                            validateDoubleArrayMeta(tmplMeta, vals, templateParam.getLen(), name);
                    }
                    else
                    {
                        retVal += ito::RetVal(
                            ito::retWarning,
                            0,
                            QObject::tr("index-based parameter cannot be validated since non-index "
                                        "based parameter has an unhandled type")
                                .toLatin1()
                                .data());
                    }
                }
                break;
                default: {
                    retVal += ito::RetVal(
                        ito::retError,
                        0,
                        QObject::tr("Index-based parameter name requires an array-type parameter.")
                            .toLatin1()
                            .data());
                }
                break;
                }
            }
        }
    }
    else if (!strict)
    {
        bool ok = false;
        ito::ParamBase p = convertParam(param, templateParam.getType(), &ok);
        if (ok)
        {
            retVal += validateParam(templateParam, p, true);
        }
        else
        {
            retVal += ito::RetVal(
                ito::retError,
                0,
                QObject::tr("Parameter could not be converted to destination type.")
                    .toLatin1()
                    .data());
        }
    }
    else
    {
        retVal += ito::RetVal(
            ito::retError,
            0,
            QObject::tr("type of parameter does not fit to requested parameter type")
                .toLatin1()
                .data());
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::validateAndCastParam(
    const ito::Param& templateParam,
    ito::ParamBase& param,
    bool strict /*= true*/,
    bool mandatory /*= false*/,
    bool roundToSteps /*= false*/)
{
    bool hasIndex = false;
    int index;
    QString paramName;
    QString additionalName;
    ito::RetVal retVal = parseParamName(param.getName(), paramName, hasIndex, index, additionalName);

    if (retVal.containsError())
    {
        return retVal;
    }

    const char* name = param.getName();

    if (!hasIndex && (templateParam.getType() == param.getType()))
    {
        switch (templateParam.getType())
        {
        case ito::ParamBase::Char: {
            retVal += validateCharMeta(
                dynamic_cast<const ito::CharMeta*>(templateParam.getMeta()),
                param.getVal<char>(),
                name);
        }
        break;
        case ito::ParamBase::Int: {
            retVal += validateIntMeta(
                dynamic_cast<const ito::IntMeta*>(templateParam.getMeta()),
                param.getVal<int>(),
                name);
        }
        break;
        case ito::ParamBase::Double: {
            retVal += validateDoubleMetaAndRoundToStepSize(
                dynamic_cast<const ito::DoubleMeta*>(templateParam.getMeta()),
                param,
                roundToSteps,
                name);
        }
        break;
        case ito::ParamBase::CharArray: {
            retVal += validateCharArrayMeta(
                templateParam.getMeta(), param.getVal<const char*>(), param.getLen(), name);
        }
        break;
        case ito::ParamBase::IntArray: {
            retVal += validateIntArrayMeta(
                templateParam.getMeta(), param.getVal<const int*>(), param.getLen(), name);
        }
        break;
        case ito::ParamBase::DoubleArray: {
            retVal += validateDoubleArrayMeta(
                templateParam.getMeta(), param.getVal<const double*>(), param.getLen(), name);
        }
        break;
        case ito::ParamBase::String: {
            retVal += validateStringMeta(
                dynamic_cast<const ito::StringMeta*>(templateParam.getMeta()),
                param.getVal<char*>(),
                mandatory,
                name);
        }
        break;
        case ito::ParamBase::HWRef & ito::paramTypeMask: {
            retVal += validateHWMeta(
                dynamic_cast<const ito::HWMeta*>(templateParam.getMeta()),
                (ito::AddInBase*)param.getVal<void*>(),
                mandatory,
                name);
        }
        case ito::ParamBase::DObjPtr & ito::paramTypeMask: {
            retVal += validateDObjMeta(
                dynamic_cast<const ito::DObjMeta*>(templateParam.getMeta()),
                param.getVal<const ito::DataObject*>(),
                mandatory,
                name);
        }
        break;
        }
    }
    else if (
        hasIndex && (templateParam.getType() & ito::ParamBase::Pointer) &&
        (templateParam.getType() == (param.getType() ^ ito::ParamBase::Pointer)))
    {
        if (index < 0 || index >= templateParam.getLen())
        {
            retVal += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("Index value is out of range [0, %i]").toLatin1().data(),
                templateParam.getLen() - 1);
        }

        const ito::ParamMeta* tmplMeta = templateParam.getMeta();

        if (tmplMeta)
        {
            switch (templateParam.getType())
            {
            case ito::ParamBase::CharArray: {
                // for charArray there is only CharArrayMeta possible, this allows only checking
                // single values
                retVal += validateCharMeta(
                    dynamic_cast<const ito::CharMeta*>(tmplMeta), param.getVal<char>(), name);
            }
            break;
            case ito::ParamBase::IntArray: {
                if (tmplMeta->getType() == ito::ParamMeta::rttiIntArrayMeta)
                {
                    retVal += validateIntMeta(
                        dynamic_cast<const ito::IntMeta*>(tmplMeta), param.getVal<int>(), name);
                }
                else
                {
                    // temporarily set new value into given template, check it and reset the value
                    // in the template.
                    int* values = templateParam.getVal<int*>();
                    int old = values[index];
                    values[index] = param.getVal<int>();
                    retVal += validateIntArrayMeta(tmplMeta, values, templateParam.getLen(), name);
                    values[index] = old;
                }
            }
            break;
            case ito::ParamBase::DoubleArray: {
                if (tmplMeta->getType() == ito::ParamMeta::rttiDoubleArrayMeta)
                {
                    retVal += validateDoubleMeta(
                        dynamic_cast<const ito::DoubleMeta*>(tmplMeta),
                        param.getVal<double>(),
                        name);
                }
                else
                {
                    retVal += ito::RetVal(
                        ito::retWarning,
                        0,
                        QObject::tr("index-based parameter cannot be validated since non-index "
                                    "based parameter is an interval")
                            .toLatin1()
                            .data());
                }
            }
            break;
            default: {
                retVal += ito::RetVal(
                    ito::retError,
                    0,
                    QObject::tr("Index-based parameter name requires an array-type parameter.")
                        .toLatin1()
                        .data());
            }
            break;
            }
        }
    }
    else if (!strict)
    {
        bool ok = false;
        ito::ParamBase p = convertParam(param, templateParam.getType(), &ok);
        if (ok)
        {
            retVal += validateAndCastParam(templateParam, p, true, roundToSteps);
            param = p;
        }
        else
        {
            retVal += ito::RetVal(
                ito::retError,
                0,
                QObject::tr("Parameter could not be converted to destination type.")
                    .toLatin1()
                    .data());
        }
    }
    else
    {
        retVal += ito::RetVal(
            ito::retError,
            0,
            QObject::tr("type of parameter does not fit to requested parameter type")
                .toLatin1()
                .data());
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::ParamBase ParamHelper::convertParam(
    const ito::ParamBase& source, int destType, bool* ok /*= nullptr*/)
{
    int sourceType = source.getType();
    bool ok2;

    if (ok)
    {
        *ok = true;
    }

    if (sourceType == (destType & (int)ito::paramTypeMask))
    {
        return source;
    }


    switch (destType & ito::paramTypeMask)
    {
    case ito::ParamBase::Int: {
        if (source.isNumeric())
        {
            return ito::ParamBase(source.getName(), destType, source.getVal<int32>());
        }
        else if (sourceType & ito::ParamBase::String)
        {
            int val = QByteArray(source.getVal<char*>()).toInt(&ok2);
            if (ok2)
            {
                return ito::ParamBase(source.getName(), destType, val);
            }
        }
    }
    break;

    case ito::ParamBase::Char: {
        if (source.isNumeric())
        {
            return ito::ParamBase(source.getName(), destType, source.getVal<char>());
        }
        else if (sourceType & ito::ParamBase::String)
        {
            char val = QByteArray(source.getVal<char*>()).toShort(&ok2);
            if (ok2)
            {
                return ito::ParamBase(source.getName(), destType, val);
            }
        }
    }
    break;

    case ito::ParamBase::Double: {
        if (source.isNumeric())
        {
            return ito::ParamBase(source.getName(), destType, source.getVal<float64>());
        }
        else if (sourceType & ito::ParamBase::String)
        {
            double val = QByteArray(source.getVal<char*>()).toDouble(&ok2);
            if (ok2)
            {
                return ito::ParamBase(source.getName(), destType, val);
            }
        }
    }
    break;

    case ito::ParamBase::String: {
        if (source.isNumeric())
        {
            return ito::ParamBase(
                source.getName(),
                destType,
                QString::number(source.getVal<double>()).toLatin1().data());
        }
    }
    break;
    }

    if (ok)
    {
        *ok = false;
    }

    return ParamBase();
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamHelper::getParamFromMapByKey(
    QMap<QString, ito::Param>& paramMap,
    const QString& key,
    QMap<QString, ito::Param>::iterator& found,
    bool errorIfReadOnly)
{
    if (key == "")
    {
        return ito::RetVal(
            ito::retError, 0, QObject::tr("Name of given parameter is empty.").toLatin1().data());
    }

    QMap<QString, ito::Param>::iterator it = paramMap.find(key);
    if (it != paramMap.end())
    {
        if (errorIfReadOnly && (it->getFlags() & ito::ParamBase::Readonly))
        {
            return ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("Parameter '%1' is read only.").arg(key).toLatin1().data());
        }
        else if (it->getFlags() & ito::ParamBase::NotAvailable)
        {
            return ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("Parameter '%1' is (temporarily) not available.")
                    .arg(key)
                    .toLatin1()
                    .data());
        }

        found = it;
    }
    else
    {
        return ito::RetVal::format(
            ito::retError, 0, QObject::tr("Parameter '%1' not found.").arg(key).toLatin1().data());
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! parses parameter name with respect to regular expression, assigned for parameter-communication
//! with plugins
/*!
    This method parses any parameter-name with respect to the rules defined for possible names of
   plugin-parameters.

    The regular expression used for the check is "^([a-zA-Z]+\\w*)(\\[(\\d+)\\]){0,1}(:(.*)){0,1}$"

    Then the components are:

    [0] full string
    [1] PARAMNAME
    [2] [INDEX] or empty-string if no index is given
    [3] INDEX or empty-string if no index is given
    [4] :ADDITIONALTAG or empty-string if no tag is given
    [5] ADDITIONALTAG or empty-string if no tag is given

    \param [in] name is the raw parameter name
    \param [out] paramName is the real parameter name (first part of name; part before the first
   opening bracket ('[') or if not available the first colon (':')) \param [out] hasIndex indicates
   whether the name contains an index part (defined by a number within two brackets (e.g.
   '[NUMBER]'), which has to be appended to the paramName \param [out] index is the fixed-point
   index value or -1 if hasIndex is false \param [out] additionalTag is the remaining string of name
   which is the part after the first colon (':'). If an index part exists, the first colon after the
   index part is taken.
*/
ito::RetVal ParamHelper::parseParamName(
    const QString& name, QString& paramName, bool& hasIndex, int& index, QString& additionalTag)
{
    ito::RetVal retValue;
    QRegularExpression rx("^([a-zA-Z]+\\w*)(\\[(\\d+)\\]){0,1}(:(.*)){0,1}$");

    paramName = QString();
    hasIndex = false;
    index = -1;
    additionalTag = QString();
    auto match = rx.match(name);

    if (!match.hasMatch())
    {
        retValue +=
            ito::RetVal(ito::retError, 0, QObject::tr("invalid parameter name").toLatin1().data());
    }
    else
    {
        QStringList pname = match.capturedTexts();
        paramName = pname[1];

        if (pname.size() >= 4)
        {
            if (!pname[3].isEmpty())
            {
                index = pname[3].toInt(&hasIndex);
            }
        }

        if (pname.size() >= 6)
        {
            additionalTag = pname[5];
        }
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::RetVal ParamHelper::getItemFromArray(
    const ito::Param& arrayParam, const int index, ito::Param& itemParam)
{
    ito::RetVal retval;
    int len = arrayParam.getLen();

    if (index < 0 || index >= len)
    {
        return ito::RetVal::format(
            ito::retError,
            0,
            QObject::tr("index is ouf of range [0, %i]").toLatin1().data(),
            len - 1);
    }

    switch (arrayParam.getType())
    {
    case ito::ParamBase::CharArray& ito::paramTypeMask:
    case ito::ParamBase::IntArray& ito::paramTypeMask:
    case ito::ParamBase::DoubleArray& ito::paramTypeMask:
    case ito::ParamBase::ComplexArray& ito::paramTypeMask: {
        itemParam = arrayParam[index];
        break;
    }
    default:
        retval += ito::RetVal(ito::retError, 0, QObject::tr("param is no array").toLatin1().data());
        break;
    }
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::Param ParamHelper::getParam(
    const ito::Param& param, const bool hasIndex, const int index, ito::RetVal& ret)
{
    if (!hasIndex)
    {
        return param;
    }
    else
    {
        // check whether param is an array type
        ito::uint32 type = param.getType();
        if (type == (ito::ParamBase::IntArray & ito::paramTypeMask) ||
            type == (ito::ParamBase::DoubleArray & ito::paramTypeMask) ||
            type == (ito::ParamBase::CharArray & ito::paramTypeMask) ||
            type == (ito::ParamBase::ComplexArray & ito::paramTypeMask))
        {
            ito::Param p;
            ret += getItemFromArray(param, index, p);
            return p;
        }
        else
        {
            ret += ito::RetVal(
                ito::retError,
                0,
                QObject::tr("Paramater is no array type. Indexing not possible.")
                    .toLatin1()
                    .data());
        }
    }
    return ito::Param();
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::RetVal ParamHelper::updateParameters(
    QMap<QString, ito::Param>& paramMap, const QVector<QSharedPointer<ito::ParamBase>>& values)
{
    ito::RetVal retval;
    QString name;

    for (int i = 0; i < values.size(); ++i)
    {
        name = QLatin1String(values[i]->getName());

        if (paramMap.contains(name))
        {
            retval += paramMap[name].copyValueFrom(values[i].data());
        }
        else
        {
            retval += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("Parameter '%s' does not exist").toLatin1().data(),
                name.toLatin1().data());
        }
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ bool ParamHelper::fitToDoubleStepSize(double min, double step, double val)
{
    static double eps = std::numeric_limits<double>::epsilon();

    if (step >= eps)
    {
        // the following inequation must hold for an integer value R:
        // minVal - eps + R(step - eps) < value < minVal + eps + R(step + eps)
        // this leads to a comparison of R1 and R2 as follows:

        int R1 = std::floor((val - min + eps) / (step - eps)); // R for left inequation
        int R2 = std::ceil((val - min - eps) / (step + eps)); // R for right inequation
        if (R1 != R2)
        {
            return false;
        }
    }

    return true;
}

} // end namespace ito
