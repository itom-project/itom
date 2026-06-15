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

#ifndef ALGOINTERFACEVALIDATOR_H
#define ALGOINTERFACEVALIDATOR_H

#if !defined(Q_MOC_RUN) || defined(ADDINMGR_DLL) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

#include "addInMgrDefines.h"
#include "../common/addInInterface.h"

#include <qscopedpointer.h>

namespace ito {

class AlgoInterfaceValidatorPrivate; //forward declaration

class ADDINMGR_EXPORT AlgoInterfaceValidator : public QObject
{
public:
    AlgoInterfaceValidator(ito::RetVal &retValue);
    ~AlgoInterfaceValidator();

    ito::RetVal addInterface(ito::AddInAlgo::tAlgoInterface iface, QVector<ito::Param> &mandParams, QVector<ito::Param> &outParams, int maxNumMand, int maxNumOpt, int maxNumOut);
    bool isValidFilter(const ito::AddInAlgo::FilterDef &filter, ito::RetVal &ret, QStringList &tags) const;
    bool isValidWidget(const ito::AddInAlgo::AlgoWidgetDef &widget, ito::RetVal &ret, QStringList &tags) const;
    ito::RetVal getInterfaceParameters(ito::AddInAlgo::tAlgoInterface iface, QVector<ito::ParamBase> &mandParams, QVector<ito::ParamBase> &outParams) const;

protected:
    ito::RetVal init(void);
    bool isValid(const ito::AddInAlgo::tAlgoInterface iface, const ito::AddInAlgo::t_filterParam filterParamFunc, ito::RetVal &ret) const;
    bool getTags(const ito::AddInAlgo::tAlgoInterface iface, const QString &metaInformation, QStringList &tags) const;

private:
    QScopedPointer<AlgoInterfaceValidatorPrivate> d_ptr; //!> self-managed pointer to the private class container (deletes itself if d_ptr is destroyed)
    Q_DECLARE_PRIVATE(AlgoInterfaceValidator);
};

} //end namespace ito

#endif // #if !defined(Q_MOC_RUN) || defined(ADDINMGR_DLL)

#endif
