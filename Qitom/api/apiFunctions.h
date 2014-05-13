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

#ifndef APIFUNCTIONS_H
#define APIFUNCTINOS_H

#include "../../common/apiFunctionsInc.h"

namespace ito 
{
    class AbstractAddInConfigDialog;

    class ApiFunctions 
    {
        public:
            ito::RetVal setFPointer(void);
            ApiFunctions();
            ~ApiFunctions();

            static ito::RetVal mfilterGetFunc(const QString &name, ito::AddInAlgo::FilterDef *&FilterDef);
            static ito::RetVal mfilterCall(const QString &name, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut);
            static ito::RetVal mfilterParamBase(const QString &name, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut);
            static ito::RetVal mfilterParam(const QString &name, QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut);
            static ito::RetVal maddInGetInitParams(const QString &name, const int pluginType, int *pluginNum, QVector<ito::Param> *&paramsMand, QVector<ito::Param> *&paramsOpt);
            static ito::RetVal maddInOpenActuator(const QString &name, const int pluginNum, const bool autoLoadParams, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ito::AddInActuator *&actuator);
            static ito::RetVal maddInOpenDataIO(const QString &name, const int pluginNum, const bool autoLoadParams, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ito::AddInDataIO *&dataIO);

            static ito::DataObject* mcreateFromDataObject(const ito::DataObject *dObj, int nrDims, ito::tDataType type, int *sizeLimits = NULL, ito::RetVal *retval = NULL);
            static ito::DataObject* mcreateFromNamedDataObject(const ito::DataObject *dObj, int nrDims, ito::tDataType type, const char *name = NULL, int *sizeLimits = NULL, ito::RetVal *retval = NULL);
            static QString getCurrentWorkingDir(void);

            static ito::RetVal mshowConfigurationDialog(ito::AddInBase *plugin, ito::AbstractAddInConfigDialog *configDialogInstance);

        private:
            int m_loadFPointer;
    };
}

#endif //APIFUNTIONS_H
