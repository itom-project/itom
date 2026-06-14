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

#ifndef APIFUNCTIONS_H
#define APIFUNCTINOS_H

#include "../common/apiFunctionsInc.h"

namespace ito
{
    class AbstractAddInConfigDialog;

    class ApiFunctions
    {
        public:
            ApiFunctions();
            ~ApiFunctions();

            //! function called by apiFilterGetFunc
            static ito::RetVal mfilterGetFunc(
                const QString &name,
                ito::AddInAlgo::FilterDef *&FilterDef);

            //! function called by apiFilterCall
            static ito::RetVal mfilterCall(
                const QString &name,
                QVector<ito::ParamBase> *paramsMand,
                QVector<ito::ParamBase> *paramsOpt,
                QVector<ito::ParamBase> *paramsOut);

            //! function called by apiFilterCallExt
            static ito::RetVal mfilterCallExt(
                const QString &name,
                QVector<ito::ParamBase> *paramsMand,
                QVector<ito::ParamBase> *paramsOpt,
                QVector<ito::ParamBase> *paramsOut,
                QSharedPointer<ito::FunctionCancellationAndObserver> observer);

            //! function called by apiFilterParamBase
            static ito::RetVal mfilterParamBase(
                const QString &name,
                QVector<ito::ParamBase> *paramsMand,
                QVector<ito::ParamBase> *paramsOpt,
                QVector<ito::ParamBase> *paramsOut);

            //! function called by apiFilterParam
            static ito::RetVal mfilterParam(
                const QString &name,
                QVector<ito::Param> *paramsMand,
                QVector<ito::Param> *paramsOpt,
                QVector<ito::Param> *paramsOut);

            //! function called by apiFilterVersion
            static ito::RetVal mfilterVersion(const QString &name, int &version);

            //! function called by apiFilterAuthor
            static ito::RetVal mfilterAuthor(const QString &name, QString &author);

            //! function called by apiFilterPluginName
            static ito::RetVal mfilterPluginName(const QString &name, QString &pluginName);

            //! function called by apiAddInGetInitParams
            static ito::RetVal maddInGetInitParams(
                const QString &name,
                const int pluginType,
                int *pluginNum,
                QVector<ito::Param> *&paramsMand,
                QVector<ito::Param> *&paramsOpt);

            //! function called by apiAddInOpenActuator
            static ito::RetVal maddInOpenActuator(
                const QString &name,
                const int pluginNum,
                const bool autoLoadParams,
                QVector<ito::ParamBase> *paramsMand,
                QVector<ito::ParamBase> *paramsOpt,
                ito::AddInActuator *&instance);

            //! function called by apiAddInOpenDataIO
            static ito::RetVal maddInOpenDataIO(
                const QString &name,
                const int pluginNum,
                const bool autoLoadParams,
                QVector<ito::ParamBase> *paramsMand,
                QVector<ito::ParamBase> *paramsOpt,
                ito::AddInDataIO *&instance);

            //! function called by apiAddInClose
            static ito::RetVal maddInClose(ito::AddInBase *instance);

            //! function called by apiCreateFromDataObject
            static ito::DataObject* mcreateFromDataObject(
                const ito::DataObject *dObj,
                int nrDims,
                ito::tDataType type,
                int *sizeLimits = NULL,
                ito::RetVal *retval = NULL);

            //! function called by apiCreateFromNamedDataObject
            static ito::DataObject* mcreateFromNamedDataObject(
                const ito::DataObject *dObj,
                int nrDims,
                ito::tDataType type,
                const char *name = NULL,
                int *sizeLimits = NULL,
                ito::RetVal *retval = NULL);

            //! function called by apiGetCurrentWorkingDir
            static QString getCurrentWorkingDir(void);

            //! function called by apiShowConfigurationDialog
            static ito::RetVal mshowConfigurationDialog(
                ito::AddInBase *plugin,
                ito::AbstractAddInConfigDialog *configDialogInstance);

            //! function called by apiSendParamToPyWorkspace
            // function moved to apiFunctionsGui
            //static ito::RetVal sendParamToPyWorkspaceThreadSafe(const QString &varname, const QSharedPointer<ito::ParamBase> &value);

            //! function called by apiSendParamsToPyWorkspace
            // function moved to apiFunctionsGui
            //static ito::RetVal sendParamsToPyWorkspaceThreadSafe(const QStringList &varnames, const QVector<QSharedPointer<ito::ParamBase> > &values);

            //! substitute for removed functions
            static ito::RetVal removed(...);

            //! get itom settings file
            static QString getSettingsFile(void);

            //! setter function only used on AddInManager startup
            static ito::RetVal setSettingsFile(QString settingsFile);

        private:
            int m_loadFPointer;
            static QString m_settingsFile;
    };
}

#endif //APIFUNTIONS_H
