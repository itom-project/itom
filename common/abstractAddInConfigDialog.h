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

#ifndef ABSTRACTADDINCONFIGDIALOG_H
#define ABSTRACTADDINCONFIGDIALOG_H

#include "retVal.h"
#include "typeDefs.h"
#include "sharedStructuresQt.h"
#include "commonGlobal.h"
#include "../common/interval.h"
#include "../common/qtMetaTypeDeclarations.h"

#include <QtWidgets/qdialog.h>
#include <qmap.h>
#include <qsharedpointer.h>





//forward declarations
namespace ito
{
    class AddInBase; //forward declaration
    class AbstractAddInConfigDialogPrivate; //forward declaration

    //----------------------------------------------------------------------------------------------------------------------------------
    /** @class AbstractAddInConfigDialog
    *   @brief abstract base class for configuration dialogs of plugins
    *
    *   Inherit your plugin's configuration dialog from this class in order to benefit from many simplified and unified
    *   processes. Since the plugin runs in its own thread while the configuration dialog runs in the main thread, the communication
    *   between the plugin and its configuration dialog needs to be done via signal/slot connections or general invocations.
    *
    *   The methods of this class help in this process.
    *
    *   Communication plugin -> configuration dialog:
    *   - After the construction of the configuration dialog, the plugin automatically invokes the slot parametersChanged with all internal parameters of
          the plugin.
    *   - overload parametersChanged in order to set all widgets to the current values of the plugin
    *
    *   Communication configuration dialog -> plugin:
    *   - Just emit the accept() signal if the ok-button is clicked
    *   - Emit the reject() signal for the cancel button
    *   - call the applyParameters() method once an optional apply-button is clicked
    *   - Overload the applyParameters() method in order to send all changed values to the plugin (using setPluginParameter or setPluginParameters).
    *   - If the dialog is exit using the ok-button, applyParameters is automatically called as well
    *
    *   For implementing a configuration dialog, overload ito::AddInBase::hasConfigDialog and return 1. Additionally overload ito::AddInBase::showConfigDialog
        and create an instance of the configuration dialog that is directly passed to the api-function apiShowConfigurationDialog.
    */
    class ITOMCOMMONQT_EXPORT AbstractAddInConfigDialog : public QDialog
    {
        Q_OBJECT

        public:
            //! constructor.
            AbstractAddInConfigDialog(ito::AddInBase *plugin);

            //! destructor
            virtual ~AbstractAddInConfigDialog();

            /**
            * MessageLevel enumeration
            * defines whether warnings and/or errors that might occur during some executions should be displayed with a message box.
            */
            enum MessageLevel
            {
                msgLevelNo = 0,          /*!< no messagebox should information about warnings or errors */
                msgLevelErrorOnly = 1,   /*!< a message box should only inform about errors */
                msgLevelWarningOnly = 2, /*!< a message box should only inform about warnings */
                msgLevelWarningAndError = msgLevelErrorOnly | msgLevelWarningOnly /*!< a message box should inform about warnings and errors */
            };

            //! method to send changed parameters back to the plugin (needs to be overwritten)
            /*!
                This method is automatically called once the OK-button (accept-role) of the configuration dialog
                is clicked and should also be called if the apply-button is clicked (only this needs to be done by
                the plugin).

                In this method, check all changed parameters and send them to the plugin using setPluginParameter
                or setPluginParameters. This method needs to be implemented in every single configuration dialog.
            */
            virtual ito::RetVal applyParameters() = 0;

        protected:
            //! invokes AddInBase::setParam of plugin in order to set the given parameter
            /*!
                Use this method to thread-safely set any desired parameter of the plugin. No direct call of setParam of the plugin
                is possible since the plugin usually runs in a secondary thread.

                If the method was successful, the map m_currentParameters is updated with respect to param.

                \param param is the parameter to set.
                \param msgLevel defines if any warnings or errors should be displayed within an appropriate message box.
                \return RetVal returns retOk or any other warning or error depending on success.

                \sa observeInvocation
            */
            virtual ito::RetVal setPluginParameter(QSharedPointer<ito::ParamBase> param, MessageLevel msgLevel = msgLevelWarningAndError);

            //! invokes AddInBase::setParamVector of plugin in order to set multiple given parameters
            /*!
                Use this method to thread-safely set any desired parameters of the plugin. No direct call of setParam or setParamVector of the plugin
                is possible since the plugin usually runs in a secondary thread.

                If the method was successful, the map m_currentParameters is updated with respect to params.

                \param params is a vector of parameters to set.
                \param msgLevel defines if any warnings or errors should be displayed within an appropriate message box.
                \return RetVal returns retOk or any other warning or error depending on success.

                \sa observeInvocation
            */
            virtual ito::RetVal setPluginParameters(const QVector<QSharedPointer<ito::ParamBase> > params, MessageLevel msgLevel = msgLevelWarningAndError);

            //! observes the status of the given semaphore and returns after the semaphore has been released or a timeout occurred
            /*!
                This helper method is mainly called by setPluginParameter or setPluginParameters in order to wait until the parameters
                have been set within the plugin. The corresponding return value is obtained and returned or displayed in a message box (if desired).

                Directly call this method after having invoked any other slot where an instance of ItomSharedSemaphore has been passed as wait condition.
                This method returns if the wait condition has been released by the receiver, if the invocation failed or if a timeout occurred. For any possible
                timeout the AddInBase::isAlive flag is continuously evaluated.

                \param waitCond is the wait condition passed to the invokeMethod command.
                \param msgLevel defines if any warnings or errors should be displayed within an appropriate message box.
                \return RetVal returns retOk or any other warning or error depending on success.

                \sa setPluginParameter, setPluginParameters
            */
            virtual ito::RetVal observeInvocation(ItomSharedSemaphore *waitCond, MessageLevel msgLevel) const;

            QMap<QString, ito::Param> m_currentParameters; /*! use this map to save the current values of all parameters. For instance it is conventient to copy the map given in parametersChanged to this map */

        private:
            AbstractAddInConfigDialogPrivate* d; /*! private data pointer of this class. */

        public slots:
            //! slot invoked if any parameter of the plugin has been changed.
            /*!
                overload this method in order reset the widgets depending on the current states of the parameter.
                The first invocation can also be used in order to configure the configuration dialog depending on the
                current set of available parameters. This slot is automatically invoked right after the construction
                of the configuration dialog

                \params map of parameters (usually equal to m_params member of ito::AddInBase)
            */
            virtual void parametersChanged(QMap<QString, ito::Param> params) = 0;
    };
} //end namespace ito

#endif //ABSTRACTADDINCONFIGDIALOG_H
