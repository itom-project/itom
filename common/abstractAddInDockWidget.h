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

#ifndef ABSTRACTADDINDOCKWIDGET_H
#define ABSTRACTADDINDOCKWIDGET_H

#include "retVal.h"
#include "typeDefs.h"
#include "sharedStructuresQt.h"
#include "commonGlobal.h"

#include <QtWidgets/qwidget.h>
#include <qmap.h>
#include <qsharedpointer.h>

//forward declarations
namespace ito
{
    class AddInBase; //forward declaration
    class AbstractAddInDockWidgetPrivate; //forward declaration

    //----------------------------------------------------------------------------------------------------------------------------------
    /** @class AbstractAddInDockWidget
    *   @brief abstract base class for dock widgets (toolboxes) of plugins
    *
    *   Inherit your plugin's dock widget from this class in order to benefit from many simplified and unified
    *   processes. Since the plugin runs in its own thread while the dock widget runs in the main thread, the communication
    *   between the plugin and its dock widget needs to be done via signal/slot connections or general invocations.
    *
    *   The methods of this class help in this process.
    *
    *   Communication plugin -> dock widget:
    *   - once any parameter in the plugin is changed, the signal ito::AddInBase::parametersChanged is emitted.
    *   - this signal should be connected to the slot AbstractAddInDockWidget::parametersChanged.
    *   - Overload this method in order to get informed about any changes and adapt the widgets to the current values.
    *   - Changes of the plugin's identifier (firstly set after construction of the dock widget) are signalled via the slot identifierChanged.
    *     Overload this slot in order to use the information.
    *
    *   Communication dock widget -> plugin:
    *   - use setPluginParameter in order to set one parameter in the plugin
    *   - use setPluginParameters in order to set multiple parameters in the plugin
    */
    class ITOMCOMMONQT_EXPORT AbstractAddInDockWidget : public QWidget
    {
        Q_OBJECT

        public:
            //! constructor.
            AbstractAddInDockWidget(ito::AddInBase *plugin);

            //! destructor
            virtual ~AbstractAddInDockWidget();

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

        protected:
            //! invokes AddInBase::setParam of plugin in order to set the given parameter
            /*!
                Use this method to thread-safely set any desired parameter of the plugin. No direct call of setParam of the plugin
                is possible since the plugin usually runs in a secondary thread.

                \param param is the parameter to set.
                \param msgLevel defines if any warnings or errors should be displayed within an appropriate message box.
                \return RetVal returns retOk or any other warning or error depending on success.

                \sa observeInvocation
            */
            ito::RetVal setPluginParameter(QSharedPointer<ito::ParamBase> param, MessageLevel msgLevel = msgLevelWarningAndError) const;

            //! invokes AddInBase::setParamVector of plugin in order to set multiple given parameters
            /*!
                Use this method to thread-safely set any desired parameters of the plugin. No direct call of setParam or setParamVector of the plugin
                is possible since the plugin usually runs in a secondary thread.

                \param params is a vector of parameters to set.
                \param msgLevel defines if any warnings or errors should be displayed within an appropriate message box.
                \return RetVal returns retOk or any other warning or error depending on success.

                \sa observeInvocation
            */
            ito::RetVal setPluginParameters(const QVector<QSharedPointer<ito::ParamBase> > params, MessageLevel msgLevel = msgLevelWarningAndError) const;

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
            ito::RetVal observeInvocation(ItomSharedSemaphore *waitCond, MessageLevel msgLevel) const;

            //! invokes AddInActuator::setPosRel or AddInActuator::setPosAbs of plugin in order to force a movement of one or multiple axes
            /*!
                Use this method to thread-safely position one or multiple axes of an actuator plugin. Do not directly call setPosRel
                or setPosAbs of the plugin, since this is not thread safe.

                This method waits until the movement ended (if the movement has been configured to by synchronous, plugin parameter)

                \param axes is a vector of axes indices (zero-based)
                \param positions are the relative or absolute positions (vector with the same length than axes) in mm or degree
                \param relNotAbs indicates a relative movement if true, else an absolute movement
                \param msgLevel defines if any warnings or errors should be displayed within an appropriate message box.
                \return RetVal returns retOk or any other warning or error depending on success (error as well if the plugin is no actuator plugin).

                \sa observeInvocation
            */
            ito::RetVal setActuatorPosition(QVector<int> axes, QVector<double> positions, bool relNotAbs, MessageLevel msgLevel = msgLevelWarningAndError) const;

            //! invokes AddInActuator::setPosRel or AddInActuator::setPosAbs of plugin in order to force a movement of one axis
            /*!
                Use this method to thread-safely position one axis of an actuator plugin. Do not directly call setPosRel
                or setPosAbs of the plugin, since this is not thread safe.

                This method waits until the movement ended (if the movement has been configured to by synchronous, plugin parameter)

                \param axis is the index of the axis (zero-based)
                \param position is the relative or absolute position in mm or degree
                \param relNotAbs indicates a relative movement if true, else an absolute movement
                \param msgLevel defines if any warnings or errors should be displayed within an appropriate message box.
                \return RetVal returns retOk or any other warning or error depending on success (error as well if the plugin is no actuator plugin).

                \sa observeInvocation
            */
            ito::RetVal setActuatorPosition(int axis, double position, bool relNotAbs, MessageLevel msgLevel = msgLevelWarningAndError) const;

            //! method to immediately set the interrupt flag of the actuator
            /*!
                Call this method in order to thread-safely and intermediately set the interrupt flag of the actuator.

                The actuator should check this flag with isInterrupted() and stop the movement if possible.
            */
            ito::RetVal setActuatorInterrupt() const;

            //! method to request the current status, positions and target positions from the actuator plugin
            /*!
                This method invokes the slot requestStatusAndPosition of the actuator plugin, that should get the
                current status, positions and target positions and emit these. Finally, the slots targetChanged and
                actuatorStatusChanged (depending on the boolean arguments) of this dock widget are called by means
                of a callback.
            */
            ito::RetVal requestActuatorStatusAndPositions(bool sendCurrentPos, bool sendTargetPos, MessageLevel msgLevel = msgLevelWarningAndError) const;

        private:
            AbstractAddInDockWidgetPrivate* d; /*! private data pointer of this class. */

        public slots:
            //! slot invoked if any parameter of the plugin has been changed.
            /*!
                overload this method in order reset the widgets depending on the current states of the parameter.
                The first invocation can also be used in order to configure the dock widget depending on the
                current set of available parameters.

                \params map of parameters (usually equal to m_params member of ito::AddInBase)
            */
            virtual void parametersChanged(QMap<QString, ito::Param> params) = 0;

            //! slot invoked if identifier of plugin has been set using AddInBase::setIdentifier
            /*!
                overload this method in order to get the identfier of the plugin. Usually, this identifier is only set in the
                init-method of the plugin, hence, after construction of the dock widget. Therefore this slot is invoked
                once the identifier has been changed.

                \param identifier new identifier name of the plugin
            */
            virtual void identifierChanged(const QString &identifier) = 0;

            //! slot invoked if the status or current position of an actuator plugin has been changed
            /*!
                overload this method if you want to react on such changes.
                Usually this slot is only connected in the dockWidgetVisibilityChanged method of an actuator plugin.

                You don't need to overload this in non-actuator plugin based dock widgets.

                \param status vector with status values for each axis (usually corresponds to ito::AddInActuator::m_currentStatus)
                \param actPosition vector with current position values (absolute in mm or degree). This vector can also be empty, if only status values have been changed.
                        (usually corresponds to ito::AddInActuator::m_currentPos)
            */
            virtual void actuatorStatusChanged(QVector<int> status, QVector<double> actPosition);

            //! slot invoked if the target position of an actuator plugin has been changed
            /*!
                overload this method if you want to react on such changes.
                Usually this slot is only connected in the dockWidgetVisibilityChanged method of an actuator plugin.

                You don't need to overload this in non-actuator plugin based dock widgets.

                \param targetPositions is the vector of target positions (in mm or degree). Usually this corresponds to the member ito::AddInActuator::m_targetPos
            */
            virtual void targetChanged(QVector<double> targetPositions);
    };
} //end namespace ito

#endif //ABSTRACTADDINDOCKWIDGET_H
