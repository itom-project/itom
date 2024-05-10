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

#include "../abstractAddInDockWidget.h"

#include "addInInterface.h"

#include <QtWidgets/qmessagebox.h>
#include <qmetaobject.h>
#include <qdebug.h>
#include <QTime>
#include <qcoreapplication.h>

namespace ito
{

class AbstractAddInDockWidgetPrivate
{
public:
    AbstractAddInDockWidgetPrivate() : m_pPlugin(NULL)
    {}

    ito::AddInBase *m_pPlugin;
};

//-------------------------------------------------------------------------------------------------------------------------------------------------
AbstractAddInDockWidget::AbstractAddInDockWidget(ito::AddInBase *plugin) : d(NULL)
{
    d = new AbstractAddInDockWidgetPrivate();
    d->m_pPlugin = plugin;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
AbstractAddInDockWidget::~AbstractAddInDockWidget()
{
    delete d;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractAddInDockWidget::setPluginParameter(QSharedPointer<ito::ParamBase> param, MessageLevel msgLevel /*= msgLevelWarningAndError*/) const
{
    ito::RetVal retval;

    if (d->m_pPlugin)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        if (QMetaObject::invokeMethod(d->m_pPlugin, "setParam", Q_ARG(QSharedPointer<ito::ParamBase>, param), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
        {
            retval += observeInvocation(locker.getSemaphore(),msgLevelNo);
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, tr("slot 'setParam' could not be invoked since it does not exist.").toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("pointer to plugin is invalid.").toLatin1().data());
    }

    if (retval.containsError() && (msgLevel & msgLevelErrorOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Error while setting parameter"));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.exec();
    }
    else if (retval.containsWarning() && (msgLevel & msgLevelWarningOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Warning while setting parameter"));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.exec();
    }

    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractAddInDockWidget::setPluginParameters(const QVector<QSharedPointer<ito::ParamBase> > params, MessageLevel msgLevel /*= msgLevelWarningAndError*/) const
{
    ito::RetVal retval;

    if (d->m_pPlugin)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        if (QMetaObject::invokeMethod(d->m_pPlugin, "setParamVector", Q_ARG(const QVector<QSharedPointer<ito::ParamBase> >, params), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
        {
            retval += observeInvocation(locker.getSemaphore(),msgLevelNo);
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, tr("slot 'setParamVector' could not be invoked since it does not exist.").toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("pointer to plugin is invalid.").toLatin1().data());
    }

    if (retval.containsError() && (msgLevel & msgLevelErrorOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Error while setting parameter"));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.exec();
    }
    else if (retval.containsWarning() && (msgLevel & msgLevelWarningOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Warning while setting parameter"));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.exec();
    }

    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractAddInDockWidget::observeInvocation(ItomSharedSemaphore *waitCond, MessageLevel msgLevel) const
{
    ito::RetVal retval;

    if (d->m_pPlugin)
    {
        bool timeout = false;

        while(!timeout && waitCond->waitAndProcessEvents(PLUGINWAIT) == false)
        {
            if (d->m_pPlugin->isAlive() == false)
            {
                retval += ito::RetVal(ito::retError, 0, tr("Timeout while waiting for answer from plugin instance.").toLatin1().data());
                timeout = true;
            }
        }

        if (!timeout)
        {
            retval += waitCond->returnValue;
        }

        if (retval.containsError() && (msgLevel & msgLevelErrorOnly))
        {
            QMessageBox msgBox;
            msgBox.setText(tr("Error while execution"));
            if (retval.hasErrorMessage())
            {
                msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
            }
            msgBox.setIcon(QMessageBox::Critical);
            msgBox.exec();
        }
        else if (retval.containsWarning() && (msgLevel & msgLevelWarningOnly))
        {
            QMessageBox msgBox;
            msgBox.setText(tr("Warning while execution"));
            if (retval.hasErrorMessage())
            {
                msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
            }
            msgBox.setIcon(QMessageBox::Warning);
            msgBox.exec();
        }
    }

    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractAddInDockWidget::setActuatorPosition(QVector<int> axes, QVector<double> positions, bool relNotAbs, MessageLevel msgLevel) const
{
    ito::RetVal retval;

    QByteArray funcName = relNotAbs ? "setPosRel" : "setPosAbs";

    if (d->m_pPlugin)
    {
        if (qobject_cast<ito::AddInActuator*>(d->m_pPlugin) == NULL)
        {
            retval += ito::RetVal(ito::retError, 0, tr("setActuatorPosition can only be called for actuator plugins").toLatin1().data());
        }
        else
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

            if (QMetaObject::invokeMethod(d->m_pPlugin, funcName, Q_ARG(const QVector<int>, axes), Q_ARG(QVector<double>, positions), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
            {
                retval += observeInvocation(locker.getSemaphore(),msgLevelNo);
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, tr("slot '%1' could not be invoked since it does not exist.").arg(QLatin1String(funcName)).toLatin1().data());
            }
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("pointer to plugin is invalid.").toLatin1().data());
    }

    if (retval.containsError() && (msgLevel & msgLevelErrorOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Error while calling %1").arg(QLatin1String(funcName)));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.exec();
    }
    else if (retval.containsWarning() && (msgLevel & msgLevelWarningOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Warning while calling %1").arg(QLatin1String(funcName)));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.exec();
    }

    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractAddInDockWidget::setActuatorPosition(int axis, double position, bool relNotAbs, MessageLevel msgLevel) const
{
    ito::RetVal retval;

    QByteArray funcName = relNotAbs ? "setPosRel" : "setPosAbs";

    if (d->m_pPlugin)
    {
        if (qobject_cast<ito::AddInActuator*>(d->m_pPlugin) == NULL)
        {
            retval += ito::RetVal(ito::retError, 0, tr("setActuatorPosition can only be called for actuator plugins").toLatin1().data());
        }
        else
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

            if (QMetaObject::invokeMethod(d->m_pPlugin, funcName, Q_ARG(int, axis), Q_ARG(double, position), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
            {
                retval += observeInvocation(locker.getSemaphore(),msgLevelNo);
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, tr("slot '%1' could not be invoked since it does not exist.").arg(QLatin1String(funcName)).toLatin1().data());
            }
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("pointer to plugin is invalid.").toLatin1().data());
    }

    if (retval.containsError() && (msgLevel & msgLevelErrorOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Error while calling %1").arg(QLatin1String(funcName)));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.exec();
    }
    else if (retval.containsWarning() && (msgLevel & msgLevelWarningOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Warning while calling %1").arg(QLatin1String(funcName)));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.exec();
    }

    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractAddInDockWidget::requestActuatorStatusAndPositions(bool sendCurrentPos, bool sendTargetPos, MessageLevel msgLevel) const
{
    ito::RetVal retval;

    if (d->m_pPlugin)
    {
        if (qobject_cast<ito::AddInActuator*>(d->m_pPlugin) == NULL)
        {
            retval += ito::RetVal(ito::retError, 0, tr("setActuatorPosition can only be called for actuator plugins").toLatin1().data());
        }
        else
        {
            if (!QMetaObject::invokeMethod(d->m_pPlugin, "requestStatusAndPosition", Qt::QueuedConnection, Q_ARG(bool, sendCurrentPos), Q_ARG(bool, sendTargetPos)))
            {
                retval += ito::RetVal(ito::retError, 0, tr("slot 'requestStatusAndPosition' could not be invoked since it does not exist.").toLatin1().data());
            }
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("pointer to plugin is invalid.").toLatin1().data());
    }

    if (retval.containsError() && (msgLevel & msgLevelErrorOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Error while calling 'requestStatusAndPosition'"));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.exec();
    }
    else if (retval.containsWarning() && (msgLevel & msgLevelWarningOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Warning while calling 'requestStatusAndPosition'"));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.exec();
    }

    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractAddInDockWidget::setActuatorInterrupt() const
{
    ito::RetVal retval;

    if (d->m_pPlugin)
    {
        if (qobject_cast<ito::AddInActuator*>(d->m_pPlugin) == NULL)
        {
            retval += ito::RetVal(ito::retError, 0, tr("setActuatorInterrupt can only be called for actuator plugins").toLatin1().data());
        }
        else
        {
            qobject_cast<ito::AddInActuator*>(d->m_pPlugin)->setInterrupt();
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("pointer to plugin is invalid.").toLatin1().data());
    }

    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void AbstractAddInDockWidget::actuatorStatusChanged(QVector<int> status, QVector<double> actPosition)
{
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void AbstractAddInDockWidget::targetChanged(QVector<double> targetPositions)
{
}

} //namespace ito
