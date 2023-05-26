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

#include "../abstractAddInConfigDialog.h"

#include "addInInterface.h"

#include <QtWidgets/qmessagebox.h>
#include <qmetaobject.h>
#include <qdebug.h>
#include <QTime>
#include <qcoreapplication.h>
#include "apiFunctionsInc.h"

namespace ito
{

class AbstractAddInConfigDialogPrivate
{
public:
    AbstractAddInConfigDialogPrivate() : m_pPlugin(NULL)
    {}

    ito::AddInBase *m_pPlugin;
};

//-------------------------------------------------------------------------------------------------------------------------------------------------
AbstractAddInConfigDialog::AbstractAddInConfigDialog(ito::AddInBase *plugin) : QDialog(), d(NULL)
{
    d = new AbstractAddInConfigDialogPrivate();
    d->m_pPlugin = plugin;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
AbstractAddInConfigDialog::~AbstractAddInConfigDialog()
{
    delete d;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AbstractAddInConfigDialog::setPluginParameter(QSharedPointer<ito::ParamBase> param, MessageLevel msgLevel /*= msgLevelWarningAndError*/)
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

    if (!retval.containsError())
    {
        QVector<QSharedPointer<ito::ParamBase> > vec(1, param);
        retval += apiUpdateParameters(m_currentParameters, vec);
    }

    if (retval.containsError() && (msgLevel & msgLevelErrorOnly))
    {
        QMessageBox msgBox;
        if (param->getName())
        {
            msgBox.setText(tr("Error while setting parameter '%1'").arg(QLatin1String(param->getName())));
        }
        else
        {
            msgBox.setText(tr("Error while setting parameter"));
        }
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
        if (param->getName())
        {
            msgBox.setText(tr("Warning while setting parameter '%1'").arg(QLatin1String(param->getName())));
        }
        else
        {
            msgBox.setText(tr("Warning while setting parameter"));
        }
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
ito::RetVal AbstractAddInConfigDialog::setPluginParameters(const QVector<QSharedPointer<ito::ParamBase> > params, MessageLevel msgLevel /*= msgLevelWarningAndError*/)
{
    ito::RetVal retval;

    if (d->m_pPlugin)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        if (QMetaObject::invokeMethod(d->m_pPlugin, "setParamVector", Q_ARG(QVector<QSharedPointer<ito::ParamBase> >, params), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
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

    if (!retval.containsError())
    {
        retval += apiUpdateParameters(m_currentParameters, params);
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
ito::RetVal AbstractAddInConfigDialog::observeInvocation(ItomSharedSemaphore *waitCond, MessageLevel msgLevel) const
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

} //namespace ito
