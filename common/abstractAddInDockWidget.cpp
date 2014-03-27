/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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

#include "abstractAddInDockWidget.h"

#include "addInInterface.h"

#include <qmessagebox.h>
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
        bool success = false;
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
        msgBox.setText(tr("Error while setting parameter").toLatin1().data());
        if (retval.errorMessage())
        {
            msgBox.setInformativeText(retval.errorMessage());
        }
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.exec();
    }
    else if (retval.containsWarning() && (msgLevel & msgLevelWarningOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Warning while setting parameter").toLatin1().data());
        if (retval.errorMessage())
        {
            msgBox.setInformativeText(retval.errorMessage());
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
        bool success = false;
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
        msgBox.setText(tr("Error while setting parameter").toLatin1().data());
        if (retval.errorMessage())
        {
            msgBox.setInformativeText(retval.errorMessage());
        }
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.exec();
    }
    else if (retval.containsWarning() && (msgLevel & msgLevelWarningOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Warning while setting parameter").toLatin1().data());
        if (retval.errorMessage())
        {
            msgBox.setInformativeText(retval.errorMessage());
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
    bool timeout = false;
    QTime time;
    time.start();
    if (d->m_pPlugin)
    {
        while(!timeout && waitCond->wait(10) == false)
        {
            QCoreApplication::processEvents();
            if (time.elapsed() > PLUGINWAIT)
            {
                if (d->m_pPlugin->isAlive() == false)
                {
                    retval += ito::RetVal(ito::retError, 0, tr("Timeout while waiting for answer from plugin instance.").toLatin1().data());
                    timeout = true;
                }
                time.restart();
            }
        }
        
        if (!timeout)
        {
            retval += waitCond->returnValue;
        }
        
        if (retval.containsError() && (msgLevel & msgLevelErrorOnly))
        {
            QMessageBox msgBox;
            msgBox.setText(tr("Error while execution").toLatin1().data());
            if (retval.errorMessage())
            {
                msgBox.setInformativeText(retval.errorMessage());
            }
            msgBox.setIcon(QMessageBox::Critical);
            msgBox.exec();
        }
        else if (retval.containsWarning() && (msgLevel & msgLevelWarningOnly))
        {
            QMessageBox msgBox;
            msgBox.setText(tr("Warning while execution").toLatin1().data());
            if (retval.errorMessage())
            {
                msgBox.setInformativeText(retval.errorMessage());
            }
            msgBox.setIcon(QMessageBox::Warning);
            msgBox.exec();
        }
    }
    
    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void AbstractAddInDockWidget::parametersChanged(QMap<QString, ito::Param> params)
{
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void AbstractAddInDockWidget::identifierChanged(const QString &identifier)
{
    qDebug() << "identifier changed to " << identifier;
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