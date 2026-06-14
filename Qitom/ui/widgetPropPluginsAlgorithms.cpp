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

#include "widgetPropPluginsAlgorithms.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>
#include <qthread.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropPluginsAlgorithms::WidgetPropPluginsAlgorithms(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);

    ui.txtMaxNumThreads->setText(QString::number(QThread::idealThreadCount()));
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropPluginsAlgorithms::~WidgetPropPluginsAlgorithms()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPluginsAlgorithms::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("AddInManager");
    if (QThread::idealThreadCount() < 0)
    {
        ui.spinMaxNumThreads->setMaximum(1);
        ui.spinMaxNumThreads->setValue(qBound(1, settings.value("maximumThreadCount", 2).toInt(), 2));
    }
    else
    {
        ui.spinMaxNumThreads->setMaximum(QThread::idealThreadCount());
        ui.spinMaxNumThreads->setValue(qBound(1, settings.value("maximumThreadCount", 2).toInt(), QThread::idealThreadCount()));
    }
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPluginsAlgorithms::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("AddInManager");
    settings.setValue("maximumThreadCount", ui.spinMaxNumThreads->value() );
    settings.endGroup();
}

} //end namespace ito
