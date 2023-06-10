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

#include "widgetPropWorkspaceUnpack.h"

#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>

namespace ito
{
	WidgetPropWorkspaceUnpack::WidgetPropWorkspaceUnpack(QWidget *parent) :
		AbstractPropertyPageWidget(parent)
	{
		ui.setupUi(this);
	}

//----------------------------------------------------------------------------------------------------------------------------------
	void WidgetPropWorkspaceUnpack::readSettings()
	{
		QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
		settings.beginGroup("Workspace");
		ui.checkBoxUnpackDictionary->setChecked(settings.value("importIdcMatUnpackDict", "true").toBool());
		settings.endGroup();
	}

//----------------------------------------------------------------------------------------------------------------------------------
	void WidgetPropWorkspaceUnpack::writeSettings()
	{
		QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
		settings.beginGroup("Workspace");
		settings.setValue("importIdcMatUnpackDict", ui.checkBoxUnpackDictionary->isChecked());
		settings.endGroup();
	}

}//endNamespace ito
