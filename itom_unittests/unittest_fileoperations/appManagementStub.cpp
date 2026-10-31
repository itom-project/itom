/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut für Technische Optik (ITO),
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

// Minimal replacement of Qitom/AppManagement.cpp for the unittest. The real
// implementation reads the settings file from the UserOrganizer, which would
// pull the whole application into the test. The settings file is a temporary,
// empty ini file (set in unittest_fileoperations.cc).

#include "AppManagement.h"

#include <qstring.h>

namespace ito {

QObject* AppManagement::m_pe = nullptr;
QObject* AppManagement::m_psp = nullptr;
QObject* AppManagement::m_sew = nullptr;
QObject* AppManagement::m_dwo = nullptr;
QObject* AppManagement::m_plo = nullptr;
QObject* AppManagement::m_app = nullptr;
QObject* AppManagement::m_mainWin = nullptr;
QObject* AppManagement::m_addInManager = nullptr;
QObject* AppManagement::m_uiOrganizer = nullptr;
QObject* AppManagement::m_processOrganizer = nullptr;
QObject* AppManagement::m_projectOrganizer = nullptr;
QObject* AppManagement::m_userOrganizer = nullptr;
QObject* AppManagement::m_cerrStream = nullptr;
QObject* AppManagement::m_coutStream = nullptr;
QObject* AppManagement::m_logger = nullptr;
QMutex AppManagement::m_mutex;
AppManagement::Timeouts AppManagement::timeouts;

namespace test {
static QString g_settingsFilePath;

void setSettingsFilePath(const QString& path)
{
    g_settingsFilePath = path;
}
} // namespace test

/*static*/ QString AppManagement::getSettingsFile(void)
{
    QMutexLocker locker(&m_mutex);
    return test::g_settingsFilePath;
}

} // end namespace ito
