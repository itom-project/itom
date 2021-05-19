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

#include "AppManagement.h"
#include "organizer/userOrganizer.h"

#include <qtextcodec.h>

namespace ito
{

/*!
    \class AppManagement
    \brief static class only administrating QObject-pointers to main organization and management units of the main application
*/

//initialization of members
QObject* AppManagement::m_pe = nullptr;
QObject* AppManagement::m_psp = nullptr;
QObject* AppManagement::m_sew = nullptr;
QObject* AppManagement::m_dwo = nullptr;
QObject* AppManagement::m_plo = nullptr;
QObject* AppManagement::m_app = nullptr;
QObject* AppManagement::m_mainWin = nullptr;
QObject* AppManagement::m_uiOrganizer = nullptr;
QObject* AppManagement::m_processOrganizer = nullptr;
QObject* AppManagement::m_userOrganizer = nullptr;
QObject* AppManagement::m_addInManager = nullptr;
QObject* AppManagement::m_cerrStream = nullptr;
QObject* AppManagement::m_coutStream = nullptr;
QMutex AppManagement::m_mutex;
AppManagement::Timeouts AppManagement::timeouts;
QTextCodec* AppManagement::m_scriptTextCodec = nullptr;

//-------------------------------------------------------------------------------------------
/*static*/ QString AppManagement::getSettingsFile(void)
{
    QMutexLocker locker(&m_mutex);
    return ((ito::UserOrganizer*)m_userOrganizer)->getCurrentUserSettingsFile();
}

//-------------------------------------------------------------------------------------------
/*static*/ QTextCodec* AppManagement::getScriptTextCodec()
{
    QMutexLocker locker(&m_mutex);
    if (m_scriptTextCodec)
    {
        return m_scriptTextCodec;
    }
    else
    {
        return QTextCodec::codecForLocale();
    }
}

//-------------------------------------------------------------------------------------------
/*static*/ void AppManagement::setScriptTextCodec(QTextCodec *codec)
{
    QMutexLocker locker(&m_mutex);
    m_scriptTextCodec = codec;
}

} //end namespace ito
