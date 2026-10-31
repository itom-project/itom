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

#include "projectSession.h"

#include "../AppManagement.h"
#include "projectOrganizer.h"

#include <qdir.h>
#include <qfileinfo.h>

namespace ito {

//-------------------------------------------------------------------------------------
ProjectSession::ProjectSession(ProjectOrganizer* organizer, QObject* parent) :
    QObject(parent), m_pOrganizer(organizer)
{
    if (m_pOrganizer)
    {
        connect(m_pOrganizer, &ProjectOrganizer::projectOpened, this, &ProjectSession::projectOpened);
        connect(m_pOrganizer, &ProjectOrganizer::projectClosed, this, &ProjectSession::projectClosed);
    }
}

//-------------------------------------------------------------------------------------
ProjectSession::~ProjectSession()
{
}

//-------------------------------------------------------------------------------------
/*static*/ QString ProjectSession::pyStringList(const QStringList& paths)
{
    QStringList items;

    foreach (const QString& p, paths)
    {
        QString escaped = QDir::toNativeSeparators(p);
        escaped.replace("\\", "\\\\").replace("'", "\\'");
        items << "'" + escaped + "'";
    }

    return "[" + items.join(", ") + "]";
}

//-------------------------------------------------------------------------------------
/*static*/ QString ProjectSession::addToSysPathStatement(const QStringList& paths)
{
    if (paths.isEmpty())
    {
        return QString();
    }

    // reversed, so that the first entry of the project ends up first in sys.path
    QStringList reversed = paths;
    std::reverse(reversed.begin(), reversed.end());

    return QString("import sys as _itom_sys\n"
                   "for _itom_p in %1:\n"
                   "    if _itom_p not in _itom_sys.path:\n"
                   "        _itom_sys.path.insert(0, _itom_p)\n"
                   "del _itom_sys, _itom_p\n")
        .arg(pyStringList(reversed));
}

//-------------------------------------------------------------------------------------
/*static*/ QString ProjectSession::removeFromSysPathStatement(const QStringList& paths)
{
    if (paths.isEmpty())
    {
        return QString();
    }

    return QString("import sys as _itom_sys\n"
                   "for _itom_p in %1:\n"
                   "    while _itom_p in _itom_sys.path:\n"
                   "        _itom_sys.path.remove(_itom_p)\n"
                   "del _itom_sys, _itom_p\n")
        .arg(pyStringList(paths));
}

//-------------------------------------------------------------------------------------
//! calls ScriptEditorOrganizer::switchScriptSession (via the meta object, such that this
//! file does not depend on the script editor and can be compiled into the unittests)
void ProjectSession::switchScripts(const QString& saveGroup, const QString& restoreGroup)
{
    QObject* seo = AppManagement::getScriptEditorOrganizer();

    if (seo)
    {
        bool ok = false;
        QMetaObject::invokeMethod(
            seo,
            "switchScriptSession",
            Qt::DirectConnection,
            Q_RETURN_ARG(bool, ok),
            Q_ARG(QString, saveGroup),
            Q_ARG(QString, restoreGroup));
    }
}

//-------------------------------------------------------------------------------------
void ProjectSession::runPython(const QString& statement)
{
    QObject* pyEngine = AppManagement::getPythonEngine();

    if (pyEngine && !statement.isEmpty())
    {
        // the engine lives in the python thread; the call is queued
        QMetaObject::invokeMethod(pyEngine, "pythonRunString", Q_ARG(QString, statement));
    }
}

//-------------------------------------------------------------------------------------
void ProjectSession::projectOpened(const QString& filePath)
{
    if (!m_pOrganizer || !m_pOrganizer->hasProject())
    {
        return;
    }

    if (m_pOrganizer->restoreScripts())
    {
        // the scripts open so far belong to the global session
        switchScripts(QString(), ProjectOrganizer::settingsGroup(filePath));
    }

    m_currentPythonPath.clear();

    foreach (const QString& p, m_pOrganizer->project().pythonPath())
    {
        if (QFileInfo(p).isDir())
        {
            m_currentPythonPath << p;
        }
    }

    runPython(addToSysPathStatement(m_currentPythonPath));
}

//-------------------------------------------------------------------------------------
void ProjectSession::projectClosed(const QString& filePath)
{
    if (m_pOrganizer && m_pOrganizer->restoreScripts())
    {
        switchScripts(ProjectOrganizer::settingsGroup(filePath), QString());
    }

    runPython(removeFromSysPathStatement(m_currentPythonPath));
    m_currentPythonPath.clear();
}

} // namespace ito
