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

#include "workspaceApplyService.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>

namespace ito
{
namespace llm
{

RetVal WorkspaceApplyService::apply(
    const QString &workspaceRoot,
    const WorkspaceChangeSet &changeSet,
    WorkspaceApplyResult *resultOut) const
{
    if (workspaceRoot.trimmed().isEmpty())
    {
        const QByteArray msg =
            QCoreApplication::translate("WorkspaceApplyService", "Workspace root is empty.").toUtf8();
        return RetVal(retError, 0, msg.constData());
    }

    if (changeSet.isEmpty())
    {
        const QByteArray msg =
            QCoreApplication::translate("WorkspaceApplyService", "No workspace changes to apply.").toUtf8();
        return RetVal(retError, 0, msg.constData());
    }

    RetVal retval(retOk);

    for (int i = 0; i < changeSet.changes.size(); ++i)
    {
        RetVal single = applySingleChange(workspaceRoot, changeSet.changes[i], resultOut);
        if (single.containsError())
        {
            retval += single;
        }
    }

    return retval;
}

QString WorkspaceApplyService::resolveSafePath(
    const QString &workspaceRoot,
    const QString &relativeOrAbsolutePath) const
{
    QDir root(workspaceRoot);
    const QString rootCanonical = QFileInfo(root.absolutePath()).canonicalFilePath();

    QFileInfo targetInfo(relativeOrAbsolutePath);
    QString absolutePath;

    if (targetInfo.isAbsolute())
    {
        absolutePath = targetInfo.absoluteFilePath();
    }
    else
    {
        absolutePath = root.absoluteFilePath(relativeOrAbsolutePath);
    }

    QFileInfo absoluteInfo(absolutePath);
    const QString parentCanonical = QFileInfo(absoluteInfo.absolutePath()).canonicalFilePath();
    if (parentCanonical.isEmpty() || rootCanonical.isEmpty() ||
        !parentCanonical.startsWith(rootCanonical, Qt::CaseInsensitive))
    {
        return QString();
    }

    const QString canonicalCandidate = absoluteInfo.exists()
        ? absoluteInfo.canonicalFilePath()
        : parentCanonical + QDir::separator() + absoluteInfo.fileName();

    if (!canonicalCandidate.startsWith(rootCanonical, Qt::CaseInsensitive))
    {
        return QString();
    }

    return absoluteInfo.absoluteFilePath();
}

RetVal WorkspaceApplyService::applySingleChange(
    const QString &workspaceRoot,
    const WorkspaceChange &change,
    WorkspaceApplyResult *resultOut) const
{
    const QString path = resolveSafePath(workspaceRoot, change.filePath);
    if (path.isEmpty())
    {
        if (resultOut)
        {
            resultOut->errors << QCoreApplication::translate("WorkspaceApplyService", "Unsafe path blocked: %1")
                                     .arg(change.filePath);
        }
        const QByteArray msg = QCoreApplication::translate("WorkspaceApplyService", "Unsafe path blocked.").toUtf8();
        return RetVal(retError, 0, msg.constData());
    }

    QFileInfo info(path);

    if (change.type == WorkspaceChangeType::DeleteFile)
    {
        if (info.exists() && !QFile::remove(path))
        {
            if (resultOut)
            {
                resultOut->errors << QCoreApplication::translate("WorkspaceApplyService", "Delete failed: %1")
                                         .arg(path);
            }
            const QByteArray msg = QCoreApplication::translate("WorkspaceApplyService", "Delete failed.").toUtf8();
            return RetVal(retError, 0, msg.constData());
        }

        if (resultOut)
        {
            resultOut->deletedFiles += 1;
        }

        return RetVal(retOk);
    }

    QDir dir = info.dir();
    if (!dir.exists() && !dir.mkpath("."))
    {
        if (resultOut)
        {
            resultOut->errors << QCoreApplication::translate("WorkspaceApplyService", "Could not create directory for: %1")
                                     .arg(path);
        }
        const QByteArray msg =
            QCoreApplication::translate("WorkspaceApplyService", "Could not create target directory.").toUtf8();
        return RetVal(retError, 0, msg.constData());
    }

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate))
    {
        if (resultOut)
        {
            resultOut->errors << QCoreApplication::translate("WorkspaceApplyService", "Write failed: %1")
                                     .arg(path);
        }
        const QByteArray msg = QCoreApplication::translate("WorkspaceApplyService", "Write failed.").toUtf8();
        return RetVal(retError, 0, msg.constData());
    }

    QTextStream stream(&file);
    stream << change.content;
    file.close();

    if (resultOut)
    {
        if (change.type == WorkspaceChangeType::CreateFile)
        {
            resultOut->createdFiles += 1;
        }
        else
        {
            resultOut->updatedFiles += 1;
        }
    }

    return RetVal(retOk);
}

} // namespace llm
} // namespace ito
