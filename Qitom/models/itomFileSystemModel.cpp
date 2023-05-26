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

#include "itomFileSystemModel.h"
#include <qmimedata.h>
#include <qurl.h>
#include <qfileinfo.h>
#include <qdir.h>
#include <qmessagebox.h>

namespace ito
{

/*!
    \class ItomFileSystemModel
    \brief  Instead of the default QFileSystemModel, this model is able to provide a more flexible support for drop or
        paste operations if the destination file already exists. If the file should be duplicated (from one folder to the same folder),
        a (copy)-suffix is appended to the base-filename. If the source is another directory, but the file has the same name,
        the user is prompted if he wants to overwrite the source file or not.
*/

//----------------------------------------------------------------------------------------------------------------------------------
ItomFileSystemModel::ItomFileSystemModel(QObject *parent) :
    QFileSystemModel(parent)
{
}



//----------------------------------------------------------------------------------------------------------------------------------
bool ItomFileSystemModel::dropMimeData(const QMimeData *data, Qt::DropAction action,
                              int row, int column, const QModelIndex &parent)
{
    Q_UNUSED(row);
    Q_UNUSED(column);
    if (!parent.isValid() || isReadOnly())
        return false;


    bool success = true;

    QString to = filePath(parent) + QDir::separator();


    QList<QUrl> urls = data->urls();
    QList<QUrl>::const_iterator it = urls.constBegin();
    QFileInfo toFile;
    QString toFileAbs;
    QFileInfo fromFile;
    QString fromFileAbs;
    QMessageBox::StandardButton standard = QMessageBox::Yes;

    for (; it != urls.constEnd(); ++it)
    {
        fromFile.setFile(it->toLocalFile());
        fromFileAbs = fromFile.absoluteFilePath();

        if (!fromFile.exists())
        {
            QMessageBox::critical(0, tr("file does not exist."), tr("The source file '%s' does not exist and can not be moved or pasted").arg(fromFileAbs));
            continue;
        }

        toFile.setFile(to + fromFile.fileName());
        toFileAbs = toFile.absoluteFilePath();

        switch (action)
        {
        case Qt::MoveAction:
            if (fromFileAbs != toFileAbs)
            {
                success = QFile::rename(fromFileAbs, toFileAbs) && success;
            }
            break;
        case Qt::CopyAction:
            if (toFile.exists() == false)
            {
                success = QFile::copy(fromFileAbs, toFileAbs) && success;
            }
            else if (toFileAbs == fromFileAbs)
            {
                //duplicate in same directory
                toFileAbs = toFile.dir().absoluteFilePath(toFile.baseName() + tr(" (copy)") + "." + toFile.suffix());
                success = QFile::copy(fromFileAbs, toFileAbs) && success;
            }
            else //the filename already exists in destination, but it is not the same the source file
            {
                if (standard & QMessageBox::NoToAll)
                {
                    continue;
                }
                else if (standard & QMessageBox::YesToAll)
                {
                    if (QFile::remove(toFileAbs))
                    {
                        success = QFile::copy(fromFileAbs, toFileAbs) && success;
                    }
                    else
                    {
                        success = false;
                    }
                }
                else
                {
                    standard = QMessageBox::question(0, tr("Destination already exists."), tr("The file '%1' already exists. Should it be overwritten?").arg(toFileAbs), QMessageBox::Yes | QMessageBox::No | QMessageBox::NoToAll | QMessageBox::YesToAll, QMessageBox::No);

                    if (standard & (QMessageBox::YesToAll | QMessageBox::Yes))
                    {
                        if (QFile::remove(toFileAbs))
                        {
                            success = QFile::copy(fromFileAbs, toFileAbs) && success;
                        }
                        else
                        {
                            success = false;
                        }
                    }
                }
            }
            break;
        case Qt::LinkAction:
            if (toFile.exists() == false)
            {
                success = QFile::copy(fromFileAbs, toFileAbs) && success;
            }
            else if (toFileAbs == fromFileAbs)
            {
                //duplicate in same directory
                toFileAbs = toFile.dir().absoluteFilePath(toFile.baseName() + tr("- Copy") + "." + toFile.suffix());
                success = QFile::link(fromFileAbs, toFileAbs) && success;
            }
            else //the filename already exists in destination, but it is not the same the source file
            {
                if (standard & QMessageBox::NoToAll)
                {
                    continue;
                }
                else if (standard & QMessageBox::YesToAll)
                {
                    QFile::remove(toFileAbs);
                    success = QFile::link(fromFileAbs, toFileAbs) && success;
                }
                else
                {
                    standard = QMessageBox::question(0, tr("Destination already exists."), tr("The file '%1' already exists. Should it be overwritten by the new link?").arg(toFileAbs), QMessageBox::Yes | QMessageBox::No | QMessageBox::NoToAll | QMessageBox::YesToAll, QMessageBox::No);

                    if (standard & (QMessageBox::YesToAll | QMessageBox::Yes))
                    {
                        QFile::remove(toFileAbs);
                        success = QFile::link(fromFileAbs, toFileAbs) && success;
                    }
                }
            }
            break;
        }
    }

     return success;
 }




} //end namespace ito
