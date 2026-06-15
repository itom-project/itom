/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut fuer Technische Optik (ITO),
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

#pragma once

#include "../python/pythonJedi.h"
#include <qabstractbutton.h>
#include <qdialog.h>
#include <qdialogbuttonbox.h>
#include <qevent.h>
#include <qlineedit.h>
#include <qtreewidget.h>
#include <qwidget.h>

#include "common/retVal.h"

namespace ito {

class PyCodeReferenceRenamer : public QObject
{
    Q_OBJECT
public:
    PyCodeReferenceRenamer(QWidget* parent = nullptr);
    ~PyCodeReferenceRenamer();
    RetVal rename(const int& line, const int& column, const QString& filepath);

    enum RenamerRetVal
    {
        CodeYetNotAvailable = 100,
        CouldNotOpenFile = 101
    };

    struct RenameItem
    {
    };

private:
    QObject* m_pPythonEngine;
    QDialog* m_renameDialog;
    QLineEdit* m_newNameUserInput;
    QTreeWidget* m_treeWidgetReferences;
    QDialogButtonBox* m_dialogButtonBox;
    ito::JediRenameRequest m_request;
    QWidget* m_pParent;

    enum RenamerRole
    {
        RoleFilePath = Qt::UserRole,
        RoleMainFile = Qt::UserRole + 1,
        RoleFileOpened = Qt::UserRole + 2,
        RoleFileModified = Qt::UserRole + 3,
        RoleFileRenameItem = Qt::UserRole + 4
    };

    QStringList readFirstNLinesFromFile(const QString& filepath, int n) const;

private slots:
    void onJediRenameResultAvailable(
        const QVector<ito::JediRename>& filesToChange,
        const QString& oldValue,
        bool success,
        QString errorText);
    void onApply();
    void onCanceled();
    void onItemChanged(QTreeWidgetItem* item, int column);
    void keyPressEvent(QKeyEvent* event);
    void clearAndHideTreeWidget();
    void onItemDoubleClick(QTreeWidget* treeWidget, QTreeWidgetItem* item);


    ito::RetVal replaceOccurencesInFile(
        const QString& filePath,
        const QString& newValue,
        const QVector<ito::FileRenameItem>& renameItems);

signals:
};

}; // namespace ito
