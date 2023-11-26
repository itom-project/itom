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

namespace ito {

class PyCodeReferenceRenamer : public QObject
{
    Q_OBJECT
public:
    PyCodeReferenceRenamer(QObject* parent = nullptr);
    ~PyCodeReferenceRenamer();
    void rename(const int& line, const int& column, const QString& fileName);

private:
    QObject* m_pPythonEngine;
    QDialog* m_renameDialog;
    QLineEdit* m_newNameUserInput;
    QTreeWidget* m_treeWidgetReferences;
    QDialogButtonBox* m_dialogButtonBox;
    QVector<ito::JediRename> m_filesToChange;
    ito::JediRenameRequest m_request;


private slots:
    void onJediRenameResultAvailable(const QVector<ito::JediRename>& filesToChange);
    void onApply();
    void onCanceled();
    void onItemChanged(QTreeWidgetItem* item, int column);
    void keyPressEvent(QKeyEvent* event);
    void clearAndHideTreeWidget();
    void onItemDoubleClick(QTreeWidgetItem* item, int column);

    QString getAbsoluteFilePath(const QString& fileName);
    void replaceWordInFile(
        const QString& filePath,
        int lineNumber,
        int columnNumber,
        const QString& value,
        const QString& newValue);

signals:
};

}; // namespace ito