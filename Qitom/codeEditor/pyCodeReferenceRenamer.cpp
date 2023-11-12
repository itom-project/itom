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

    --------------------------------
    This class is a modified version of the class QToolTip of the
    Qt framework (licensed under LGPL):
    https://code.woboq.org/qt5/qtbase/src/widgets/kernel/qtooltip.cpp.html
*********************************************************************** */

#include "pyCodeReferenceRenamer.h"
#include "../AppManagement.h"
#include "../python/pythonEngine.h"
#include "../python/pythonJedi.h"

#include <qfileinfo.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qlistwidget.h>
#include <qmetaobject.h>
#include <qpushbutton.h>

namespace ito {

//-------------------------------------------------------------------------------------
PyCodeReferenceRenamer::PyCodeReferenceRenamer(QObject* parent) : QObject(parent)
{
    m_pPythonEngine = AppManagement::getPythonEngine();

    // create dialog
    m_renameDialog = new QDialog();
    m_renameDialog->setWindowTitle(tr("Rename references").toLatin1().data());

    m_newNameUserInput = new QLineEdit();
    QHBoxLayout* newNameLayout = new QHBoxLayout();
    newNameLayout->addWidget(new QLabel(tr("New value: ").toLatin1().data()));
    newNameLayout->addWidget(m_newNameUserInput);

    m_treeWidgetReferences = new QTreeWidget;
    m_treeWidgetReferences->setAlternatingRowColors(true);
    m_treeWidgetReferences->setColumnCount(3);


    QStringList headerLabels;
    headerLabels << tr("Value").toLatin1().data() << tr("Line").toLatin1().data()
                 << tr("Column").toLatin1().data();
    m_treeWidgetReferences->setHeaderLabels(headerLabels);
    m_dialogButtonBox = new QDialogButtonBox(
        QDialogButtonBox::StandardButton::Apply | QDialogButtonBox::StandardButton::Cancel);

    QVBoxLayout* dialogLayout = new QVBoxLayout(m_renameDialog);
    dialogLayout->addLayout(newNameLayout);
    dialogLayout->addWidget(m_treeWidgetReferences);
    dialogLayout->addWidget(m_dialogButtonBox);

    // connect dialog
    connect(
        m_dialogButtonBox, &QDialogButtonBox::clicked, this, &PyCodeReferenceRenamer::onClicked);
}

//-------------------------------------------------------------------------------------
PyCodeReferenceRenamer::~PyCodeReferenceRenamer()
{
}

//-------------------------------------------------------------------------------------
void PyCodeReferenceRenamer::rename(const int& line, const int& column, const QString& fileName)
{
    ito::JediRenameRequest request;
    request.m_code = "";
    request.m_callbackFctName = "onJediRenameResultAvailable";
    request.m_col = column;
    request.m_line = line;
    request.m_fileName = fileName;
    request.m_sender = this;
    PythonEngine* pyEng = (PythonEngine*)m_pPythonEngine;
    pyEng->enqueueJediRenameRequest(request);
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onJediRenameResultAvailable(QVector<ito::JediRename> filesToChange)
{
    if (filesToChange.size() != 0)
    {
        foreach (const JediRename& file, filesToChange)
        {
            QTreeWidgetItem* fileItem = new QTreeWidgetItem(m_treeWidgetReferences);
            fileItem->setFlags(fileItem->flags() | Qt::ItemIsUserCheckable);
            fileItem->setCheckState(0, Qt::Checked);

            QFileInfo* fileInfo = new QFileInfo(file.m_filePath);
            fileItem->setText(0, fileInfo->fileName());
            m_treeWidgetReferences->addTopLevelItem(fileItem);

            int idxLine = 0;
            foreach (const int& line, file.m_lines)
            {
                QTreeWidgetItem* lineItem = new QTreeWidgetItem(fileItem);
                lineItem->setFlags(lineItem->flags() | Qt::ItemIsUserCheckable);
                lineItem->setCheckState(idxLine, Qt::Checked);
                lineItem->setText(0, file.m_values.at(idxLine));
                lineItem->setText(1, QString::number(line));
                lineItem->setText(2, QString::number(file.m_columns.at(idxLine)));
            }
        }

        m_treeWidgetReferences->expandAll();
        for (int i = 0; i < m_treeWidgetReferences->columnCount(); ++i)
        {
            m_treeWidgetReferences->resizeColumnToContents(i);
        }
        m_renameDialog->resize(500, 500);
        m_renameDialog->show();
    }
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onClicked(QAbstractButton* button)
{
    QDialogButtonBox::ButtonRole role = m_dialogButtonBox->buttonRole(button);
    if (role == QDialogButtonBox::ApplyRole)
    {
        onAccept();
    }
    else
    {
        onCanceled();
    }
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onAccept()
{
    m_renameDialog->hide();
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onCanceled()
{
    m_treeWidgetReferences->clear();
    m_renameDialog->hide();
}

} // namespace ito
