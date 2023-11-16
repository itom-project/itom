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
#include "../python/pythonEngine.h"
#include "../python/pythonJedi.h"
#include "AppManagement.h"
#include "global.h"
#include "organizer/scriptEditorOrganizer.h"

#include <qfile.h>
#include <qfileinfo.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qlistwidget.h>
#include <qmessagebox.h>
#include <qmetaobject.h>
#include <qpushbutton.h>

namespace ito {

//-------------------------------------------------------------------------------------
PyCodeReferenceRenamer::PyCodeReferenceRenamer(QObject* parent) :
    QObject(parent), m_pPythonEngine(nullptr), m_renameDialog(nullptr), m_newNameUserInput(nullptr),
    m_treeWidgetReferences(nullptr), m_dialogButtonBox(nullptr), m_filesToChange()
{
    m_pPythonEngine = AppManagement::getPythonEngine();

    if (!m_renameDialog)
    {
        // create dialog
        m_renameDialog = new QDialog();
        m_renameDialog->setWindowTitle(tr("Rename reference").toUtf8().data());
        m_renameDialog->setModal(true);

        m_newNameUserInput = new QLineEdit();
        QHBoxLayout* newNameLayout = new QHBoxLayout();
        newNameLayout->addWidget(new QLabel(tr("New value: ").toUtf8().data()));
        newNameLayout->addWidget(m_newNameUserInput);

        m_treeWidgetReferences = new QTreeWidget;
        m_treeWidgetReferences->setAlternatingRowColors(true);
        m_treeWidgetReferences->setColumnCount(3);


        QStringList headerLabels;
        headerLabels << tr("Value").toUtf8().data() << tr("Line").toUtf8().data()
                     << tr("Column").toUtf8().data();
        m_treeWidgetReferences->setHeaderLabels(headerLabels);

        m_dialogButtonBox = new QDialogButtonBox(
            QDialogButtonBox::StandardButton::Ok | QDialogButtonBox::StandardButton::Cancel);

        QVBoxLayout* dialogLayout = new QVBoxLayout(m_renameDialog);
        dialogLayout->addLayout(newNameLayout);
        dialogLayout->addWidget(new QLabel(tr("References changes:").toUtf8().data()));
        dialogLayout->addWidget(m_treeWidgetReferences);
        dialogLayout->addWidget(m_dialogButtonBox);

        // connect dialog
        connect(
            m_treeWidgetReferences,
            &QTreeWidget::itemChanged,
            this,
            &PyCodeReferenceRenamer::onItemChanged);

        connect(
            m_dialogButtonBox, &QDialogButtonBox::accepted, this, &PyCodeReferenceRenamer::onApply);
        connect(
            m_dialogButtonBox,
            &QDialogButtonBox::rejected,
            this,
            &PyCodeReferenceRenamer::onCanceled);

        connect(
            m_treeWidgetReferences,
            &QTreeWidget::itemDoubleClicked,
            this,
            &PyCodeReferenceRenamer::onItemDoubleClick);
    }
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
void PyCodeReferenceRenamer::onJediRenameResultAvailable(
    const QVector<ito::JediRename>& filesToChange)
{
    if (filesToChange.size() != 0)
    {
        m_filesToChange = filesToChange;
        // set current value to new value line edit
        m_newNameUserInput->setText(filesToChange.at(0).m_values.at(0));
        m_newNameUserInput->selectAll();

        ScriptEditorOrganizer* seo =
            qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());

        foreach (const JediRename& file, m_filesToChange)
        {
            QTreeWidgetItem* fileItem = new QTreeWidgetItem(m_treeWidgetReferences);
            fileItem->setFlags(fileItem->flags() | Qt::ItemIsUserCheckable);
            fileItem->setCheckState(0, Qt::Checked);

            QFileInfo* fileInfo = new QFileInfo(file.m_filePath);

            fileItem->setText(0, fileInfo->fileName());
            QFont font = QFont(fileItem->font(0));
            font.setBold(true);
            fileItem->setFont(0, font);

            m_treeWidgetReferences->addTopLevelItem(fileItem);

            QFile* scriptFile = new QFile(file.m_filePath);
            if (!scriptFile->open(QIODevice::ReadOnly | QIODevice::Text))
            {
                qDebug() << "Failed to open the file";
            }

            QTextStream in(scriptFile);

            int idxLine = 0;
            int iterLine = 1;
            foreach (const int& line, file.m_lines)
            {
                QTreeWidgetItem* lineItem = new QTreeWidgetItem(fileItem);
                QString lineText;

                for (iterLine; iterLine <= line; ++iterLine)
                {
                    lineText = in.readLine();
                }

                lineItem->setFlags(lineItem->flags() | Qt::ItemIsUserCheckable);
                lineItem->setCheckState(0, Qt::Checked);
                lineItem->setText(0, lineText);

                lineItem->setText(1, QString::number(line));
                lineItem->setText(2, QString::number(file.m_columns.at(idxLine)));
                idxLine++;
            }
            scriptFile->close();
        }

        m_treeWidgetReferences->expandAll();
        for (int i = 0; i < m_treeWidgetReferences->columnCount(); ++i)
        {
            m_treeWidgetReferences->resizeColumnToContents(i);
        }
        m_renameDialog->resize(800, 600);
        m_renameDialog->show();
        m_newNameUserInput->setFocus();
    }
    else
    {
        m_filesToChange.clear();
    }
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::clearAndHideTreeWidget()
{
    m_treeWidgetReferences->clear();
    m_renameDialog->hide();
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onApply()
{
    QTreeWidgetItem* topItem;
    QString filePath;
    QString value;
    int line;
    int column;
    QString newValue = m_newNameUserInput->text();

    if (newValue.length() == 0)
    {
        QMessageBox msgBox(
            QMessageBox::Warning,
            tr("Warning"),
            tr("No new reference name was given.\n"
               "Do you want to continue?"),
            QMessageBox::Ok | QMessageBox::Cancel);

        msgBox.setDefaultButton(QMessageBox::Cancel);
        int result = msgBox.exec();

        if (result == QMessageBox::Cancel)
        {
            clearAndHideTreeWidget();
            return;
        }
    }

    // iter all files
    for (int idxTopLevel = 0; idxTopLevel < m_treeWidgetReferences->topLevelItemCount();
         ++idxTopLevel)
    {
        topItem = m_treeWidgetReferences->topLevelItem(idxTopLevel); // get files on top level

        if (topItem->checkState(0) == Qt::Checked) // iter lines and columns of checked files
        {
            filePath = topItem->text(0);
            for (int idxSecondLevel = topItem->childCount() - 1; idxSecondLevel >= 0;
                 --idxSecondLevel)
            {
                QTreeWidgetItem* secondLevelItem = topItem->child(idxSecondLevel);
                filePath = getAbsoluteFilePath(filePath);
                value = secondLevelItem->text(0);
                line = secondLevelItem->text(1).toInt();
                column = secondLevelItem->text(2).toInt();

                replaceWordInFile(
                    filePath, line, column, m_filesToChange.at(0).m_values.at(0), newValue);
            }
        }
    }
    clearAndHideTreeWidget();
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Enter || event->key() == Qt::Key_Return)
    {
        onApply();
    }
    else if (event->key() == Qt::Key_Escape)
    {
        onCanceled();
    }
    else
    {
        PyCodeReferenceRenamer::keyPressEvent(event);
    }
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onCanceled()
{
    clearAndHideTreeWidget();
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onItemChanged(QTreeWidgetItem* item, int column)
{
    if (item && item->childCount() > 0 && column == 0)
    {
        Qt::CheckState state = item->checkState(column);

        for (int iter = 0; iter < item->childCount(); iter++)
        {
            item->child(iter)->setCheckState(column, state);
            if (state == Qt::Unchecked)
            {
                item->setExpanded(false);
            }
            else
            {
                item->setExpanded(true);
            }
        }
    }
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onItemDoubleClick(QTreeWidgetItem* item, int column)
{
    int line = 0;
    QString fileToOpen;
    QTreeWidgetItem* topLevelItem = nullptr;

    ScriptEditorOrganizer* seo =
        qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());

    if (seo)
    {
        if (m_treeWidgetReferences->indexOfTopLevelItem(item) != -1)
        {
            topLevelItem = item;
            line = 0;
        }
        else
        {
            topLevelItem = item->parent();
            line = item->text(1).toInt() - 1;
        }

        fileToOpen = topLevelItem->text(0);
        fileToOpen = getAbsoluteFilePath(fileToOpen);

        seo->openScript(fileToOpen, nullptr, line);
    }
}

//-------------------------------------------------------------------
QString PyCodeReferenceRenamer::getAbsoluteFilePath(const QString& fileName)
{
    int index;
    QString absolutePath;

    foreach (auto files, m_filesToChange)
    {
        index = files.m_filePath.indexOf(fileName);
        if (index != -1)
        {
            absolutePath = files.m_filePath;
            break;
        }
    }
    return absolutePath;
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::replaceWordInFile(
    const QString& filePath,
    int lineNumber,
    int columnNumber,
    const QString& value,
    const QString& newValue)
{
    // Open the file
    QFile file(filePath);
    if (!file.open(QIODevice::ReadWrite | QIODevice::Text))
    {
        qDebug() << "Could not open file" << filePath;
        return;
    }

    // Create a QTextStream to read from and write to the file
    QTextStream in(&file);

    // Create a list to store modified lines
    QStringList modifiedLines;

    // Read and modify lines one by one
    int currentLineNumber = 1;
    while (!in.atEnd())
    {
        QString line = in.readLine();

        // Check if this is the line to be modified
        if (currentLineNumber == lineNumber)
        {
            // Check if the specified column exists in the line
            if (columnNumber > 0 && columnNumber <= line.length())
            {
                // Delete the word at the specified column
                line.remove(columnNumber, value.length());

                // Insert the new word at the specified column
                line.insert(columnNumber, newValue);
            }
            else
            {
                qDebug() << "Invalid column number";
            }
        }

        // Add the modified or unmodified line to the list
        modifiedLines << line;

        // Move to the next line
        currentLineNumber++;
    }

    // Close the file
    file.close();

    // Open the file in write mode to update its content
    if (file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QTextStream out(&file);

        // Write the modified lines back to the file
        for (const QString& modifiedLine : modifiedLines)
        {
            out << modifiedLine << "\n";
        }

        // Close the file
        file.close();
        qDebug() << "Word replaced successfully.";
    }
    else
    {
        qDebug() << "Could not open file" << filePath << "for writing";
    }
}

} // namespace ito
