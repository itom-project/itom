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
#include "codeEditor.h"
#include "global.h"
#include "helper/IOHelper.h"
#include "organizer/scriptEditorOrganizer.h"
#include "widgets/scriptDockWidget.h"
#include "widgets/scriptEditorWidget.h"
#include "delegates/htmlItemDelegate.h"

#include <qfile.h>
#include <qfileinfo.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qlistwidget.h>
#include <qmessagebox.h>
#include <qmetaobject.h>
#include <qpushbutton.h>
#include <qstandarditemmodel.h>

namespace ito {

//-------------------------------------------------------------------------------------
PyCodeReferenceRenamer::PyCodeReferenceRenamer(QObject* parent) :
    QObject(parent), m_pPythonEngine(nullptr), m_renameDialog(nullptr), m_newNameUserInput(nullptr),
    m_treeWidgetReferences(nullptr), m_dialogButtonBox(nullptr), m_filesToChange(), m_request()
{
    m_pPythonEngine = AppManagement::getPythonEngine();

    if (!m_renameDialog)
    {
        // create dialog
        m_renameDialog = new QDialog();
        m_renameDialog->setWindowTitle(tr("Rename reference"));
        m_renameDialog->setModal(true);

        m_newNameUserInput = new QLineEdit(m_renameDialog);
        QHBoxLayout* newNameLayout = new QHBoxLayout();
        newNameLayout->addWidget(new QLabel(tr("New value: ")));
        newNameLayout->addWidget(m_newNameUserInput);

        m_treeWidgetReferences = new QTreeWidget(m_renameDialog);
        m_treeWidgetReferences->setAlternatingRowColors(true);
        m_treeWidgetReferences->setItemDelegateForColumn(0, new HtmlItemDelegate());
        m_treeWidgetReferences->setColumnCount(3);


        QStringList headerLabels;
        headerLabels << tr("Value").toUtf8().data() << tr("Line").toUtf8().data()
                     << tr("Column").toUtf8().data();
        m_treeWidgetReferences->setHeaderLabels(headerLabels);

        m_dialogButtonBox = new QDialogButtonBox(
            QDialogButtonBox::StandardButton::Ok | QDialogButtonBox::StandardButton::Cancel, m_renameDialog);

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
    DELETE_AND_SET_NULL(m_renameDialog);
}

//-------------------------------------------------------------------------------------
void PyCodeReferenceRenamer::rename(const int& line, const int& column, const QString& fileName)
{
    PythonEngine* pyEng = (PythonEngine*)m_pPythonEngine;

    if (pyEng)
    {
        if (pyEng->tryToLoadJediIfNotYetDone())
        {
            ScriptEditorOrganizer* seo =
                qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());

            const ScriptDockWidget* sdw = seo->getActiveDockWidget();
            const ScriptEditorWidget* sew = sdw->getCurrentEditor();
            QString code = sew->toPlainText();

            m_request.m_code = code;
            m_request.m_callbackFctName = "onJediRenameResultAvailable";
            m_request.m_col = column;
            m_request.m_line = line;
            m_request.m_fileName = fileName;
            m_request.m_sender = this;
            PythonEngine* pyEng = (PythonEngine*)m_pPythonEngine;

            seo->saveAllScripts(true, true);

            pyEng->enqueueJediRenameRequest(m_request);
        }
    }
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onJediRenameResultAvailable(
    const QVector<ito::JediRename>& filesToChange)
{
    if (filesToChange.isEmpty())
    {
        clearAndHideTreeWidget();
        return;
    }

    m_filesToChange = filesToChange;

    // set current value to new value line edit
    m_newNameUserInput->setText(filesToChange.first().m_values.first());
    m_newNameUserInput->selectAll();

    QDir rootDir = QDir(m_request.m_fileName);

    for (const auto& file : m_filesToChange)
    {
        QTreeWidgetItem* fileItem = new QTreeWidgetItem(m_treeWidgetReferences);
        fileItem->setFlags(fileItem->flags() | Qt::ItemIsUserCheckable);
        QFileInfo fileInfo(file.m_filePath);
        QString displayedPath = fileInfo.absoluteFilePath();
        displayedPath.replace("/", "\\");
        IOHelper::elideFilepathMiddle(displayedPath, 300);
        fileItem->setText(0, displayedPath);
        fileItem->setToolTip(0, fileInfo.absoluteFilePath());
        QFont font(fileItem->font(0));

        QString relativePath = rootDir.relativeFilePath(file.m_filePath);

        if (relativePath.startsWith(".."))
        {
            fileItem->setCheckState(0, Qt::Unchecked);
            fileItem->setExpanded(false);
            font.setBold(false);
        }
        else
        {
            fileItem->setCheckState(0, Qt::Checked);
            fileItem->setExpanded(true);
            font.setBold(true);
        }

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
        int previousLine = -1;
        QString lineText;
        QString textLeft, textRight;
        int lineNumber;

        for (int idx = 0; idx < file.m_lines.size(); ++idx)
        {
            lineNumber = file.m_lines[idx];
            QTreeWidgetItem* lineItem = new QTreeWidgetItem(fileItem);

            if (lineNumber > previousLine)
            {
                for (iterLine; iterLine <= lineNumber; ++iterLine)
                {
                    lineText = in.readLine();
                }
            }

            lineItem->setFlags(lineItem->flags() | Qt::ItemIsUserCheckable);
            lineItem->setCheckState(0, fileItem->checkState(0));

            textLeft = lineText.left(file.m_columns[idx]);
            textRight = lineText.mid(file.m_columns[idx] + file.m_values[idx].size());
            lineItem->setText(0, textLeft + "<b>" + file.m_values[idx] + "</b>" + textRight);

            lineItem->setText(1, QString::number(lineNumber));
            lineItem->setText(2, QString::number(file.m_columns.at(idxLine)));
            idxLine++;
            previousLine = lineNumber;
        }

        scriptFile->close();
        DELETE_AND_SET_NULL(scriptFile);
    }

    for (int i = 0; i < m_treeWidgetReferences->columnCount(); ++i)
    {
        m_treeWidgetReferences->resizeColumnToContents(i);
    }

    m_renameDialog->resize(800, 600);
    m_renameDialog->show();
    m_newNameUserInput->setFocus();
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

    if (newValue.isEmpty())
    {
        QMessageBox msgBox(
            QMessageBox::Warning,
            tr("Warning"),
            tr("No new reference name was given.\n"
               "Do you want to continue?"),
            QMessageBox::Ok | QMessageBox::Cancel);
        msgBox.setDefaultButton(QMessageBox::Cancel);

        if (msgBox.exec() == QMessageBox::Cancel)
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
            filePath = getAbsoluteFilePath(m_filesToChange.at(idxTopLevel).m_filePath);
            for (int idxSecondLevel = topItem->childCount() - 1; idxSecondLevel >= 0;
                 --idxSecondLevel)
            {
                QTreeWidgetItem* secondLevelItem = topItem->child(idxSecondLevel);
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
    switch (event->key())
    {
    case Qt::Key_Enter:
    case Qt::Key_Return:
        onApply();
        break;

    case Qt::Key_Escape:
        onCanceled();
        break;

    default:
        PyCodeReferenceRenamer::keyPressEvent(event);
        break;
    }
    return;
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onCanceled()
{
    clearAndHideTreeWidget();
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onItemChanged(QTreeWidgetItem* item, int column)
{
    if (!item || item->childCount() == 0 || column != 0)
    {
        return;
    }

    Qt::CheckState state = item->checkState(column);

    for (int iter = 0; iter < item->childCount(); ++iter)
    {
        auto childItem = item->child(iter);
        childItem->setCheckState(column, state);

        item->setExpanded(state == Qt::Checked);
        QFont font = QFont(item->font(0));
        font.setBold(state == Qt::Checked);
        item->setFont(0, font);
    }
    return;
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onItemDoubleClick(QTreeWidgetItem* item, int column)
{
    int line = 0;
    QString fileToOpen;
    QTreeWidgetItem* topLevelItem = nullptr;

    ScriptEditorOrganizer* seo =
        qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());

    if (!seo)
        return;

    if (m_treeWidgetReferences->indexOfTopLevelItem(item) != -1)
    {
        topLevelItem = item;
    }
    else
    {
        topLevelItem = item->parent();
        line = item->text(1).toInt() - 1;
    }

    QFileInfo* fileInfo = new QFileInfo(topLevelItem->text(0));
    fileToOpen = getAbsoluteFilePath(fileInfo->fileName());
    seo->openScript(getAbsoluteFilePath(fileToOpen), nullptr, line);
    return;
}

//-------------------------------------------------------------------
QString PyCodeReferenceRenamer::getAbsoluteFilePath(const QString& fileName)
{
    for (const auto& files : m_filesToChange)
    {
        if (files.m_filePath.indexOf(fileName) != -1)
        {
            return files.m_filePath;
        }
    }

    return QString(); // Return an empty string if the file is not found
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
            if (columnNumber >= 0 && columnNumber <= line.length())
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
