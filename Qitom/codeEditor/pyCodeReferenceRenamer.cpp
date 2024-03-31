/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2024, Institut fuer Technische Optik (ITO),
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

#include "pyCodeReferenceRenamer.h"
#include "../python/pythonEngine.h"
#include "../python/pythonJedi.h"
#include "AppManagement.h"
#include "codeEditor.h"
#include "delegates/htmlItemDelegate.h"
#include "global.h"
#include "helper/IOHelper.h"
#include "helper/guiHelper.h"
#include "organizer/scriptEditorOrganizer.h"
#include "widgets/scriptDockWidget.h"
#include "widgets/scriptEditorWidget.h"
#include "widgets/itomQWidgets.h"

#include <qfile.h>
#include <qfileinfo.h>
#include <qheaderview.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qlistwidget.h>
#include <qmessagebox.h>
#include <qmetaobject.h>
#include <qpushbutton.h>
#include <qstandarditemmodel.h>

namespace ito {

//-------------------------------------------------------------------------------------
PyCodeReferenceRenamer::PyCodeReferenceRenamer(QWidget* parent) :
    QObject(parent), m_pParent(parent), m_pPythonEngine(nullptr), m_renameDialog(nullptr),
    m_newNameUserInput(nullptr), m_treeWidgetReferences(nullptr), m_dialogButtonBox(nullptr)
{
    m_pPythonEngine = AppManagement::getPythonEngine();

    if (!m_renameDialog)
    {
        // create dialog
        m_renameDialog = new QDialog(parent);
        m_renameDialog->setWindowTitle(tr("Rename reference"));
        m_renameDialog->setModal(true);

        m_newNameUserInput = new QLineEdit(m_renameDialog);
        QHBoxLayout* newNameLayout = new QHBoxLayout();
        newNameLayout->addWidget(new QLabel(tr("New value: ")));
        newNameLayout->addWidget(m_newNameUserInput);

        m_treeWidgetReferences = new QTreeWidgetItom(m_renameDialog);
        m_treeWidgetReferences->setAlternatingRowColors(true);
        HtmlItemDelegate* htmlDelegate = new HtmlItemDelegate();
        m_treeWidgetReferences->setItemDelegateForColumn(0, htmlDelegate);
        m_treeWidgetReferences->setColumnCount(3);

        m_treeWidgetReferences->header()->setDefaultSectionSize(GuiHelper::screenDpiFactor() * 55);
        m_treeWidgetReferences->header()->setStretchLastSection(false);
        m_treeWidgetReferences->header()->setSectionResizeMode(0, QHeaderView::Stretch);
        m_treeWidgetReferences->header()->setSectionResizeMode(1, QHeaderView::Fixed);
        m_treeWidgetReferences->header()->setSectionResizeMode(2, QHeaderView::Fixed);

        QStringList headerLabels;
        headerLabels << tr("Value").toUtf8().data() << tr("Line").toUtf8().data()
                     << tr("Column").toUtf8().data();
        m_treeWidgetReferences->setHeaderLabels(headerLabels);

        m_dialogButtonBox = new QDialogButtonBox(
            QDialogButtonBox::StandardButton::Ok | QDialogButtonBox::StandardButton::Cancel,
            m_renameDialog);

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
            htmlDelegate,
            &HtmlItemDelegate::itemDoubleClicked,
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
//! starts a variable rename operation for the word under the given cursor position
/*!
    setups connections to python engine and to get a notification about focus changes.

    \param filepath is the filepath of the word under the cursor or a NULL-string, if 
        the script is unnamed, hence, has not been safed yet.
*/
ito::RetVal PyCodeReferenceRenamer::rename(
    const int& line, const int& column, const QString& filepath)
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

            if (filepath.isNull())
            {
                // unnamed file --> create a non-existing dummy filename
                QDir currDir = QDir::currentPath();
                int idx = 1;
                QString filenamePattern = "__untitled%1__.py";

                while (currDir.exists(filenamePattern.arg(idx)))
                {
                    idx++;
                }

                m_request.m_filepath = currDir.filePath(filenamePattern.arg(idx));
                m_request.m_fileModified = true;
                m_request.m_untitledFile = true;
                m_request.m_untitledName = sew->getUntitledName();
            }
            else
            {
                m_request.m_filepath = filepath;
                m_request.m_fileModified = sew->isModified();
                m_request.m_untitledFile = false;
                m_request.m_untitledName = QString();
            }
            
            m_request.m_callbackFctName = "onJediRenameResultAvailable";
            m_request.m_col = column;
            m_request.m_line = line;
            
            m_request.m_sender = this;
            PythonEngine* pyEng = (PythonEngine*)m_pPythonEngine;

            QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
            pyEng->enqueueJediRenameRequest(m_request);
        }
        else
        {
            return RetVal(
                retError,
                CodeYetNotAvailable,
                tr("Python module Jedi is not available or could not be loaded. The rename feature "
                   "will be disabled.")
                    .toLatin1()
                    .data());
        }
    }

    return ito::retOk;
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onJediRenameResultAvailable(
    const QVector<ito::JediRename>& filesToChange,
    const QString& oldValue,
    bool success,
    QString errorText)
{
    QApplication::restoreOverrideCursor();

    if (!success || filesToChange.isEmpty())
    {
        if (errorText != "")
        {
            QMessageBox::critical(m_pParent, tr("Rename error"), errorText);
        }

        clearAndHideTreeWidget();
        return;
    }

    // set current value to new value line edit
    m_newNameUserInput->setText(oldValue);

    QDir rootDir = QDir(m_request.m_filepath);

    ScriptEditorOrganizer* seo =
        qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());

    const auto openedScripts = seo->getAllOpenedScriptsWithModificationState();
    QString canonicalFilePath, displayedPath;
    bool modified, scriptOpened;

    for (const auto& file : filesToChange)
    {
        QTreeWidgetItem* fileItem = new QTreeWidgetItem(m_treeWidgetReferences);
        fileItem->setFlags(fileItem->flags() | Qt::ItemIsUserCheckable);
        QFileInfo fileInfo(file.m_filePath);
        canonicalFilePath = fileInfo.canonicalFilePath();
        displayedPath = canonicalFilePath;
        modified = true;
        scriptOpened = false;

        // openedScripts only contains saved scripts, Untitled scripts are not considered.
        foreach (const auto& item, openedScripts)
        {
            if (item.first == canonicalFilePath)
            {
                modified = item.second;
                scriptOpened = true;
                break;
            }
        }

        if (file.m_untitledFile)
        {
            fileItem->setData(0, RoleFilePath, file.m_untitledFilename);
            fileItem->setData(0, RoleMainFile, file.m_untitledFilename);
            fileItem->setData(0, RoleFileUntitled, true);
            displayedPath = file.m_untitledFilename; 
            scriptOpened = true; // an untitled script must be opened, anything else is not possible
        }
        else
        {
            fileItem->setData(0, RoleFilePath, canonicalFilePath);
            fileItem->setData(0, RoleMainFile, file.m_mainFile);
            fileItem->setData(0, RoleFileUntitled, false);
            
        }

        fileItem->setData(0, RoleFileModified, modified);
        fileItem->setData(0, RoleFileOpened, scriptOpened);
        IOHelper::elideFilepathMiddle(displayedPath, 300);

        if (modified)
        {
            fileItem->setText(0, "<b>" + displayedPath + "*</b>");

            if (!file.m_untitledFile)
            {
                fileItem->setData(
                    0,
                    Qt::ToolTipRole,
                    fileInfo.absoluteFilePath() + " " + tr("(Script contains unsaved changes)"));
            }
        }
        else
        {
            fileItem->setText(0, "<b>" + displayedPath + "</b>");

            if (!file.m_untitledFile)
            {
                fileItem->setData(0, Qt::ToolTipRole, fileInfo.absoluteFilePath());
            }
        }

        if (!file.m_fileInProject)
        {
            fileItem->setCheckState(0, Qt::Unchecked);
            fileItem->setExpanded(false);
        }
        else
        {
            fileItem->setCheckState(0, Qt::Checked);
            fileItem->setExpanded(true);
        }

        m_treeWidgetReferences->addTopLevelItem(fileItem);
        fileItem->setFirstColumnSpanned(true);

        QString lineText;
        QString textLeft, textRight, value;
        QStringList content;

        if (file.m_items.size() > 0)
        {
            if (file.m_mainFile)
            {
                const ScriptEditorWidget* sew;

                if (file.m_untitledFile)
                {
                    sew = seo->getActiveDockWidget() ? seo->getActiveDockWidget()->getCurrentEditor() : nullptr;

                    if (sew && sew->getUntitledName() != file.m_untitledFilename)
                    {
                        sew = nullptr;
                    }
                }
                else
                {
                    sew = seo->getEditorFromCanonicalFilepath(file.m_filePath);
                }

                if (sew)
                {
                    content = sew->toPlainText().split("\n");
                }
            }
            else
            {
                content = readFirstNLinesFromFile(file.m_filePath, file.m_items.last().lineNumber);
            }
        }

        for (int idx = 0; idx < file.m_items.size(); ++idx)
        {
            const ito::FileRenameItem& renameItem = file.m_items[idx];
            QTreeWidgetItem* lineItem = new QTreeWidgetItem(fileItem);
            lineItem->setData(0, Qt::UserRole, canonicalFilePath);

            if (content.size() >= renameItem.lineNumber)
            {
                lineText = content[renameItem.lineNumber - 1];
            }
            else
            {
                lineText = "";
            }

            lineItem->setFlags(lineItem->flags() | Qt::ItemIsUserCheckable);
            lineItem->setCheckState(0, fileItem->checkState(0));

            textLeft = lineText.left(renameItem.startColumnIndex);
            textRight = lineText.mid(renameItem.startColumnIndex + renameItem.oldWordSize);
            value = lineText.mid(renameItem.startColumnIndex, renameItem.oldWordSize);
            lineItem->setText(0, textLeft + "<b>" + value + "</b>" + textRight);

            lineItem->setText(1, QString::number(renameItem.lineNumber));
            lineItem->setText(2, QString::number(renameItem.startColumnIndex));
            lineItem->setData(0, RoleFileRenameItem, QVariant::fromValue(file.m_items[idx]));
        }
    }

    auto f = GuiHelper::screenDpiFactor();
    m_renameDialog->resize(f * 800.0, f * 600.0);
    m_renameDialog->show();
    m_newNameUserInput->setFocus();
    m_newNameUserInput->selectAll();
}

//-------------------------------------------------------------------
QStringList PyCodeReferenceRenamer::readFirstNLinesFromFile(const QString& filepath, int n) const
{
    QFile scriptFile(filepath);

    if (!scriptFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        qDebug() << "Failed to open the file";
        return QStringList();
    }

    QTextStream in(&scriptFile);
    QStringList content;
    int i = 0;

    while (i < n && !in.atEnd())
    {
        content << in.readLine();
        i++;
    }

    scriptFile.close();
    return content;
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
    QTreeWidgetItem* fileItem;
    QTreeWidgetItem* changeItem;
    QString value;
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

    // Approach:
    /* 1. collect all files with at least one replacement and a list of replacements
       2. Check if there is at least one file, that is currently opened in itom, modified and not
       the main file. If this exists, ask if the modifications should be done. If yes, the affected
          files will be modified on the hard drive and a modification notification will be shown in
       itom.
    */

    ScriptEditorOrganizer* seo =
        qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());
    ScriptEditorWidget* sew = nullptr;

    struct RenameFile
    {
        QString canonicalFileName; // is the file path or the name of the untitled script (fileUntitled = true)
        QVector<FileRenameItem> items;
        bool fileOpened;
        bool mainFile;
        bool fileModified;
        bool fileUntitled;
    };

    QVector<RenameFile> renameFiles;
    bool needToAskTheUser = false;

    // iterate over all files
    for (int fileIdx = 0; fileIdx < m_treeWidgetReferences->topLevelItemCount(); ++fileIdx)
    {
        fileItem = m_treeWidgetReferences->topLevelItem(fileIdx); // get files on top level

        if (fileItem->checkState(0) == Qt::Checked) // iter lines and columns of checked files
        {
            RenameFile renameFile;
            renameFile.canonicalFileName = fileItem->data(0, RoleFilePath).toString();
            renameFile.mainFile = fileItem->data(0, RoleMainFile).toBool();
            renameFile.fileOpened = fileItem->data(0, RoleFileOpened).toBool();
            renameFile.fileModified = fileItem->data(0, RoleFileModified).toBool();
            renameFile.fileUntitled = fileItem->data(0, RoleFileUntitled).toBool();

            for (int itemIdx = fileItem->childCount() - 1; itemIdx >= 0; --itemIdx)
            {
                changeItem = fileItem->child(itemIdx);

                if (changeItem->checkState(0) == Qt::Checked)
                {
                    renameFile.items
                        << changeItem->data(0, RoleFileRenameItem).value<FileRenameItem>();
                }
            }

            if (renameFile.items.size() > 0)
            {
                renameFiles << renameFile;

                if (renameFile.fileOpened && !renameFile.mainFile && renameFile.fileModified)
                {
                    needToAskTheUser = true;
                }
            }
        }
    }

    // check if there is at least one opened file, which is not the main file and which is modified
    if (needToAskTheUser)
    {
        QMessageBox msgBox(
            QMessageBox::Question,
            tr("Changes in modified files"),
            tr("Some renames affect other opened and modified scripts. If you continue, "
               "these files will be modified based on their latest saved state. "
               "Do you want to continue?"),
            QMessageBox::Yes | QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::No);

        if (msgBox.exec() == QMessageBox::No)
        {
            return;
        }
    }

    // do the replacement
    foreach (const RenameFile& renameFile, renameFiles)
    {
        if (renameFile.fileUntitled)
        {
            if (renameFile.fileOpened)
            {
                sew = seo->getActiveDockWidget() ? seo->getActiveDockWidget()->getCurrentEditor()
                                                 : nullptr;

                if (sew && sew->getUntitledName() == renameFile.canonicalFileName)
                {
                    sew->replaceOccurencesInCurrentScript(newValue, renameFile.items);
                }
                else
                {
                    qDebug() << "The main script could not be referenced or is not the wanted "
                                "untitled script. This should not be possible!";
                }
            }
            else
            {
                qDebug() << "Renames should be done in an untitled script, which however is not "
                            "opened. This should not be possible!";
            }
        }
        else if (!renameFile.fileOpened ||
            (renameFile.fileOpened && !renameFile.mainFile && renameFile.fileModified))
        {
            ito::RetVal retValue =
                replaceOccurencesInFile(renameFile.canonicalFileName, newValue, renameFile.items);

            if (retValue.containsError())
            {
                QMessageBox::warning(
                    m_renameDialog,
                    tr("File error"),
                    tr("An error occurred when replacing occurrences in the file '%1': %2")
                        .arg(renameFile.canonicalFileName)
                        .arg(QLatin1String(retValue.errorMessage())));
            }
        }
        else if (renameFile.mainFile || (renameFile.fileOpened && !renameFile.fileModified))
        {
            sew = seo->getEditorFromCanonicalFilepath(renameFile.canonicalFileName);

            if (sew)
            {
                sew->replaceOccurencesInCurrentScript(newValue, renameFile.items);
            }
            else
            {
                qDebug() << "The main script could not be referenced. This should not be possible!";
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
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onItemDoubleClick(QTreeWidget* treeWidget, QTreeWidgetItem* item)
{
    ScriptEditorOrganizer* seo =
        qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());
    if (!seo)
        return;

    // top level item with filename information
    QTreeWidgetItem* topLevelItem =
        (treeWidget->indexOfTopLevelItem(item) != -1) ? item : item->parent();

    QString filename = topLevelItem->data(0, RoleFilePath).toString();
    QFileInfo fileInfo(filename);

    int line = item->data(1, Qt::DisplayRole).toInt() - 1;
    line = (line == -1) ? 0 : line; // set to 0 to open file in the first line

    seo->openScript(fileInfo.filePath(), nullptr, line);
}

//-------------------------------------------------------------------
ito::RetVal PyCodeReferenceRenamer::replaceOccurencesInFile(
    const QString& filePath,
    const QString& newValue,
    const QVector<ito::FileRenameItem>& renameItems)
{
    // sort items by starting with the last one first
    QVector<ito::FileRenameItem> items = renameItems;
    std::sort(
        items.begin(), items.end(), [](const ito::FileRenameItem& a, const ito::FileRenameItem& b) {
            if (a.lineNumber != b.lineNumber)
            {
                return a.lineNumber >= b.lineNumber;
            }
            else
            {
                return a.startColumnIndex >= b.startColumnIndex;
            }
        });

    // Open the file
    QFile file(filePath);

    if (!file.open(QIODevice::ReadWrite | QIODevice::Text))
    {
        return ito::RetVal(ito::retError, CouldNotOpenFile, "could not open the file");
    }

    // Create a QTextStream to read from and write to the file
    QTextStream stream(&file);

    // Create a list to store modified lines
    QStringList content;
    QString line;

    while (!stream.atEnd())
    {
        content << stream.readLine();
    }

    // modify content
    foreach (const ito::FileRenameItem& item, items)
    {
        line = content[item.lineNumber - 1];
        line = line.left(item.startColumnIndex) + newValue +
            line.mid(item.startColumnIndex + item.oldWordSize);
        content[item.lineNumber - 1] = line;
    }

    // clear file
    file.resize(0);

    // go back to the start
    stream.seek(0);

    foreach (const QString& line, content)
    {
        stream << line << "\n";
    }

    file.close();

    return ito::retOk;
}

} // namespace ito
