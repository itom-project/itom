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

// File handling of the script editor (moved from scriptEditorWidget.cpp): loading and
// saving with charset encodings and line endings, the file system watcher.

#include "scriptEditorWidget.h"

#include "../AppManagement.h"
#include "../global.h"
#include "../helper/IOHelper.h"
#include "../helper/charsetEncoding.h"
#include "../helper/lineEnding.h"
#include "../organizer/scriptEditorOrganizer.h"
#include "../python/pythonEngine.h"
#include "../ui/dialogScriptCharsetEncoding.h"

#include <qdebug.h>
#include <qdir.h>
#include <qfile.h>
#include <qfiledialog.h>
#include <qfileinfo.h>
#include <qmessagebox.h>
#include <qsettings.h>
#include <qtextdocument.h>

namespace ito {

//-------------------------------------------------------------------------------------
void ScriptEditorWidget::changeFileSaveEncoding(const IOHelper::CharsetEncodingItem& encoding)
{
    m_charsetEncoding = encoding;
    setModified(true);
    onTextChanged();
    emit cursorPositionChanged();
}

//-------------------------------------------------------------------------------------
void ScriptEditorWidget::menuScriptCharsetEncoding()
{
    DialogScriptCharsetEncoding* dialog =
        new DialogScriptCharsetEncoding(charsetEncoding(), !hasNoFilename(), this);

    connect(dialog, &DialogScriptCharsetEncoding::accepted, [=]() {
        this->changeFileSaveEncoding(dialog->getSaveCharsetEncoding());
    });

    connect(
        dialog,
        &DialogScriptCharsetEncoding::reloadWithEncoding,
        [=](const IOHelper::CharsetEncodingItem& item) {
            bool oldGuess = m_charsetEncodingAutoGuess;
            m_charsetEncodingAutoGuess = false;
            m_charsetEncoding = item;

            openFile(getFilename(), false, dialog);

            m_charsetEncodingAutoGuess = oldGuess;
        });

    connect(
        dialog,
        &DialogScriptCharsetEncoding::addPythonEncodingComment,
        [=](const QString& encoding) {
            if (encoding != "")
            {
                auto items = encoding.split(" ");

                if (encoding.size() > 0)
                {
                    QString newText = QString("# coding=%1\n").arg(items[0]);
                    QTextCursor currentCursor = textCursor();
                    currentCursor.movePosition(QTextCursor::Start, QTextCursor::MoveAnchor);
                    currentCursor.insertText(newText);
                    setModified(true);
                    onTextChanged();
                }
            }
        });

    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}


//-------------------------------------------------------------------------------------
IOHelper::CharsetEncodingItem ScriptEditorWidget::guessEncoding(const QByteArray& content) const
{
    // check for BOMs first (e.g. UTF-8, UTF-16 or UTF-32 files).
    bool bomFound = false;
    auto bomEncoding = CharsetEncoding::fromBom(content, &bomFound);

    if (bomFound)
    {
        return bomEncoding;
    }

    // UTF-16 files without BOM (e.g. from some Windows tools)
    bool utf16Found = false;
    auto utf16Encoding = CharsetEncoding::guessUtf16WithoutBom(content, &utf16Found);

    if (utf16Found)
    {
        return utf16Encoding;
    }

    QList<QByteArray> firstLines;

    int from = 0;
    int idx = content.indexOf("\n", from);

    if (idx >= 0)
    {
        firstLines << content.mid(from, idx - from).trimmed();
        from = idx + 1;

        idx = content.indexOf("\n", from);

        if (idx >= from)
        {
            firstLines << content.mid(from, idx - from).trimmed();
        }
    }

    QRegularExpression re("#.*?coding[:=][ \\t]*([-_.a-zA-Z0-9]+)");

    foreach (const QByteArray& ba, firstLines)
    {
        auto match = re.match(QString::fromUtf8(ba));

        if (match.hasMatch())
        {
            return IOHelper::getEncodingFromAlias(match.captured(1));
        }
    }

    return m_charsetEncoding;
}

//-------------------------------------------------------------------------------------
void ScriptEditorWidget::setLineEnding(LineEnding::Type lineEnding)
{
    if (lineEnding != m_lineEnding)
    {
        m_lineEnding = lineEnding;
        document()->setModified(true);
        emit cursorPositionChanged(); // updates the status bar
    }
}

//-------------------------------------------------------------------------------------
void ScriptEditorWidget::menuScriptLineEndingLF()
{
    setLineEnding(LineEnding::LF);
}

//-------------------------------------------------------------------------------------
void ScriptEditorWidget::menuScriptLineEndingCRLF()
{
    setLineEnding(LineEnding::CRLF);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::openFile(
    const QString& fileName, bool ignorePresentDocument, QWidget* parent /*= nullptr*/)
{
    if (parent == nullptr)
    {
        parent = this;
    }

    //!< check for modifications in the present document first
    if (!ignorePresentDocument)
    {
        if (isModified())
        {
            int ret = QMessageBox::information(
                parent,
                tr("Unsaved Changes"),
                tr("There are unsaved changes in the current document. Do you want to save it "
                   "first?"),
                QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel,
                QMessageBox::Yes);

            if (ret & QMessageBox::Cancel)
            {
                return RetVal(retOk);
            }
            else if (ret & QMessageBox::Yes)
            {
                saveFile(false);
                setModified(false);
            }
        }
    }

    QFile file(fileName);

    // The file is read in binary mode. QIODevice::Text would remove '\r' bytes before
    // the decoding, which destroys the alignment of UTF-16 / UTF-32 files. The line
    // endings are normalized after the decoding (see LineEnding::normalize).
    if (!file.open(QIODevice::ReadOnly))
    {
        QMessageBox::warning(
            parent,
            tr("Error while opening file"),
            tr("File %1 could not be loaded").arg(fileName));
    }
    else
    {
        // in Qt4, QString(QByteArray) created the string with fromAscii(byteArray), in Qt5 it is
        // fromUtf8(byteArray) therefore there is a setting property telling the encoding of saved
        // python files and the files are loaded assuming this special encoding. If no encoding is
        // given, latin1 is always assumed.
        QByteArray content = file.readAll();
        QString text;

        if (m_charsetEncodingAutoGuess)
        {
            m_charsetEncoding = guessEncoding(content);
        }

        bool encodingOk = true;
        text = CharsetEncoding::decode(content, m_charsetEncoding, &encodingOk);
        m_lineEnding = LineEnding::detect(text);
        LineEnding::normalize(text);

        if (!encodingOk)
        {
            qDebug() << "unknown encoding " << m_charsetEncoding.encodingName << ". Assume UTF_8";
        }

        file.close();

        clearAllBreakpoints();

        bool wasLargeFile = m_largeFileMode;
        m_largeFileMode = content.size() > m_largeFileThresholdBytes;

        // set the file type (and syntax highlighter) before the text is loaded
        updateFileType(fileName);

        if (m_largeFileMode != wasLargeFile)
        {
            // reloads the enabled states of all modes and applies the large file mode
            loadSettings();
        }

        // block signals avoid that nrOfLinesChanges is emitted, that
        // might cause a crash uue to the outline parser and codeOrErrorString cells
        blockSignals(true);
        setPlainText("");
        blockSignals(false);

        setPlainText(text);

        changeFilename(fileName);

        QList<BookmarkItem> currentBookmarks = m_pBookmarkModel->getBookmarks(fileName);

        foreach (const BookmarkItem& item, currentBookmarks)
        {
            onBookmarkAdded(item);
        }

        QStringList watchedFiles = m_pFileSysWatcher->files();

        if (watchedFiles.size() > 0)
        {
            m_pFileSysWatcher->removePaths(watchedFiles);
        }

        m_pFileSysWatcher->addPath(m_filename);

        //!< check if BreakPointModel already contains breakpoints for this editor and load them
        if (getFilename() != "")
        {
            const BreakPointModel* bpModel = getBreakPointModel();

            if (bpModel)
            {
                QModelIndexList modelIndexList = bpModel->getBreakPointIndizes(getFilename());
                QList<BreakPointItem> bpItems = bpModel->getBreakPoints(modelIndexList);

                for (int i = 0; i < bpItems.size(); i++)
                {
                    breakPointAdd(bpItems.at(i), i);
                }
            }
        }

        setModified(false);

        // update the outline
        outlineTimerElapsed();

        QObject* seo = AppManagement::getScriptEditorOrganizer();
        if (seo)
        {
            QMetaObject::invokeMethod(seo, "fileOpenedOrSaved", Q_ARG(QString, m_filename));
        }
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::saveFile(bool askFirst)
{
    if (!isModified())
    {
        return RetVal(retOk);
    }

    if (getFilename().isNull())
    {
        return saveAsFile(askFirst);
    }

    if (m_autoCodeFormatOnSave && isPythonFile())
    {
        ito::RetVal retValue = formatPythonCode(500);

        if (retValue.containsError())
        {
            return retValue;
        }
    }

    if (askFirst)
    {
        int ret = QMessageBox::information(
            this,
            tr("Unsaved Changes"),
            tr("There are unsaved changes in the document '%1'. Do you want to save it first?")
                .arg(getFilename()),
            QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel,
            QMessageBox::Yes);

        if (ret & QMessageBox::Cancel)
        {
            return RetVal(retError);
        }
        else if (ret & QMessageBox::No)
        {
            return RetVal(retOk);
        }
    }
    return writeFileContent(getFilename());
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::writeFileContent(const QString &filename)
{
    QFile file(filename);

    // The file is written in binary mode. QIODevice::Text would convert '\n' to "\r\n"
    // on the byte level (Windows), which destroys UTF-16 / UTF-32 files. The platform
    // line ending of the file is applied to the text instead (see LineEnding::apply).
    if (!file.open(QIODevice::WriteOnly))
    {
        QMessageBox::warning(
            this,
            tr("Error while accessing file"),
            tr("File %1 could not be accessed").arg(filename));
        return RetVal(retError);
    }

    m_pFileSysWatcher->removePath(filename);

    QString t = LineEnding::apply(toPlainText(), m_lineEnding);
    QByteArray text;

    if (m_charsetEncodingAutoGuess)
    {
        auto encoding = guessEncoding(t.toUtf8());

        if (encoding.encodingName != m_charsetEncoding.encodingName)
        {
            int ret = QMessageBox::question(
                this,
                tr("Inconsistent encoding"),
                tr("The coding tag %1 does not correspond to the file encoding %2. Use %1 as new "
                   "file encoding?")
                    .arg(encoding.encodingName)
                    .arg(m_charsetEncoding.encodingName),
                QMessageBox::Yes | QMessageBox::No,
                QMessageBox::Yes);

            if (ret & QMessageBox::Yes)
            {
                m_charsetEncoding = encoding;
            }
        }
    }

    bool encodingOk = true;
    text = CharsetEncoding::encode(t, m_charsetEncoding, &encodingOk);

    if (!encodingOk)
    {
        QMessageBox::warning(
            this,
            tr("Unsupported encoding"),
            tr("The encoding %1 is unsupported on this computer. Switch to the default encoding "
               "UTF-8")
                .arg(m_charsetEncoding.encodingName));
        m_charsetEncoding = IOHelper::getDefaultScriptEncoding();
        text = CharsetEncoding::encode(t, m_charsetEncoding);
    }

    file.write(text);
    file.close();

    changeFilename(filename);

    QFileInfo fi(filename);

    if (fi.exists())
    {
        QObject* seo = AppManagement::getScriptEditorOrganizer();

        if (seo)
        {
            QMetaObject::invokeMethod(seo, "fileOpenedOrSaved", Q_ARG(QString, filename));
        }
    }

    setModified(false);

    m_pFileSysWatcher->addPath(filename);

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::saveAsFile(bool askFirst)
{
    if (m_autoCodeFormatOnSave && isPythonFile())
    {
        ito::RetVal retValue = formatPythonCode(500);

        if (retValue.containsError())
        {
            return retValue;
        }
    }

    if (askFirst)
    {
        int ret = QMessageBox::information(
            this,
            tr("Unsaved Changes"),
            tr("There are unsaved changes in the current document. Do you want to save it first?"),
            QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel,
            QMessageBox::Yes);
        if (ret & QMessageBox::Cancel)
        {
            return RetVal(retError);
        }
        else if (ret & QMessageBox::No)
        {
            return RetVal(retOk);
        }
    }

    QString defaultPath = QDir::currentPath();

    // we need to block the signals from the file system watcher, since a crash will occur if this
    // file is renamed during the save as process (the 'remove file due to rename' dialog will
    // appear during the save-as dialog if the signal is not blocked)
    m_pFileSysWatcher->blockSignals(true);
    QString tempFileName =
        QFileDialog::getSaveFileName(
            this, tr("Save As..."), defaultPath, EditorFileType::saveFileFilter(m_fileType));
    m_pFileSysWatcher->blockSignals(false);
    if (!tempFileName.isEmpty())
    {
        QDir::setCurrent(QFileInfo(tempFileName).path());
        return writeFileContent(tempFileName);
    }
    else
    {
        return RetVal(retError);
    }
}


//-------------------------------------------------------------------------------------
// this signal may be emitted multiple times at once for the same file, therefore the mutex
// protection is introduced
void ScriptEditorWidget::fileSysWatcherFileChanged(const QString& path)
{
    if (path == getFilename())
    {
        QFileInfo file(path);

        // due to a bug in file system watcher, the fileChanged signal
        // is sometimes emitted more than one time. Therefore block it here
        // and release it in the fileSysWatcherFileChangedStep2 method after
        // a while.
        m_pFileSysWatcher->blockSignals(true);

        if (!file.exists()) // file deleted
        {
            // if git updates a file, the file is deleted and then the modified file is created.
            // this will cause a 'delete' notification, however the 'modified' notification would be
            // correct. try to sleep for a while and recheck the state of the file again...
            QTimer::singleShot(
                500,
                this,
                std::bind(&ScriptEditorWidget::fileSysWatcherFileChangedStep2, this, path));
        }
        else // file changed
        {
            QTimer::singleShot(
                100,
                this,
                std::bind(&ScriptEditorWidget::fileSysWatcherFileChangedStep2, this, path));
        }
    }
}

//-------------------------------------------------------------------------------------
/* this method is called by fileSysWatcherFileChanged. If the file
is modified, it is called immediately. If the file is pretended to be deleted,
this 2nd step method is called with a certain delay, since it might be,
that the file exists again after this delay (e.g. during a git modification step)
*/
void ScriptEditorWidget::fileSysWatcherFileChangedStep2(const QString& path)
{
    if (path == getFilename())
    {
        QFile file(path);

        if (!file.exists()) // file deleted
        {
            QMessageBox* msgBox = new QMessageBox(this);
            msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No);
            msgBox->setDefaultButton(QMessageBox::Yes);
            msgBox->setText(tr("The file '%1' does not exist any more.").arg(path));
            msgBox->setInformativeText(tr("Keep this file in editor?"));

            int ret = msgBox->exec();
            DELETE_AND_SET_NULL(msgBox);

            if (ret == QMessageBox::No)
            {
                emit closeRequest(this, true);
            }
            else
            {
                document()->setModified(true);
            }
        }
        else // file changed
        {
            // compare the content of the file with the content of this script
            // (decoded and with normalized line endings, independent of the encoding)
            QString fileText;

            if (file.open(QIODevice::ReadOnly))
            {
                fileText = CharsetEncoding::decode(file.readAll(), m_charsetEncoding);
                LineEnding::normalize(fileText);
                file.close();
            }

            if (fileText != toPlainText()) // does not ask user in the case of same texts
            {
                QMessageBox* msgBox = new QMessageBox(this);
                msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No);
                msgBox->setDefaultButton(QMessageBox::Yes);
                msgBox->setText(
                    tr("The file '%1' has been modified by another program.").arg(path));
                msgBox->setInformativeText(tr("Do you want to reload it?"));
                int ret = msgBox->exec();
                DELETE_AND_SET_NULL(msgBox);

                if (ret == QMessageBox::Yes)
                {
                    int line, col;
                    cursorPosition(line, col);

                    openFile(path, true);

                    // try to apply the cursor position again
                    setCursorPosAndEnsureVisible(line);
                    setCursorPosition(line, col);
                }
                else
                {
                    document()->setModified(true);
                }
            }
        }
    }

    m_pFileSysWatcher->blockSignals(false);
}

//-------------------------------------------------------------------------------------
// Outline
//-------------------------------------------------------------------------------------

} // end namespace ito
