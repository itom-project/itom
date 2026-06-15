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

    Further hints:
    ------------------------

    This file belongs to the code editor of itom. The code editor is
    in major parts a fork / rewritten version of the python-based source
    code editor PyQode from Colin Duquesnoy and others
    (see https://github.com/pyQode). PyQode itself is licensed under
    the MIT License (MIT).

    Some parts of the code editor of itom are also inspired by the
    source code editor of the Spyder IDE (https://github.com/spyder-ide),
    also licensed under the MIT License and developed by the Spyder Project
    Contributors.

*********************************************************************** */

#include "pyGotoAssignment.h"

#include "../codeEditor.h"
#include "../utils/utils.h"
#include "../managers/textDecorationsManager.h"
#include "../delayJobRunner.h"
#include "AppManagement.h"
#include "../../widgets/scriptEditorWidget.h"
#include "../../helper/guiHelper.h"

#include "python/pythonEngine.h"

#include <qinputdialog.h>
#include <qmessagebox.h>
#include <qdir.h>

namespace ito {

//----------------------------------------------------------
/*
*/
PyGotoAssignmentMode::PyGotoAssignmentMode(const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    WordClickMode("PyGotoAssignment", description, parent),
    m_gotoRequested(false),
    m_pPythonEngine(NULL),
    m_pActionGotoDefinition(NULL),
    m_pActionGotoAssignment(NULL),
    m_pActionGotoAssignmentExtended(NULL),
    m_defaultMode(0),
    m_mouseClickEnabled(true),
    m_gotoRequestedTimerId(0)
{
    qRegisterMetaType<PyAssignment>("PyAssignment");

    connect(this, SIGNAL(wordClicked(QTextCursor)), this, SLOT(onWordClicked(QTextCursor)));

    m_pPythonEngine = AppManagement::getPythonEngine();

    // the shortcut cannot be handled by a shortcut, set to the action,
    // since it has the wrong context. Therefore the shortcut is caught
    // by the keyRelease event and the shortcut string is appended via
    // a tab sign to the string, such that it appears on the right side
    // of the menu entry.
    auto keySequence = QKeySequence(tr("F12", "QShortcut"));
    m_pActionGotoDefinition = new QAction(tr("Go To Definition") + "\t" + keySequence.toString(QKeySequence::NativeText), this);
    // m_pActionGotoDefinition->setShortcut(keySequence);
    connect(m_pActionGotoDefinition, SIGNAL(triggered()), this, SLOT(requestGotoDefinition()));

    m_pActionGotoAssignment = new QAction(tr("Go To Assignment"), this);
    connect(m_pActionGotoAssignment, SIGNAL(triggered()), this, SLOT(requestGotoAssignment()));

    m_pActionGotoAssignmentExtended = new QAction(tr("Go To Assignment (Follow Imports)"), this);
    connect(m_pActionGotoAssignmentExtended, SIGNAL(triggered()), this, SLOT(requestGotoAssignmentEx()));
}

//----------------------------------------------------------
/*
*/
PyGotoAssignmentMode::~PyGotoAssignmentMode()
{
    m_pActionGotoDefinition->deleteLater();
    m_pActionGotoAssignment->deleteLater();
    m_pActionGotoAssignmentExtended->deleteLater();
}

//----------------------------------------------------------
/*
*/
/*virtual*/ QList<QAction*> PyGotoAssignmentMode::actions() const
{
    return (QList<QAction*>() << m_pActionGotoDefinition << m_pActionGotoAssignment << m_pActionGotoAssignmentExtended);
}


//----------------------------------------------------------
/*
*/
/*virtual*/ void PyGotoAssignmentMode::onStateChanged(bool state)
{
    if (m_pPythonEngine)
    {
        WordClickMode::onStateChanged(state && m_mouseClickEnabled);

        if (state)
        {
            m_pActionGotoDefinition->setVisible(true);
            m_pActionGotoAssignment->setVisible(true);
            m_pActionGotoAssignmentExtended->setVisible(true);
            connect(editor(), SIGNAL(keyReleased(QKeyEvent*)), this, SLOT(onKeyReleasedGoto(QKeyEvent*)));
        }
        else
        {
            m_pActionGotoDefinition->setVisible(false);
            m_pActionGotoAssignment->setVisible(false);
            m_pActionGotoAssignmentExtended->setVisible(false);
            disconnect(editor(), SIGNAL(keyReleased(QKeyEvent*)), this, SLOT(onKeyReleasedGoto(QKeyEvent*)));
        }
    }
}

//----------------------------------------------------------
void PyGotoAssignmentMode::setMouseClickEnabled(bool enabled)
{
    if (enabled != m_mouseClickEnabled)
    {
        m_mouseClickEnabled = enabled;
        WordClickMode::onStateChanged(this->enabled() && m_mouseClickEnabled);
    }
}


//--------------------------------------------------------------
/*
Request a goto action for the word under the text cursor. (goto definition)
*/
void PyGotoAssignmentMode::requestGotoDefinition()
{
    m_gotoRequested = true;
    checkWordCursorWithMode(editor()->wordUnderMouseCursor(), 0);
}

//--------------------------------------------------------------
/*
Request a goto action for the word under the text cursor. (goto assignment - do not follow imports)
*/
void PyGotoAssignmentMode::requestGotoAssignment()
{
    m_gotoRequested = true;
    checkWordCursorWithMode(editor()->wordUnderMouseCursor(), 1);
}

//--------------------------------------------------------------
/*
Request a goto action for the word under the text cursor. (goto assignment and follow imports)
*/
void PyGotoAssignmentMode::requestGotoAssignmentEx()
{
    m_gotoRequested = true;
    checkWordCursorWithMode(editor()->wordUnderMouseCursor(), 2);
}

//--------------------------------------------------------------
/*
*/
void PyGotoAssignmentMode::onJediAssignmentResultsAvailable(QVector<ito::JediAssignment> assignments)
{
    if (m_gotoRequestedTimerId > 0)
    {
        QApplication::restoreOverrideCursor();
        killTimer(m_gotoRequestedTimerId);
        m_gotoRequestedTimerId = 0;
    }

    foreach (const ito::JediAssignment &d, assignments)
    {
        m_assignments.append(PyAssignment(d.m_path, d.m_line, d.m_column, d.m_fullName));
    }

    m_assignments = unique(m_assignments);
    if (m_assignments.size() >= 1)
    {
        if (m_gotoRequested)
        {
            performGoto(m_assignments);
        }
        else
        {
            selectWordCursor();
            editor()->setMouseCursor(Qt::PointingHandCursor);
        }
    }
    else
    {
        if (m_gotoRequested)
        {
            QMessageBox::information(editor(), tr("No definition found"), tr("No definition could be found."));
        }

        clearSelection();
        editor()->setMouseCursor(Qt::IBeamCursor);
    }

    m_gotoRequested = false;
}

//--------------------------------------------------------------
/*
*/
void PyGotoAssignmentMode::doGoto(const PyAssignment &assignment)
{
    ScriptEditorWidget *sew = qobject_cast<ScriptEditorWidget*>(editor());
    QString filename;
    if (sew)
    {
        filename = sew->getFilename();
    }

    if (!filename.isNull())
    {
        filename.replace(".pyc", ".py");
    }

    if (assignment.m_modulePath == "" || assignment.m_modulePath == filename) //module path is empty if this script currently has no filename
    {
        int line = assignment.m_line;
        int col = assignment.m_column;

        editor()->gotoLine(line, col, true);
        //_logger().debug("Go to %s" % assignment)
    }
    else
    {
        //_logger().debug("Out of doc: %s" % assignment)
        emit outOfDoc(assignment);
    }
}

//--------------------------------------------------------------
/*
*/
void PyGotoAssignmentMode::onKeyReleasedGoto(QKeyEvent *e)
{
    if (e->key() == Qt::Key_F12)
    {
        QTextCursor cursor = editor()->textCursor();

        if (!cursor.isNull())
        {
            m_gotoRequested = true;
            checkWordCursorWithMode(cursor, 0);
        }
    }
}

//--------------------------------------------------------------
/*
Request a go to assignment.

    :param tc: Text cursor which contains the text that we must look for
                its assignment. Can be None to go to the text that is under
                the text cursor.
    :type tc: QtGui.QTextCursor
*/
void PyGotoAssignmentMode::checkWordCursor(const QTextCursor &cursor)
{
    checkWordCursorWithMode(cursor, m_defaultMode);
}

//--------------------------------------------------------------
/*
Request a go to assignment.

    :param tc: Text cursor which contains the text that we must look for
                its assignment. Can be None to go to the text that is under
                the text cursor.
    :type tc: QtGui.QTextCursor
    :param mode: 0: goto definition, 1: goto assignment (no follow imports), 2: goto assignment (follow imports)
*/
void PyGotoAssignmentMode::checkWordCursorWithMode(const QTextCursor &cursor, int mode)
{
    QTextCursor tc = cursor;

    if (tc.isNull())
    {
        tc = editor()->wordUnderCursor(false);
    }

    PythonEngine *pyEng = qobject_cast<PythonEngine*>(m_pPythonEngine);

    if (pyEng)
    {
        ScriptEditorWidget *sew = qobject_cast<ScriptEditorWidget*>(editor());
        QString filename;

        if (sew)
        {
            filename = sew->getFilename();
        }

        if (filename == "")
        {
            filename = QDir::cleanPath(QDir::current().absoluteFilePath("__temporaryfile__.py"));
        }

        if (pyEng->tryToLoadJediIfNotYetDone())
        {
            if (m_gotoRequestedTimerId > 0)
            {
                QApplication::restoreOverrideCursor();
                killTimer(m_gotoRequestedTimerId);
            }

            QApplication::setOverrideCursor(Qt::WaitCursor);

            m_gotoRequestedTimerId = startTimer(gotoRequestedTimeoutMs);

            //store current position as go back navigation point before jumping to another position
            editor()->reportPositionAsGoBackNavigationItem(tc, "goto");

            JediAssignmentRequest request;
            request.m_sender = this;
            request.m_callbackFctName = "onJediAssignmentResultsAvailable";
            request.m_col = tc.columnNumber();
            request.m_line = tc.blockNumber();
            request.m_mode = mode;
            request.m_path = filename;
            request.m_source = editor()->toPlainText();

            pyEng->enqueueGoToAssignmentRequest(request);
        }
        else
        {
            onStateChanged(false);
        }
    }
}

//--------------------------------------------------------------
/*
*/
void PyGotoAssignmentMode::timerEvent(QTimerEvent *event)
{
    qDebug() << "Timeout to obtain a goto assignment result from python. Probably python is busy.";
    QApplication::restoreOverrideCursor();

    if (m_gotoRequestedTimerId > 0)
    {
        killTimer(m_gotoRequestedTimerId);
        m_gotoRequestedTimerId = 0;
    }
}

//--------------------------------------------------------------
/*
*/
QList<PyAssignment> PyGotoAssignmentMode::unique(const QList<PyAssignment> &assignments) const
{
    // order preserving
    QList<PyAssignment> checked;
    bool present;

    foreach (const PyAssignment &a, assignments)
    {
        present = false;
        foreach (const PyAssignment &c, checked)
        {
            if (c == a)
            {
                present = true;
                break;
            }
        }

        if (!present)
        {
            checked.append(a);
        }
    }

    return checked;
}

//--------------------------------------------------------------
/*
*/
void PyGotoAssignmentMode::clearSelection()
{
    WordClickMode::clearSelection();
    m_assignments.clear();
}

//--------------------------------------------------------------
/*
*/
void PyGotoAssignmentMode::performGoto(const QList<PyAssignment> &assignments)
{
    if (assignments.size() == 0)
    {
        QMessageBox::information(editor(), tr("No definition found"), tr("No definition could be found."));
        return;
    }

    QList<PyAssignment> assignmentsValid;

    foreach(const PyAssignment& a, assignments)
    {
        if (a.m_line >= 0 && \
            a.m_fullName != "" && \
            !a.m_modulePath.endsWith(".pyi", Qt::CaseInsensitive))
        {
            assignmentsValid.append(a);
        }
    }

    switch (assignmentsValid.size())
    {
    case 0:
        QMessageBox::information(editor(), tr("Source of definition not reachable"), tr("The source of this definition is not reachable."));
        break;
    case 1:
        doGoto(assignmentsValid[0]);
        break;
    default:
    {
        /*_logger().debug(
        "More than 1 assignments in different modules, user "
        "need to make a choice: %s" % assignments)*/
        QStringList items;
        foreach(const PyAssignment &a, assignmentsValid)
        {
            QString fullname = a.m_fullName;

            if (fullname.size() > 30)
            {
                fullname = QString("...%1").arg(fullname.right(27));
            }

            QString path = a.m_modulePath;
            int lastIndex = std::max(path.lastIndexOf("/"), path.lastIndexOf("\\"));

            if (lastIndex >= 0)
            {
                path = path.mid(lastIndex + 1);
            }
            else if (path.size() > 30)
            {
                path = QString("...%1").arg(path.right(27));
            }

            if (fullname != "")
            {
                if (a.m_line >= 0 && a.m_column >= 0)
                {
                    items << QString("%1: %2 (line %3, column %4)").arg(path).arg(fullname).arg(a.m_line + 1).arg(a.m_column);
                }
                else if (a.m_line >= 0)
                {
                    items << QString("%1: %2 (line %3)").arg(path).arg(fullname).arg(a.m_line + 1);
                }
                else
                {
                    items << QString("%1: %2").arg(path).arg(fullname);
                }
            }
            else
            {
                if (a.m_line >= 0 && a.m_column >= 0)
                {
                    items << QString("%1: line %2, column %3").arg(path).arg(a.m_line + 1).arg(a.m_column);
                }
                else if (a.m_line >= 0)
                {
                    items << QString("%1: line %2").arg(path).arg(a.m_line + 1);
                }
                else
                {
                    items << a.m_modulePath;
                }
            }
        }

        float guiFactor = GuiHelper::screenDpiFactor();

        QSharedPointer<QInputDialog> dialog(new QInputDialog(editor(), QFlag(0)));
        dialog->setWindowTitle(tr("Choose a definition"));
        dialog->setLabelText(tr("Choose the definition you want to go to:"));
        dialog->setComboBoxItems(items);
        dialog->setTextValue(items[0]);
        dialog->setComboBoxEditable(false);
        dialog->setInputMethodHints(Qt::ImhNone);
        dialog->resize(500 * guiFactor, 250 * guiFactor);
        dialog->setOption(QInputDialog::UseListViewForComboBoxItems);
        const int ret = dialog->exec();
        bool ok = !!ret;
        if (ok && ret)
        {
            int idx = items.indexOf(dialog->textValue());
            doGoto(assignmentsValid[idx]);
        }
    }
    break;
    }
}

} //end namespace ito
