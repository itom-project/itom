/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#include "python/pythonEngine.h"

#include <qinputdialog.h>

namespace ito {

//----------------------------------------------------------
/*
*/
PyGotoAssignmentMode::PyGotoAssignmentMode(const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    WordClickMode("PyGotoAssignment", description, parent),
    m_gotoRequested(false),
    m_pPythonEngine(NULL),
    m_pActionGoto(NULL)
{
    qRegisterMetaType<PyAssignment>("PyAssignment");

    connect(this, SIGNAL(wordClicked(QTextCursor)), this, SLOT(onWordClicked(QTextCursor)));

    m_pPythonEngine = AppManagement::getPythonEngine();
    if (m_pPythonEngine)
    {
        connect(this, SIGNAL(jediDefinitionRequested(QString,int,int,QString,QByteArray)), m_pPythonEngine, SLOT(jediDefinitionRequested(QString,int,int,QString,QByteArray)));
    }

    m_pActionGoto = new QAction(tr("Go To Definition"), this);
    connect(m_pActionGoto, SIGNAL(triggered()), this, SLOT(requestGoto()));
}

//----------------------------------------------------------
/*
*/
PyGotoAssignmentMode::~PyGotoAssignmentMode()
{
    m_pActionGoto->deleteLater();
}


//----------------------------------------------------------
/*
*/
/*virtual*/ void PyGotoAssignmentMode::onStateChanged(bool state)
{
    if (m_pPythonEngine)
    {
        WordClickMode::onStateChanged(state);
        if (state)
        {
            editor()->addContextAction(m_pActionGoto, name());
            m_pActionGoto->setVisible(true);
        }
        else
        {
            m_pActionGoto->setVisible(false);
        }
    }
}


//--------------------------------------------------------------
/*
Request a goto action for the word under the text cursor.
*/
void PyGotoAssignmentMode::requestGoto()
{
    m_gotoRequested = true;
    checkWordCursor(QTextCursor());
}

//--------------------------------------------------------------
/*
*/
void PyGotoAssignmentMode::onJediDefinitionResultsAvailable(QVector<ito::JediDefinition> definitions)
{
    //_logger().debug("Got %r" % definitions)
    foreach (const ito::JediDefinition &d, definitions)
    {
        m_definitions.append(PyAssignment(d.m_path, d.m_line, d.m_column, d.m_fullName));
    }

    m_definitions = unique(m_definitions);
    if (validateDefinitions(m_definitions))
    {
        if (m_gotoRequested)
        {
            performGoto(m_definitions);
        }
        else
        {
            selectWordCursor();
            editor()->setMouseCursor(Qt::PointingHandCursor);
        }
    }
    else
    {
        clearSelection();
        editor()->setMouseCursor(Qt::IBeamCursor);
    }

    m_gotoRequested = false;
}

//--------------------------------------------------------------
/*
*/
void PyGotoAssignmentMode::doGoto(const PyAssignment &definition)
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

    if (definition.m_modulePath == "" || definition.m_modulePath == filename) //module path is empty if this script currently has no filename
    {
        int line = definition.m_line;
        int col = definition.m_column;
        editor()->gotoLine(line, col, true);
        //_logger().debug("Go to %s" % definition)
    }
    else
    {
        //_logger().debug("Out of doc: %s" % definition)
        emit outOfDoc(definition);
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
    QTextCursor tc = cursor;
    if (tc.isNull())
    {
        tc = editor()->wordUnderCursor(false);
    }

    ScriptEditorWidget *sew = qobject_cast<ScriptEditorWidget*>(editor());
    QString filename;
    if (sew)
    {
        filename = sew->getFilename();
    }

    PythonEngine *pyEng = (PythonEngine*)m_pPythonEngine;
    if (pyEng)
    {
        if (pyEng->tryToLoadJediIfNotYetDone())
        {
            emit jediDefinitionRequested(editor()->toPlainText(), tc.blockNumber(), tc.columnNumber(), filename, "onJediDefinitionResultsAvailable");
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
QList<PyAssignment> PyGotoAssignmentMode::unique(const QList<PyAssignment> &definitions) const
{
    // order preserving
    QList<PyAssignment> checked;
    bool present;

    foreach (const PyAssignment &a, definitions)
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
    m_definitions.clear();
}

//--------------------------------------------------------------
/*
*/
bool PyGotoAssignmentMode::validateDefinitions(const QList<PyAssignment> &definitions) const
{
    switch (definitions.size())
    {
    case 0:
        return false;
    case 1:
        return definitions[0].m_line >= 0;
    default:
        return true;
    }
}

//--------------------------------------------------------------
/*
*/
void PyGotoAssignmentMode::performGoto(const QList<PyAssignment> &definitions)
{
    if (definitions.size() == 1)
    {
        doGoto(definitions[0]);
    }
    else if (definitions.size() > 1)
    {
        /*_logger().debug(
            "More than 1 assignments in different modules, user "
            "need to make a choice: %s" % definitions)*/
        QStringList items;
        foreach (const PyAssignment &a, definitions)
        {
            if (a.m_line >= 0 && a.m_column >= 0)
            {
                items << QString("%1 (line %2, column %3)").arg(a.m_fullName).arg(a.m_line+1).arg(a.m_column);
            }
            else
            {
                items << a.m_fullName;
            }
        }

        QString result = QInputDialog::getItem(editor(), tr("Choose a definition"), tr("Choose the definition you want to go to:"), items, 0, false);
        
        if (result.isNull() == false)
        {
            int idx = items.indexOf(result);
            doGoto(definitions[idx]);
        }
    }
}


} //end namespace ito