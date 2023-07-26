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

#ifndef PYGOTOASSIGNMENT_H
#define PYGOTOASSIGNMENT_H

/*
Contains the go to assignments mode.
*/

#include "wordclick.h"

#include "../../python/pythonJedi.h"

namespace ito {

//-----------------------------------------------------------
/*
Defines an assignment. Used by :class:`PyGotoAssignmentsMode`.
*/
struct PyAssignment
{
    PyAssignment() :
        m_line(-1),
        m_column(-1)
    {
    }

    PyAssignment(const QString &path, int line, int column, const QString &fullName) :
        m_line(line),
        m_column(column),
        m_fullName(fullName)
    {
        m_modulePath = path;
        m_modulePath.replace(".pyc", ".py");
    }

    bool operator==(const PyAssignment &rhs) const
    {
        return (m_modulePath == rhs.m_modulePath) && \
            (m_line == rhs.m_line) && \
            (m_column == rhs.m_column) && \
            (m_fullName == rhs.m_fullName);
    }

    QString m_modulePath; // File path of the module where the assignment can be found
    int m_line; //line number
    int m_column; //column number
    QString m_fullName; //assignement full name
};

/*
Goes to the assignments (using jedi.Script.goto_assignments) when the user
execute the shortcut or click word. If there are more than one assignments,
an input dialog is used to ask the user to choose the desired assignment.

This mode will emit the :attr:`out_of_doc` signal if the definition can
not be reached in the current document. IDE will typically connects a slot
that open a new editor tab and goes to the definition position.
*/
class PyGotoAssignmentMode : public WordClickMode
{
    Q_OBJECT
public:
    PyGotoAssignmentMode(const QString &description = "", QObject *parent = NULL);
    virtual ~PyGotoAssignmentMode();

    virtual void onStateChanged(bool state);

    void setDefaultWordClickMode(int mode) { m_defaultMode = mode; }
    int defaultWordClickMode() const { return m_defaultMode; }

    bool mouseClickEnabled() const { return m_mouseClickEnabled; }
    void setMouseClickEnabled(bool enabled);

    virtual QList<QAction*> actions() const;

protected:
    void doGoto(const PyAssignment &assignment);
    virtual void checkWordCursor(const QTextCursor &cursor);
    void checkWordCursorWithMode(const QTextCursor &cursor, int mode);
    QList<PyAssignment> unique(const QList<PyAssignment> &assignments) const;
    virtual void clearSelection();
    void performGoto(const QList<PyAssignment> &assignments);

    void timerEvent(QTimerEvent *event) override;

private:
    QObject *m_pPythonEngine;
    bool m_gotoRequested;
    int m_gotoRequestedTimerId;
    static const int gotoRequestedTimeoutMs = 3000;

    QList<PyAssignment> m_assignments;
    QAction *m_pActionGotoDefinition;
    QAction *m_pActionGotoAssignment;
    QAction *m_pActionGotoAssignmentExtended;
    int m_defaultMode; //mode, if word is clicked: 0: goto definition, 1: goto assignment (no follow imports), 2: goto assignment (follow imports)
    bool m_mouseClickEnabled;

private slots:
    void requestGotoDefinition();
    void requestGotoAssignment();
    void requestGotoAssignmentEx();
    void onJediAssignmentResultsAvailable(QVector<ito::JediAssignment> assignments);
    void onWordClicked(const QTextCursor &cursor) { performGoto(m_assignments); }
    void onKeyReleasedGoto(QKeyEvent *e);


signals:
    void outOfDoc(PyAssignment assignment); //Signal emitted when the definition cannot be reached in the current document
    void noResultsFound(); //Signal emitted when no results could be found.


};

} //end namespace ito

#endif
