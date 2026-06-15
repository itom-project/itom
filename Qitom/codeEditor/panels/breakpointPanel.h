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

#ifndef BREAKPOINTPANEL_H
#define BREAKPOINTPANEL_H

/*
Breakpoint panels
*/

#include "../panel.h"
#include "../utils/utils.h"
#include "../textBlockUserData.h"

#include <qevent.h>
#include <qsize.h>
#include <qcolor.h>
#include <qicon.h>
#include <qmap.h>

class QMenu;
class QAction;

namespace ito {

/*
Shows messages collected by one or more checker modes
*/
class BreakpointPanel : public Panel
{
    Q_OBJECT
public:
    BreakpointPanel(const QString &description = "", QWidget *parent = NULL);
    virtual ~BreakpointPanel();

    virtual QSize sizeHint() const;

    void setCurrentLine(int line); //!< line = -1 removes the current line icon
    void setSelectedCallstackLine(int line); //! adds a green arrow if another than the top line of the callstack (during debug) is selected, line = -1 removes the icon again.
    void removeAllLineSelectors(); //! this is equal than setCurrentLine(-1); setSelectedCallstackLine(-1);

protected:
    virtual void paintEvent(QPaintEvent *e);
    virtual void mouseReleaseEvent(QMouseEvent *e);
    virtual void contextMenuEvent (QContextMenuEvent *e);

protected:

private:
    QMap<TextBlockUserData::BreakpointType, QIcon> m_icons;
    QIcon m_currentLineIcon;
    int m_currentLine;
    QIcon m_selectedCallstackLineIcon;
    int m_selectedCallstackLine;

    QMenu *m_pContextMenu;
    QMap<QString, QAction*> m_contextMenuActions;
    int m_contextMenuLine;

signals:
    void toggleBreakpointRequested(int);
    void toggleEnableBreakpointRequested(int);
    void editBreakpointRequested(int);
    void clearAllBreakpointsRequested();
    void gotoNextBreakPointRequested();
    void gotoPreviousBreakRequested();

private slots:
    void menuToggleBreakpoint();
    void menuToggleEnableBreakpoint();
    void menuEditBreakpoint();
    void menuGotoNextBreakPoint();
    void menuGotoPreviousBreakPoint();
    void menuClearAllBreakpoints();
};

} //end namespace ito

#endif
