

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

#pragma once

/*
This module contains the WordHoverTooltipMode
*/

#include "../mode.h"
#include "../textDecoration.h"
#include "../delayJobRunner.h"
#include "../../python/pythonJedi.h"
#include <qobject.h>
#include <qstring.h>
#include <qtextcursor.h>
#include <qevent.h>
#include <qvector.h>
#include <qpair.h>

namespace ito {

/*
Adds support for tooltips when hovering words in a script.


*/
class WordHoverTooltipMode : public QObject, public Mode
{
    Q_OBJECT
public:
    WordHoverTooltipMode(const QString &name = "WordHoverTooltipMode", const QString &description = "", QObject *parent = nullptr);
    virtual ~WordHoverTooltipMode();

    virtual void onStateChanged(bool state);

    void hideTooltip();

protected:
    QTextCursor m_cursor;

    void emitWordHover(QTextCursor cursor);

    /*
    \returns (stringlist os signatures, docstring)
    */
    QPair<QStringList, QString> parseTooltipDocstring(const QString &docstring) const;

private:
    DelayJobRunnerBase *m_pTimer;
    QObject *m_pPythonEngine;
    int m_requestCount;
    int m_tooltipsMaxLength;
    bool m_tooltipVisible;

private slots:
    void onMouseMoved(QMouseEvent *e);
    void onJediGetHelpResultAvailable(QVector<ito::JediGetHelp> helps);

};

} //end namespace ito
