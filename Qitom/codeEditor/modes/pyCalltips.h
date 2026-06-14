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

#ifndef PYCALLTIPS_H
#define PYCALLTIPS_H


#include "../../python/pythonJedi.h"

#include "../utils/utils.h"
#include "../toolTip.h"
#include "../mode.h"
#include <qevent.h>
#include <qobject.h>
#include <qpair.h>
#include <qstring.h>
#include <qlist.h>

namespace ito {

/*
Contains the JediCompletionProvider class implementation.
*/


/*
Shows function calltips.

This mode shows function/method call tips in a QToolTip using
:meth:`jedi.Script.call_signatures`.
*/
class PyCalltipsMode : public QObject, public Mode
{
    Q_OBJECT
public:
    PyCalltipsMode(const QString &name, const QString &description = "", QObject *parent = NULL);
    virtual ~PyCalltipsMode();

    virtual void onStateChanged(bool state);

private slots:
    void onKeyReleased(QKeyEvent *e);
    void onJediCalltipResultAvailable(QVector<ito::JediCalltip> calltips);

signals:
    //void jediCalltipRequested(const QString &source, int line, int col, const QString &path, const QString &encoding, QByteArray callbackFctName);

protected:
    void requestCalltip(const QString &source, int line, int col, const QString &encoding);
    bool isLastChardEndOfWord() const;

private:
    QObject *m_pPythonEngine;
    QList<int> m_disablingKeys;
    int m_requestCount;
};

} //end namespace ito

#endif
