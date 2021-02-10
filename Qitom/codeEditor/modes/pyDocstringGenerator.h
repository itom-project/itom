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

#ifndef PYDOCSTRINGGENERATOR_H
#define PYDOCSTRINGGENERATOR_H

#include "../utils/utils.h"
#include "../mode.h"
#include "../../models/outlineItem.h"
#include <qevent.h>
#include <qobject.h>
#include <qtextcursor.h>
#include <qstring.h>
#include <qsharedpointer.h>

namespace ito {

/*
Contains a generator for a template of a docstring of the current method or class.
*/

class PyDocstringGeneratorMode : public QObject, public Mode
{
    Q_OBJECT
public:
    PyDocstringGeneratorMode(const QString &name, const QString &description = "", QObject *parent = nullptr);
    virtual ~PyDocstringGeneratorMode();

    virtual void onStateChanged(bool state);

    void insertDocstring(const QTextCursor &cursor) const;
    QSharedPointer<OutlineItem> getOutlineOfLineIdx(int lineIdx) const;

protected:
    int lastLineIdxOfDefinition(const QSharedPointer<OutlineItem> &item) const;
    

private slots:
    void onKeyPressed(QKeyEvent *e) {};


};

} //end namespace ito

#endif
