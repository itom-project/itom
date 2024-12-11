/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2024, Institut für Technische Optik (ITO),
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
This module contains the error line highlighter mode
*/

#include "../textDecoration.h"
#include "../mode.h"
#include "models/outlineItem.h"

#include <qcolor.h>
#include <qlist.h>
#include <qevent.h>

namespace ito {

class ScriptEditorWidget;

/*
Handles code cells
*/
class CodeCellHighlighterMode : public QObject, public Mode
{
    Q_OBJECT
public:
    CodeCellHighlighterMode(const QString &description = "", QObject *parent = NULL);
    virtual ~CodeCellHighlighterMode();

    virtual void onInstall(CodeEditor *editor);
    virtual void onStateChanged(bool state);

public slots:
    void outlineModelChanged(ScriptEditorWidget* sew, QSharedPointer<OutlineItem> rootItem);
    void updateActiveCodeCell();

protected:
    QColor m_headlineBgColor;
    QColor m_activeCellBgColor;
    QList<TextDecoration::Ptr> m_codeCellHeadlineDecorators;
    TextDecoration::Ptr m_activeCodeCellDecorator;
    QSharedPointer<OutlineItem> m_rootOutline;

    void clearAllDecorators(bool removeActiveCodeCell, bool removeCodeCellHeadings);

};

} //end namespace ito
