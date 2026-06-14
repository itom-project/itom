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

#ifndef AUTOINDENT_H
#define AUTOINDENT_H

#include "../utils/utils.h"
#include "../mode.h"
#include <qevent.h>
#include <qobject.h>
#include <qpair.h>
#include <qstring.h>

namespace ito {

/*
Contains the automatic generic indenter
*/


/*
Indents text automatically.
Generic indenter mode that indents the text when the user press RETURN.

You can customize this mode by overriding
:meth:`pyqode.core.modes.AutoIndentMode._get_indent`

This mode contains two features:

1. The auto indentation itself
2. It is possible to remove trailing whitespaces and tabs
   from the current line before the newline character is
   applied. This improves the accordance to Pep8, which
   does not allow empty lines, that contains only spaces.

To handle these two features, always enable the mode and
control the enable/disable state of the two features using
the specific setters enableAutoIndent,
setAutoStripTrailingSpacesAfterReturn.
*/
class AutoIndentMode : public QObject, public Mode
{
    Q_OBJECT
public:
    AutoIndentMode(const QString &name, const QString &description = "", QObject *parent = NULL);
    virtual ~AutoIndentMode();

    virtual void onStateChanged(bool state);

    void setKeyPressedModifiers(Qt::KeyboardModifiers modifiers);
    Qt::KeyboardModifiers keyPressedModifiers() const;

    void setAutoStripTrailingSpacesAfterReturn(bool strip);
    bool autoStripTrailingSpacesAfterReturn() const;

    void enableAutoIndent(bool autoIndent);
    bool isAutoIndentEnabled() const;

private slots:
    void onKeyPressed(QKeyEvent *e);

protected:
    QChar indentChar() const;
    QString singleIndent() const;
    virtual QPair<QString, QString> getIndent(const QTextCursor &cursor) const;

private:
    Qt::KeyboardModifiers m_keyPressedModifiers;
    bool m_autoStripTrailingSpacesAfterReturn;
    bool m_enableAutoIndent;

};

} //end namespace ito

#endif
