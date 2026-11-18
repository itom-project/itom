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


#ifndef LINEENDING_H
#define LINEENDING_H

#include <qstring.h>

namespace ito {

//! Line endings of text files, that are opened in the script editor (QtCore only, unit tested).
class LineEnding
{
public:
    enum Type
    {
        LF, //!< "\n" (Unix)
        CRLF //!< "\r\n" (Windows)
    };

    //! the line ending of the platform (CRLF on Windows, LF else)
    static Type platformDefault();

    //! short name for the status bar: "LF" or "CRLF"
    static QString shortName(Type type);

    //! detects the line ending of a text (majority of "\r\n" vs. "\n", platform default if no newline)
    static Type detect(const QString& text);

    //! converts "\r\n" to "\n" (the editor only works with "\n")
    static void normalize(QString& text);

    //! returns the text with the given line ending (the text must only contain "\n")
    static QString apply(const QString& text, Type type);
};

} // end namespace ito

#endif
