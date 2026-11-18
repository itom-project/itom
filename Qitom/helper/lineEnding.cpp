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


#include "lineEnding.h"

namespace ito {

//-------------------------------------------------------------------------------------
/*static*/ LineEnding::Type LineEnding::platformDefault()
{
#ifdef Q_OS_WIN
    return CRLF;
#else
    return LF;
#endif
}

//-------------------------------------------------------------------------------------
/*static*/ QString LineEnding::shortName(Type type)
{
    return type == CRLF ? "CRLF" : "LF";
}

//-------------------------------------------------------------------------------------
/*static*/ LineEnding::Type LineEnding::detect(const QString& text)
{
    const int crlf = text.count("\r\n");
    const int lf = text.count('\n') - crlf;

    if (crlf == 0 && lf == 0)
    {
        return platformDefault();
    }

    return crlf > lf ? CRLF : LF;
}

//-------------------------------------------------------------------------------------
/*static*/ void LineEnding::normalize(QString& text)
{
    text.replace("\r\n", "\n");
}

//-------------------------------------------------------------------------------------
/*static*/ QString LineEnding::apply(const QString& text, Type type)
{
    if (type == CRLF)
    {
        QString t = text;
        t.replace("\n", "\r\n");
        return t;
    }

    return text;
}

} // end namespace ito
