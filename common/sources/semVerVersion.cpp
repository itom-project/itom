/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "../semVerVersion.h"

#include "sharedStructures.h"

#include <qstringlist.h>

namespace ito
{
/*!
\class SemVerVersion
\brief  provides version string parsing and comparison based on semantic versioning
*/

//-----------------------------------------------------------------------------------
SemVerVersion::SemVerVersion(int major, int minor, int patch /*= 0*/) :
    m_major(major),
    m_minor(minor),
    m_patch(patch)
{
}

//-----------------------------------------------------------------------------------
SemVerVersion::SemVerVersion() :
    m_major(0),
    m_minor(0),
    m_patch(0)
{
}

//-----------------------------------------------------------------------------------
QString SemVerVersion::toString() const
{
    return QString("%1.%2.%3").arg(m_major).arg(m_minor).arg(m_patch);
}

//-----------------------------------------------------------------------------------
int SemVerVersion::toInt() const
{
    return CREATEVERSION(m_major, m_minor, m_patch);
}

//-----------------------------------------------------------------------------------
/*static*/ SemVerVersion SemVerVersion::fromString(const QString &versionString, bool *ok /*= NULL*/)
{
    QStringList items = versionString.split(".");

    int major = 0;
    int minor = 0;
    int patch = 0;
    bool ok2 = true;

    switch (items.size())
    {
    case 3:
    case 4:
        patch = items[2].toInt(&ok2);
    case 2:
        if (ok2)
            minor = items[1].toInt(&ok2);
    case 1:
        if (ok2)
            major = items[0].toInt(&ok2);
        break;
    default:
        ok2 = false;
    }

    if (ok)
    {
        *ok = ok2;
    }

    if (ok2)
    {
        return SemVerVersion(major, minor, patch);
    }
    else
    {
        return SemVerVersion();
    }
}

//-----------------------------------------------------------------------------------
/*static*/ SemVerVersion SemVerVersion::fromInt(int versionNumber)
{
    int major = MAJORVERSION(versionNumber);
    int minor = MINORVERSION(versionNumber);
    int patch = PATCHVERSION(versionNumber);

    return SemVerVersion(major, minor, patch);
}

//-----------------------------------------------------------------------------------
bool SemVerVersion::operator==(const SemVerVersion &other) const //returns true if this and other are exactly the same
{
    return (m_major == other.m_major) && \
        (m_minor == other.m_minor) && \
        (m_patch == other.m_patch);
}

//-----------------------------------------------------------------------------------
bool SemVerVersion::operator>=(const SemVerVersion &other) const
{
    return toInt() >= other.toInt();
}

//-----------------------------------------------------------------------------------
bool SemVerVersion::operator>(const SemVerVersion &other) const
{
    return toInt() > other.toInt();
}

//-----------------------------------------------------------------------------------
bool SemVerVersion::operator<=(const SemVerVersion &other) const
{
    return toInt() <= other.toInt();
}

//-----------------------------------------------------------------------------------
bool SemVerVersion::operator<(const SemVerVersion &other) const
{
    return toInt() < other.toInt();
}

//-----------------------------------------------------------------------------------
bool SemVerVersion::isCompatible(const SemVerVersion &other) const //returns true if this and other have the same major and if the minor of other is <= this.minor.
{
    return (m_major == other.m_major) && \
        (m_minor >= other.m_minor);
}

//-----------------------------------------------------------------------------------
int SemVerVersion::svMajor() const
{
    return m_major;
}

//-----------------------------------------------------------------------------------
int SemVerVersion::svMinor() const
{
    return m_minor;
}

//-----------------------------------------------------------------------------------
int SemVerVersion::svPatch() const
{
    return m_patch;
}

//-----------------------------------------------------------------------------------
bool SemVerVersion::isValid() const
{
    return m_major != 0 || m_minor != 0 || m_patch != 0;
}

} //end namespace ito
