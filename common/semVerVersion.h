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

#ifndef SEMVERVERSION_H
#define SEMVERVERSION_H

/* includes */

#include "commonGlobal.h"
#include "typeDefs.h"
#include <qstring.h>


#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{
    class ITOMCOMMONQT_EXPORT SemVerVersion
    {
    public:
        SemVerVersion(int major, int minor, int patch = 0);
        SemVerVersion(); //constructs an empty version (major = minor = patch = 0)

        QString toString() const;
        int toInt() const; //returns the major, minor and patch in the form 0xAABBCC (AA: major, BB: minor, CC: patch)

        static SemVerVersion fromString(const QString &versionString, bool *ok = NULL);
        static SemVerVersion fromInt(int versionNumber); //the integer must be 0xAABBCC where AA is the major, BB is the minor, CC is the patch

        bool operator==(const SemVerVersion &other) const; //returns true if this and other are exactly the same
        bool isCompatible(const SemVerVersion &other) const; //returns true if this and other have the same major and if the minor of other is <= this.minor.
        bool isValid() const; //returns false if major=minor==patch==0
        bool operator>=(const SemVerVersion &other) const;
        bool operator>(const SemVerVersion &other) const;
        bool operator<=(const SemVerVersion &other) const;
        bool operator<(const SemVerVersion &other) const;

        int svMajor() const;
        int svMinor() const;
        int svPatch() const;

    private:
        int m_major;
        int m_minor;
        int m_patch;
    };

};   // end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif
