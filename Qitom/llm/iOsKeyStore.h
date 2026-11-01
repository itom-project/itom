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

#pragma once

#include "../../common/retVal.h"

#include <QString>

namespace ito
{
namespace llm
{

class IOsKeyStore
{
    public:
        virtual ~IOsKeyStore() {}

        virtual RetVal writeSecret(const QString &key, const QString &value) = 0;
        virtual RetVal readSecret(const QString &key, QString &valueOut) = 0;
        virtual RetVal removeSecret(const QString &key) = 0;
};

} // namespace llm
} // namespace ito
