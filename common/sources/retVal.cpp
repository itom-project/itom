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

#include "../retVal.h"

namespace ito {


//----------------------------------------------------------------------------------------------------------------------------------
RetVal::RetVal(ito::tRetValue retValue, int retCode, const char *pRetMessage)
    : m_retValue(retValue),
    m_retCode(retCode),
    m_retMessage(pRetMessage)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal &RetVal::operator=(const RetVal &rhs)
{
    m_retValue = rhs.m_retValue;
    m_retCode = rhs.m_retCode;
    m_retMessage = rhs.m_retMessage;
    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal & RetVal::operator += (const RetVal &rhs)
{
    if (rhs.m_retValue > m_retValue) //rhs is more severe
    {
        m_retCode = rhs.m_retCode;
        m_retMessage = rhs.m_retMessage;
        m_retValue = rhs.m_retValue;
    }
    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
void RetVal::appendRetMessage(const char *addRetMessage)
{
    m_retMessage.append(addRetMessage);
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ RetVal RetVal::format(ito::tRetValue retValue, int retCode, const char *pRetMessage, ...)
{
    if (pRetMessage != NULL)
    {
        va_list args;
        va_start (args, pRetMessage);
        char buffer[2048];
        int len = 0;
        len = vsprintf_s(buffer, 2048, pRetMessage, args);
        va_end(args);
        if (len < 0)
        {
            return RetVal(retValue, retCode, pRetMessage);
        }
        buffer[len] = '\0';
        return  RetVal(retValue, retCode, buffer);
    }
    else
    {
        return RetVal(retValue, retCode, NULL);
    }
}


} //end namespace ito
