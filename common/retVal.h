/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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

#ifndef RETVAL_H
#define RETVAL_H

#ifdef __APPLE__
extern "C++" {
#endif

/* includes */
#include "commonGlobal.h"
#include "typeDefs.h"
#include "byteArray.h"

#include <stdarg.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
/** @class RetVal
*   @brief  Class for error value management
*
*   The RetVal class is used for handling return codes. All classes should use this class.
*   In case an error occurs, only the first error is stored and will not be overridden
*   by potentially subsequent occurring errors.
*/
class ITOMCOMMON_EXPORT RetVal
{       
    private:
        tRetValue m_retValue;    /*!< can be one of enumeration \ref tLogLevel values or an or-combination of these values*/
        int m_retCode;           /*!< the error code itself */
        ByteArray m_retMessage;

    public:
        inline RetVal() : m_retValue(ito::retOk), m_retCode(0) {}
        
        RetVal(tRetValue retValue) : m_retValue(retValue), m_retCode(0) {}
        
        RetVal(int retValue) : m_retValue((tRetValue)retValue), m_retCode(0) {}
        
        //RetVal(tRetValue retValue, int retCode, char *pRetMessage)
        /**
        *   constructor with retValue, retCode and errorMessage
        *   @param [in]  retValue     type of RetVal; for possible values see \ref tRetValue
        *   @param [in]  retCode      user definable return code
        *   @param [in]  pRetMessage  error message to be passed or NULL, string is copied
        *   Makes a deep copy of RetVal, i.e. a copy of the error message
        */
        RetVal(ito::tRetValue retValue, int retCode, const char *pRetMessage);
        
        inline ~RetVal() {}

        //RetVal & operator = (const RetVal &rhs);
        /**
        *   assignment operator, copies values of rhs to current RetVal. Before copiing current errorMessage is freed
        */
        RetVal &operator=(const RetVal &rhs);


        //----------------------------------------------------------------------------------------------------------------------------------
        /**
        *   Concatenation of RetVal
        *   "Adds" RetVals, i.e. returns the most serious error. In case of
        *   equally serious errors the first is retained
        */
        RetVal & operator += (const RetVal &rhs);

        //----------------------------------------------------------------------------------------------------------------------------------
        /**
        *   Concatenation of RetVal
        *   See operator RetVal::operator+=
        */
        RetVal operator + (const RetVal &rhs)
        {
            return (*this += rhs);
        }

        //----------------------------------------------------------------------------------------------------------------------------------
        /**
        *   equality operator compares retValue with with retValue of rhs RetVal. For possible constant values see \ref tRetValue
        */
        inline char operator == (const RetVal &rhs)
        {
            return m_retValue == rhs.m_retValue;
        }

        //----------------------------------------------------------------------------------------------------------------------------------
        /**
        *   unequality operator compares retValue with with retValue of rhs RetVal. For possible constant values see \ref tRetValue
        */
        inline char operator != (const RetVal &rhs)
        {
            return !(m_retValue == rhs.m_retValue);
        }

        //----------------------------------------------------------------------------------------------------------------------------------
        /**
        *   equality operator compares retValue with tRetValue constant. For possible constant values see \ref tRetValue
        */
        inline char operator == (const tRetValue rhs)
        {
            return m_retValue == rhs;
        }

        //----------------------------------------------------------------------------------------------------------------------------------
        /**
        *   unequality operator compares retValue with tRetValue constant. For possible constant values see \ref tRetValue
        */
        inline char operator != (const tRetValue rhs)
        {
            return !(m_retValue == rhs);
        }

        //----------------------------------------------------------------------------------------------------------------------------------
        void appendRetMessage(const char *addRetMessage);

        //----------------------------------------------------------------------------------------------------------------------------------
        inline int containsWarning() { return (m_retValue & retWarning); }              /*!< checks if any warning has occurred in this return value (true), else (false) */
        inline int containsError() { return (m_retValue & retError); }                 /*!< checks if any error has occurred in this return value (true), else (false) */
        inline int containsWarningOrError() { return (m_retValue & (retError | retWarning)); }  /*!< checks if any warning or error has occurred in this return value (true), else (false) */

        inline const char *errorMessage() { return m_retMessage.data(); }
        inline int errorCode() const { return m_retCode; }

        //----------------------------------------------------------------------------------------------------------------------------------
        static RetVal format(ito::tRetValue retValue, int retCode, const char *pRetMessage, ...);

};


} //end namespace ito

#ifdef __APPLE__
}
#endif

#endif
