/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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
*   @brief  Class for managing status values (like errors or warning)
*
*   The RetVal class is used for handling return codes. All classes should use this class.
*   In case an error occurs, only the first error is stored and will not be overridden
*   by potentially subsequent occurring errors. An error is more severe than a warning and will
*   overwrite a warning if it is added to an existing RetVal.
*/
class ITOMCOMMON_EXPORT RetVal
{       
    private:
        tRetValue m_retValue;    /*!< can be one of enumeration \ref tLogLevel values or an or-combination of these values*/
        int m_retCode;           /*!< the error code itself */
        ByteArray m_retMessage;  /*!< message of this RetVal using ByteArray (using implicit sharing) */

    public:
        //! default constructor that creates a RetVal with status ito::retOk, code 0 and no message.
        inline RetVal() : m_retValue(ito::retOk), m_retCode(0) {}
        
        //! default constructor that creates a RetVal with the status given by retValue, code 0 and no message.
        RetVal(tRetValue retValue) : m_retValue(retValue), m_retCode(0) {}
        
        //! default constructor that creates a RetVal with the status given by retValue, code 0 and no message.
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
        
        //! destructor
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
        inline int containsWarning() const { return (m_retValue & retWarning); }              /*!< check if any warning has occurred in this return value (true), else (false) */
        inline int containsError() const { return (m_retValue & retError); }                 /*!< check if any error has occurred in this return value (true), else (false) */
        inline int containsWarningOrError() const { return (m_retValue & (retError | retWarning)); }  /*!< check if any warning or error has occurred in this return value (true), else (false) */

        inline bool hasErrorMessage() const { return m_retMessage.size() > 0; } /*!< return true if an error or warning message is available */

        inline int errorCode() const { return m_retCode; } /*!< return the error code (default: 0) */

        inline const char *errorMessage() const { return m_retMessage.data(); } /*!< return zero-terminated error message or empty, zero-terminated string if no error message has been set */

        //----------------------------------------------------------------------------------------------------------------------------------
        //! create RetVal with message that may contain placeholders using the formalism of the default sprintf method.
        /*!
        \param retValue is the type of RetVal (retOk, retWarning, retError)
        \param retCode is the code number of RetVal
        \param pRetMessage is the zero-terminated message string that may contain placeholders like %s, %i, ... (see sprintf)
        \param ... are further arguments. Their number and type must correspond to the placeholders in pRetMessage
        \return created RetVal
        */
        static RetVal format(ito::tRetValue retValue, int retCode, const char *pRetMessage, ...);

};


} //end namespace ito

#ifdef __APPLE__
}
#endif

#endif
