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

#ifndef SHAREDSTRUCTURES_H
#define SHAREDSTRUCTURES_H

/* includes */

#include "../common/typeDefs.h"
//#include "./helper/paramHelper.h"

//#include <string>
#include <string.h>
//#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <limits>
#include <stdexcept>

/* definition and macros */
/* global variables (avoid) */
/* content */

namespace ito
{
    #define CREATEVERSION(major,minor,patch)    (major << 16) + (minor << 8) + patch
    #define MAJORVERSION(version)               version >> 16
    #define MINORVERSION(version)               (version >> 8) - (MAJORVERSION(version) << 8)
    #define PATCHVERSION(version)               version - ((version >> 8) << 8)
    #define MAXVERSION                          CREATEVERSION(999999,0,0)    //maximum possible version (that means no maximum version is indicated)
    #define MINVERSION                          CREATEVERSION(0,0,0)         //minimum possible version

    #define ItomDoc_VAR(name) static char name[]
    #define ItomDoc_STRVAR(name,str) ItomDoc_VAR(name) = str


    //class tParam;
    class Param;
    class ParamHelper;
    class RetVal;
    template<typename _Tp> struct ItomParamHelper;
   // template<typename _Tp> void tParam_setVal(ito::tParam *param, double &dVal, int &iVal, char *&cVal, _Tp val);

    const uint32 paramFlagMask = 0xFFFF0000; //!< bits of type lying within this mask are flags (e.g. typeNoAutosave, typeReadonly...)
    const uint32 paramTypeMask = 0x0000FFFF; //!< bits of param type lying withing this mask describe the type (typeNoAutosave must be included there)


    //----------------------------------------------------------------------------------------------------------------------------------
    /** @class RetVal
    *   @brief  Class for error value management
    *
    *   The RetVal class is used for handling return codes. All classes should use this class.
    *   In case an error occurs, only the first error is stored and will not be overridden
    *   by potentially subsequent occurring errors.
    */
    class RetVal
    {
        public:
            inline RetVal(tRetValue retValue = retOk) : m_retValue(retValue), m_retCode(0), m_pRetMessage(NULL) {}
            inline RetVal(int retValue) : m_retValue((tRetValue)retValue), m_retCode(0), m_pRetMessage(NULL) {}
            //RetVal(tRetValue retValue = retOk); //Auskommentiert sonstwer

            //----------------------------------------------------------------------------------------------------------------------------------
            //RetVal(tRetValue retValue, int retCode, char *pRetMessage) // Auskommentiert Ly bei umstellen wegen dataObj 05.10.2011
            /**
            *   constructor with retValue, retCode and errorMessage
            *   @param [in]  retValue     return Value constant, for possible values see \ref tRetValue
            *   @param [in]  retCode      user definable return code
            *   @param [in]  pRetMessage  error message to be passed, string is copied
            *   Makes a deep copy of RetVal, i.e. a copy of the error message
            */
            RetVal(ito::tRetValue retValue, int retCode, const char *pRetMessage) : m_retValue(retValue), m_retCode(retCode), m_pRetMessage(NULL) // Copied from sharedStructure.cpp 05.10.2011
            {
                if (pRetMessage != NULL)
                {
                    int messageLen = (int)strlen(pRetMessage);
                    m_pRetMessage = new char [messageLen + 1];
                    memcpy(m_pRetMessage, pRetMessage, sizeof(*pRetMessage) * messageLen);
                    m_pRetMessage[messageLen] = '\0';
                }
            }

            //----------------------------------------------------------------------------------------------------------------------------------
            //RetVal(const RetVal& copyConstr); // Auskommentiert Ly bei umstellen wegen dataObj 05.10.2011
            /**
            *   Deep copy constructor
            *   Makes a deep copy of RetVal, i.e. a copy of the error message
            */
            RetVal(const RetVal& copyConstr) // Copied from sharedStructure.cpp 05.10.2011
            {
                m_retValue = copyConstr.m_retValue;
                m_retCode = copyConstr.m_retCode;

                if (copyConstr.m_pRetMessage != NULL)
                {
                    int messageLen = (int)strlen(copyConstr.m_pRetMessage);
                    m_pRetMessage = new char [messageLen + 1];
                    memcpy(m_pRetMessage, copyConstr.m_pRetMessage, sizeof(*copyConstr.m_pRetMessage) * messageLen);
                    m_pRetMessage[messageLen] = '\0';
                }
                else
                {
                    m_pRetMessage = NULL;
                }
            }

            //----------------------------------------------------------------------------------------------------------------------------------
            //~RetVal(); // Auskommentiert Ly bei umstellen wegen dataObj 05.10.2011
            /**
            *   Destructor, in case frees errorMessage string
            */
            ~RetVal() // Copied from sharedStructure.cpp 05.10.2011
            {
                if (this->m_pRetMessage)
                {
                    delete [] this->m_pRetMessage;
                    this->m_pRetMessage = NULL;
                }
            }

            //----------------------------------------------------------------------------------------------------------------------------------
            //RetVal & operator = (const RetVal rhs); // Auskommentiert Ly bei umstellen wegen dataObj 05.10.2011
            /**
            *   assignment operator, copies values of rhs to current RetVal. Before copiing current errorMessage is freed
            */
            RetVal & operator = (const RetVal rhs) // Copied from sharedStructure.cpp 05.10.2011
            {
                if (this == &rhs)
                {
                    return *this;
                }

                this->m_retCode = rhs.m_retCode;
                this->m_retValue = rhs.m_retValue;

                if (this->m_pRetMessage)
                {
                    delete [] this->m_pRetMessage;
                    this->m_pRetMessage = NULL;
                }

                if (rhs.m_pRetMessage != NULL)
                {
                    int messageLen = (int)strlen(rhs.m_pRetMessage);
                    this->m_pRetMessage = new char [messageLen + 1];
                    memcpy(this->m_pRetMessage, rhs.m_pRetMessage, sizeof(*rhs.m_pRetMessage) * messageLen);
                    this->m_pRetMessage[messageLen] = '\0';
                }

                return *this;
            }

            //----------------------------------------------------------------------------------------------------------------------------------
            //RetVal & operator += (const RetVal rhs); // Auskommentiert Ly bei umstellen wegen dataObj 05.10.2011
            /**
            *   Concatenation of RetVal
            *   "Adds" RetVals, i.e. returns the most serious error. In case of
            *   equally serious errors the first is retained
            */
            RetVal & operator += (const RetVal rhs) // Copied from sharedStructure.cpp 05.10.2011
            {
                if (rhs.m_retValue > this->m_retValue)
                {
                    this->m_retCode = rhs.m_retCode;
                    if (this->m_pRetMessage)
                    {
                        delete [] this->m_pRetMessage;
                        this->m_pRetMessage = NULL;
                    }

                    if (rhs.m_pRetMessage != NULL)
                    {
                        int messageLen = (int)strlen(rhs.m_pRetMessage);
                        this->m_pRetMessage = new char [messageLen + 1];
                        memcpy(this->m_pRetMessage, rhs.m_pRetMessage, sizeof(*rhs.m_pRetMessage) * messageLen);
                        this->m_pRetMessage[messageLen] = '\0';
                    }
                }
                this->m_retValue = static_cast<tRetValue>(this->m_retValue | rhs.m_retValue); // |= rhs.m_retValue;
                return *this;
            }

            //----------------------------------------------------------------------------------------------------------------------------------
            //RetVal operator + (const RetVal rhs); // Auskommentiert Ly bei umstellen wegen dataObj 05.10.2011
            /**
            *   Concatenation of RetVal
            *   See operator RetVal::operator+=
            */
            RetVal operator + (const RetVal rhs) // Copied from sharedStructure.cpp 05.10.2011
            {
                static RetVal result = *this;
                result += rhs;
                return result;
            }

            //----------------------------------------------------------------------------------------------------------------------------------
            //char operator == (const RetVal rhs); // Auskommentiert Ly bei umstellen wegen dataObj 05.10.2011
            /**
            *   equality operator compares retValue with with retValue of rhs RetVal. For possible constant values see \ref tRetValue
            */
            char operator == (const RetVal rhs) // Copied from sharedStructure.cpp 05.10.2011
            {
                return m_retValue == rhs.m_retValue;
            }

            //----------------------------------------------------------------------------------------------------------------------------------
            //char operator != (const RetVal rhs); // Auskommentiert Ly bei umstellen wegen dataObj 05.10.2011
            /**
            *   unequality operator compares retValue with with retValue of rhs RetVal. For possible constant values see \ref tRetValue
            */
            char operator != (const RetVal rhs) // Copied from sharedStructure.cpp 05.10.2011
            {
                return !(m_retValue == rhs.m_retValue);
            }

            //----------------------------------------------------------------------------------------------------------------------------------
            //char operator == (const tRetValue rhs); // Auskommentiert Ly bei umstellen wegen dataObj 05.10.2011
            /**
            *   equality operator compares retValue with tRetValue constant. For possible constant values see \ref tRetValue
            */
            char operator == (const tRetValue rhs) // Copied from sharedStructure.cpp 05.10.2011
            {
                return m_retValue == rhs;
            }

            //----------------------------------------------------------------------------------------------------------------------------------
            //char operator != (const tRetValue rhs); // Auskommentiert Ly bei umstellen wegen dataObj 05.10.2011
            /**
            *   unequality operator compares retValue with tRetValue constant. For possible constant values see \ref tRetValue
            */
            char operator != (const tRetValue rhs) // Copied from sharedStructure.cpp 05.10.2011
            {
                return !(m_retValue == rhs);
            }

            //----------------------------------------------------------------------------------------------------------------------------------
            inline int containsWarning() { return (m_retValue & retWarning); }              /*!< checks if any warning has occurred in this return value (true), else (false) */
            inline int containsError() { return (m_retValue & retError); }                 /*!< checks if any error has occurred in this return value (true), else (false) */
            inline int containsWarningOrError() { return (m_retValue & (retError | retWarning)); }  /*!< checks if any warning or error has occurred in this return value (true), else (false) */

            inline char *errorMessage() { return m_pRetMessage; }
            inline int errorCode() const { return m_retCode; }

            //----------------------------------------------------------------------------------------------------------------------------------
            static RetVal format(ito::tRetValue retValue, int retCode, const char *pRetMessage, ...)
            {
                if (pRetMessage != NULL)
                {
                    //int messageLen = strlen(pRetMessage);
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

            //----------------------------------------------------------------------------------------------------------------------------------
            inline void appendRetMessage(const char *addRetMessage)
            {
                if (addRetMessage != NULL)
                {
                    int messageLen = 0;
                    int messageLen2 = (int)strlen(addRetMessage);
                    char *tempRetMessage = NULL;
                    
                    if (m_pRetMessage != NULL)
                    {
                        messageLen = (int)strlen(m_pRetMessage);
                        tempRetMessage = new char [messageLen + messageLen2 + 1];
                        memcpy(tempRetMessage, m_pRetMessage, sizeof(*m_pRetMessage) * messageLen);
                        delete [] m_pRetMessage;
                    }
                    else
                    {
                        tempRetMessage = new char [messageLen2 + 1];
                    }

                    memcpy(tempRetMessage + messageLen, addRetMessage, sizeof(*addRetMessage) * messageLen2);
                    tempRetMessage[messageLen2 + messageLen] = '\0';
                    
                    m_pRetMessage = tempRetMessage;
                    tempRetMessage = NULL;
                }
            } 

        private:
            tRetValue m_retValue;    /*!< can be one of enumeration \ref tLogLevel values or an or-combination of these values*/
            int m_retCode;           /*!< the error code itself */
            char *m_pRetMessage;  /*!< error text if available, else NULL*/
    };

    class ParamBase
    {
    protected:
        uint32 m_type;
        char *m_pName;      //!< parameter name

        void InOutCheck();

    private:
        double m_dVal;      //!< internal value for double typed values
        int m_iVal;         //!< internal value for integer typed values
        char *m_cVal;       //!< internal pointer for pointer type values (also strings)

    public:
        enum Type {
            //flags (bit 17-32)
            NoAutosave      = 0x010000,
            Readonly        = 0x020000, // this flag is not used inside tParam but you can check for it
            In              = 0x040000,
            Out             = 0x080000,

            //type (bit 1-16)
            Pointer         = 0x000001,
            Char            = 0x000002,
            Int             = 0x000004,
            Double          = 0x000008,
            //DObj            = 0x000010,
            String          = 0x000020 | Pointer,
            HWRef           = 0x000040 | Pointer | NoAutosave,
            DObjPtr         = 0x000010 | Pointer | NoAutosave,
            CharArray       = Char     | Pointer,
            IntArray        = Int      | Pointer,
            DoubleArray     = Double   | Pointer,
            PointCloudPtr   = 0x000080 | Pointer | NoAutosave,
            PointPtr        = 0x000100 | Pointer | NoAutosave,
            PolygonMeshPtr  = 0x000200 | Pointer | NoAutosave
        };

        static inline uint32 typeFilter(uint32 type) { return type & paramTypeMask; }

        //--------------------------------------------------------------------------------------------
        //  CONSTRUCTORS, COPY-CONSTRUCTOR, DESTRUCTOR
        //--------------------------------------------------------------------------------------------
        //! default constructor, creates "empty" tParam
        ParamBase() : m_type(0), m_pName(NULL), m_dVal(0.0), m_iVal(0), m_cVal(NULL) {}
        ParamBase(const char *name);                                                               // type-less ParamBase with name only
        ParamBase(const char *name, const uint32 type);                                               // constructor with type and name
        ParamBase(const char *name, const uint32 type, const char *val);                              // constructor with name and type, char val
        ParamBase(const char *name, const uint32 type, const double val);                             // constructor with name and type, double val and optional info
        ParamBase(const char *name, const uint32 type, const int val);                                // constructor with name and type, int val and optional info
        ParamBase(const char *name, const uint32 type, const unsigned int size, const char *values);  // array constructor with name and type, size and array
        ParamBase(const char *name, const uint32 type, const unsigned int size, const int *values);   // array constructor with name and type, size and array
        ParamBase(const char *name, const uint32 type, const unsigned int size, const double *values);// array constructor with name and type, size and array
        virtual ~ParamBase(); //Destructor
        ParamBase(const ParamBase &copyConstr); //Copy-Constructor

        //--------------------------------------------------------------------------------------------
        //  ASSIGNMENT AND OPERATORS
        //--------------------------------------------------------------------------------------------
        const ParamBase operator [] (const int num) const;     //!< braces operator for element-wise access in arrays
        ParamBase& operator = (const ParamBase &rhs);          //!< assignment operator (sets values of lhs to values of rhs tParam, strings are copied)
        ito::RetVal copyValueFrom(const ito::ParamBase *rhs);  //!< just copies the value from the right-hand-side tParam (rhs) to this tParam.

        //--------------------------------------------------------------------------------------------
        //  SET/GET FURTHER PROPERTIES
        //--------------------------------------------------------------------------------------------

        //! returns true if Param is of type char, int or double
        inline bool isNumeric(void) const
        {
            switch(this->getType())
            {
                case Char:
                case Int:
                case Double:
                    return true;
                default:
                    return false;
            }
        }

        inline int addNameSuffix(const char *suffix) 
        { 
            if (suffix && m_pName)  
            { 
                int newSize = (int)strlen(m_pName) + (int)strlen(suffix) + 1;
                m_pName = (char *)realloc(m_pName, newSize);
                strcat_s(m_pName, newSize, suffix);
                return ito::retOk;
            }
            return ito::retWarning;
        }
        //! returns whether tParam contains a valid type (true) or is an empty parameter (false, type == 0). The default tParam-constructor is always an invalid tParam.
        inline bool isValid(void) const { return m_type != 0; }

        //! returns parameter type (autosave flag and optional flag are only included if filterFlags is set false)
        inline uint32 getType(bool filterFlags = true) const 
        { 
            if (filterFlags) 
                return m_type & paramTypeMask; 
            else
                return m_type;
        }

        //! returns parameter flags (parameter type is not included)
        inline uint32 getFlags(void) const { return m_type & paramFlagMask; }
        
        //! sets parameter flags (parameter type remains untouched), for possible flags see \ref tParamType
        inline uint32 setFlags(const uint32 flags) { m_type = getType() | (flags & paramFlagMask); return 0; }
   
        //! returns parameter name (string is not copid)
        inline const char * getName(void) const { return m_pName; }
        //! returns content of autosave flag - this flag determines whether the parameter value gets automagically saved to xml file
        //! when an instance of a plugin class is deleted (closed)
        inline uint32 getAutosave(void) const { return ((m_type & NoAutosave) > 0 ? 0 : 1); }
        //! sets content of autosave flag - this flag determines whether the parameter value gets automagically saved to xml file
        //! when an instance of a plugin class is deleted (closed)
        inline void setAutosave(const uint32 autosave) { m_type = autosave > 0 ? m_type & ~NoAutosave : m_type | NoAutosave; return; }

        inline int getLen(void) const
        {
            switch (m_type & paramTypeMask)
            {
                case DoubleArray:
                case IntArray:
                case CharArray:
                    if (m_cVal)
                    {
                        return m_iVal;
                    }
                    else
                    {
                        return -1;
                    }
                break;

                case String:
                    if (m_cVal)
                    {
                        return static_cast<int>(strlen(m_cVal));
                    }
                    else
                    {
                        return 0;
                    }
                break;

                case Double:
                case Int:
                    return 1;
                break;

                default:
                    return -1;
                break;
            }
        }

            
        /** setVal  set parameter value - templated version
        *   @param [in] val  value to set to
        *   @return     RetVal with operation status
        *   sets the parameter value to the passed value, if the parameter type is inadequate it is set to the
        *   maximum value of template type
        */
        template<typename _Tp> inline ito::RetVal setVal(_Tp val)
        {
            return ItomParamHelper<_Tp>::setVal(getType(), m_cVal, m_iVal, m_dVal, val, 0);
        }

        /** setVal  set parameter value - templated version for arrays
        *   @param [in] val  value to set to
        *   @param [in] len  length of array
        *   @return     RetVal with operation status
        *   sets the parameter value to the passed value, if the length is below 1 or a Null pointer is passed
        *   an error is returned
        */
        template<typename _Tp> inline ito::RetVal setVal(_Tp val, int len)
        {
            return ItomParamHelper<_Tp>::setVal(getType(), m_cVal, m_iVal, m_dVal, val, len);
        }

        /** getVal  read parameter value - templated version
        *   @return parameter value (numeric, casted)
        *
        *   returns the actual parameter value casted to the template parameter type. If the tParam has a non numeric type
        *   the largest value for the template type is passed.
        */
        template<typename _Tp> inline _Tp getVal(void) const
        {
            int len = 0;
            return ItomParamHelper<_Tp>::getVal(getType(), m_cVal, m_iVal, m_dVal, len);
        }

        /** getVal  read parameter value - templated version for arrays
        *   @param [out] len  length of array
        *
        *   returns the actual parameter value casted to the template parameter type. In 'len' is returned what is supposed
        *   to be the length of the array. As only array references are used within tParam the actual size may differ.
        */
        template<typename _Tp> inline _Tp getVal(int &len) const
        {
            return ItomParamHelper<_Tp>::getVal(getType(), m_cVal, m_iVal, m_dVal, len);
        }

    };

    /*!
    \class ParamMeta
    \brief Base class for all meta-information classes
    \sa IntMeta, DoubleMeta, CharMeta, StringMeta, HWMeta, DObjMeta
    */
    class ParamMeta
    {
    public:
        ParamMeta() : m_type(0) {}
        ParamMeta(uint32 type) : m_type(type) {}
        virtual ~ParamMeta() {}
        inline uint32 getType() const { return m_type; }
    protected:
        uint32 m_type;
    };

    /*!
    \class CharMeta
    \brief Meta-information for Param of type Char or CharArray.
    \sa ito::Param
    */
    class CharMeta : public ParamMeta
    {
    public:
        //! constructor with minimum and maximum value
        explicit CharMeta(char minVal, char maxVal) : ParamMeta(ParamBase::Char), m_minVal(minVal), m_maxVal(maxVal) { if(m_maxVal < m_minVal) std::swap(m_minVal,m_maxVal); }
        static CharMeta* all() { return new CharMeta(std::numeric_limits<char>::min(), std::numeric_limits<char>::max() ); } //!< returns a new instance of CharMeta, where the min and max are set to the full range available for char.
        inline char getMin() const { return m_minVal; } //!< returns minimum value
        inline char getMax() const { return m_maxVal; } //!< returns maximum value

        //! sets the minimum value
        /*!
            \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is changed to val, too
        */
        inline void setMin(char val) { m_minVal = val; m_maxVal = std::max(m_maxVal,m_minVal); }
        
        //! sets the maximum value
        /*!
            \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is changed to val, too
        */
        inline void setMax(char val) { m_maxVal = val; m_minVal = std::min(m_maxVal,m_minVal); }
    private:
        char m_minVal;
        char m_maxVal;
    };

    /*!
    \class IntMeta
    \brief Meta-information for Param of type Int or IntArray.
    \sa ito::Param
    */
    class IntMeta : public ParamMeta
    {
    public:
        //! constructor with minimum and maximum value
        explicit IntMeta(int minVal, int maxVal) : ParamMeta(ParamBase::Int), m_minVal(minVal), m_maxVal(maxVal) { if(m_maxVal < m_minVal) std::swap(m_minVal,m_maxVal); }
        static IntMeta* all() { return new IntMeta(std::numeric_limits<int>::min(), std::numeric_limits<int>::max() ); } //!< returns a new instance of IntMeta, where the min and max are set to the full range available for integers.
        inline int getMin() const { return m_minVal; } //!< returns minimum value
        inline int getMax() const { return m_maxVal; } //!< returns maximum value
        
        //! sets the minimum value
        /*!
            \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is changed to val, too
        */
        inline void setMin(int val) { m_minVal = val; m_maxVal = std::max(m_maxVal,m_minVal); }
        
        //! sets the maximum value
        /*!
            \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is changed to val, too
        */
        inline void setMax(int val) { m_maxVal = val; m_minVal = std::min(m_maxVal,m_minVal); }
    private:
        int m_minVal;
        int m_maxVal;
    };

    /*!
    \class DoubleMeta
    \brief Meta-information for Param of type Double or DoubleArray.
    \sa ito::Param
    */
    class DoubleMeta : public ParamMeta
    {
    public:
        //! constructor with minimum and maximum value
        explicit DoubleMeta(double minVal, double maxVal) : ParamMeta(ParamBase::Double), m_minVal(minVal), m_maxVal(maxVal) { if(m_maxVal < m_minVal) std::swap(m_minVal,m_maxVal); }
        static DoubleMeta* all() { return new DoubleMeta(-std::numeric_limits<double>::max(), std::numeric_limits<double>::max() ); } //!< returns a new instance of DoubleMeta, where the min and max are set to the full range available for double.
        inline double getMin() const { return m_minVal; } //!< returns minimum value
        inline double getMax() const { return m_maxVal; } //!< returns maximum value
        
        //! sets the minimum value
        /*!
            \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is changed to val, too
        */
        inline void setMin(double val) { m_minVal = val; m_maxVal = std::max(m_maxVal,m_minVal); }
        
        //! sets the maximum value
        /*!
            \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is changed to val, too
        */
        inline void setMax(double val) { m_maxVal = val; m_minVal = std::min(m_maxVal,m_minVal); }
    private:
        double m_minVal;
        double m_maxVal;
    };

    /*!
    \class HWMeta
    \brief Meta-information for Param of type HWPtr.
    \sa ito::Param
    */
    class HWMeta : public ParamMeta
    {
        public:
            //! constructor
            /*!
                creates HWMeta-information struct where you can pass a bitmask which consists of values of the enumeration
                ito::tPluginType. The plugin reference of the corresponding Param should then only accept plugins, where
                all bits are set, too.
                \sa ito::Plugin, ito::tPluginType
            */
            explicit HWMeta(uint32 minType) : ParamMeta(ParamBase::HWRef), m_minType(minType), m_pHWAddInName(NULL) {}

            //! constructor
            /*!
                creates HWMeta-information struct where you can pass a specific name of a plugin, which only is
                allowed by the corresponding plugin-instance.
                \sa ito::Plugin
            */
            explicit HWMeta(const char *HWAddInName) : ParamMeta(ParamBase::HWRef), m_minType(0), m_pHWAddInName(NULL)
            {
                if(HWAddInName) m_pHWAddInName = _strdup(HWAddInName);
            }
            HWMeta(const HWMeta& cpy) : ParamMeta(ParamBase::HWRef), m_minType(cpy.m_minType), m_pHWAddInName(NULL)
            {
                if(cpy.m_pHWAddInName) m_pHWAddInName = _strdup(cpy.m_pHWAddInName);
            }
            ~HWMeta() { if(m_pHWAddInName) free(m_pHWAddInName); }            //!< destructor
            inline uint32 getMinType() const { return m_minType; }                //!< returns type-bitmask which is minimally required by plugin-reference. Default 0. \sa ito::tPluginType
            inline char * getHWAddInName() const { return m_pHWAddInName; } //!< returns zero-terminated name of specific plugin-name or NULL if not specified.
        private:
            uint32 m_minType;            //!< type-bitmask which is minimally required. default: 0
            char *m_pHWAddInName;    //!< zero-terminated name of specific plugin-name of NULL if not specified.
    };

    /*!
    \class StringMeta
    \brief Meta-information for Param of type String.
    \sa ito::Param
    */
    class StringMeta : public ParamMeta
    {
        public:
            enum tType {
                String, //!< string elements should be considered as strings (exact match)
                Wildcard, //!< string elements should be considered as wildcard-expressions (e.g. *.doc)
                RegExp    //!< string elements should be considered as regular expressions (e.g. ^(.*)[abc]{1,5}$)
            };

            //! constructor
            /*!
                Returns a meta information class for string-types.
                \param type indicates how the string elements should be considered
                \sa tType
            */
            StringMeta(tType type) : ParamMeta(ParamBase::String), m_stringType(type), m_len(0), m_val(NULL) {}

            //! constructor
            /*!
                Returns a meta information class for string-types.
                \param type indicates how the string elements should be considered
                \param val adds a first string to the element list
                \sa tType
            */
            StringMeta(tType type, const char* val) : ParamMeta(ParamBase::String), m_stringType(type), m_len(1)
            {
                if(val)
                {
                    m_val = (char**) calloc(1, sizeof(char*));
                    m_val[0] = _strdup(val);
                }
                else
                {
                    m_len = 0;
                    m_val = NULL;
                }
            }

            //! copy constructor
            StringMeta(const StringMeta& cpy) : ParamMeta(ParamBase::String), m_stringType(cpy.m_stringType), m_len(cpy.m_len), m_val(NULL)
            {
                if(m_len > 0)
                {
                    m_val = (char**) calloc(m_len, sizeof(char*));
                    for(int i=0;i<m_len;++i) m_val[i] = _strdup(cpy.m_val[i]);
                }
            }

            //! destructor
            ~StringMeta()
            {
                for(int i=0;i<m_len;++i) free(m_val[i]);
                free(m_val);
            }

            inline tType getStringType() const { return m_stringType; } //!< returns the type how strings in list should be considered. \sa tType
            inline int getLen() const { return m_len; } //!< returns the number of string elements in meta information class.
            inline const char* getString(int idx = 0) const { return (idx >= m_len) ? NULL : m_val[idx]; } //!< returns string from list at index position or NULL, if index is out of range.
            void addItem(const char *val) //!< adds another element to the string list.
            {
                if(m_val)
                {
                    m_val = (char**)realloc(m_val, sizeof(char*) * (++m_len) );
                }
                else
                {
                    m_val = (char**) calloc(++m_len, sizeof(char*));
                }
                m_val[m_len-1] = _strdup(val);
            }
            StringMeta & operator += (const char *val)
            {
                addItem(val);
                return *this;
            }

        private:
            tType m_stringType;
            int m_len;
            char **m_val;
    };

    /*!
    \class DObjMeta
    \brief Meta-information for Param of type DObjPtr.
    \sa ito::Param
    */
    class DObjMeta : public ParamMeta
    {
        public:
            explicit DObjMeta(uint32 allowedTypes = 0xFFFF, int minDim = 0, int maxDim = std::numeric_limits<int>::max()) : ParamMeta(ParamBase::DObjPtr), m_allowedTypes(allowedTypes), m_minDim(minDim), m_maxDim(maxDim) {}
            inline int getAllowedTypes() const { return m_allowedTypes; }
            inline int getMinDim() const { return m_minDim; } //!< returns maximum allowed dimensions of data object
            inline int getMaxDim() const { return m_maxDim; } //!< returns minimum number of dimensions of data object

        private:
            uint32 m_allowedTypes;
            int m_minDim;
            int m_maxDim;
    };



    //----------------------------------------------------------------------------------------------------------------------------------
    /** @class ExtParam
    *   @brief  class for parameter handling e.g. to pass paramters to plugins
    *
    *   The plugins use this class to organize their parameters (internally) and for the paramList which is used
    *   for type checking whilst parsing parameters passed from python to c.
    */
    class Param : public ParamBase
    {
        private:
            ParamMeta *m_pMeta;
            char *m_pInfo;

        public:

            //--------------------------------------------------------------------------------------------
            //  CONSTRUCTORS, COPY-CONSTRUCTOR, DESTRUCTOR
            //--------------------------------------------------------------------------------------------
            //! default constructor, creates "empty" tParam
            Param() : ParamBase(), m_pMeta(NULL), m_pInfo(NULL) {}
            Param(const char *name) : ParamBase(name), m_pMeta(NULL), m_pInfo(NULL) {}                         // type-less Param with name only
            Param(const char *name, const uint32 type) : ParamBase(name, type), m_pMeta(NULL), m_pInfo(NULL) {}   // constructor with type and name
            Param(const char *name, const uint32 type, const char *val, const char *info);                        // constructor with name and type, char val and optional info
            Param(const char *name, const uint32 type, const double minVal, const double maxVal, const double val, const char *info); // constructor with name and type, double val, double minVal, double maxVal and optional info
            Param(const char *name, const uint32 type, const int minVal, const int maxVal, const int val, const char *info); // constructor with name and type, int val, int minVal, int maxVal and optional info
            Param(const char *name, const uint32 type, const char minVal, const char maxVal, const char val, const char *info); // constructor with name and type, int val, int minVal, int maxVal and optional info
            Param(const char *name, const uint32 type, const unsigned int size, const char *values, const char *info);  // array constructor with name and type, size and array
            Param(const char *name, const uint32 type, const unsigned int size, const int *values, const char *info);   // array constructor with name and type, size and array
            Param(const char *name, const uint32 type, const unsigned int size, const double *values, const char *info);// array constructor with name and type, size and array
            Param(const char *name, const uint32 type, const int val, ParamMeta *meta, const char *info);
            Param(const char *name, const uint32 type, const double val, ParamMeta *meta, const char *info);
            Param(const char *name, const uint32 type, const char val, ParamMeta *meta, const char *info);
            Param(const char *name, const uint32 type, const unsigned int size, const double *values, ParamMeta *meta, const char *info);
            Param(const char *name, const uint32 type, const unsigned int size, const int *values, ParamMeta *meta, const char *info);
            Param(const char *name, const uint32 type, const unsigned int size, const char *values, ParamMeta *meta, const char *info);
            ~Param();                        //!< Destructor
            Param(const Param &copyConstr); //!< Copy-Constructor

            //--------------------------------------------------------------------------------------------
            //  ASSIGNMENT AND OPERATORS
            //--------------------------------------------------------------------------------------------
            const Param operator [] (const int num) const;    //!< braces operator for element-wise access in arrays
            Param& operator = (const Param &rhs);             //!< assignment operator (sets values of lhs to values of rhs Param, strings are copied)
            ito::RetVal copyValueFrom(const ParamBase *rhs);  //!< just copies the value from the right-hand-side ParamBase (rhs) to this tParam.

            //--------------------------------------------------------------------------------------------
            //  SET/GET FURTHER PROPERTIES
            //--------------------------------------------------------------------------------------------
            //!< returns content of info string (string is not copied)
            inline const char * getInfo(void) const { return m_pInfo; }
            //!< sets content of info string, if necessary the info buffer is freed first, passed string is copied
            inline int setInfo(const char *info)
            {
                if (m_pInfo) free(m_pInfo);
                if (info)
                {
                    if (!(m_pInfo = _strdup(info)))
                    {
                        return ito::retError;
                    }
                }
                else
                {
                    m_pInfo = NULL;
                }
                return ito::retOk;
            }

            inline const ParamMeta* getMeta(void) const { return m_pMeta; } //!< returns const-pointer to meta-information instance or NULL if not available
            inline ParamMeta* getMeta(void) { return m_pMeta; }                //!< returns pointer to meta-information instance or NULL if not available

            //! sets a new ParamMeta-instance as meta information for this Param
            /*!
                \param meta is the pointer to any instance derived from ParamMeta
                \param takeOwnership (default: false) defines, whether this Param should take the ownership of the ParamMeta-instance
                \sa ito::ParamMeta
            */
            bool setMeta(ParamMeta* meta, bool takeOwnership = false);

            bool copyMetaFrom(const ParamMeta *meta);

            double getMin() const;
            double getMax() const;
    };

    //---------------------------------------------------------------------------------------------------------------------
    template<typename _Tp>
    struct ItomParamHelper
    {
        static ito::RetVal setVal(uint32 type, char *&cVal, int &iVal, double &/*dVal*/, _Tp val, int len = 0)
        {
            switch (type & paramTypeMask)
            {
                case (ito::ParamBase::HWRef & paramTypeMask):
                case (ito::ParamBase::DObjPtr & paramTypeMask):
                case ito::ParamBase::PointCloudPtr & paramTypeMask:
                case ito::ParamBase::PointPtr & paramTypeMask:
                case ito::ParamBase::PolygonMeshPtr & paramTypeMask:
//                case ito::ParamBase::Pointer & paramTypeMask:
                    cVal = reinterpret_cast<char*>(val);
                    return ito::retOk;
                break;

                case ito::ParamBase::String & paramTypeMask:
                    {
                        char *cVal_ = cVal;
                        if (val)
                        {
                            cVal = _strdup(const_cast<const char*>((char*)val));
                            iVal = static_cast<int>(strlen(cVal));
                        }
                        else
                        {
                            cVal = 0;
                            iVal = -1;
                        }
                        if (cVal_)
                        {
                            free(cVal_);
                        }
                    }
                    return ito::retOk;
                break;

                case ito::ParamBase::CharArray & ito::paramTypeMask:
                    {
                        char *cVal_ = cVal;
                        if ((val) && (len > 0))
                        {
                            cVal = (char*)malloc(len * sizeof(char));
                            memcpy(cVal, val, len * sizeof(char));
                            iVal = len;
                        }
                        else
                        {
                            cVal = NULL;
                            iVal = -1;
                        }
                        if (cVal_)
                        {
                            free(cVal_);
                        }
                    }
                    return ito::retOk;
                break;

                case ito::ParamBase::IntArray & ito::paramTypeMask:
                    {
                        char *cVal_ = cVal;
                        if ((val) && (len > 0))
                        {
                            cVal = (char*)malloc(len * sizeof(int));
                            memcpy(cVal, val, len * sizeof(int));
                            iVal = len;
                        }
                        else
                        {
                            cVal = NULL;
                            iVal = -1;
                        }
                        if (cVal_)
                        {
                            free(cVal_);
                        }
                    }
                    return ito::retOk;
                break;

                case ito::ParamBase::DoubleArray & ito::paramTypeMask:
                    {
                        char *cVal_ = cVal;
                        if ((val) && (len > 0))
                        {
                            cVal = (char*)malloc(len * sizeof(double));
                            memcpy(cVal, val, len * sizeof(double));
                            iVal = len;
                        }
                        else
                        {
                            cVal = NULL;
                            iVal = -1;
                        }
                        if (cVal_)
                        {
                            free(cVal_);
                        }
                    }
                    return ito::retOk;
                break;

                default:
                    return ito::retError;
                break;
            }
        }

        static _Tp getVal(const uint32 type, const char *cVal, const int &iVal, const double &/*dVal*/, int &len)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::String & paramTypeMask:
                    if (cVal)
                    {
                        len = static_cast<int>(strlen(cVal));
                        //return reinterpret_cast<_Tp>(_strdup(cVal));
                        return reinterpret_cast<_Tp>(const_cast<char*>(cVal));
                    }
                    else
                    {
                        len = 0;
                        return 0;
                    }
                break;

                case ito::ParamBase::CharArray & ito::paramTypeMask:
                case ito::ParamBase::IntArray & ito::paramTypeMask:
                case ito::ParamBase::DoubleArray & ito::paramTypeMask:
                    if (cVal)
                    {
                        len = iVal;
                        return reinterpret_cast<_Tp>(const_cast<char*>(cVal));
                    }
                    else
                    {
                        len = 0;
                        return 0;
                    }
                break;

                case (ito::ParamBase::HWRef & paramTypeMask):
                case (ito::ParamBase::DObjPtr & paramTypeMask):
                case ito::ParamBase::PointCloudPtr & paramTypeMask:
                case ito::ParamBase::PointPtr & paramTypeMask:
                case ito::ParamBase::PolygonMeshPtr & paramTypeMask:
                    return reinterpret_cast<_Tp>(const_cast<char*>(cVal));
                break;

                default:
                    throw std::logic_error("Non-matching type!");
                    return (_Tp)0;
                break;
            }
        }
    };

    template<>
    struct ItomParamHelper<double>
    {
        static ito::RetVal setVal(uint32 type, char *&/*cVal*/, int &iVal, double &dVal, double val, int /*len = 0*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Double & ito::paramTypeMask:
                    dVal = static_cast<double>(val);
                    return ito::retOk;
                break;

                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    iVal = static_cast<int>(val);
                    return ito::retOk;
                break;

                default:
                    return ito::retError;
                break;
            }
        }

        static double getVal(const uint32 type, const char * /*cVal*/, const int &iVal, const double &dVal, int & /*len*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    return static_cast<double>(iVal);
                break;
                case ito::ParamBase::Double & ito::paramTypeMask:
                    return dVal;
                break;

                default:
                    throw std::logic_error("Non-matching type!");
                    return 0;
                break;
            }
        }
    };

    template<>
    struct ItomParamHelper<int>
    {
        static ito::RetVal setVal(uint32 type, char *&/*cVal*/, int &iVal, double &dVal, int val, int /*len = 0*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Double & ito::paramTypeMask:
                    dVal = static_cast<int>(val);
                    return ito::retOk;
                break;

                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    iVal = val;
                    return ito::retOk;
                break;

                default:
                    return ito::retError;
                break;
            }
        }

        static int getVal(const uint32 type, const char * /*cVal*/, const int &iVal, const double &dVal, int & /*len*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    return iVal;
                break;

                case ito::ParamBase::Double & ito::paramTypeMask:
                    return static_cast<int>(dVal);
                break;

                case 0:
                    throw std::invalid_argument("non existent parameter");
                    return 0;
                break;

                default:
                    throw std::logic_error("Non-matching type!");
                    return 0;
                break;
            }
        }
    };

    template<>
    struct ItomParamHelper<char>
    {
        static ito::RetVal setVal(uint32 type, char *&/*cVal*/, int &iVal, double &dVal, char val, int /*len = 0*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Double & ito::paramTypeMask:
                    dVal = static_cast<char>(val);
                    return ito::retOk;
                break;

                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    iVal = static_cast<char>(val);
                    return ito::retOk;
                break;

                default:
                    return ito::retError;
                break;
            }
        }

        static char getVal(const uint32 type, const char * /*cVal*/, const int &iVal, const double &dVal, int & /*len*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    return static_cast<char>(iVal);
                break;

                case ito::ParamBase::Double & ito::paramTypeMask:
                    return static_cast<char>(dVal);
                break;

                case 0:
                    throw std::invalid_argument("non existent parameter");
                    return 0;
                break;

                default:
                    throw std::logic_error("Non-matching type!");
                    return 0;
                break;
            }
        }
    };

    template<>
    struct ItomParamHelper<unsigned char>
    {
        static ito::RetVal setVal(uint32 type, char *&/*cVal*/, int &iVal, double &dVal, char val, int /*len = 0*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Double & ito::paramTypeMask:
                    dVal = static_cast<unsigned char>(val);
                    return ito::retOk;
                break;

                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    iVal = static_cast<unsigned char>(val);
                    return ito::retOk;
                break;

                default:
                    return ito::retError;
                break;
            }
        }

        static char getVal(const uint32 type, const char * /*cVal*/, const int &iVal, const double &dVal, int & /*len*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    return static_cast<unsigned char>(iVal);
                break;

                case ito::ParamBase::Double & ito::paramTypeMask:
                    return static_cast<unsigned char>(dVal);
                break;

                case 0:
                    throw std::invalid_argument("non existent parameter");
                    return 0;
                break;

                default:
                    throw std::logic_error("Non-matching type!");
                    return 0;
                break;
            }
        }
    };

} //end namespace ito

#endif
