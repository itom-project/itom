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

#ifndef PARAM_H
#define PARAM_H

/* includes */

#include "commonGlobal.h"
#include "typeDefs.h"
#include "byteArray.h"
#include "retVal.h"
#include "paramMeta.h"

#include <string.h>
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
    class Param;
    class ParamHelper;
    template<typename _Tp> struct ItomParamHelper;

    const uint32 paramFlagMask = 0xFFFF0000; //!< bits of type lying within this mask are flags (e.g. typeNoAutosave, typeReadonly...)
    const uint32 paramTypeMask = 0x0000FFFF; //!< bits of param type lying withing this mask describe the type (typeNoAutosave must be included there)

    //! wrapper class for a complex128 value. This class is used, since the std::complex stl class is not exported over DLLs
    struct ITOMCOMMON_EXPORT complex128_
    {
        complex128_(float64 r, float64 i) : real(r), imag(i) {};
        float64 real;
        float64 imag;
    };

    class ITOMCOMMON_EXPORT ParamBase
    {
    protected:
        uint32 m_type;
        ByteArray m_name;      //!< parameter name

        void inOutCheck();

        static inline uint32 typeFilter(uint32 type) { return type & paramTypeMask; }

    private:
        complex128_ m_dVal;    //!< internal value for float64 and complex128 typed values
        ito::int32 m_iVal;         //!< internal value for integer typed values
        char *m_cVal;         //!< internal pointer for pointer type values (also strings)

    public:
        enum Type {
            //flags (bit 17-32)
            NoAutosave      = 0x010000,
            Readonly        = 0x020000, // this flag is not used inside Param but you can check for it
            In              = 0x040000,
            Out             = 0x080000,
			NotAvailable    = 0x100000,

            //type (bit 1-16)
            Pointer         = 0x000001,
            Char            = 0x000002,
            Int             = 0x000004,
            Double          = 0x000008,
            Complex         = 0x000400,
            DObjPtr         = 0x000010 | Pointer | NoAutosave,
            String          = 0x000020 | Pointer,
            HWRef           = 0x000040 | Pointer | NoAutosave,
            CharArray       = Char     | Pointer,
            IntArray        = Int      | Pointer,
            DoubleArray     = Double   | Pointer,
            ComplexArray    = Complex  | Pointer,
            PointCloudPtr   = 0x000080 | Pointer | NoAutosave,
            PointPtr        = 0x000100 | Pointer | NoAutosave,
            PolygonMeshPtr  = 0x000200 | Pointer | NoAutosave
        };

        
        //--------------------------------------------------------------------------------------------
        //  CONSTRUCTORS, COPY-CONSTRUCTOR, DESTRUCTOR
        //--------------------------------------------------------------------------------------------
        //! default constructor, creates "empty" ParamBase
        ParamBase() : m_type(0), m_name(NULL), m_dVal(0.0, 0.0), m_iVal(0), m_cVal(NULL) {}
        ParamBase(const ByteArray &name);                                                                  // type-less ParamBase with name only
        ParamBase(const ByteArray &name, const uint32 type);                                               // constructor with type and name
        ParamBase(const ByteArray &name, const uint32 type, const char *val);                              // constructor with name and type and char val
        ParamBase(const ByteArray &name, const uint32 type, const float64 val);                            // constructor with name and type and float64 val
        ParamBase(const ByteArray &name, const uint32 type, const int32 val);                              // constructor with name and type and int32 val
        ParamBase(const ByteArray &name, const uint32 type, const complex128 val);                         // constructor with name and type and complex128 val
        ParamBase(const ByteArray &name, const uint32 type, const uint32 size, const char *values);        // array constructor with name and type, size and array
        ParamBase(const ByteArray &name, const uint32 type, const uint32 size, const int32 *values);       // array constructor with name and type, size and array
        ParamBase(const ByteArray &name, const uint32 type, const uint32 size, const float64 *values);     // array constructor with name and type, size and array
        ParamBase(const ByteArray &name, const uint32 type, const uint32 size, const complex128 *values);  // array constructor with name and type, size and array
        virtual ~ParamBase(); //Destructor
        ParamBase(const ParamBase &copyConstr); //Copy-Constructor

        //--------------------------------------------------------------------------------------------
        //  ASSIGNMENT AND OPERATORS
        //--------------------------------------------------------------------------------------------
        const ParamBase operator [] (const int num) const;     //!< braces operator for element-wise access in arrays
        ParamBase& operator = (const ParamBase &rhs);          //!< assignment operator (sets values of lhs to values of rhs Param, strings are copied)
        ito::RetVal copyValueFrom(const ito::ParamBase *rhs);  //!< just copies the value from the right-hand-side tParam (rhs) to this tParam.

        //--------------------------------------------------------------------------------------------
        // COMPARISON OPERATORS (two values are equal if both their type and the content is equal)
        //--------------------------------------------------------------------------------------------
        bool operator ==(const ParamBase &rhs) const;

        inline bool operator !=(const ParamBase &rhs) const
        {
            return !(*this == rhs);
        }

        //--------------------------------------------------------------------------------------------
        //  SET/GET FURTHER PROPERTIES
        //--------------------------------------------------------------------------------------------

        //! returns true if Param is of type char, int, double or complex
        inline bool isNumeric(void) const
        {
            static int numericTypeMask = ito::ParamBase::Char | ParamBase::Int | ParamBase::Double | ParamBase::Complex;
            return ((m_type & numericTypeMask) > 0) && !(m_type & ito::ParamBase::Pointer);
        }

        //! returns true if Param is of type char array, int array, double array or complex array
        inline bool isNumericArray(void) const
        {
            static int numericTypeMask = ito::ParamBase::Char | ParamBase::Int | ParamBase::Double | ParamBase::Complex;
            return (m_type & (numericTypeMask | ito::ParamBase::Pointer)) > 0;
        }

        //! returns whether Param contains a valid type (true) or is an empty parameter (false, type == 0). The default tParam-constructor is always an invalid tParam.
        inline bool isValid(void) const { return m_type != 0; }

        //! returns parameter type (autosave flag and other flags (like in, out or readonly) are only included if filterFlags is set false)
        inline uint32 getType(bool filterFlags = true) const
        {
            if (filterFlags)
                return m_type & paramTypeMask;
            else
                return m_type;
        }

        //! returns parameter flags
        inline uint32 getFlags(void) const { return m_type & paramFlagMask; }
        
        //! sets parameter flagsfor possible flags see \ref tParamType
        inline void setFlags(const uint32 flags) { m_type = getType() | (flags & paramFlagMask); }
   
        //! returns parameter name (returned string is no copy, do not delete it)
        inline const char* getName(void) const { return m_name.data(); }

        //! returns content of autosave flag - this flag determines whether the parameter value gets automagically saved to xml file
        //! when an instance of a plugin class is deleted (closed)
        inline bool getAutosave(void) const { return (m_type & NoAutosave) > 0; }

        //! sets content of autosave flag - this flag determines whether the parameter value gets automagically saved to xml file
        //! when an instance of a plugin class is deleted (closed)
        inline void setAutosave(const bool autosave) { m_type = autosave ? (m_type | NoAutosave) : (m_type & ~NoAutosave); }

        //! returns length of array parameters or -1 if no array is given. For string parameter returns length of string or 0 if not given, for number parameters return 1. In all other cases -1.
        int getLen(void) const;
        
            
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

    //----------------------------------------------------------------------------------------------------------------------------------
    /** @class Param
    *   @brief  class for parameter handling e.g. to pass paramters to plugins
    *
    *   The plugins use this class to organize their parameters (internally) and for the paramList which is used
    *   for type checking whilst parsing parameters passed from python to c.
    */
    class ITOMCOMMON_EXPORT Param : public ParamBase
    {
        private:
            ParamMeta *m_pMeta;
            ByteArray m_info;

        public:

            //--------------------------------------------------------------------------------------------
            //  CONSTRUCTORS, COPY-CONSTRUCTOR, DESTRUCTOR
            //--------------------------------------------------------------------------------------------
            //! default constructor, creates "empty" tParam
            Param() : ParamBase(), m_pMeta(NULL), m_info(NULL) {}
            Param(const ByteArray &name) : ParamBase(name), m_pMeta(NULL), m_info(NULL) {}                         // type-less Param with name only
            Param(const ByteArray &name, const uint32 type) : ParamBase(name, type), m_pMeta(NULL), m_info(NULL) {}   // constructor with type and name
            Param(const ByteArray &name, const uint32 type, const char *val, const char *info);                        // constructor with name and type, char val and optional info
            Param(const ByteArray &name, const uint32 type, const char minVal, const char maxVal, const char val, const char *info); // constructor with name and type, int val, int minVal, int maxVal and optional info
            Param(const ByteArray &name, const uint32 type, const int32 minVal, const int32 maxVal, const int32 val, const char *info); // constructor with name and type, int val, int minVal, int maxVal and optional info
            Param(const ByteArray &name, const uint32 type, const float64 minVal, const float64 maxVal, const float64 val, const char *info); // constructor with name and type, double val, double minVal, double maxVal and optional info
            Param(const ByteArray &name, const uint32 type, const unsigned int size, const char *values, const char *info);  // array constructor with name and type, size and array
            Param(const ByteArray &name, const uint32 type, const unsigned int size, const int32 *values, const char *info);   // array constructor with name and type, size and array
            Param(const ByteArray &name, const uint32 type, const unsigned int size, const float64 *values, const char *info);// array constructor with name and type, size and array
            Param(const ByteArray &name, const uint32 type, const unsigned int size, const complex128 *values, const char *info);// array constructor with name and type, size and array
            Param(const ByteArray &name, const uint32 type, const char val, ParamMeta *meta, const char *info);
            Param(const ByteArray &name, const uint32 type, const int32 val, ParamMeta *meta, const char *info);
            Param(const ByteArray &name, const uint32 type, const float64 val, ParamMeta *meta, const char *info);
            Param(const ByteArray &name, const uint32 type, const complex128 val, ParamMeta *meta, const char *info);
            Param(const ByteArray &name, const uint32 type, const unsigned int size, const char *values, ParamMeta *meta, const char *info);
            Param(const ByteArray &name, const uint32 type, const unsigned int size, const int32 *values, ParamMeta *meta, const char *info);
            Param(const ByteArray &name, const uint32 type, const unsigned int size, const float64 *values, ParamMeta *meta, const char *info);
            Param(const ByteArray &name, const uint32 type, const unsigned int size, const complex128 *values, ParamMeta *meta, const char *info);
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
            inline const char * getInfo(void) const { return m_info.data(); }

            //!< sets content of info string, if necessary the info buffer is freed first, passed string is copied
            inline void setInfo(const char *info)
            {
                m_info = info;
            }

            inline void setInfo(const ByteArray &info)
            {
                m_info = info;
            }

            inline const ParamMeta* getMeta(void) const { return m_pMeta; }  //!< returns const-pointer to meta-information instance or NULL if not available. Cast this pointer to the right class of the parameter.
            inline ParamMeta* getMeta(void) { return m_pMeta; }              //!< returns pointer to meta-information instance or NULL if not available. Cast this pointer to the right class of the parameter.

			//!< returns const-pointer to meta-information instance casted to 'const _Tp*' or NULL if not available or cast failed.
			/*
			Example: intParam.getMetaT<ito::IntMeta>();
			*/
			template<typename _Tp> inline const _Tp* getMetaT(void) const 
			{ 
				return static_cast<const _Tp*>(m_pMeta);
			}  

			//!< returns pointer to meta-information instance casted to '_Tp*' or NULL if not available or cast failed.
			/*
			Example: intParam.getMetaT<ito::IntMeta>();
			*/
			template<typename _Tp> inline _Tp* getMetaT(void)
			{
				return static_cast<_Tp*>(m_pMeta);
			}

            //! sets a new ParamMeta-instance as meta information for this Param
            /*!
                \param meta is the pointer to any instance derived from ParamMeta
                \param takeOwnership (default: false) defines, whether this Param should take the ownership of the ParamMeta-instance
                \sa ito::ParamMeta
            */
            void setMeta(ParamMeta* meta, bool takeOwnership = false);

            float64 getMin() const;
            float64 getMax() const;
    };

    //---------------------------------------------------------------------------------------------------------------------
    template<typename _Tp>
    struct ItomParamHelper
    {
        static ito::RetVal setVal(uint32 type, char *&cVal, int32 &iVal, complex128_ &/*dVal*/, const _Tp val, int len = 0)
        {
            switch (type & paramTypeMask)
            {
                case (ito::ParamBase::HWRef & paramTypeMask):
                case (ito::ParamBase::DObjPtr & paramTypeMask):
                case ito::ParamBase::PointCloudPtr & paramTypeMask:
                case ito::ParamBase::PointPtr & paramTypeMask:
                case ito::ParamBase::PolygonMeshPtr & paramTypeMask:
//                case ito::ParamBase::Pointer & paramTypeMask:
                    cVal = (char*)(reinterpret_cast<const char*>(val));
                    return ito::retOk;

                case ito::ParamBase::String & paramTypeMask:
                    {
                        char *cVal_ = cVal;
                        if (val)
                        {
                            cVal = _strdup((const char*)val);
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

                case ito::ParamBase::IntArray & ito::paramTypeMask:
                    {
                        char *cVal_ = cVal;
                        if ((val) && (len > 0))
                        {
                            cVal = (char*)malloc(len * sizeof(ito::int32));
                            memcpy(cVal, val, len * sizeof(ito::int32));
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

                case ito::ParamBase::DoubleArray & ito::paramTypeMask:
                    {
                        char *cVal_ = cVal;
                        if ((val) && (len > 0))
                        {
                            cVal = (char*)malloc(len * sizeof(ito::float64));
                            memcpy(cVal, val, len * sizeof(ito::float64));
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

                case ito::ParamBase::ComplexArray & ito::paramTypeMask:
                    {
                        char *cVal_ = cVal;
                        if ((val) && (len > 0))
                        {
                            cVal = (char*)malloc(len * sizeof(ito::complex128));
                            memcpy(cVal, val, len * sizeof(ito::complex128));
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

                default:
                    return ito::RetVal(ito::retError, 0, "_Tp parameter of setVal<_Tp> does not match the type of the parameter");
            }
        }

        static _Tp getVal(const uint32 type, const char *cVal, const int32 &iVal, const complex128_ &/*dVal*/, int &len)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::String & paramTypeMask:
                    if (cVal)
                    {
                        len = static_cast<int>(strlen(cVal));
                        return reinterpret_cast<_Tp>(const_cast<char*>(cVal));
                    }
                    else
                    {
                        len = 0;
                        return 0;
                    }

                case ito::ParamBase::CharArray & ito::paramTypeMask:
                case ito::ParamBase::IntArray & ito::paramTypeMask:
                case ito::ParamBase::DoubleArray & ito::paramTypeMask:
                case ito::ParamBase::ComplexArray & ito::paramTypeMask:
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

                case (ito::ParamBase::HWRef & paramTypeMask):
                case (ito::ParamBase::DObjPtr & paramTypeMask):
                case ito::ParamBase::PointCloudPtr & paramTypeMask:
                case ito::ParamBase::PointPtr & paramTypeMask:
                case ito::ParamBase::PolygonMeshPtr & paramTypeMask:
                    return reinterpret_cast<_Tp>(const_cast<char*>(cVal));

                default:
                    throw std::logic_error("Param::getVal<_Tp>: Non-matching type!");
            }
        }
    };

    template<>
    struct ItomParamHelper<float64>
    {
        static ito::RetVal setVal(uint32 type, char *&/*cVal*/, int32 &iVal, complex128_ &dVal, float64 val, int /*len = 0*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Complex & ito::paramTypeMask:
                case ito::ParamBase::Double & ito::paramTypeMask:
                    dVal.real = static_cast<float64>(val);
                    dVal.imag = 0.0;
                    return ito::retOk;

                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    iVal = static_cast<int32>(val);
                    return ito::retOk;

                default:
                    return ito::RetVal(ito::retError, 0, "float64 value passed to setVal<float64> does not match the type of the parameter");
            }
        }

        static float64 getVal(const uint32 type, const char * /*cVal*/, const int32 &iVal, const complex128_ &dVal, int & /*len*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    return static_cast<float64>(iVal);
                
                case ito::ParamBase::Complex & ito::paramTypeMask:
                case ito::ParamBase::Double & ito::paramTypeMask:
                    return dVal.real;

                default:
                    throw std::logic_error("Param::getVal<float64>: Non-matching type!");
            }
        }
    };

    template<>
    struct ItomParamHelper<int32>
    {
        static ito::RetVal setVal(uint32 type, char *&/*cVal*/, int32 &iVal, complex128_ &dVal, int32 val, int /*len = 0*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Complex & ito::paramTypeMask:
                case ito::ParamBase::Double & ito::paramTypeMask:
                    dVal.real = static_cast<float64>(val);
                    dVal.imag = 0.0;
                    return ito::retOk;

                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    iVal = val;
                    return ito::retOk;

                default:
                    return ito::RetVal(ito::retError, 0, "int32 value passed to setVal<int32> does not match the type of the parameter");
            }
        }

        static int32 getVal(const uint32 type, const char * /*cVal*/, const int32 &iVal, const complex128_ &dVal, int & /*len*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    return iVal;

                case ito::ParamBase::Complex & ito::paramTypeMask:
                case ito::ParamBase::Double & ito::paramTypeMask:
                    return static_cast<int32>(dVal.real);

                case 0:
                    throw std::invalid_argument("Param::getVal<int32>: non existent parameter");

                default:
                    throw std::logic_error("Param::getVal<int32>: Non-matching type!");
            }
        }
    };

    template<>
    struct ItomParamHelper<char>
    {
        static ito::RetVal setVal(uint32 type, char *&/*cVal*/, int32 &iVal, complex128_ &dVal, char val, int /*len = 0*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Complex & ito::paramTypeMask:
                case ito::ParamBase::Double & ito::paramTypeMask:
                    dVal.real = static_cast<float64>(val);
                    dVal.imag = 0.0;
                    return ito::retOk;

                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    iVal = static_cast<char>(val);
                    return ito::retOk;

                default:
                    return ito::RetVal(ito::retError, 0, "char value passed to setVal<char> does not match the type of the parameter");
            }
        }

        static char getVal(const uint32 type, const char * /*cVal*/, const int32 &iVal, const complex128_ &dVal, int & /*len*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    return static_cast<char>(iVal);

                case ito::ParamBase::Complex & ito::paramTypeMask:
                case ito::ParamBase::Double & ito::paramTypeMask:
                    return static_cast<char>(dVal.real);

                case 0:
                    throw std::invalid_argument("Param::getVal<char>: non existent parameter");

                default:
                    throw std::logic_error("Param::getVal<char>: Non-matching type!");
            }
        }
    };

    template<>
    struct ItomParamHelper<unsigned char>
    {
        static ito::RetVal setVal(uint32 type, char *&/*cVal*/, int32 &iVal, complex128_ &dVal, char val, int /*len = 0*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Complex & ito::paramTypeMask:
                case ito::ParamBase::Double & ito::paramTypeMask:
                    dVal.real = static_cast<float64>(val);
                    dVal.imag = 0.0;
                    return ito::retOk;

                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    iVal = static_cast<unsigned char>(val);
                    return ito::retOk;

                default:
                    return ito::RetVal(ito::retError, 0, "unsigned char value passed to setVal<unsigned char> does not match the type of the parameter");
            }
        }

        static char getVal(const uint32 type, const char * /*cVal*/, const int32 &iVal, const complex128_ &dVal, int & /*len*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    return static_cast<unsigned char>(iVal);

                case ito::ParamBase::Complex & ito::paramTypeMask:
                case ito::ParamBase::Double & ito::paramTypeMask:
                    return static_cast<unsigned char>(dVal.real);

                case 0:
                    throw std::invalid_argument("Param::getVal<uchar>: non existent parameter");

                default:
                    throw std::logic_error("Param::getVal<uchar>: Non-matching type!");
            }
        }
    };

    template<>
    struct ItomParamHelper<complex128>
    {
        static ito::RetVal setVal(uint32 type, char *&/*cVal*/, int32 &iVal, complex128_ &dVal, complex128 val, int /*len = 0*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Complex & ito::paramTypeMask:
                    dVal.real = val.real();
                    dVal.imag = val.imag();
                    return ito::retOk;

                default:
                    return ito::RetVal(ito::retError, 0, "complex128 value passed to setVal<complex128> does not match the type of the parameter");
            }
        }

        static complex128 getVal(const uint32 type, const char * /*cVal*/, const int32 &iVal, const complex128_ &dVal, int & /*len*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    return complex128(iVal, 0.0);
                
                case ito::ParamBase::Complex & ito::paramTypeMask:
                    return complex128(dVal.real, dVal.imag);

                case ito::ParamBase::Double & ito::paramTypeMask:
                    return dVal.real;

                default:
                    throw std::logic_error("Param::getVal<complex128>: Non-matching type!");
            }
        }
    };

} //end namespace ito

#endif
