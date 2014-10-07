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

#ifndef PARAM_H
#define PARAM_H

/* includes */

#include "commonGlobal.h"
#include "typeDefs.h"
#include "byteArray.h"
#include "retVal.h"

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


    class ITOMCOMMON_EXPORT ParamBase
    {
    protected:
        uint32 m_type;
        ByteArray m_name;      //!< parameter name

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
        ParamBase() : m_type(0), m_name(NULL), m_dVal(0.0), m_iVal(0), m_cVal(NULL) {}
        ParamBase(const char *name);                                                               // type-less ParamBase with name only
        ParamBase(const char *name, const uint32 type);                                               // constructor with type and name
        ParamBase(const char *name, const uint32 type, const char *val);                              // constructor with name and type, char val
        ParamBase(const char *name, const uint32 type, const double val);                             // constructor with name and type, double val and optional info
        ParamBase(const char *name, const uint32 type, const int val);                                // constructor with name and type, int val and optional info
        ParamBase(const ByteArray &name, const uint32 type, const char *val);                              // constructor with name and type, char val
        ParamBase(const ByteArray &name, const uint32 type, const double val);                             // constructor with name and type, double val and optional info
        ParamBase(const ByteArray &name, const uint32 type, const int val);                                // constructor with name and type, int val and optional info
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
            static int numericTypeMask = ito::ParamBase::Char | ParamBase::Int | ParamBase::Double;
            int type = getType();
            return (type & numericTypeMask) && !(type & ito::ParamBase::Pointer);
        }

        inline ito::RetVal addNameSuffix(const char *suffix) 
        { 
            if (suffix)  
            { 
                m_name.append(suffix);
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
        inline const char * getName(void) const { return m_name.data(); }
        //! returns content of autosave flag - this flag determines whether the parameter value gets automagically saved to xml file
        //! when an instance of a plugin class is deleted (closed)
        inline uint32 getAutosave(void) const { return ((m_type & NoAutosave) > 0 ? 0 : 1); }
        //! sets content of autosave flag - this flag determines whether the parameter value gets automagically saved to xml file
        //! when an instance of a plugin class is deleted (closed)
        inline void setAutosave(const uint32 autosave) { m_type = autosave > 0 ? m_type & ~NoAutosave : m_type | NoAutosave; return; }

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

    /*!
    \class ParamMeta
    \brief Base class for all meta-information classes
    \sa IntMeta, DoubleMeta, CharMeta, StringMeta, HWMeta, DObjMeta
    */
    class ITOMCOMMON_EXPORT ParamMeta
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
    class ITOMCOMMON_EXPORT CharMeta : public ParamMeta
    {
    public:
        //! constructor with minimum and maximum value
        explicit CharMeta(char minVal, char maxVal, char stepSize=1) : ParamMeta(ParamBase::Char), m_minVal(minVal), m_maxVal(maxVal), m_stepSize(stepSize) { if(m_maxVal < m_minVal) std::swap(m_minVal,m_maxVal); }
        static CharMeta* all() { return new CharMeta(std::numeric_limits<char>::min(), std::numeric_limits<char>::max() ); } //!< returns a new instance of CharMeta, where the min and max are set to the full range available for char.
        inline char getMin() const { return m_minVal; } //!< returns minimum value
        inline char getMax() const { return m_maxVal; } //!< returns maximum value
        inline char getStepSize() const { return m_stepSize; } //!< returns step size

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

        //! sets the step size
        /*!
            \param val is the new step size, hence only discrete values [minVal, minVal+stepSize, minVal+2*stepSize...,maxVal] are allowed
        */
        inline void setStepSize(char val) { m_stepSize = val; }
    private:
        char m_minVal;
        char m_maxVal;
        char m_stepSize; // >= 1
    };

    /*!
    \class IntMeta
    \brief Meta-information for Param of type Int or IntArray.
    \sa ito::Param
    */
    class ITOMCOMMON_EXPORT IntMeta : public ParamMeta
    {
    public:
        //! constructor with minimum and maximum value
        explicit IntMeta(int minVal, int maxVal, int stepSize=1) : ParamMeta(ParamBase::Int), m_minVal(minVal), m_maxVal(maxVal), m_stepSize(stepSize) { if(m_maxVal < m_minVal) std::swap(m_minVal,m_maxVal); }
        static IntMeta* all() { return new IntMeta(std::numeric_limits<int>::min(), std::numeric_limits<int>::max() ); } //!< returns a new instance of IntMeta, where the min and max are set to the full range available for integers.
        inline int getMin() const { return m_minVal; } //!< returns minimum value
        inline int getMax() const { return m_maxVal; } //!< returns maximum value
        inline int getStepSize() const { return m_stepSize; } //!< returns step size
        
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

        //! sets the step size
        /*!
            \param val is the new step size, hence only discrete values [minVal, minVal+stepSize, minVal+2*stepSize...,maxVal] are allowed
        */
        inline void setStepSize(int val) { m_stepSize = val; }
    private:
        int m_minVal;
        int m_maxVal;
        int m_stepSize; // >= 1
    };

    /*!
    \class DoubleMeta
    \brief Meta-information for Param of type Double or DoubleArray.
    \sa ito::Param
    */
    class ITOMCOMMON_EXPORT DoubleMeta : public ParamMeta
    {
    public:
        //! constructor with minimum and maximum value
        explicit DoubleMeta(double minVal, double maxVal, double stepSize=0.0 /*0.0 means no specific step size*/) : ParamMeta(ParamBase::Double), m_minVal(minVal), m_maxVal(maxVal), m_stepSize(stepSize) { if(m_maxVal < m_minVal) std::swap(m_minVal,m_maxVal); }
        static DoubleMeta* all() { return new DoubleMeta(-std::numeric_limits<double>::max(), std::numeric_limits<double>::max() ); } //!< returns a new instance of DoubleMeta, where the min and max are set to the full range available for double.
        inline double getMin() const { return m_minVal; } //!< returns minimum value
        inline double getMax() const { return m_maxVal; } //!< returns maximum value
        inline double getStepSize() const { return m_stepSize; } //!< returns step size
        
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

        //! sets the step size
        /*!
            \param val is the new step size, hence only discrete values [minVal, minVal+stepSize, minVal+2*stepSize...,maxVal] are allowed
        */
        inline void setStepSize(double val) { m_stepSize = val; }
    private:
        double m_minVal;
        double m_maxVal;
        double m_stepSize; // >= 0, 0.0 means no specific step size
    };

    /*!
    \class HWMeta
    \brief Meta-information for Param of type HWPtr.
    \sa ito::Param
    */
    class ITOMCOMMON_EXPORT HWMeta : public ParamMeta
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
    class ITOMCOMMON_EXPORT StringMeta : public ParamMeta
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
            bool addItem(const char *val) //!< adds another element to the string list.
            {
                if(m_val)
                {
                    char **m_val_old = m_val;
					m_val = (char**)realloc(m_val, sizeof(char*) * (++m_len) ); //m_val can change its address. if NULL, reallocation failed and m_val_old still contains old values
					if (!m_val)
					{
                        m_val = m_val_old;
						m_len--; //failed to add new value
                        return false;
					}
                }
                else
                {
                    m_val = (char**) calloc(++m_len, sizeof(char*));
                }
                m_val[m_len-1] = _strdup(val);
                return true;
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
    class ITOMCOMMON_EXPORT DObjMeta : public ParamMeta
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
            Param(const char *name) : ParamBase(name), m_pMeta(NULL), m_info(NULL) {}                         // type-less Param with name only
            Param(const char *name, const uint32 type) : ParamBase(name, type), m_pMeta(NULL), m_info(NULL) {}   // constructor with type and name
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
            inline const char * getInfo(void) const { return m_info.data(); }

            //!< sets content of info string, if necessary the info buffer is freed first, passed string is copied
            inline void setInfo(const char *info)
            {
                m_info = info;
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

                default:
                    return ito::retError;
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

                case (ito::ParamBase::HWRef & paramTypeMask):
                case (ito::ParamBase::DObjPtr & paramTypeMask):
                case ito::ParamBase::PointCloudPtr & paramTypeMask:
                case ito::ParamBase::PointPtr & paramTypeMask:
                case ito::ParamBase::PolygonMeshPtr & paramTypeMask:
                    return reinterpret_cast<_Tp>(const_cast<char*>(cVal));

                default:
                    throw std::logic_error("Param::getVal<_Tp>: Non-matching type!");
                    return (_Tp)0;
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

                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    iVal = static_cast<int>(val);
                    return ito::retOk;

                default:
                    return ito::retError;
            }
        }

        static double getVal(const uint32 type, const char * /*cVal*/, const int &iVal, const double &dVal, int & /*len*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    return static_cast<double>(iVal);

                case ito::ParamBase::Double & ito::paramTypeMask:
                    return dVal;

                default:
                    throw std::logic_error("Param::getVal<double>: Non-matching type!");
                    return 0;
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

                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    iVal = val;
                    return ito::retOk;

                default:
                    return ito::retError;
            }
        }

        static int getVal(const uint32 type, const char * /*cVal*/, const int &iVal, const double &dVal, int & /*len*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    return iVal;

                case ito::ParamBase::Double & ito::paramTypeMask:
                    return static_cast<int>(dVal);

                case 0:
                    throw std::invalid_argument("Param::getVal<int>: non existent parameter");

                default:
                    throw std::logic_error("Param::getVal<int>: Non-matching type!");
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

                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    iVal = static_cast<char>(val);
                    return ito::retOk;

                default:
                    return ito::retError;
            }
        }

        static char getVal(const uint32 type, const char * /*cVal*/, const int &iVal, const double &dVal, int & /*len*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    return static_cast<char>(iVal);

                case ito::ParamBase::Double & ito::paramTypeMask:
                    return static_cast<char>(dVal);

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
        static ito::RetVal setVal(uint32 type, char *&/*cVal*/, int &iVal, double &dVal, char val, int /*len = 0*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Double & ito::paramTypeMask:
                    dVal = static_cast<unsigned char>(val);
                    return ito::retOk;

                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    iVal = static_cast<unsigned char>(val);
                    return ito::retOk;

                default:
                    return ito::retError;
            }
        }

        static char getVal(const uint32 type, const char * /*cVal*/, const int &iVal, const double &dVal, int & /*len*/)
        {
            switch (type & paramTypeMask)
            {
                case ito::ParamBase::Int & ito::paramTypeMask:
                case ito::ParamBase::Char & ito::paramTypeMask:
                    return static_cast<unsigned char>(iVal);

                case ito::ParamBase::Double & ito::paramTypeMask:
                    return static_cast<unsigned char>(dVal);

                case 0:
                    throw std::invalid_argument("Param::getVal<uchar>: non existent parameter");
                    return 0;

                default:
                    throw std::logic_error("Param::getVal<uchar>: Non-matching type!");
                    return 0;
            }
        }
    };

} //end namespace ito

#endif
