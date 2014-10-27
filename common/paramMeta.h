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

#ifndef PARAMMETA_H
#define PARAMMETA_H

/* includes */

#include "commonGlobal.h"
#include "typeDefs.h"
#include "byteArray.h"
#include "retVal.h"


/* definition and macros */
/* global variables (avoid) */
/* content */

namespace ito
{
    /*!
    \class ParamMeta
    \brief Base class for all meta-information classes
    \sa IntMeta, DoubleMeta, CharMeta, StringMeta, HWMeta, DObjMeta
    */
    class ITOMCOMMON_EXPORT ParamMeta
    {
    public:
        /*!
            \brief Runtime type information

            MetaRtti is used to cast param meta objects, without
            having to enable runtime type information of the compiler.
        */
        enum MetaRtti
        {
            rttiUnknown = 0,
            rttiCharMeta = 1,
            rttiIntMeta = 2,
            rttiDoubleMeta = 3,
            rttiStringMeta = 4,
            rttiHWMeta = 5,
            rttiDObjMeta = 6,
            rttiIntArrayMeta = 7,
            rttiDoubleArrayMeta = 8,
            rttiCharArrayMeta = 9,
            rttiRangeMeta = 10,
            rttiDoubleRangeMeta = 11,
            rttiRectMeta = 12
        };

        ParamMeta() : m_type(rttiUnknown) {}
        ParamMeta(MetaRtti type) : m_type(type) {}
        virtual ~ParamMeta() {}
        inline MetaRtti getType_() const { return m_type; }
    protected:
        MetaRtti m_type;
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
        explicit CharMeta(char minVal, char maxVal, char stepSize = 1); //!< constructor with minimum and maximum value as well as optional step size (default: 1)
        static CharMeta* all();                                 //!< returns a new instance of CharMeta, where the min and max are set to the full range available for char.
        inline char getMin() const { return m_minVal; }         //!< returns minimum value
        inline char getMax() const { return m_maxVal; }         //!< returns maximum value
        inline char getStepSize() const { return m_stepSize; }  //!< returns step size

        //! sets the minimum value
        /*!
            \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is changed to val, too
        */
        void setMin(char val);
        
        //! sets the maximum value
        /*!
            \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is changed to val, too
        */
        void setMax(char val);

        //! sets the step size
        /*!
            \param val is the new step size, hence only discrete values [minVal, minVal+stepSize, minVal+2*stepSize...,maxVal] are allowed
        */
        void setStepSize(char val);
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
        explicit IntMeta(int minVal, int maxVal, int stepSize = 1); //!< constructor with minimum and maximum value as well as optional step size (default: 1)
        static IntMeta* all();                                      //!< returns a new instance of IntMeta, where the min and max are set to the full range available for integers.
        inline int getMin() const { return m_minVal; }              //!< returns minimum value
        inline int getMax() const { return m_maxVal; }              //!< returns maximum value
        inline int getStepSize() const { return m_stepSize; }       //!< returns step size
        
        //! sets the minimum value
        /*!
            \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is changed to val, too
        */
        void setMin(int val);
        
        //! sets the maximum value
        /*!
            \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is changed to val, too
        */
        void setMax(int val);

        //! sets the step size
        /*!
            \param val is the new step size, hence only discrete values [minVal, minVal+stepSize, minVal+2*stepSize...,maxVal] are allowed
        */
        void setStepSize(int val);
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
        explicit DoubleMeta(double minVal, double maxVal, double stepSize = 0.0 /*0.0 means no specific step size*/);
        static DoubleMeta* all();                                //!< returns a new instance of DoubleMeta, where the min and max are set to the full range available for double.
        inline double getMin() const { return m_minVal; }        //!< returns minimum value
        inline double getMax() const { return m_maxVal; }        //!< returns maximum value
        inline double getStepSize() const { return m_stepSize; } //!< returns step size
        
        //! sets the minimum value
        /*!
            \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is changed to val, too
        */
        void setMin(double val);
        
        //! sets the maximum value
        /*!
            \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is changed to val, too
        */
        void setMax(double val);

        //! sets the step size
        /*!
            \param val is the new step size, hence only discrete values [minVal, minVal+stepSize, minVal+2*stepSize...,maxVal] are allowed
        */
        void setStepSize(double val);
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
            explicit HWMeta(uint32 minType) : ParamMeta(rttiHWMeta), m_minType(minType) 
            {
            }

            //! constructor
            /*!
                creates HWMeta-information struct where you can pass a specific name of a plugin, which only is
                allowed by the corresponding plugin-instance.
                \sa ito::Plugin
            */
            explicit HWMeta(const char *HWAddInName) : ParamMeta(rttiHWMeta), m_minType(0), m_HWName(HWAddInName)
            {
            }

            HWMeta(const HWMeta& cpy) : ParamMeta(rttiHWMeta), m_minType(cpy.m_minType), m_HWName(cpy.m_HWName)
            {
            }

            inline uint32 getMinType() const { return m_minType; }                //!< returns type-bitmask which is minimally required by plugin-reference. Default 0. \sa ito::tPluginType
            inline ito::ByteArray getHWAddInName() const { return m_HWName; }  //!< returns name of specific hardware plugin
        private:
            uint32 m_minType;            //!< type-bitmask which is minimally required. default: 0
            ito::ByteArray m_HWName;     //!< zero-terminated name of specific plugin-name or invalid if not defined
    };

    /*!
    \class StringMeta
    \brief Meta-information for Param of type String.
    \sa ito::Param
    */
    class ITOMCOMMON_EXPORT StringMeta : public ParamMeta
    {
        public:
            enum tType 
            {
                String,   //!< string elements should be considered as strings (exact match)
                Wildcard, //!< string elements should be considered as wildcard-expressions (e.g. *.doc)
                RegExp    //!< string elements should be considered as regular expressions (e.g. ^(.*)[abc]{1,5}$)
            };

            //! constructor
            /*!
                Returns a meta information class for string-types.
                \param type indicates how the string elements should be considered
                \sa tType
            */
            StringMeta(tType type);

            //! constructor
            /*!
                Returns a meta information class for string-types.
                \param type indicates how the string elements should be considered
                \param val adds a first string to the element list
                \sa tType
            */
            StringMeta(tType type, const char* val);

            //! copy constructor
            StringMeta(const StringMeta& cpy);

            //! destructor
            virtual ~StringMeta();

            inline tType getStringType() const { return m_stringType; } //!< returns the type how strings in list should be considered. \sa tType
            inline int getLen() const { return m_len; }                 //!< returns the number of string elements in meta information class.
            const char* getString(int idx = 0) const;                   //!< returns string from list at index position or NULL, if index is out of range.
            bool addItem(const char *val);                              //!< adds another element to the string list.
            StringMeta & operator += (const char *val);

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
            explicit DObjMeta(uint32 allowedTypes = 0xFFFF, int minDim = 0, int maxDim = std::numeric_limits<int>::max()) : ParamMeta(rttiDObjMeta), m_allowedTypes(allowedTypes), m_minDim(minDim), m_maxDim(maxDim) {}
            inline int getAllowedTypes() const { return m_allowedTypes; }
            inline int getMinDim() const { return m_minDim; } //!< returns maximum allowed dimensions of data object
            inline int getMaxDim() const { return m_maxDim; } //!< returns minimum number of dimensions of data object

        private:
            uint32 m_allowedTypes;
            int m_minDim;
            int m_maxDim;
    };

    /*!
    \class CharArrayMeta
    \brief Meta-information for Param of type CharArrayMeta.
    \sa ito::Param
    */
    class ITOMCOMMON_EXPORT CharArrayMeta : public CharMeta
    {
    public:
        explicit CharArrayMeta(char minVal, char maxVal, char stepSize = 1);
        explicit CharArrayMeta(char minVal, char maxVal, char stepSize, size_t numMin, size_t numMax, size_t numStepSize = 1);
        inline size_t getNumMin() const { return m_numMin; }         //!< returns minimum number of values
        inline size_t getNumMax() const { return m_numMax; }         //!< returns maximum number of values
        inline size_t getNumStepSize() const { return m_numStep; }   //!< returns step size of number of values

        //! sets the minimum number of values
        /*!
            \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is changed to val, too
        */
        void setNumMin(size_t val);
        
        //! sets the maximum number of values
        /*!
            \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is changed to val, too
        */
        void setNumMax(size_t val);

        //! sets the step size of the number of values
        /*!
            \param val is the new step size, hence only discrete values [minVal, minVal+stepSize, minVal+2*stepSize...,maxVal] are allowed
        */
        void setNumStepSize(size_t val);

    private:
        size_t m_numMin;
        size_t m_numMax;
        size_t m_numStep;
    };

    /*!
    \class CharArrayMeta
    \brief Meta-information for Param of type CharArrayMeta.
    \sa ito::Param
    */
    class ITOMCOMMON_EXPORT IntArrayMeta : public IntMeta
    {
    public:
        explicit IntArrayMeta(int minVal, int maxVal, int stepSize = 1);
        explicit IntArrayMeta(int minVal, int maxVal, int stepSize, size_t numMin, size_t numMax, size_t numStepSize = 1);
        inline size_t getNumMin() const { return m_numMin; }         //!< returns minimum number of values
        inline size_t getNumMax() const { return m_numMax; }         //!< returns maximum number of values
        inline size_t getNumStepSize() const { return m_numStep; }   //!< returns step size of number of values

        //! sets the minimum number of values
        /*!
            \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is changed to val, too
        */
        void setNumMin(size_t val);
        
        //! sets the maximum number of values
        /*!
            \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is changed to val, too
        */
        void setNumMax(size_t val);

        //! sets the step size of the number of values
        /*!
            \param val is the new step size, hence only discrete values [minVal, minVal+stepSize, minVal+2*stepSize...,maxVal] are allowed
        */
        void setNumStepSize(size_t val);

    private:
        size_t m_numMin;
        size_t m_numMax;
        size_t m_numStep;
    };

    /*!
    \class DoubleArrayMeta
    \brief Meta-information for Param of type DoubleArrayMeta.
    \sa ito::Param
    */
    class ITOMCOMMON_EXPORT DoubleArrayMeta : public DoubleMeta
    {
    public:
        explicit DoubleArrayMeta(double minVal, double maxVal, double stepSize = 0.0);
        explicit DoubleArrayMeta(double minVal, double maxVal, double stepSize, size_t numMin, size_t numMax, size_t numStepSize = 1);
        inline size_t getNumMin() const { return m_numMin; }         //!< returns minimum number of values
        inline size_t getNumMax() const { return m_numMax; }         //!< returns maximum number of values
        inline size_t getNumStepSize() const { return m_numStep; }   //!< returns step size of number of values

        //! sets the minimum number of values
        /*!
            \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is changed to val, too
        */
        void setNumMin(size_t val);
        
        //! sets the maximum number of values
        /*!
            \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is changed to val, too
        */
        void setNumMax(size_t val);

        //! sets the step size of the number of values
        /*!
            \param val is the new step size, hence only discrete values [minVal, minVal+stepSize, minVal+2*stepSize...,maxVal] are allowed
        */
        void setNumStepSize(size_t val);

    private:
        size_t m_numMin;
        size_t m_numMax;
        size_t m_numStep;
    };


    /*!
    \class RangeMeta
    \brief Meta-information for Param of type RangeMeta.
    \sa ito::Param
    */
    class ITOMCOMMON_EXPORT RangeMeta : public IntMeta
    {
    public:
        explicit RangeMeta(int minVal, int maxVal, int stepSize = 1);
        explicit RangeMeta(int minVal, int maxVal, int stepSize, int rangeMin, int rangeMax, int rangeStepSize = 1);
        inline int getRangeMin() const { return m_rangeMin; }         //!< returns minimum size of range
        inline int getRangeMax() const { return m_rangeMax; }         //!< returns maximum size of range
        inline int getRangeStepSize() const { return m_rangeStep; }   //!< returns step size of size of range

        //! sets the minimum size of the range (= 1+max-min)
        /*!
            \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is changed to val, too
        */
        void setRangeMin(int val);
        
        //! sets the maximum size of the range (= 1+max-min)
        /*!
            \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is changed to val, too
        */
        void setRangeMax(int val);

        //! sets the step size of the size of the range (= 1+max-min)
        /*!
            \param val is the new step size, hence only discrete values [minVal, minVal+stepSize, minVal+2*stepSize...,maxVal] are allowed
        */
        void setRangeStepSize(int val);

    private:
        int m_rangeMin;
        int m_rangeMax;
        int m_rangeStep;
    };

    /*!
    \class DoubleRangeMeta
    \brief Meta-information for Param of type DoubleRangeMeta.
    \sa ito::Param
    */
    class ITOMCOMMON_EXPORT DoubleRangeMeta : public DoubleMeta
    {
    public:
        explicit DoubleRangeMeta(double minVal, double maxVal, double stepSize = 0.0);
        explicit DoubleRangeMeta(double minVal, double maxVal, double stepSize, double rangeMin, double rangeMax, double rangeStepSize = 0.0);
        inline double getRangeMin() const { return m_rangeMin; }         //!< returns minimum size of range
        inline double getRangeMax() const { return m_rangeMax; }         //!< returns maximum size of range
        inline double getRangeStepSize() const { return m_rangeStep; }   //!< returns step size of size of range

        //! sets the minimum size of the range (= 1+max-min)
        /*!
            \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is changed to val, too
        */
        void setRangeMin(double val);
        
        //! sets the maximum size of the range (= 1+max-min)
        /*!
            \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is changed to val, too
        */
        void setRangeMax(double val);

        //! sets the step size of the size of the range (= 1+max-min)
        /*!
            \param val is the new step size, hence only discrete values [minVal, minVal+stepSize, minVal+2*stepSize...,maxVal] are allowed
        */
        void setRangeStepSize(double val);

    private:
        double m_rangeMin;
        double m_rangeMax;
        double m_rangeStep;
    };

    /*!
    \class RangeMeta
    \brief Meta-information for Param of type RangeMeta.
    \sa ito::Param
    */
    class ITOMCOMMON_EXPORT RectMeta : public ParamMeta
    {
    public:
        explicit RectMeta(const ito::RangeMeta &widthMeta, const ito::RangeMeta &heightMeta);
        inline ito::RangeMeta getWidthRangeMeta() const { return m_widthMeta; }
        inline ito::RangeMeta getHeightRangeMeta() const { return m_heightMeta; }

        void setWidthRangeMeta(const ito::RangeMeta &widthMeta);
        void setHeightRangeMeta(const ito::RangeMeta &heightMeta);

    private:
        ito::RangeMeta m_heightMeta;
        ito::RangeMeta m_widthMeta;
    };



} //end namespace ito

#endif
