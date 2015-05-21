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

#include <limits>


/* definition and macros */
/* global variables (avoid) */
/* content */

namespace ito
{
    /*!
    \class ParamMeta
    \brief Base class for all meta-information classes

    Parameters of type ito::Param can have a pointer to this class. Consider this base class to be abstract, such that
    it is only allowed to pass the right implementation (derived from this class) that fits to the type of the parameter.
    The runtime type information value m_type indicates the real type of this pointer, such that a direct cast
    can be executed.

    \sa ito::CharMeta, ito::IntMeta, ito::DoubleMeta, ito::StringMeta, ito::HWMeta, ito::DObjMeta, ito::CharArrayMeta, ito::IntArrayMeta, ito::DoubleArrayMeta
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
            rttiUnknown = 0,      /*!< unknown parameter */ 
            rttiCharMeta = 1,     /*!< meta for a char parameter */ 
            rttiIntMeta = 2,      /*!< meta for an integer parameter */ 
            rttiDoubleMeta = 3,   /*!< meta for a double parameter */ 
            rttiStringMeta = 4,   /*!< meta for a string parameter */
            rttiHWMeta = 5,       /*!< meta for a hardware plugin parameter */
            rttiDObjMeta = 6,     /*!< meta for a data object parameter */
            rttiIntArrayMeta = 7, /*!< meta for an integer array parameter */
            rttiDoubleArrayMeta = 8, /*!< meta for a double array parameter */
            rttiCharArrayMeta = 9, /*!< meta for a char array parameter */
            rttiIntervalMeta = 10, /*!< meta for an integer array with two values that represent an interval [value1, value2] parameter */
            rttiDoubleIntervalMeta = 11, /*!< meta for a double array with two values that represent an interval [value1, value2] parameter (size of the interval is value2-value1) */
            rttiRangeMeta = 12,    /*!< meta for an integer array with two values that represent a range [value1, value2] parameter (size of a range is 1+value2-value1) */
            rttiRectMeta = 13      /*!< meta for an integer array with four values that consists of two ranges (vertical and horizontal, e.g. for ROIs of cameras) */
        };

        ParamMeta() : m_type(rttiUnknown) {}               //!< default constructor with an unknown meta information type
        ParamMeta(MetaRtti type) : m_type(type) {}         //!< constructor used by derived classes to indicate their real type
        virtual ~ParamMeta() {}                            //!< destructor
        inline MetaRtti getType() const { return m_type; } //!< returns runtime type information value
    protected:
        MetaRtti m_type;
    };

    /*!
    \class CharMeta
    \brief meta-information for Param of type Char.
    
    An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::Char. If set, the given char number
    can be limited with respect to given minimum and maximum values as well as an optional step size (default: 1).

    \sa ito::Param, ito::ParamMeta
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
    \brief Meta-information for Param of type Int.

    An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::Int. If set, the given integer number
    can be limited with respect to given minimum and maximum values as well as an optional step size (default: 1).

    \sa ito::Param, ito::ParamMeta
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
    \brief Meta-information for ito::Param of type Double.
    
    An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::Double. If set, the given double number
    can be limited with respect to given minimum and maximum values as well as an optional step size (default: 0.0 -> no step size).

    \sa ito::Param, ito::ParamMeta
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
    
    An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::HWPtr, that is an instance of another hardware plugin. 
    If set, it is possible to restrict the given hardware plugin to a specific type (e.g. dataIO, dataIO + grabber, actuator...) and/or to limit it
    to a specific name of the plugin (e.g. SerialIO).

    \sa ito::Param, ito::ParamMeta
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

            inline uint32 getMinType() const { return m_minType; }             //!< returns type-bitmask which is minimally required by plugin-reference. Default 0. \sa ito::tPluginType
            inline ito::ByteArray getHWAddInName() const { return m_HWName; }  //!< returns name of specific hardware plugin
        private:
            uint32 m_minType;            //!< type-bitmask which is minimally required. default: 0
            ito::ByteArray m_HWName;     //!< zero-terminated name of specific plugin-name or invalid if not defined
    };

    /*!
    \class StringMeta
    \brief Meta-information for Param of type String.
    
    An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::String. 
    If set, it is possible to restrict the a given string to fit to a given list of strings. This list of strings
    might be interpreted in an exact way (tType::String), as wildcard expressions (tType::Wildcard) or as regular expressions (tType::RegExp).

    \sa ito::Param, ito::ParamMeta
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
            bool addItem(const char *val);                              //!< adds another element to the list of patterns.
            StringMeta & operator += (const char *val);                 //!< add another pattern string to the list of patterns.

        private:
            tType m_stringType;
            int m_len;
            char **m_val;
    };

    /*!
    \class DObjMeta
    \brief Meta-information for Param of type DObjPtr.

    (not used yet)

    \sa ito::Param, ito::ParamMeta
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
    
    An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::CharArray. 
    Since this meta information class is derived from ito::CharMeta, it is possible to restrict each value to the single value contraints of ito::CharMeta.
    Furthermore, this class allows restricting the minimum and maximum length of the array as well as the optional step size of the array's length.

    \sa ito::Param, ito::ParamMeta, ito::CharMeta
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
    \brief Meta-information for Param of type IntArrayMeta.
    
    An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::IntArray. 
    Since this meta information class is derived from ito::IntMeta, it is possible to restrict each value to the single value contraints of ito::IntMeta.
    Furthermore, this class allows restricting the minimum and maximum length of the array as well as the optional step size of the array's length.

    \sa ito::Param, ito::ParamMeta, ito::IntArray
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
    
    An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::DoubleArray. 
    Since this meta information class is derived from ito::DoubleArray, it is possible to restrict each value to the single value contraints of ito::DoubleArray.
    Furthermore, this class allows restricting the minimum and maximum length of the array as well as the optional step size of the array's length.

    \sa ito::Param, ito::ParamMeta, ito::DoubleMeta
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
    \class DoubleIntervalMeta
    \brief Meta-information for Param of type DoubleIntervalMeta.
    
    An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::DoubleArray. 
    This meta information class indicates that the corresponding double array parameter is interpreted as an interval, hence, only an array
    consisting of two values is accepted. The size of the interval is defined by the difference (value[1] - value[0]). You can restrict this
    size to a certain minimum and maximum value as well as indicating a specific step size (default: 0.0 -> no step size).

    \sa ito::Param, ito::ParamMeta, ito::DoubleMeta
    */
    class ITOMCOMMON_EXPORT DoubleIntervalMeta : public DoubleMeta
    {
    public:
        explicit DoubleIntervalMeta(double minVal, double maxVal, double stepSize = 0.0);
        explicit DoubleIntervalMeta(double minVal, double maxVal, double stepSize, double sizeMin, double sizeMax, double sizeStep = 0.0);
        inline double getSizeMin() const { return m_sizeMin; }         //!< returns minimum size of range
        inline double getSizeMax() const { return m_sizeMax; }         //!< returns maximum size of range
        inline double getSizeStepSize() const { return m_sizeStep; }   //!< returns step size of size of range

        //! sets the minimum size of the interval (= max-min)
        /*!
            \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is changed to val, too
        */
        void setSizeMin(double val);
        
        //! sets the maximum size of the interval (= max-min)
        /*!
            \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is changed to val, too
        */
        void setSizeMax(double val);

        //! sets the step size of the size of the interval (= max-min)
        /*!
            \param val is the new step size, hence only discrete values [minVal, minVal+stepSize, minVal+2*stepSize...,maxVal] are allowed
        */
        void setSizeStep(double val);

    private:
        double m_sizeMin;
        double m_sizeMax;
        double m_sizeStep;
    };


    /*!
    \class IntervalMeta
    \brief Meta-information for Param of type IntArrayMeta that represent an interval [minimum, maximum).
    
    An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::IntArray. 
    This meta information class indicates that the corresponding integer array parameter is interpreted as an interval, hence, only an array
    consisting of two values is accepted. The size of the interval is defined by the difference (value[1] - value[0]). You can restrict this
    size to a certain minimum and maximum value as well as indicating a specific step size (default: 1).

    An example for an interval might be a certain interval of allowed radius values when optimizing a cylinder fit.

    \sa ito::Param, ito::ParamMeta, ito::RangeMeta, ito::IntMeta, ito::IntervalMeta
    */
    class ITOMCOMMON_EXPORT IntervalMeta : public IntMeta
    {
    public:
        explicit IntervalMeta(int minVal, int maxVal, int stepSize = 1);
        explicit IntervalMeta(int minVal, int maxVal, int stepSize, int sizeMin, int sizeMax, int intervalStep = 1);
        inline int getSizeMin() const { return m_sizeMin; }         //!< returns minimum size of interval or range
        inline int getSizeMax() const { return m_sizeMax; }         //!< returns maximum size of interval or range
        inline int getSizeStepSize() const { return m_sizeStep; }   //!< returns step size of size of interval or range
        inline bool isIntervalNotRange() const { return m_isIntervalNotRange; }

        //! sets the minimum size of the interval (= max-min)
        /*!
            \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is changed to val, too
        */
        void setIntervalMin(int val);
        
        //! sets the maximum size of the interval (= max-min)
        /*!
            \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is changed to val, too
        */
        void setIntervalMax(int val);

        //! sets the step size of the size of the interval (= max-min)
        /*!
            \param val is the new step size, hence only discrete values [minVal, minVal+stepSize, minVal+2*stepSize...,maxVal] are allowed
        */
        void setIntervalStep(int val);

    protected:
        int m_sizeMin;
        int m_sizeMax;
        int m_sizeStep;
        bool m_isIntervalNotRange; //!< this flag describes if this object is an interval where its interval/range is (end-begin) or a range with (1+end-begin)
    };


    /*!
    \class RangeMeta
    \brief Meta-information for Param of type IntArrayMeta that represent a range [minVal, maxVal].
    
    An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::IntArray. 
    This meta information class indicates that the corresponding integer array parameter is interpreted as a range, hence, only an array
    consisting of two values is accepted. The size of the interval is defined by the difference (1 + value[1] - value[0]). You can restrict this
    size to a certain minimum and maximum value as well as indicating a specific step size (default: 1).

    An example for a range might be a one dimension (vertical or horizontal) of a ROI (region of interest) of a camera, where the range
    determines the first and last pixel value inside of the ROI, such that the total size is the difference between both limits + 1.

    The range object is defined by its first and last value, that are both inside of the range, hence the size of the range is (1+last-first).
    This is the difference to IntervalMeta, where the size of the interval is last-first only.

    \sa ito::Param, ito::ParamMeta, ito::IntervalMeta, ito::IntArrayMeta
    */
    class ITOMCOMMON_EXPORT RangeMeta : public IntervalMeta
    {
    public:
        explicit RangeMeta(int minVal, int maxVal, int stepSize = 1);
        explicit RangeMeta(int minVal, int maxVal, int stepSize, int sizeMin, int sizeMax, int sizeStep = 1);
    };


    /*!
    \class RectMeta
    \brief Meta-information for Param of type IntArrayMeta that represent a rectangle (left, top, width, height).
    
    An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::IntArray. 
    This meta information class indicates that the corresponding integer array parameter is interpreted as a rectangle, hence, only an array
    consisting of four values is accepted. This meta information consists of two object of type ito::RangeMeta, describing the
    contraints of the horizontal and vertical axes of the rectangle.

    \sa ito::Param, ito::ParamMeta, ito::RangeMeta, ito::IntArrayMeta
    */
    class ITOMCOMMON_EXPORT RectMeta : public ParamMeta
    {
    public:
        explicit RectMeta(const ito::RangeMeta &widthMeta, const ito::RangeMeta &heightMeta);
        inline const ito::RangeMeta& getWidthRangeMeta() const { return m_widthMeta; }
        inline const ito::RangeMeta& getHeightRangeMeta() const { return m_heightMeta; }

        void setWidthRangeMeta(const ito::RangeMeta &widthMeta);
        void setHeightRangeMeta(const ito::RangeMeta &heightMeta);

    protected:
        ito::RangeMeta m_heightMeta;
        ito::RangeMeta m_widthMeta;
    };



} //end namespace ito

#endif
