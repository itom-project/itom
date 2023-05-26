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

#ifndef NUMERIC_H
#define NUMERIC_H

#include "typeDefs.h"
#include "color.h"
#include <limits>

namespace ito
{
	//! method returns whether a given variable is not equal to zero.
	/*!
	For floating point variables, this method considers a variable not to be zero, if its value
	lie without the boundaries (-epsilon,epsilon). Epsilon is retrieved from
	std::numeric_limits<_Tp>::epsilon(). For floating point values only parameters of type
	float32, float64, complex64 or complex128 are treated in the desired way.

	\param value is value to check
	\return true if value is zero or within the epsilon boundaries, else false
	*/
    template<typename _Tp> inline bool isNotZero(_Tp value)
    {
        return !(value == 0);
    }

    //! Check if a value is equal to zero for float32
    template<> inline bool isNotZero<float32>(float32 value)
    {
        if (fabs(value) < std::numeric_limits<float32>::epsilon())
            return false;
        else
            return true;
    }

    //! Check if a value is equal to zero for float64
    template<> inline bool isNotZero<float64>(float64 value)
    {
        if (fabs(value) < std::numeric_limits<float64>::epsilon())
            return false;
        else
            return true;
    }

    //! Check if a value is equal to zero for complex64
    template<> inline bool isNotZero<complex64>(complex64 value)
    {
        if (fabs(value.real()) < std::numeric_limits<float32>::epsilon() &&
            fabs(value.imag()) < std::numeric_limits<float32>::epsilon())
            return false;
        else
            return true;
    }

    //! Check if a value is equal to zero for complex128
    template<> inline bool isNotZero<complex128>(complex128 value)
    {
        if (fabs(value.real()) < std::numeric_limits<float64>::epsilon() &&
            fabs(value.imag()) < std::numeric_limits<float64>::epsilon())
            return false;
        else
            return true;
    }


	//! method returns whether a given variable is finite.
	/*!
	For floating point variables, this method considers a variable to be finite if the bitmask
	is neither NaN nor Inf. For floating point values only parameters of type
	float32, float64, complex64 or complex128 are treated in the desired way.
	For integer types --> always true

	\param value is value to check
	\return true if value is neither Inf nor NaN, else false
	*/
    template<typename _Tp> inline bool isFinite(_Tp /*value*/)
    {
        return true;
    }

    //! Check if a value is finite float32 values
    template<> inline bool isFinite<float32>(float32 value)
    {
        unsigned char *ch = (unsigned char *)&value;
        return (ch[3] & 0x7f) != 0x7f || (ch[2] & 0x80) != 0x80;
    }

    //! Check if a value is finite float64 values
    template<> inline bool isFinite<float64>(float64 value)
    {
        unsigned char *ch = (unsigned char *)&value;
        return (ch[7] & 0x7f) != 0x7f || (ch[6] & 0xf0) != 0xf0;
    }

    //! Check if both components of complex64 value are finite
    template<> inline bool isFinite<complex64>(complex64 value)
    {
        float32 realVal = value.real();
        float32 imagVal = value.imag();
        unsigned char *chreal = (unsigned char *)&realVal;
        unsigned char *chimag = (unsigned char *)&imagVal;
        return ((chreal[3] & 0x7f) != 0x7f || (chreal[2] & 0x80) != 0x80) && ((chimag[3] & 0x7f) != 0x7f || (chimag[2] & 0x80) != 0x80);
    }

    //! Check if both components of complex128 value are finite
    template<> inline bool isFinite<complex128>(complex128 value)
    {
        float64 realVal = value.real();
        float64 imagVal = value.imag();
        unsigned char *chreal = (unsigned char *)&realVal;
        unsigned char *chimag = (unsigned char *)&imagVal;
        return ((chreal[7] & 0x7f) != 0x7f || (chreal[6] & 0xf0) != 0xf0) && ((chimag[7] & 0x7f) != 0x7f || (chimag[6] & 0xf0) != 0xf0);
    }

	//! method returns whether a given variable is NaN / not a Number but maybe Inf.
	/*!
	For floating point variables, this method considers a variable to be NaN if the bitmask
	is NaN. For floating point values only parameters of type
	float32, float64, complex64 or complex128 are treated in the desired way.
	For integer types --> always false

	\param value is value to check
	\return true if value is NaN else false
	*/
    template<typename _Tp> inline bool isNaN(_Tp value)
    {
        return false;
    }

    //! Check if a value is isNaN float32 values
    template<> inline bool isNaN<float32>(float32 value)
    {
        unsigned char *ch = (unsigned char *)&value;
        return (ch[3] & 0x7f) == 0x7f && ch[2] > 0x80;
    }
    //! Check if a value is isNaN float64 values
    template<> inline bool isNaN<float64>(float64 value)
    {
        unsigned char *ch = (unsigned char *)&value;
        return (ch[7] & 0x7f) == 0x7f && ch[6] > 0xf0;
    }

    //! Check if one of the components of complex64 values are not a number
    template<> inline bool isNaN<complex64>(complex64 value)
    {
        float32 realVal = value.real();
        float32 imagVal = value.imag();
        unsigned char *chreal = (unsigned char *)&realVal;
        unsigned char *chimag = (unsigned char *)&imagVal;
        return ((chreal[3] & 0x7f) == 0x7f && chreal[2] > 0x80) || ((chimag[3] & 0x7f) == 0x7f && chimag[2] > 0x80);
    }

    //! Check if one of the components of complex128 values are not a number
    template<> inline bool isNaN<complex128>(complex128 value)
    {
        float64 realVal = value.real();
        float64 imagVal = value.imag();
        unsigned char *chreal = (unsigned char *)&realVal;
        unsigned char *chimag = (unsigned char *)&imagVal;
        return ((chreal[7] & 0x7f) == 0x7f && chreal[6] > 0xf0) || ((chimag[7] & 0x7f) == 0x7f && chimag[6] > 0xf0);
    }

	//! method returns whether a given variable is Inf / not may be NaN.
	/*!
	For floating point variables, this method considers a variable to be Inf if the bitmask
	is Inf. For floating point values only parameters of type
	float32, float64, complex64 or complex128 are treated in the desired way.
	For integer types --> always false

	\param value is value to check
	\return true if value is Inf else false
	*/
    template<typename _Tp> inline bool isInf(_Tp /*value*/)
    {
        return false;
    }

    //! Check if a value is infinite float32 values
    template<> inline bool isInf<float32>(float32 value)
    {
        unsigned char *ch = (unsigned char *)&value;
        return (ch[3] & 0x7f) == 0x7f && ch[2] == 0x80;
    }

    //! Check if a value is infinite float64 values
    template<> inline bool isInf<float64>(float64 value)
    {
        unsigned char *ch = (unsigned char *)&value;
        return (ch[7] & 0x7f) == 0x7f && ch[6] == 0xf0;
    }

    //! Check if one of the components of complex64 values are infinite
    template<> inline bool isInf<complex64>(complex64 value)
    {
        float32 realVal = value.real();
        float32 imagVal = value.imag();
        unsigned char *chreal = (unsigned char *)&realVal;
        unsigned char *chimag = (unsigned char *)&imagVal;
        return ((chreal[3] & 0x7f) == 0x7f && chreal[2] == 0x80) || ((chimag[3] & 0x7f) == 0x7f && chimag[2] == 0x80);
    }

    //! Check if one of the components of complex128 values are infinite
    template<> inline bool isInf<complex128>(complex128 value)
    {
        float64 realVal = value.real();
        float64 imagVal = value.imag();
        unsigned char *chreal = (unsigned char *)&realVal;
        unsigned char *chimag = (unsigned char *)&imagVal;
        return ((chreal[7] & 0x7f) == 0x7f && chreal[6] == 0xf0) || ((chimag[7] & 0x7f) == 0x7f && chimag[6] == 0xf0);
    }

	//! method returns whether a given variable is equal to zero.
	/*!
	For floating point variables, this method considers a variable to be zero, if its value
	lie within the boundaries (-epsilon,epsilon). Epsilon can for example be obtained by
	std::numeric_limits<_Tp>::epsilon(). For floating point values only parameters of type
	float32, float64, complex64 or complex128 are treated in the desired way.

	\param v is value to check
	\param epsilon is epsilon boundary, for fixed-point values this value is ignored.
	\return true if value is zero or within the epsilon boundaries, else false
	*/
	template<typename _Tp> inline bool isZeroValue(_Tp v, _Tp /*epsilon*/)
	{
		return v == 0;
	}

	template<> inline bool isZeroValue(Rgba32 v, Rgba32 /*epsilon*/)
	{
		return v == Rgba32::zeros();
	}

	template<> inline bool isZeroValue(float32 v, float32 epsilon)
	{
		return v >= epsilon ? false : (v <= -epsilon ? false : true);
	}

	template<> inline bool isZeroValue(float64 v, float64 epsilon)
	{
		return v >= epsilon ? false : (v <= -epsilon ? false : true);
	}

	template<> inline bool isZeroValue(std::complex<ito::float32> v, std::complex<ito::float32> epsilon)
	{
		return isZeroValue<ito::float32>(v.real(), epsilon.real()) && isZeroValue<ito::float32>(v.imag(), epsilon.real());
	}

	template<> inline bool isZeroValue(std::complex<ito::float64> v, std::complex<ito::float64> epsilon)
	{
		return isZeroValue<ito::float64>(v.real(), epsilon.real()) && isZeroValue<ito::float64>(v.imag(), epsilon.real());
	}


    //! method returns whether two given numbers of the same type are equal.
    /*!

    */
    template<typename _Tp> inline bool areEqual(_Tp a, _Tp b)
    {
        //for fixed-point and Rgba32 data types
        return a == b;
    }

    template<> inline bool areEqual(float32 a, float32 b)
    {
        return fabs(a - b) < std::numeric_limits<float32>::epsilon();
    }

    template<> inline bool areEqual(float64 a, float64 b)
    {
        return fabs(a - b) < std::numeric_limits<float64>::epsilon();
    }

    template<> inline bool areEqual(complex64 a, complex64 b)
    {
        return areEqual<ito::float32>(a.real(), b.real()) && areEqual<ito::float32>(a.imag(), b.imag());
    }

    template<> inline bool areEqual(complex128 a, complex128 b)
    {
        return areEqual<ito::float64>(a.real(), b.real()) && areEqual<ito::float64>(a.imag(), b.imag());
    }


} // namespace ito

#endif // NUMERIC_H
