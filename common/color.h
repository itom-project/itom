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

#ifndef COLOR_H
#define COLOR_H

#include "typeDefs.h"

namespace ito
{

    ////////////////////////////////////////////////////////// DO NOT ADD ANY MEMBER VARIABLES TO RGBA_T / CHANNEL_T ///////////////////////////////////////////////////////////////////////
    /** \class  Rgba32
    *   \brief  This class implements basic functionality for color handling in itom.
    *   \detail This class implements ARGB32 into itom. In openCV-mat this will be represented by an 4channel uint8-plane.
    *           The functions of this class are inspired by http://virjo.googlecode.com/svn/trunk/SFML_Windows/src/SFML/Graphics/Color.cpp, written by Laurent Gomila (laurent.gom@gmail.com)
    *
    *   \author lyda
    *   \date   2013
    *   \sa     ito::rgba32
    */
        class Rgba32 : public RgbaBase32
        {
        public:

            enum RGBSelectionFlags
            {
                RGBA_B    = 0, //!< blue
                RGBA_G    = 1, //!< green
                RGBA_R    = 2, //!< red
                RGBA_A    = 3, //!< alpha
                RGBA_Y    = 4, //!< gray
                RGBA_RGB  = 5
            };

            Rgba32()  /*! < Constructor for basic values */
            {
                //memset(m_value, 0, 4*sizeof(ito::uint8));
            }

            static Rgba32 zeros()
            {
                Rgba32 temp;
                temp.rgba = 0;
                return temp;
            }

            static Rgba32 black()
            {
                Rgba32 temp(255, 0,0,0);
                return temp;
            }

            //! static constructor to create Rgba32 from uint32 containing the values argb
            /*!
             \param val is the uint32 value that is defined as (alpha << 24 + red << 16 + green << 8 + blue)
             */
            static Rgba32 fromUnsignedLong(const uint32 val)
            {
                Rgba32 temp;
                memcpy(temp.u8ptr(), &val, 4*sizeof(ito::uint8));
                return temp;
            }

            Rgba32(const uint8 &a, const uint8 &r, const uint8 &g, const uint8 &b) /*! < Constructor for ARGB by 4 channels*/
            {
                this->b = b;
                this->g = g;
                this->r = r;
                this->a = a;
            }

            //! Constructor which will set color channels to gray uint8 and alpha to 255.
            /*!
             \param gray is the gray value. Alpha is 255, R=G=B are set to this value 'gray'.
             */
            explicit Rgba32(const uint8 gray)
            {
                a = 0xFF;
                r = gray;
                b = gray;
                g = gray;
            }

            Rgba32(const Rgba32 &rhs)/*! < Copy-Constructor for lvalues */
            {
                rgba = rhs.rgba;
            };

            Rgba32& operator +=(const Rgba32 &rhs)/*! < Implementation of += operator with overflow handling */
            {
                b = static_cast<uint8>(std::min<int16>(b + rhs.b, 255));
                g = static_cast<uint8>(std::min<int16>(g + rhs.g, 255));
                r = static_cast<uint8>(std::min<int16>(r + rhs.r, 255));
                a = static_cast<uint8>(std::min<int16>(a + rhs.a, 255));
                return *this;
            }

            Rgba32& operator =(const Rgba32 &rhs)/*! < Implementation of = operator */
            {
                rgba = rhs.rgba;
                return *this;
            }

            Rgba32& operator =(const uint32 &rhs)/*! < Implementation of = for uint32 by direct copy*/
            {
                rgba = rhs;
                return *this;
            }

            Rgba32& operator -=(const Rgba32 &rhs)/*! < Implementation of -= operator with overflow handling */
            {
                b = static_cast<uint8>(std::max<int16>(b - rhs.b, 0));
                g = static_cast<uint8>(std::max<int16>(g - rhs.g, 0));
                r = static_cast<uint8>(std::max<int16>(r - rhs.r, 0));
                a = static_cast<uint8>(std::max<int16>(a - rhs.a, 0));
                return *this;
            }

            //! Multiplication by another Rgba32 value.
            /*!
             \param All channels are multiplied by each other and then divided by 255 (integer division leads to a ceil operation in any case)
             */
            Rgba32& operator *=(const Rgba32 &rhs)/*! < Implementation of *= operator with overflow handling and normalisation */
            {
                b = static_cast<uint8>(b * rhs.b / 255);
                g = static_cast<uint8>(g * rhs.g / 255);
                r = static_cast<uint8>(r * rhs.r / 255);
                a = static_cast<uint8>(a * rhs.a / 255);
                return *this;
            }

            //! Multiplication by a float32 grayFactor (alpha is unchanged).
            /*!
            \param The channels R,G and B are multiplied by the given grayFactor and cropped to the range [0,255]
            \throws runtime_error if grayFactor < 0
            */
            Rgba32& operator *=(const ito::float32 &grayFactor)
            {
                if (grayFactor < 0.0)
                {
                    throw std::runtime_error("Multiplication factor must be >= 0.0");
                }
                unsigned int t = static_cast<unsigned int>(b * grayFactor);
                b = static_cast<uint8>(t <= 255 ? t : 255);
				t = static_cast<unsigned int>(g * grayFactor);
                g = static_cast<uint8>(t <= 255 ? t : 255);
				t = static_cast<unsigned int>(r * grayFactor);
                r = static_cast<uint8>(t <= 255 ? t : 255);
                return *this;
            }

            Rgba32& operator /=(const Rgba32 &rhs)/*! < Implementation of /= operator with overflow handling and normalisation */
            {
                if(rhs.b == 0 || rhs.g == 0 || rhs.r == 0 || rhs.a == 0)
                {
                    throw std::runtime_error("Division by zero not allowed for rgba32-values");
                }
                b = static_cast<uint8>(std::min<int16>((b * (ito::int16)255) / rhs.b, 255));
                g = static_cast<uint8>(std::min<int16>((g * (ito::int16)255) / rhs.g, 255));
                r = static_cast<uint8>(std::min<int16>((r * (ito::int16)255) / rhs.r, 255));
                a = static_cast<uint8>(std::min<int16>((a * (ito::int16)255) / rhs.a, 255));
                return *this;
            }

            Rgba32 operator +(const Rgba32 &second) const /*! < Implementation of + operator using += operator */
            {
                Rgba32 first(*this);
                first += second;
                return first;
            }

            Rgba32 operator -(const Rgba32 &second) const /*! < Implementation of - operator using -= operator */
            {
                Rgba32 first(*this);
                first -= second;
                return first;
            }

            Rgba32 operator *(const Rgba32 &second) const /*! < Implementation of * operator using *= operator */
            {
                Rgba32 first(*this);
                first *= second;
                return first;
            }

            Rgba32 operator *(const ito::float32 &second) const /*! < Implementation of * operator using *= operator */
            {
                Rgba32 first(*this);
                first *= second;
                return first;
            }

            Rgba32 operator /(const Rgba32 &second) const /*! < Implementation of * operator using *= operator */
            {
                Rgba32 first(*this);
                first /= second;
                return first;
            }

            bool operator ==(const Rgba32 &rhs) const /*! < Implementation of == operator comparing each element including alpha channel, true if all are equal */
            {
                return (b == rhs.b) && (g == rhs.g) && (r == rhs.r && (a == rhs.a));
            }

            bool operator !=(const Rgba32 &rhs) const /*! < Implementation of != operator comparing each element including alpha channel, true if one is different */
            {
                return (b != rhs.b) || (g != rhs.g) || (r != rhs.r || (a != rhs.a));
            }

            inline float32 gray() const /*! < Return the gray-value of the current RGB-Value*/
			{
				return static_cast<float32>(0.299 * r + 0.587 * g + 0.114 * b);
			}

            uint8& alpha() {return a;}; /*! < Access to alpha-Channel*/
            uint8& red()   {return r;}; /*! < Access to red-Channel*/
            uint8& green() {return g;}; /*! < Access to green-Channel*/
            uint8& blue()  {return b;}; /*! < Access to blue-Channel*/

            uint8 alpha() const {return a;}; /*! < Read out alpha-Channel*/
            uint8 red()   const {return r;}; /*! < Read out red-Channel*/
            uint8 green() const {return g;}; /*! < Read out green-Channel*/
            uint8 blue()  const {return b;}; /*! < Read out blue-Channel*/

            uint32& argb() {return rgba;}; /*! < Access to argb-Channel*/
            uint32 argb() const {return rgba;}; /*! < Read out argb-Channel*/

            uint32* u32ptr() {return (&rgba);}
            uint8* u8ptr() {return ((uint8*)(&rgba));}
        };


        template<uint8 _COLOR> class RGBChannel : public Rgba32
        {
        public:
            RGBChannel()  /*! < Constructor for basic values */
            {
                //memset(m_value, 0, 4*sizeof(ito::uint8));
            };

            explicit RGBChannel(const uint8 gray) /*! < Constructor which will set color channels to gray uint8 and alpha to 255 */
            {
                r = g = b = 0;
                a = 0xFF;
                items[_COLOR] = gray;
            }

            static RGBChannel<_COLOR> zeros()
            {
                RGBChannel<_COLOR> temp;
                temp.rgba = 0;
                return temp;
            }

            static RGBChannel<_COLOR> black()
            {
                RGBChannel<_COLOR> temp;
                temp.rgba = 0;
                temp.a = 0xFF;
                return temp;
            }

            RGBChannel(const RGBChannel &rhs)/*! < Copy-Constructor for lvalues */
            {
                rgba = 0;
                a = 0xFF;
                items[_COLOR] = rhs.items[_COLOR];
            }

            RGBChannel& operator +=(const RGBChannel &rhs)/*! < Implementation of += operator with overflow handling */
            {
                items[_COLOR] = static_cast<uint8>(std::min<int16>(items[_COLOR] + rhs.items[_COLOR], 255));
                return *this;
            }

            RGBChannel& operator =(const RGBChannel &rhs)/*! < Implementation of = operator */
            {
                items[_COLOR] = rhs.items[_COLOR];
                return *this;
            }

            RGBChannel& operator =(const uint32 &rhs)/*! < Implementation of = for uint32 by direct copy*/
            {
                items[_COLOR] = ((unsigned char*)&rhs)[_COLOR];
                return *this;
            }

            RGBChannel& operator -=(const RGBChannel &rhs)/*! < Implementation of -= operator with overflow handling */
            {
                items[_COLOR] = static_cast<uint8>(std::max<int16>(items[_COLOR] - rhs.items[_COLOR], 0));
                return *this;
            }

            RGBChannel& operator *=(const RGBChannel &rhs)/*! < Implementation of *= operator with overflow handling and normalisation */
            {
                items[_COLOR] = static_cast<uint8>(items[_COLOR] * rhs.items[_COLOR] / 255);
                return *this;
            }

            RGBChannel& operator /=(const RGBChannel &rhs)/*! < Implementation of *= operator with overflow handling and normalisation */
            {
                if(rhs.items[_COLOR] == 0)
                {
                    throw std::runtime_error("Division by zero not allowed for rgba32-values");
                }
                items[_COLOR] = static_cast<uint8>(std::min<int16>((items[_COLOR] * (ito::int16)255)/ rhs.items[_COLOR], 255));
                return *this;
            }

            RGBChannel operator +(const RGBChannel &second) const /*! < Implementation of + operator using += operator */
            {
                RGBChannel<_COLOR> first(*this);
                first += second;
                return first;
            }

            RGBChannel operator -(const RGBChannel &second) const /*! < Implementation of - operator using -= operator */
            {
                RGBChannel<_COLOR> first(*this);
                first -= second;
                return first;
            }

            RGBChannel operator *(const RGBChannel &second) const /*! < Implementation of * operator using *= operator */
            {
                RGBChannel<_COLOR> first(*this);
                first *= second;
                return first;
            }

            RGBChannel operator /(const RGBChannel &second) const /*! < Implementation of * operator using *= operator */
            {
                RGBChannel<_COLOR> first(*this);
                first /= second;
                return first;
            }

            bool operator ==(const RGBChannel &rhs) const /*! < Implementation of == operator comparing each element including alpha channel, true if all are equal */
            {
                return items[_COLOR] == rhs.items[_COLOR];
            }

            bool operator !=(const RGBChannel &rhs) const /*! < Implementation of != operator comparing each element including alpha channel, true if one is different */
            {
                return items[_COLOR] != rhs.items[_COLOR];
            }

            bool operator <(const RGBChannel &rhs) const /*! < Implementation of != operator comparing each element including alpha channel, true if one is different */
            {
                return (items[_COLOR] < rhs.items[_COLOR]);
            }

            bool operator >(const RGBChannel &rhs) const /*! < Implementation of != operator comparing each element including alpha channel, true if one is different */
            {
                return (items[_COLOR] < rhs.items[_COLOR]);
            }

            inline float32 gray() const {return static_cast<float32> (items[_COLOR]);}; /*! < Read out single channel and return as gray value*/

            uint8& value() {return items[_COLOR];}; /*! < Access to single-Channel*/
            uint8 value() const {return items[_COLOR];}; /*! < Read out single-Channel*/
        };

        typedef RGBChannel<Rgba32::RGBA_A> AlphaChannel;
        typedef RGBChannel<Rgba32::RGBA_R> RedChannel;
        typedef RGBChannel<Rgba32::RGBA_G> GreenChannel;
        typedef RGBChannel<Rgba32::RGBA_B> BlueChannel;
    ////////////////////////////////////////////////////////// DO NOT ADD ANY MEMBER VARIABLES TO RGBA_T / CHANNEL_T ///////////////////////////////////////////////////////////////////////

} //end namespace ito

#endif //COLOR_H
