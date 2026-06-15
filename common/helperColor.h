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

#ifndef HELPERCOLOR_H
#define HELPERCOLOR_H

#include "typeDefs.h"


namespace ito
{
    namespace colorHelper
    {
        inline float32 grayAlpha(const Rgba32_t &in) /*! < Return the gray-value of the current ARGB-Value*/
        {
            return in.gray() * m_value[0];
        }

        inline void rgb2yuv(const Rgba32_t &in, float32 &Y, float32 &U, float32 &V)
        {
            Y = in.gray();
            U = static_cast<float32>(-0.14713 * in.red() - 0.28886 * in.green() + 0.436 * in.blue());
            V = static_cast<float32>(0.615 * in.red() - 0.51499 * in.green() - 0.10001 * in.blue());
            return;
        }

        inline void rgb2cmyk(const Rgba32_t &in, float32 &C, float32 &M, float32 &Y, float32 &K)
        {
            K = (255 - (in.red() > in.blue() ? (in.red() > in.green() ? in.red() : in.green()) : (in.blue() > in.green() ? in.blue() : in.green()))) / 255.0;

            if( fabs(K - 1) < std::numeric_limits<float32>::epsilon())
            {
                C = 0.0;
                M = 0.0;
                Y = 0.0;
            }
            else
            {
                C = (1.0 - in.red()/ 255.0 - K) / (1.0 - K);
                M = (1.0 - in.green()/ 255.0 - K) / (1.0 - K);
                Y = (1.0 - in.green()/ 255.0 - K) / (1.0 - K);
            }
            return;
        }


        template<uint8 _CHANNEL> inline Rgba32_t max(const Rgba32_t &first, const Rgba32_t &second) const
        {
            if (((RGBChannel_t<_CHANNEL>)first) > ((RGBChannel_t<_CHANNEL>)second)
            {
                return first;
            }
            else
            {
                return second;
            }
        }

        inline Rgba32_t max(const Rgba32_t &first, const Rgba32_t &second, const uint8 &mode)
        {
            switch(mode)
            {
                case Rgba32_t::RGBA_B:
                case Rgba32_t::RGBA_G:
                case Rgba32_t::RGBA_R:
                case Rgba32_t::RGBA_A:
                    return max<mode>(first, second);
                break;
                case Rgba32_t::RGBA_Y:
                    if(first.gray() < second.gray()) return second;
                    else return first;
                break;
                default:
                case Rgba32_t::RGBA_RGB:
                {
                    uint8 max1 = (first.red() > first.blue() ? (first.red() > first.green() ? first.red() : first.green()) : (first.blue() > first.green() ? first.blue() : first.green()));
                    uint8 max2 = (second.red() > second.blue() ? (second.red() > second.green() ? second.red() : second.green()) : (second.blue() > second.green() ? second.blue() : second.green()));
                    if(max1 < max2) return second;
                    else return first;
                }
                break;
            }
        }

    }   // end namespace colorHelper
};   // end namespace ito

#endif
