#ifndef COMMONCHANNEL_H
#define COMMONCHANNEL_H

#include "../../Common/sharedStructures.h"
#include "gtest/gtest.h"
//#include "test_global.h"
//commonChannel
/*!
	This Header File consists of declaration of all userdefined datatypes used for all Test cases.
*/

typedef ::testing::Types<ito::uint8, ito::int8, ito::uint16, ito::int16,/* ito::uint32,*/ ito::int32, ito::float32, ito::float64 > ItomRealDataTypes;
typedef ::testing::Types<ito::complex64, ito::complex128 > ItomComplexDataTypes;
typedef ::testing::Types<ito::float32, ito::float64 > ItomFloatDoubleDataTypes;
typedef ::testing::Types<ito::uint8, ito::int8, ito::uint16, ito::int16, ito::int32 > ItomIntDataTypes;
typedef ::testing::Types<ito::int8, ito::int16, ito::int32 > ItomIntDataTypes2;
typedef ::testing::Types<ito::uint8, ito::uint16 > ItomUIntDataTypes;
typedef ::testing::Types<ito::uint8, ito::int8, ito::uint16, ito::int16,/* ito::uint32,*/ ito::int32, ito::float32, ito::float64, ito::complex64, ito::complex128 > ItomDataStandardTypes;
typedef ::testing::Types<ito::uint8, ito::int8, ito::uint16, ito::int16,/* ito::uint32,*/ ito::int32, ito::float32, ito::float64, ito::complex64, ito::complex128, ito::rgba32> ItomDataAllTypes;

typedef ::testing::Types<ito::rgba32> ItomColorTypes;
typedef ::testing::Types<ito::alphaChannel, ito::redChannel, ito::greenChannel, ito::blueChannel> ItomColorChannelTypes;
typedef ::testing::Types<ito::rgba32, ito::alphaChannel, ito::redChannel, ito::greenChannel, ito::blueChannel> ItomColorAllTypes;
typedef ::testing::Types<ito::uint8, ito::uint16, ito::uint32, ito::int32, ito::float32, ito::float64> ItomColorCompatibleTypes;
typedef ::testing::Types<ito::int8, ito::int16, ito::complex64, ito::complex128> ItomColorNotCompTypes;

//template<typename _Tp> int getTypeNumber() { return -1; };
//template<> int getTypeNumber<ito::uint8>() { return ito::tUInt8; };
//template<> int getTypeNumber<ito::int8>() { return ito::tInt8; };
//template<> int getTypeNumber<ito::uint16>() { return ito::tUInt16; };
//template<> int getTypeNumber<ito::int16>() { return ito::tInt16; };
//template<> int getTypeNumber<ito::uint32>() { return ito::tUInt32; };
//template<> int getTypeNumber<ito::int32>() { return ito::tInt32; };
//template<> int getTypeNumber<ito::float32>() { return ito::tFloat32; };
//template<> int getTypeNumber<ito::float64>() { return ito::tFloat64; };
//template<> int getTypeNumber<ito::complex64>() { return ito::tComplex64; };
//template<> int getTypeNumber<ito::complex128>() { return ito::tComplex128; };

#endif