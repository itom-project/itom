#ifndef TEMPHEADER_H
#define TEMPHEADER_H

//#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
//#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
//#pragma once
//#include "opencv/cv.h"
//#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"

typedef ::testing::Types<uint8, int8, uint16, int16, /*uint32*/ int32, float32, float64 > ItomRealDataTypes;
typedef ::testing::Types<uint8, int8, uint16, int16,  /*uint32*/ int32, float32, float64, complex64, complex128 > ItomDataTypes;

template<typename _Tp> int getTypeNumber() { return -1; };
template<> int getTypeNumber<uint8>() { return ito::tUInt8; };
template<> int getTypeNumber<int8>() { return ito::tInt8; };
template<> int getTypeNumber<uint16>() { return ito::tUInt16; };
template<> int getTypeNumber<int16>() { return ito::tInt16; };
template<> int getTypeNumber<uint32>() { return ito::tUInt32; };
template<> int getTypeNumber<int32>() { return ito::tInt32; };
template<> int getTypeNumber<float32>() { return ito::tFloat32; };
template<> int getTypeNumber<float64>() { return ito::tFloat64; };
template<> int getTypeNumber<complex64>() { return ito::tComplex64; };
template<> int getTypeNumber<complex128>() { return ito::tComplex128; };

#endif