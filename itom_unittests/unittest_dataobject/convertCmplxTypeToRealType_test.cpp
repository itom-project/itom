#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv/cv.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
//#include "test_global.h"
#include "commonChannel.h"


/*! \class complexToReal_Test
    \brief Test for convertCmplxTypeToRealType(...) method for different datatypes. 
    This test converts different complex type variables into different real type variables using convertCmplxTypeToRealType(...) method and checks if the method works as desired.
*/
template <typename _Tp> class complexToReal_Test : public ::testing::Test { };

TYPED_TEST_CASE(complexToReal_Test, ItomDataStandardTypes);

//! Test for convertCmplxTypeToRealType() function with different variables of different Datatypes.
TYPED_TEST(complexToReal_Test, convertCmplxTypeToRealType_Test)
{
    //!< Declaration of different variables of different datatypes.
    ito::tDataType cmplxType1 = ito::tInt8;
    ito::tDataType cmplxType2 = ito::tInt16;
    ito::tDataType cmplxType3 = ito::tInt32;
    ito::tDataType cmplxType4 = ito::tUInt8;
    ito::tDataType cmplxType5 = ito::tUInt16;
    ito::tDataType cmplxType6 = ito::tFloat32;
    ito::tDataType cmplxType7 = ito::tFloat64;
    ito::tDataType cmplxType8 = ito::tComplex64;
    ito::tDataType cmplxType9 = ito::tComplex128;
    ito::tDataType cmplxType10 = ito::tRGBA32;

    
    //!     Test Details.

    /*!  ito::t(U)Int[8/16/32] must return the same type constants. 
         ito::tFloat32/64 must return the same type constants.
         ito::tComplex64 must return ito::tFloat32.
         ito::tComplex128 must return ito::tFloat64. !
    */
    EXPECT_EQ(cmplxType1, ito::convertCmplxTypeToRealType(cmplxType1) );    
    EXPECT_EQ(cmplxType2, ito::convertCmplxTypeToRealType(cmplxType2) );
    EXPECT_EQ(cmplxType3, ito::convertCmplxTypeToRealType(cmplxType3) );
    EXPECT_EQ(cmplxType4, ito::convertCmplxTypeToRealType(cmplxType4) );
    EXPECT_EQ(cmplxType5, ito::convertCmplxTypeToRealType(cmplxType5) );
    EXPECT_NE(cmplxType5, ito::convertCmplxTypeToRealType(cmplxType4) );    /*!< Testing if two datatype variables do not show equality in result. */
    EXPECT_NE(cmplxType4, ito::convertCmplxTypeToRealType(cmplxType1) );    /*!< Testing if two datatype variables do not show equality in result. */
    EXPECT_EQ(cmplxType6, ito::convertCmplxTypeToRealType(cmplxType6) );    /*!< Test for ito::tFloat32 type variable. */
    EXPECT_EQ(cmplxType7, ito::convertCmplxTypeToRealType(cmplxType7) );    /*!< Test for ito::tFloat64 type variable. */
    EXPECT_EQ(cmplxType6, ito::convertCmplxTypeToRealType(cmplxType8) );    /*!< Testing if the function returns ito::tFloat32 type for ito::tComplex64 type variable. */
    EXPECT_EQ(cmplxType7, ito::convertCmplxTypeToRealType(cmplxType9) );    /*!< Testing if the function returns ito::tFloat64 type for ito::tComplex128 type variable. */
}



