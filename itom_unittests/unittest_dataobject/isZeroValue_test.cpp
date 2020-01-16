#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv2\opencv.hpp"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
//#include "test_global.h"
#include "commonChannel.h"
#include "../../common/numeric.h"


/*! \class IsZeroValueRealTest
    \brief Test for isZeroValue method for all real data types

    This test checks the functionality of isZeroValue(...) method with different variables of real datatypes.
*/
template <typename _Tp> class IsZeroValueRealTest : public ::testing::Test { };

TYPED_TEST_CASE(IsZeroValueRealTest, ItomRealDataTypes);

TYPED_TEST(IsZeroValueRealTest, checkZeroValueReal_Test)
{
    TypeParam ZeroVal= 0 ;
    TypeParam NonZeroVal1= 5;
    TypeParam NonZeroVal2= 23;
    TypeParam NonZeroVal3= 9;
    TypeParam NonZeroVal4= -4;
    TypeParam min;
    TypeParam max;

    if(std::numeric_limits<TypeParam>::is_exact)
    {
        EXPECT_TRUE( ito::isZeroValue<TypeParam>(ZeroVal,0) );            //!< isZeroValue<_Tp>(...) function must return true if the variable passed contains value '0'.
        EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal1,0) );        //!< isZeroValue<_Tp>(...) function must return false if the variable passed contains non-zero value.
        EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal2,0) );        //!< isZeroValue<_Tp>(...) function must return false if the variable passed contains non-zero value.
        EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal3,0) );        //!< isZeroValue<_Tp>(...) function must return false if the variable passed contains non-zero value.
        EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal4,0) );        //!< isZeroValue<_Tp>(...) function must return false if the variable passed contains non-zero value.
    }
    else
    {
        min= -std::numeric_limits<TypeParam>::epsilon();        //!< declaring negative epsilon boundary
        max= std::numeric_limits<TypeParam>::epsilon();            //!< declaring positive epsilon boundary

        EXPECT_TRUE( ito::isZeroValue<TypeParam>( ZeroVal , std::numeric_limits<TypeParam>::epsilon() ) );    //!< isZeroValue<_Tp>(...) function must return true if the variable passed contains value '0'.
        EXPECT_FALSE( ito::isZeroValue<TypeParam>( min, std::numeric_limits<TypeParam>::epsilon() ) );        //!< Note:------> Test is failing .. It should return false as the variable value is on the boundary "-epsilon".
        EXPECT_FALSE( ito::isZeroValue<TypeParam>( max, std::numeric_limits<TypeParam>::epsilon() ) );        //!< Note:------> Test is failing .. It should return false as the variable value is on the boundary "epsilon".
        EXPECT_FALSE( ito::isZeroValue<TypeParam>( min+1, std::numeric_limits<TypeParam>::epsilon() ) );    //!< isZeroValue<_Tp>(...) function must return false if the variable contains value lying outside of boundary of 0 to epsilon.
        EXPECT_FALSE( ito::isZeroValue<TypeParam>( max-1, std::numeric_limits<TypeParam>::epsilon() ) );    //!< isZeroValue<_Tp>(...) function must return false if the variable contains value lying outside of boundary of 0 to epsilon.
    }
}


/*! \class IsZeroValueComplexTest
    \brief Test for isZeroValue method for complex types

    This test checks the functionality of isZeroValue(...) method with different variables of complex datatypes.
*/
template <typename _Tp> class IsZeroValueComplexTest : public ::testing::Test { };

TYPED_TEST_CASE(IsZeroValueComplexTest, ItomComplexDataTypes);

TYPED_TEST(IsZeroValueComplexTest, checkZeroValueComplex_Test)
{
    //!< Declaration for complex32 type variables.
    std::complex<ito::float32> ZeroVal1(0.0,0.0); 
    std::complex<ito::float32> NonZeroVal1(0.01,0.0);
    std::complex<ito::float32> NonZeroVal2(0.0,-0.05);
    std::complex<ito::float32> NonZeroVal3(-0.03,0.01);

    //!< Declaration for complex64 type variables.
    std::complex<ito::float64> ZeroVal2(0.0,0.0); 
    std::complex<ito::float64> NonZeroVal4(0.01,0.0);
    std::complex<ito::float64> NonZeroVal5(0.0,-0.05);
    std::complex<ito::float64> NonZeroVal6(-0.03,0.01);

    //!< Declaration for zero value variables. 
    std::complex<ito::float32> epsilon1( std::numeric_limits<ito::float32>::epsilon() ,0.0) ;
    std::complex<ito::float64> epsilon2( std::numeric_limits<ito::float64>::epsilon() ,0.0) ;
    if(std::numeric_limits<TypeParam>::is_exact)
    {

    }
    else
    {
        EXPECT_TRUE( ito::isZeroValue<TypeParam>(ZeroVal1, epsilon1 ) );            /*!< Test of isZeroValue() function for real part of Complex Type Variable with zero value.  */
        EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal1, epsilon1 ) );        /*!< Test of isZeroValue() function for real part of Complex Type Variable with nonzero positive value. */    
        EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal2, epsilon1) );            /*!< Test of isZeroValue() function for Complex Type variable with zero real part and nonzero negative imaginary part. */
        EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal3, epsilon1 ) );        /*!< Test of isZeroValue() function for real part of Complex Type Variable with nonzero negative value. */

        EXPECT_TRUE( ito::isZeroValue<TypeParam>(ZeroVal2, epsilon2 ) );            /*!< Test of isZeroValue() function for real part of Complex Type Variable with zero value.  */
        EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal4, epsilon2 ) );        /*!< Test of isZeroValue() function for real part of Complex Type Variable with nonzero positive value. */    
        EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal5, epsilon2) );            /*!< Test of isZeroValue() function for Complex Type variable with zero real part and nonzero negative imaginary part. */
        EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal6, epsilon2 ) );        /*!< Test of isZeroValue() function for real part of Complex Type Variable with nonzero negative value. */
    }
}

/*! \class IsZeroValueRGBATest
    \brief Test for isZeroValue method for color types

    This test checks the functionality of isZeroValue(...) method with different variables of rgba datatypes.
*/
template <typename _Tp> class IsZeroValueRGBATest : public ::testing::Test { };

TYPED_TEST_CASE(IsZeroValueRGBATest, ItomColorTypes);

TYPED_TEST(IsZeroValueRGBATest, checkZeroValueRGBA_Test)
{
    //!< Declaration for complex32 type variables.
    ito::Rgba32 zeroVal1 = ito::Rgba32::zeros();
    ito::Rgba32 zeroVal2 = ito::Rgba32(0, 0, 0, 0); 
    ito::Rgba32 zeroVal3; 
    zeroVal3 = (ito::uint32)0; 
    ito::Rgba32 zeroVal4;
    zeroVal4= (ito::int32)0;
    ito::Rgba32 zeroVal5 = ito::Rgba32::fromUnsignedLong(0);

    ito::Rgba32 nonZeroVal1 = ito::Rgba32((ito::uint8)0);
    ito::Rgba32 nonZeroVal2 = ito::Rgba32(1, 0, 0, 0); 
    ito::Rgba32 nonZeroVal3 = ito::Rgba32(0, 1, 0, 0); 
    ito::Rgba32 nonZeroVal4 = ito::Rgba32(0, 0, 1, 0);  
    ito::Rgba32 nonZeroVal5 = ito::Rgba32(0, 0, 0, 1);
    

    ito::Rgba32 nonZeroVal6 = ito::Rgba32((ito::int32)std::numeric_limits<ito::uint32>::max()); 
    ito::Rgba32 nonZeroVal7 = ito::Rgba32((ito::uint32)std::numeric_limits<ito::uint32>::max()); 


    //!< Declaration for zero value variables. 
    ito::Rgba32 epsilon1 = ito::Rgba32::zeros();
    if(std::numeric_limits<TypeParam>::is_exact)
    {

    }
    else
    {
        EXPECT_TRUE( ito::isZeroValue<ito::Rgba32>(zeroVal1, epsilon1 ) );            /*!< Test of isZeroValue() function for default constructor of rgba.  */
        EXPECT_TRUE( ito::isZeroValue<ito::Rgba32>(zeroVal2, epsilon1 ) );            /*!< Test of isZeroValue() function for rgba constructor of rgba.  */
        EXPECT_TRUE( ito::isZeroValue<ito::Rgba32>(zeroVal3, epsilon1 ) );            /*!< Test of isZeroValue() function for uint32 constructor of rgba.  */
        EXPECT_TRUE( ito::isZeroValue<ito::Rgba32>(zeroVal4, epsilon1 ) );            /*!< Test of isZeroValue() function for int32 constructor of rgba.  */
        EXPECT_TRUE( ito::isZeroValue<ito::Rgba32>(zeroVal5, epsilon1 ) );            /*!< Test of isZeroValue() function for int32 constructor of rgba.  */


        EXPECT_FALSE( ito::isZeroValue<ito::Rgba32>(nonZeroVal1, epsilon1 ) );        /*!< Test of isZeroValue() function for */    
        EXPECT_FALSE( ito::isZeroValue<ito::Rgba32>(nonZeroVal2, epsilon1) );            /*!< Test of isZeroValue() function for */
        EXPECT_FALSE( ito::isZeroValue<ito::Rgba32>(nonZeroVal3, epsilon1 ) );        /*!< Test of isZeroValue() function  */
        EXPECT_FALSE( ito::isZeroValue<ito::Rgba32>(nonZeroVal4, epsilon1 ) );        /*!< Test of isZeroValue() function  */    
        EXPECT_FALSE( ito::isZeroValue<ito::Rgba32>(nonZeroVal5, epsilon1) );            /*!< Test of isZeroValue() function  */
        EXPECT_FALSE( ito::isZeroValue<ito::Rgba32>(nonZeroVal6, epsilon1 ) );        /*!< Test of isZeroValue() function  */
        EXPECT_FALSE( ito::isZeroValue<ito::Rgba32>(nonZeroVal7, epsilon1 ) );        /*!< Test of isZeroValue() function  */
        

    }
}