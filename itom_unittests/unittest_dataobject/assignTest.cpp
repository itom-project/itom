#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 )
#pragma once
#include "opencv/cv.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"


/*! \class AddressTest
    \brief Address test for all data types declared as "ItomDataAllTypes"

    This is a basic test class for any matrix of any data type. This test class confirms if the different parameters of already declared matrices are alright.
*/
template <typename _Tp> class AssignTest : public ::testing::Test 
{ 
public:

    virtual void SetUp(void)
    {
        matrix = ito::DataObject(5,4,3,ito::getDataType2<_Tp*>());
        mask = ito::DataObject();
        mask.rand(5,4,3,ito::tUInt8, false);
        ito::DataObject z(5,4,3,ito::tUInt8);
        z = 127;
        mask = (mask > z);
    };

    virtual void TearDown(void) {};

    ito::DataObject matrix;  /*!< matrix created with the template type */
    ito::DataObject mask;
    ito::DObjConstIterator it;

    typedef _Tp valueType;
};


TYPED_TEST_CASE(AssignTest, ItomDataStandardTypes);

TYPED_TEST(AssignTest, AssignOperatorTest)
{
    int typeno = ito::getDataType( (const TypeParam *) NULL );

    matrix = 1;
    for (it = matrix.constBegin(); it != matrix.constEnd(); ++it)
    {
        EXPECT_EQ ( *((TypeParam*)(*it)), cv::saturate_cast<TypeParam>(1));
    }

    matrix = 2.1;
    for (it = matrix.constBegin(); it != matrix.constEnd(); ++it)
    {
        EXPECT_EQ ( *((TypeParam*)(*it)), cv::saturate_cast<TypeParam>(2.1));
    }  
    
}

TYPED_TEST(AssignTest, SetToTest)
{
    int typeno = ito::getDataType( (const TypeParam *) NULL );

    matrix.setTo(1);
    for (it = matrix.constBegin(); it != matrix.constEnd(); ++it)
    {
        EXPECT_EQ ( *((TypeParam*)(*it)), cv::saturate_cast<TypeParam>(1));
    }

    matrix.setTo(2.1);
    for (it = matrix.constBegin(); it != matrix.constEnd(); ++it)
    {
        EXPECT_EQ ( *((TypeParam*)(*it)), cv::saturate_cast<TypeParam>(2.1));
    }  

    ito::DObjConstIterator it_mask;
    matrix.setTo(0);

    matrix.setTo(1, mask);
    for (it = matrix.constBegin(), it_mask = mask.constBegin(); it != matrix.constEnd(); ++it, ++it_mask)
    {
        if (*(*it_mask))
        {
            EXPECT_EQ ( *((TypeParam*)(*it)), cv::saturate_cast<TypeParam>(1));
        }
        else
        {
            EXPECT_EQ ( *((TypeParam*)(*it)), cv::saturate_cast<TypeParam>(0));
        }
    }

    matrix.setTo(2.1, mask);
    for (it = matrix.constBegin(), it_mask = mask.constBegin(); it != matrix.constEnd(); ++it, ++it_mask)
    {
        if (*(*it_mask))
        {
            EXPECT_EQ ( *((TypeParam*)(*it)), cv::saturate_cast<TypeParam>(2.1));
        }
        else
        {
            EXPECT_EQ ( *((TypeParam*)(*it)), cv::saturate_cast<TypeParam>(0.0));
        }
    }  
    
}
