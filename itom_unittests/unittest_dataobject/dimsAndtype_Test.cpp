#include <iostream>

#include "../../common/sharedStructures.h"

// opencv
#pragma warning(disable : 4996) // C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This
                                // function or variable may be unsafe. Consider using fopen_s instead.

#include "../../DataObject/dataobj.h"
#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"
//#include "test_global.h"
#include "commonChannel.h"

/*! \class DimAndTypeTest
    \brief Test to check functionality of getDims() and getType() methods

    This test class checks functionality of getDims() and getType() methods on different data objects of different sizes
   and types.
*/
template <typename _Tp> class DimAndTypeTest : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
        // dObj1 = ito::DataObject(0,temp_size1,ito::getDataType2<_Tp*>());
        int temp_size1[] = { 10 };
        dObj1 = ito::DataObject(0, temp_size1, ito::getDataType2<_Tp *>()); //!< Empty Data Object
        dObj2 = ito::DataObject(1, temp_size1, ito::getDataType2<_Tp *>()); //!< 1 dimensional data object
        dObj3 = ito::DataObject(5, 10, ito::getDataType2<_Tp *>());         //!< 2 dimensional data object
        dObj4 = ito::DataObject(2, 3, 4, ito::getDataType2<_Tp *>());       //!< 3 dimensional data object

        int temp_size[] = { 3,4,2,10,10 };
        dObj5 = ito::DataObject(5, temp_size, ito::getDataType2<_Tp *>()); //!< 5 dimensional data object
    };

    virtual void TearDown(void){};

    ito::DataObject dObj1;
    ito::DataObject dObj2;
    ito::DataObject dObj3;
    ito::DataObject dObj4;
    ito::DataObject dObj5;
};

TYPED_TEST_CASE(DimAndTypeTest, ItomDataAllTypes);

// getDims_Test
/*!
    This test checks the functionality of getDims() method on different data objects of different sizes and types.
*/
TYPED_TEST(DimAndTypeTest, getDims_Test)
{
    ito::tDataType testType = ito::getDataType((const TypeParam *)NULL);
    EXPECT_EQ(
        0, this->dObj1
               .getDims()); //!< checking dimensions of empty data object using getDims() function which should return 0
    EXPECT_EQ(
        2, this->dObj2
               .getDims()); //!< checking dimensions of 1 dimensional data object using getDims() function which should
                            //!< return value 2 because there is no existence of 1 dimensional data object and if they
                            //!< are defined, they should be taken as 2 dimensional data object automatically.
    EXPECT_EQ(2, this->dObj3.getDims()); //!< checking dimensions of 2 dimensional data object using getDims() function
                                         //!< which should return value 2
    EXPECT_EQ(3, this->dObj4.getDims()); //!< checking dimensions of 3 dimensional data object using getDims() function
                                         //!< which should return value 3
    EXPECT_EQ(5, this->dObj5.getDims()); //!< checking dimensions of 5 dimensional data object using getDims() function
                                         //!< which should return value 5
}

// getType_Test
/*!
    This test checks the functionality of getType() method on different data objects of different types and sizes.
*/
TYPED_TEST(DimAndTypeTest, getType_Test)
{
    ito::tDataType testType = ito::getDataType((const TypeParam *)NULL);

    EXPECT_EQ(testType, this->dObj1.getType()); //!< checking expected data type of dObj1
    EXPECT_EQ(testType, this->dObj2.getType()); //!< checking expected data type of dObj2
    EXPECT_EQ(testType, this->dObj3.getType()); //!< checking expected data type of dObj3
    EXPECT_EQ(testType, this->dObj4.getType()); //!< checking expected data type of dObj4
    EXPECT_EQ(testType, this->dObj5.getType()); //!< checking expected data type of dObj5
}
