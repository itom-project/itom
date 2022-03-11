
#include "../../common/sharedStructures.h"

// opencv
#pragma warning(disable : 4996) // C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This
                                // function or variable may be unsafe. Consider using fopen_s instead.

#include "../../DataObject/dataobj.h"
#include "commonChannel.h"
#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"

/*! \class operatorComplexTest
    \brief test for div and mul methods for complex datatype (ItomComplexDataTypes) data objects

    This test class checks functionality of div and mul methods for data objects with complex data types.
*/
template <typename _Tp> class DateTimeTest : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
    };

    virtual void TearDown(void){};
};

TEST(DateTimeTest, basicDateTimeTest)
{
    ito::DataObject dObjDateTime(10, 10, ito::tDateTime);

    EXPECT_EQ(dObjDateTime.getSize()[0], 10);
    EXPECT_EQ(dObjDateTime.getSize()[1], 10);
    EXPECT_EQ(dObjDateTime.getType(), ito::tDateTime);
}

TEST(DateTimeTest, basicTimeDeltaTest)
{
    ito::DataObject dObjTimeDelta(10, 10, ito::tTimeDelta);

    EXPECT_EQ(dObjTimeDelta.getSize()[0], 10);
    EXPECT_EQ(dObjTimeDelta.getSize()[1], 10);
    EXPECT_EQ(dObjTimeDelta.getType(), ito::tTimeDelta);
}

TEST(DateTimeTest, dateTimeSetToTest)
{
    ito::DataObject dObjDateTime(10, 10, ito::tDateTime);

    EXPECT_THROW(dObjDateTime.setTo(ito::TimeDelta(100)), cv::Exception);
    
    dObjDateTime.setTo(ito::DateTime(9500005));
    const auto dt = dObjDateTime.at<ito::DateTime>(5, 5);
    EXPECT_EQ(dt.utcOffset, 0);
    EXPECT_EQ(dt.datetime, 9500005);

    ito::DataObject dObjTimeDelta(10, 10, ito::tTimeDelta);

    EXPECT_THROW(dObjTimeDelta.setTo(ito::DateTime(100)), cv::Exception);
    
    dObjDateTime.setTo(ito::DateTime(1234));
    const auto td = dObjDateTime.at<ito::DateTime>(5, 5);
    EXPECT_EQ(td.utcOffset, 0);
    EXPECT_EQ(td.datetime, 1234);
}

