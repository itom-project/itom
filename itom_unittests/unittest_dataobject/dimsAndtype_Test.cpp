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


/*! \class ROITest
	\brief ROI methods test for real data types

	This test class checks functionality of different methods dealing with ROI for data objects.
*/
template <typename _Tp> class DimAndTypeTest : public ::testing::Test 
	{ 
public:
	
	virtual void SetUp(void)
	{
		//dObj1 = ito::DataObject(0,temp_size1,ito::getDataType( (const _Tp *) NULL ));
		size_t *temp_size1 = new size_t[1];
		temp_size1[0] =10;
		dObj1 = ito::DataObject(0,temp_size1,ito::getDataType( (const _Tp *) NULL ));
		dObj2 = ito::DataObject(1,temp_size1,ito::getDataType( (const _Tp *) NULL ));
		dObj3 = ito::DataObject(5,10,ito::getDataType( (const _Tp *) NULL ));
		dObj4 = ito::DataObject(2,3,4,ito::getDataType( (const _Tp *) NULL ));
		size_t *temp_size = new size_t[5];
		temp_size[0] = 3;
		temp_size[1] = 4;
		temp_size[2] = 2;
		temp_size[3] = 10;
		temp_size[4] = 10;
		dObj5 = ito::DataObject(5,temp_size,ito::getDataType( (const _Tp *) NULL ));
	};
 
	virtual void TearDown(void) {};
	typedef _Tp valueType;	

	ito::DataObject dObj1;
	ito::DataObject dObj2;
	ito::DataObject dObj3;
	ito::DataObject dObj4;
	ito::DataObject dObj5;
	};
	
TYPED_TEST_CASE(DimAndTypeTest, ItomRealDataTypes);

//getDims_getType_Test
/*!
	This test adjust the ROI of 3 dimensional matrices to check proper functionality of "adjustROI" method. It also checks "locateROI" method by comparing obtained offsets with original values.
*/
TYPED_TEST(DimAndTypeTest, getDims_Test)
{
	ito::tDataType testType = ito::getDataType( (const TypeParam *) NULL);
	EXPECT_EQ(0,dObj1.getDims() );
	EXPECT_EQ(2,dObj2.getDims() );
	EXPECT_EQ(2,dObj3.getDims() );
	EXPECT_EQ(3,dObj4.getDims() );
	EXPECT_EQ(5,dObj5.getDims() );

	EXPECT_EQ(testType,dObj1.getType() );           
	EXPECT_EQ(testType,dObj2.getType() );
	EXPECT_EQ(testType,dObj3.getType() );
	EXPECT_EQ(testType,dObj4.getType() );
	EXPECT_EQ(testType,dObj5.getType() );
}

TYPED_TEST(DimAndTypeTest, getType_Test)
{
	ito::tDataType testType = ito::getDataType( (const TypeParam *) NULL);

	EXPECT_EQ(testType,dObj1.getType() );           
	EXPECT_EQ(testType,dObj2.getType() );
	EXPECT_EQ(testType,dObj3.getType() );
	EXPECT_EQ(testType,dObj4.getType() );
	EXPECT_EQ(testType,dObj5.getType() );
}