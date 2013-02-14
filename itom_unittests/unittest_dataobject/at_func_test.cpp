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
template <typename _Tp> class at_func_test : public ::testing::Test 
	{ 
public:
	
	virtual void SetUp(void)
	{
		size_t *temp_size1 = new size_t[2];
		temp_size1[0] =10;
		temp_size1[1] =10;
		dObj1 = ito::DataObject(0,temp_size1,ito::getDataType( (const _Tp *) NULL ));
		dObj2 = ito::DataObject(2,temp_size1,ito::getDataType( (const _Tp *) NULL ));
		dObj3 = ito::DataObject(4,5,5,ito::getDataType( (const _Tp *) NULL ));
		size_t *temp_size = new size_t[5];
		temp_size[0] = 4;
		temp_size[1] = 5;
		temp_size[2] = 5;
		temp_size[3] = 4;
		temp_size[4] = 3;
		dObj4 = ito::DataObject(5,temp_size,ito::getDataType( (const _Tp *) NULL ));
	};
 
	virtual void TearDown(void) {};

	int calcUniqueValue5D(int d1, int d2, int d3, int d4, int d5)
	{
		return  d5 + d4 * 10 + d3 * 100 + d2 * 1000 + d1 * 10000 ;
	}

	typedef _Tp valueType;	
	ito::DataObject dObj1;
	ito::DataObject dObj2;
	ito::DataObject dObj3;
	ito::DataObject dObj4;
	};
	
TYPED_TEST_CASE(at_func_test, ItomRealDataTypes);

//!< This test checks the functionality of at(ito::Range *ranges) function.
TYPED_TEST(at_func_test, at_Test)
{
	int check_arry1[] = {0,1,2,10,11,12,20,21,22,30,31,32};

	for(int i =0; i<10;i++)
	{
		for(int j=0; j<10;j++)
		{
			dObj2.at<TypeParam>(i,j)= cv::saturate_cast<TypeParam>(10*i+j);
		}
	}

	//!< Test code with valid ranges less than dObj2
	ito::DataObject dObj2_test;
	ito::Range ranges[2];
	ranges[0].start = 0;
	ranges[0].end =4;
	ranges[1].start = 0;
	ranges[1].end =3;

	dObj2_test=dObj2.at(ranges);
	EXPECT_EQ(dObj2.getDims(), dObj2_test.getDims());
	EXPECT_EQ(cv::saturate_cast<TypeParam>(4),dObj2_test.getSize(0));
	EXPECT_EQ(cv::saturate_cast<TypeParam>(3),dObj2_test.getSize(1));
	int temp=0;
	for(int i=0;i<4;i++)
	{
		for(int j=0;j<3;j++)
		{
		EXPECT_EQ(dObj2_test.at<TypeParam>(i,j), check_arry1[temp++]);
		}
	}
	

	//!< Test code with valid ranges less than dObj3
	temp=0;
	for(int i=0;i<4; i++)
	{
		for(int j=0;j<5;j++)
		{
			for(int k=0;k<5;k++)
			{
				dObj3.at<TypeParam>(i,j,k) = cv::saturate_cast<TypeParam>(temp++);
			}
		}
	}
	ito::DataObject dObj3_test;
	ito::Range ranges_3d[3];
	ranges_3d[0].start =1;
	ranges_3d[0].end =3;
	ranges_3d[1].start =0;
	ranges_3d[1].end =2;
	ranges_3d[2].start =0;
	ranges_3d[2].end =3;
	int check_arry2[] = {25,26,27,30,31,32,50,51,52,55,56,57};

	dObj3_test = dObj3.at(ranges_3d);
	EXPECT_EQ(dObj3.getDims(), dObj3_test.getDims());
	EXPECT_EQ(cv::saturate_cast<TypeParam>(2),dObj3_test.getSize(0));
	EXPECT_EQ(cv::saturate_cast<TypeParam>(2),dObj3_test.getSize(1));
	EXPECT_EQ(cv::saturate_cast<TypeParam>(3),dObj3_test.getSize(2));

	temp=0;
	for(int i=0;i<2; i++)
	{
		for(int j=0;j<2;j++)
		{
			for(int k=0;k<3;k++)
			{
				EXPECT_EQ(dObj3_test.at<TypeParam>(i,j,k),cv::saturate_cast<TypeParam>(check_arry2[temp++]));
			}
		}
	}

	std::cout<< dObj3_test<<std::endl;

	//!< Test code with full ranges
	ito::DataObject dObj2_test1;
	ito::Range ranges1[2];
	ranges1[0] = ranges1[0].all();
	ranges1[1] = ranges1[1].all();

	dObj2_test1 = dObj2.at(ranges1);
	int test_dim1 = dObj2_test1.getDims();
	size_t test_size1 = dObj2_test1.getSize(0);

	std::cout<<dObj2_test1<<std::endl;
	std::cout<<test_dim1<<std::endl;
	std::cout<<test_size1<<std::endl;



}