#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv/cv.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"


/*! \class locateROI_func_test
	\brief Test for locateROI method for real data types

	This test class checks functionality of locateROI method for different Data Objects of different real datatypes.
	This test contains 3 test cases for 2 dimensional, 3 dimensional and 5 dimensional data objects each.
*/
template <typename _Tp> class locateROI_func_test : public ::testing::Test 
	{ 
public:
	
	virtual void SetUp(void)
	{
		int *temp_size1 = new int[2];
		temp_size1[0] =10;
		temp_size1[1] =10;
		dObj2 = ito::DataObject(2,temp_size1,ito::getDataType( (const _Tp *) NULL )); //!< creating 2 dimensional data object 
		dObj3 = ito::DataObject(4,5,5,ito::getDataType( (const _Tp *) NULL ));		  //!< creating 3 dimensional data object
		int *temp_size = new int[5];
		temp_size[0] = 4;
		temp_size[1] = 5;
		temp_size[2] = 5;
		temp_size[3] = 4;
		temp_size[4] = 3;
		dObj4 = ito::DataObject(5,temp_size,ito::getDataType( (const _Tp *) NULL ));	//!< creating 5 dimesional data object
	};
 
	virtual void TearDown(void) {};

	typedef _Tp valueType;	
	ito::DataObject dObj2;		
	ito::DataObject dObj3;
	ito::DataObject dObj4;
	};
	
TYPED_TEST_CASE(locateROI_func_test, ItomRealDataTypes);

//locateROI_Test1
/*! 
	This test checks the functionality of locateROI() function on 2 dimensional data object dObj2.
*/
TYPED_TEST(locateROI_func_test, locateROI_Test1)
{			
	int matLimits2d[] = {-4,-1,-5,-3};			//!< defining ROI offsets for 2 Dimensional Data Object dObj2 
	
	int lims2d[]= {0,0,0,0};					//!< Empty array to store result of locateROI() function.
	int exptLims2d[]= {2,-3,1,-3};				//!< expected result after applying locateROI() function on dObj2
	dObj2.adjustROI(-2,-1,-1,-2);				//!< Adjusting the ROI of 2 dimensional data object dObj2 with four parameter implementation.
	dObj2.locateROI(lims2d);					//!< adjusting ROI of dObj2 using locateROI() function.
	for(int i =0;i<4; i++)
	{
		EXPECT_EQ(lims2d[i],exptLims2d[i]);		//!< Checking if the result of locateROI() function match with expected result.
	}
	dObj2.adjustROI(2,1,1,2);					//!< Adjusting back the ROI back to normal position.
	dObj2.adjustROI(2,matLimits2d);				//!< adjusting ROI of dObj2 with general 2 parameter adjustROI method to desired position
	int lims2d1[]={0,0,0,0};					//!< Empty array to store result of locateROI() function.
	int exptLims2d1[]= {4,-5,5,-8};				//!< expected result after applying locateROI() function on dObj2
	dObj2.locateROI(lims2d1);					//!< adjusting ROI of dObj2 using locateROI() function.
	for(int i =0; i <4; i++)
	{
		EXPECT_EQ(exptLims2d1[i],lims2d1[i]);   //!< checking expected result of locateROI() function on 2 dimensional data object dObj2
	}
}

//locateROI_Test2
/*!
	This test checks the functionality of locateROI() function on 3 dimensional data object dObj3
*/
TYPED_TEST(locateROI_func_test, locateROI_Test2)
{
	int matLimits3d[] = {-1,-1,-1,-1,-2,-1};	//!< defining ROI offsets for 3 Dimensional Data Object dObj3
	int lims3d[] = {0,0,0,0,0,0};				//!< Empty array to store result of locateROI() function.
	int exptLims3d[] = {1,-2,1,-2,2,-3};		//!< expected result after applying locateROI() function on dObj3
	dObj3.adjustROI(3,matLimits3d);				//!< adjusting ROI of dObj3 with general 2 parameter adjustROI method to desired position
	dObj3.locateROI(lims3d);					//!< locating ROI of dObj3 using locateROI() function.
	for(int i =0; i<6;i++)
	{
		EXPECT_EQ(lims3d[i],exptLims3d[i]);		//! Checking if the result of locateROI() function match with expected result.
	}
}

//locateROI_Test3
/*! 
	This test checks the functionality of locateROI() function on 5 dimensional data object dObj4
*/
TYPED_TEST(locateROI_func_test, locateROI_Test3)
{
	int matLimits5d[] = {0,-3,0,-4,0,0,-1,-1,-2,0}; //!< defining ROI offsets for 5 Dimensional Data Object dObj4
	int lims5d[] = {0,0,0,0,0,0,0,0,0,0};			//!< Empty array to store result of locateROI() function.
	int exptLims5d[] = {0,-3,0,-4,0,0,1,-2,2,-2};	//!< expected result after applying locateROI() function on dObj4
	dObj4.adjustROI(5,matLimits5d);					//!< adjusting ROI of dObj4 with general 2 parameter adjustROI method to desired position
	dObj4.locateROI(lims5d);						//!< locating ROI of dObj4 using locateROI() function.
	for(int i =0; i<10;i++)
	{
		EXPECT_EQ(lims5d[i],exptLims5d[i]);			//!< Checking if the result of locateROI() function match with expected result.
	}
}