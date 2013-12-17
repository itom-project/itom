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
template <typename _Tp> class miscellaneousTests : public ::testing::Test 
	{ 
public:
	
	virtual void SetUp(void)
	{
		int *temp_size1 = new int[2];
		temp_size1[0] =10;
		temp_size1[1] =10;
		dObj1 = ito::DataObject(0,temp_size1,ito::getDataType( (const _Tp *) NULL ));
		dObj2 = ito::DataObject(2,temp_size1,ito::getDataType( (const _Tp *) NULL ));
		dObj3 = ito::DataObject(3,3,10,ito::getDataType( (const _Tp *) NULL ));
		int *temp_size = new int[5];
		temp_size[0] = 3;
		temp_size[1] = 4;
		temp_size[2] = 2;
		temp_size[3] = 10;
		temp_size[4] = 10;
		dObj4 = ito::DataObject(5,temp_size,ito::getDataType( (const _Tp *) NULL ));
		int *temp_size2 = new int[1];
		temp_size2[0] = 3;
		dObj7 = ito::DataObject(1,temp_size2,ito::getDataType( (const _Tp *) NULL ));
	};
 
	virtual void TearDown(void) {};
	typedef _Tp valueType;	
	ito::DataObject dObj1;
	ito::DataObject dObj2;
	ito::DataObject dObj3;
	ito::DataObject dObj4;
	ito::DataObject dObj5;
	ito::DataObject dObj6;
	ito::DataObject dObj7;
	};
	
TYPED_TEST_CASE(miscellaneousTests, ItomRealDataTypes);

//getDims_getType_Test
/*!
	
*/
TYPED_TEST(miscellaneousTests, getValueOffset_Test)
{
	dObj5 = ito::DataObject(dObj2);
	dObj6 = dObj2;

	//!< Test for getValueOffset() function.
	EXPECT_FLOAT_EQ(0,dObj1.getValueOffset() );
	EXPECT_FLOAT_EQ(0,dObj2.getValueOffset() );
	EXPECT_FLOAT_EQ(0,dObj3.getValueOffset() );
	EXPECT_FLOAT_EQ(0,dObj4.getValueOffset() );
	EXPECT_FLOAT_EQ(0,dObj5.getValueOffset() );
	EXPECT_FLOAT_EQ(0,dObj6.getValueOffset() );
	EXPECT_FLOAT_EQ(0,dObj7.getValueOffset() );
}

TYPED_TEST(miscellaneousTests, getValueScale_Test)
{
	dObj5 = ito::DataObject(dObj2);
	dObj6 = dObj2;

	//!< Test for getValueScale() function.
	EXPECT_FLOAT_EQ(1,dObj1.getValueScale() );
	EXPECT_FLOAT_EQ(1,dObj2.getValueScale() );
	EXPECT_FLOAT_EQ(1,dObj3.getValueScale() );
	EXPECT_FLOAT_EQ(1,dObj4.getValueScale() );
	EXPECT_FLOAT_EQ(1,dObj5.getValueScale() );
	EXPECT_FLOAT_EQ(1,dObj6.getValueScale() );
	EXPECT_FLOAT_EQ(1,dObj7.getValueScale() );
}

TYPED_TEST(miscellaneousTests, getValueUnit_Test)
{
	dObj5 = ito::DataObject(dObj2);
	dObj6 = dObj2;

	//!< Test for getValueUnit() function.
	EXPECT_EQ("",dObj1.getValueUnit() );
	EXPECT_EQ("",dObj2.getValueUnit() );
	EXPECT_EQ("",dObj3.getValueUnit() );
	EXPECT_EQ("",dObj4.getValueUnit() );
	EXPECT_EQ("",dObj5.getValueUnit() );
	EXPECT_EQ("",dObj6.getValueUnit() );
	EXPECT_EQ("",dObj7.getValueUnit() );
}

TYPED_TEST(miscellaneousTests, getValueDescription_Test)
{
	dObj5 = ito::DataObject(dObj2);
	dObj6 = dObj2;
	int i;
	//!< Test for getValueDescription() function.
	EXPECT_EQ("",dObj1.getValueDescription() );
	EXPECT_EQ("",dObj2.getValueDescription() );
	EXPECT_EQ("",dObj3.getValueDescription() );
	EXPECT_EQ("",dObj4.getValueDescription() );
	EXPECT_EQ("",dObj5.getValueDescription() );
	EXPECT_EQ("",dObj6.getValueDescription() );
	EXPECT_EQ("",dObj7.getValueDescription() );
}

TYPED_TEST(miscellaneousTests, getAxisOffset_Test)
{
	int i;
	dObj5 = ito::DataObject(dObj2);
	dObj6 = dObj2;
	//!< Test for getAxisOffset() function.
	int dObj2_dim= dObj2.getDims();
	int dObj3_dim= dObj3.getDims();
	int dObj4_dim= dObj4.getDims();
	int dObj5_dim= dObj5.getDims();
	int dObj6_dim= dObj6.getDims();
	int dObj7_dim= dObj7.getDims();

	for(i = -1;i<2;i++)
	{
	EXPECT_ANY_THROW(dObj1.getAxisOffset(i) );
	}	

	//!<Test for getAxisOffset() function for dObj2.
	EXPECT_ANY_THROW(dObj2.getAxisOffset(-1) );
	for(i=0;i<dObj2_dim;i++)
	{
	EXPECT_EQ(0,dObj2.getAxisOffset(i) );
	}
	EXPECT_ANY_THROW(dObj2.getAxisOffset(dObj2_dim) );	//!< testing if this function throws an exception if the parameter is out of range.
	EXPECT_ANY_THROW(dObj2.getAxisOffset(dObj2_dim+1) );

	//!<Test for getAxisOffset() function for dObj3.
	EXPECT_ANY_THROW(dObj3.getAxisOffset(-1) );
	for(i=0;i<dObj3_dim;i++)
	{
	EXPECT_EQ(0,dObj3.getAxisOffset(i) );
	}
	EXPECT_ANY_THROW(dObj3.getAxisOffset(dObj3_dim) );	//!< testing if this function throws an exception if the parameter is out of range.
	EXPECT_ANY_THROW(dObj3.getAxisOffset(dObj3_dim+1) );

	//!<Test for getAxisOffset() function for dObj4.
	EXPECT_ANY_THROW(dObj4.getAxisOffset(-1) );
	for(i=0;i<dObj4_dim;i++)
	{
	EXPECT_EQ(0,dObj4.getAxisOffset(i) );
	}
	EXPECT_ANY_THROW(dObj4.getAxisOffset(dObj4_dim) );	//!< testing if this function throws an exception if the parameter is out of range.
	EXPECT_ANY_THROW(dObj4.getAxisOffset(dObj4_dim+1) );

	//!<Test for getAxisOffset() function for dObj5.
	EXPECT_ANY_THROW(dObj5.getAxisOffset(-1) );
	for(i=0;i<dObj5_dim;i++)
	{
	EXPECT_EQ(0,dObj5.getAxisOffset(i) );
	}
	EXPECT_ANY_THROW(dObj5.getAxisOffset(dObj5_dim) );	//!< testing if this function throws an exception if the parameter is out of range.
	EXPECT_ANY_THROW(dObj5.getAxisOffset(dObj5_dim+1) );

	//!<Test for getAxisOffset() function for dObj6.
	EXPECT_ANY_THROW(dObj6.getAxisOffset(-1) );
	for(i=0;i<dObj6_dim;i++)
	{
	EXPECT_EQ(0,dObj6.getAxisOffset(i) );
	}
	EXPECT_ANY_THROW(dObj6.getAxisOffset(dObj6_dim) );	//!< testing if this function throws an exception if the parameter is out of range.
	EXPECT_ANY_THROW(dObj6.getAxisOffset(dObj6_dim+1) );

	//!<Test for getAxisOffset() function for dObj7.
	/*< dObj7 is explicitely defined as 1 dimensional Data Object. 
	But as there is no existance of 1 dimensional Data Objects, dObj7 becomes 2 dimensional Data Object. 
	So this test checks this type of conversion and result of getAxisOffset() function accordingly.
	*/
	EXPECT_ANY_THROW(dObj7.getAxisOffset(-1) );
	for(i=0;i<dObj7_dim;i++)
	{
	EXPECT_EQ(0,dObj7.getAxisOffset(i) );
	}
	EXPECT_ANY_THROW(dObj7.getAxisOffset(dObj7_dim) );	//!< testing if this function throws an exception if the parameter is out of range.
	EXPECT_ANY_THROW(dObj7.getAxisOffset(dObj7_dim+1) );
}

TYPED_TEST(miscellaneousTests, getAxisScale_Test)
{
	dObj5 = ito::DataObject(dObj2);
	dObj6 = dObj2;
	int i;
	int dObj2_dim= dObj2.getDims();
	int dObj3_dim= dObj3.getDims();
	int dObj4_dim= dObj4.getDims();
	int dObj5_dim= dObj5.getDims();
	int dObj6_dim= dObj6.getDims();
	int dObj7_dim= dObj7.getDims();
	//!< Test for getAxisOffset() function.

	for(i = -1;i<2;i++)
	{
	EXPECT_ANY_THROW(dObj1.getAxisScale(i) );
	}	

	//!<Test for getAxisScale() function for dObj2.
	EXPECT_ANY_THROW(dObj2.getAxisScale(-1) );
	for(i=0;i<dObj2_dim;i++)
	{
	EXPECT_EQ(1.0,dObj2.getAxisScale(i) );
	}
	EXPECT_ANY_THROW(dObj2.getAxisScale(dObj2_dim) );	//!< testing if this function throws an exception if the parameter is out of range.
	EXPECT_ANY_THROW(dObj2.getAxisScale(dObj2_dim+1) );

	//!<Test for getAxisScale() function for dObj3.
	EXPECT_ANY_THROW(dObj3.getAxisScale(-1) );
	for(i=0;i<dObj3_dim;i++)
	{
	EXPECT_EQ(1.0,dObj3.getAxisScale(i) );
	}
	EXPECT_ANY_THROW(dObj3.getAxisScale(dObj3_dim) );	//!< testing if this function throws an exception if the parameter is out of range.
	EXPECT_ANY_THROW(dObj3.getAxisScale(dObj3_dim+1) );

	//!<Test for getAxisScale() function for dObj4.
	EXPECT_ANY_THROW(dObj4.getAxisScale(-1) );
	for(i=0;i<dObj4_dim;i++)
	{
	EXPECT_EQ(1.0,dObj4.getAxisScale(i) );
	}
	EXPECT_ANY_THROW(dObj4.getAxisScale(dObj4_dim) );	//!< testing if this function throws an exception if the parameter is out of range.
	EXPECT_ANY_THROW(dObj4.getAxisScale(dObj4_dim+1) );

	//!<Test for getAxisScale() function for dObj5.
	EXPECT_ANY_THROW(dObj5.getAxisScale(-1) );
	for(i=0;i<dObj5_dim;i++)
	{
	EXPECT_EQ(1.0,dObj5.getAxisScale(i) );
	}
	EXPECT_ANY_THROW(dObj5.getAxisScale(dObj5_dim) );	//!< testing if this function throws an exception if the parameter is out of range.
	EXPECT_ANY_THROW(dObj5.getAxisScale(dObj5_dim+1) );

	//!<Test for getAxisScale() function for dObj6.
	EXPECT_ANY_THROW(dObj6.getAxisScale(-1) );
	for(i=0;i<dObj6_dim;i++)
	{
	EXPECT_EQ(1.0,dObj6.getAxisScale(i) );
	}
	EXPECT_ANY_THROW(dObj6.getAxisScale(dObj6_dim) );	//!< testing if this function throws an exception if the parameter is out of range.
	EXPECT_ANY_THROW(dObj6.getAxisScale(dObj6_dim+1) );
	
	//!<Test for getAxisScale() function for dObj7. 
	/*< dObj7 is explicitely defined as 1 dimensional Data Object. 
	But as there is no existance of 1 dimensional Data Objects, dObj7 becomes 2 dimensional Data Object. 
	So this test checks this type of conversion and result of getAxisScale() function accordingly.
	*/
	EXPECT_ANY_THROW(dObj7.getAxisScale(-1) );
	for(i=0;i<dObj7_dim;i++)
	{
	EXPECT_EQ(1.0,dObj7.getAxisScale(i) );
	}
	EXPECT_ANY_THROW(dObj7.getAxisScale(dObj7_dim) );	//!< testing if this function throws an exception if the parameter is out of range.
	EXPECT_ANY_THROW(dObj7.getAxisScale(dObj7_dim+1) );
}

TYPED_TEST(miscellaneousTests, getXYRotationalMatrix_Test)
{
	dObj5 = ito::DataObject(dObj2);
	dObj6 = dObj2;
	double  r00,r01,r02,r10,r11,r12,r20,r21,r22;

	//!< Test for getXYRotationMatrix() function.
	//	dObj1.getXYRotationalMatrix( r00,r01,r02,r10,r11,r12,r20,r21,r22 ) ;		//Note: This test fails for Obj1 (Empty DataObject).............................................
	//EXPECT_EQ(1,r00);
	//EXPECT_EQ(0,r01);
	//EXPECT_EQ(0,r02);
	//EXPECT_EQ(0,r10);
	//EXPECT_EQ(1,r11);
	//EXPECT_EQ(0,r12);
	//EXPECT_EQ(0,r20);
	//EXPECT_EQ(0,r21);
	//EXPECT_EQ(1,r22);

	r00,r01,r02,r10,r11,r12,r20,r21,r22=0;
	dObj2.getXYRotationalMatrix( r00,r01,r02,r10,r11,r12,r20,r21,r22 ) ;
	EXPECT_EQ(1,r00);
	EXPECT_EQ(0,r01);
	EXPECT_EQ(0,r02);
	EXPECT_EQ(0,r10);
	EXPECT_EQ(1,r11);
	EXPECT_EQ(0,r12);
	EXPECT_EQ(0,r20);
	EXPECT_EQ(0,r21);
	EXPECT_EQ(1,r22);

	r00,r01,r02,r10,r11,r12,r20,r21,r22=0;
		dObj3.getXYRotationalMatrix( r00,r01,r02,r10,r11,r12,r20,r21,r22 ) ;
	EXPECT_EQ(1,r00);
	EXPECT_EQ(0,r01);
	EXPECT_EQ(0,r02);
	EXPECT_EQ(0,r10);
	EXPECT_EQ(1,r11);
	EXPECT_EQ(0,r12);
	EXPECT_EQ(0,r20);
	EXPECT_EQ(0,r21);
	EXPECT_EQ(1,r22);

	r00,r01,r02,r10,r11,r12,r20,r21,r22=0;
		dObj4.getXYRotationalMatrix( r00,r01,r02,r10,r11,r12,r20,r21,r22 ) ;
	EXPECT_EQ(1,r00);
	EXPECT_EQ(0,r01);
	EXPECT_EQ(0,r02);
	EXPECT_EQ(0,r10);
	EXPECT_EQ(1,r11);
	EXPECT_EQ(0,r12);
	EXPECT_EQ(0,r20);
	EXPECT_EQ(0,r21);
	EXPECT_EQ(1,r22);

	r00,r01,r02,r10,r11,r12,r20,r21,r22=0;
		dObj5.getXYRotationalMatrix( r00,r01,r02,r10,r11,r12,r20,r21,r22 ) ;
	EXPECT_EQ(1,r00);
	EXPECT_EQ(0,r01);
	EXPECT_EQ(0,r02);
	EXPECT_EQ(0,r10);
	EXPECT_EQ(1,r11);
	EXPECT_EQ(0,r12);
	EXPECT_EQ(0,r20);
	EXPECT_EQ(0,r21);
	EXPECT_EQ(1,r22);

	r00,r01,r02,r10,r11,r12,r20,r21,r22=0;
		dObj6.getXYRotationalMatrix( r00,r01,r02,r10,r11,r12,r20,r21,r22 ) ;
	EXPECT_EQ(1,r00);
	EXPECT_EQ(0,r01);
	EXPECT_EQ(0,r02);
	EXPECT_EQ(0,r10);
	EXPECT_EQ(1,r11);
	EXPECT_EQ(0,r12);
	EXPECT_EQ(0,r20);
	EXPECT_EQ(0,r21);
	EXPECT_EQ(1,r22);

	r00,r01,r02,r10,r11,r12,r20,r21,r22=0;
		dObj7.getXYRotationalMatrix( r00,r01,r02,r10,r11,r12,r20,r21,r22 ) ;
	EXPECT_EQ(1,r00);
	EXPECT_EQ(0,r01);
	EXPECT_EQ(0,r02);
	EXPECT_EQ(0,r10);
	EXPECT_EQ(1,r11);
	EXPECT_EQ(0,r12);
	EXPECT_EQ(0,r20);
	EXPECT_EQ(0,r21);
	EXPECT_EQ(1,r22);
}
