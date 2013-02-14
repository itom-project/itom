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
template <typename _Tp> class copyTests : public ::testing::Test 
	{ 
public:
	
	virtual void SetUp(void)
	{
		size_t *temp_size1 = new size_t[2];
		temp_size1[0] =10;
		temp_size1[1] =10;
		dObj1_s = ito::DataObject(0,temp_size1,ito::getDataType( (const _Tp *) NULL ));
		dObj2_s = ito::DataObject(2,temp_size1,ito::getDataType( (const _Tp *) NULL ));
		dObj3_s = ito::DataObject(4,5,5,ito::getDataType( (const _Tp *) NULL ));
		size_t *temp_size = new size_t[5];
		temp_size[0] = 10;
		temp_size[1] = 12;
		temp_size[2] = 16;
		temp_size[3] = 18;
		temp_size[4] = 10;
		dObj4_s = ito::DataObject(5,temp_size,ito::getDataType( (const _Tp *) NULL ));
		size_t *temp_size2 = new size_t[5];
		temp_size2[0] = 1;
		temp_size2[1] = 1;
		temp_size2[2] = 2;
		temp_size2[3] = 1;
		temp_size2[4] = 1;
		dObj4_s1 = ito::DataObject(5,temp_size2,ito::getDataType( (const _Tp *) NULL ));

		dObj1_d = ito::DataObject(0,temp_size1,ito::getDataType( (const _Tp *) NULL ));
		dObj2_d = ito::DataObject(2,temp_size1,ito::getDataType( (const _Tp *) NULL ));
		dObj3_d = ito::DataObject(3,3,10,ito::getDataType( (const _Tp *) NULL ));
		dObj4_d = ito::DataObject(5,temp_size,ito::getDataType( (const _Tp *) NULL ));
	};
 
	virtual void TearDown(void) {};
	typedef _Tp valueType;	
	ito::DataObject dObj1_s;
	ito::DataObject dObj2_s;
	ito::DataObject dObj3_s;
	ito::DataObject dObj4_s;
	ito::DataObject dObj4_s1;

	ito::DataObject dObj1_sr;
	ito::DataObject dObj2_sr;
	ito::DataObject dObj3_sr;
	ito::DataObject dObj4_sr;

	ito::DataObject dObj1_d;
	ito::DataObject dObj2_d;
	ito::DataObject dObj3_d;
	ito::DataObject dObj4_d;

	ito::DataObject dObj1_dr;
	ito::DataObject dObj2_dr;
	ito::DataObject dObj3_dr;
	ito::DataObject dObj4_dr;

	ito::DataObject dObj1_dr1;
	ito::DataObject dObj2_dr1;
	ito::DataObject dObj3_dr1;
	ito::DataObject dObj4_dr1;
	};
	
TYPED_TEST_CASE(copyTests, ItomRealDataTypes);

//getDims_getType_Test
/*!
	This test adjust the ROI of 3 dimensional matrices to check proper functionality of "adjustROI" method. It also checks "locateROI" method by comparing obtained offsets with original values.
*/
TYPED_TEST(copyTests, copyTo_True_Test)
{
	int temp=0;
	unsigned int res1_str[] = {41,42,43,44,45,51,52,53,54,55};			//!< Expected element values for dObj2_dr after copying ROI of dObj2_sr by copyTo() function into it.
	unsigned int res2_str[] = {32,33,37,38,42,43,57,58,62,63,67,68};	//!< Expected element values for dObj3_dr after copying ROI of dObj3_sr by copyTo() function into it.
		for(int i =0;i<10;i++) 
		{
			for(int j=0;j<10;j++)
			{dObj2_s.at<TypeParam>(i,j) = cv::saturate_cast<TypeParam>(10*i+j);}   //!< assigning unique value to each element of Data Object dObj2_s.
		}

	for(int i =0;i<4;i++) 
		{
			for(int j=0;j<5;j++)
			{  
				for(int k=0;k<5;k++)
				{
					dObj3_s.at<TypeParam>(i,j,k) = cv::saturate_cast<TypeParam>(temp++);   //!< assigning unique value to each element of 3 dimensional Data Object dObj3_s
				}
			}
		}

	dObj1_sr = dObj1_s; 
	dObj2_sr = dObj2_s;
	dObj3_sr = dObj3_s;
	dObj4_sr = dObj4_s;
			
	int matLimits1d[] = {-4,-6};			//!< defining offsets for ROI 
	int matLimits2d[] = {-4,-4,-1,-4};		//!< defining offsets for ROI 
	int matLimits3d[] = {-1,-1,-1,-1,-2,-1};
	int matLimits5d[] = {-4,-4,-1,-4,-2,-3,-1,-1,-2,-1};

	dObj1_sr.adjustROI(0,matLimits1d);			 //!< adjust ROI (shrinking because offset values are negative)
	dObj2_sr.adjustROI(2,matLimits2d);   //!< adjust ROI (shrinking because offset values are negative)
	dObj3_sr.adjustROI(3,matLimits3d);   //!< adjust ROI (shrinking because offset values are negative)
	dObj4_sr.adjustROI(5,matLimits5d);   //!< adjust ROI (shrinking because offset values are negative)
	
	//!< Testing functionality of copyTo() function for empty Data Object with regionOnly parameter = True.
	dObj1_s.copyTo(dObj1_d,true);					
	EXPECT_EQ(0,dObj1_d.getDims());					//!< Testing if dimension of dObj1_d is same as dimension of ROI of dObj1_s.
	EXPECT_EQ(dObj1_s.getSize(),dObj1_d.getSize() );	//!< Testing if size of dObj1_d is same as dimension of ROI of dObj1_s.

	//!< Testing functionality of copyTo() function for 2 dimensional Data Object with regionOnly parameter = True.
	dObj2_s.copyTo(dObj2_d,true);
	EXPECT_EQ(2,dObj2_d.getDims());
	EXPECT_EQ(dObj2_s.getSize(),dObj2_d.getSize() );

	//!< Testing functionality of copyTo() function for 3 dimensional Data Object with regionOnly parameter = True.
	dObj3_s.copyTo(dObj3_d,true);
	EXPECT_EQ(3,dObj3_d.getDims());
	EXPECT_EQ(dObj3_s.getSize(), dObj3_d.getSize() );    //!< Testing if the size of ROI of copied Data Object is same as the ROI of original Data Object.

	//!< Testing functionality of copyTo() function for 5 dimensional Data Object with regionOnly parameter = True.
	dObj4_s.copyTo(dObj4_d,true);
	EXPECT_EQ(5,dObj4_d.getDims());
	EXPECT_EQ(dObj4_s.getSize(), dObj4_d.getSize() );    //!< Testing if the size of ROI of copied Data Object is same as the ROI of original Data Object.

	//!< Testing functionality of copyTo() function for empty Data Object with regionOnly parameter = True.
	dObj1_sr.copyTo(dObj1_dr,true);
	EXPECT_EQ(0,dObj1_dr.getDims());
	EXPECT_EQ(dObj1_sr.getSize(), dObj1_dr.getSize() ); 

	//!< Testing functionality of copyTo() function for 2 dimensional Data Object with regionOnly parameter = True.
	dObj2_sr.copyTo(dObj2_dr,true);   //!< Copying only ROI from dObj2_sr to dObj2_dr.
	EXPECT_EQ(dObj2_sr.getDims() ,dObj2_dr.getDims() );	//!< Testing if the dimensions of ROI of copied Data Object are same as the ROI of original Data Object.
	EXPECT_EQ(dObj2_sr.getSize(), dObj2_dr.getSize() );    //!< Testing if the size of ROI of copied Data Object is same as the ROI of original Data Object.
	for(int i =0;i<2;i++) 
		{
			for(int j=0;j<5;j++)EXPECT_EQ(dObj2_dr.at<TypeParam>(i,j),cv::saturate_cast<TypeParam>(res1_str[i*5+j])); //!< Check if the values of elements in ROI are same as in the original				
		}

	//!< Testing functionality of copyTo() function for 3 dimensional Data Object with regionOnly parameter = True.
	dObj3_sr.copyTo(dObj3_dr,true);
	EXPECT_EQ(dObj3_sr.getDims() ,dObj3_dr.getDims() );	//!< Testing if the dimensions of ROI of copied Data Object are same as the ROI of original Data Object.
	EXPECT_EQ(dObj3_sr.getSize() ,dObj3_dr.getSize() );	 //!< Testing if the size of ROI of copied Data Object is same as the ROI of original Data Object.
	std::cout<<"dObj3_dr"<<dObj3_dr<<std::endl;

	temp=0;
	for(int i =0;i<2;i++) 
		{
			for(int j=0;j<3;j++)
			{  
				for(int k=0;k<2;k++)
				{
					EXPECT_EQ(dObj3_dr.at<TypeParam>(i,j,k),cv::saturate_cast<TypeParam>(res2_str[temp++]) );   //!< Checking if elements of ROI of original Data Object dObj3_sr has been changed while copying it to dObj3_dr.
				}
			}
		}

	//!< Testing functionality of copyTo() function for 5 dimensional Data Object with regionOnly parameter = True.
	dObj4_sr.copyTo(dObj4_dr,true);
	EXPECT_EQ(dObj4_sr.getDims(),dObj4_dr.getDims() );	//!< Testing if the dimensions of ROI of copied Data Object are same as the ROI of original Data Object.
	EXPECT_EQ(dObj4_sr.getSize() ,dObj4_dr.getSize() );	//!< Testing if the size of ROI of copied Data Object is same as the ROI of original Data Object.
}


TYPED_TEST(copyTests, copyTo_False_Test)
{

	int expSize2d_sr[] = {12,13};  
	int expOffsets2d_sr[] = {4,1};

	int orgSize2d_sr[] = {0,0};
	int offsets2d_sr[] = {0,0};
	int orgSize2d_dr[] = {0,0};
	int offsets2d_dr[] = {0,0};

	int orgSize3d_sr[] = {0,0,0};
	int offsets3d_sr[] = {0,0,0};
	int orgSize3d_dr[] = {0,0,0};
	int offsets3d_dr[] = {0,0,0};

	int temp_index[] = {0,0,0,0,0};

	int temp=0;
	unsigned int res1_str[] = {41,42,43,44,45,51,52,53,54,55};			//!< Expected element values for dObj2_dr after copying ROI of dObj2_sr by copyTo() function into it.
	unsigned int res2_str[] = {32,33,37,38,42,43,57,58,62,63,67,68};	//!< Expected element values for dObj3_dr after copying ROI of dObj3_sr by copyTo() function into it.
		for(int i =0;i<10;i++) 
		{
			for(int j=0;j<10;j++)
			{dObj2_s.at<TypeParam>(i,j) = cv::saturate_cast<TypeParam>(10*i+j);}   //!< assigning unique value to each element of Data Object dObj2_s.
		}

	for(int i =0;i<4;i++) 
		{
			for(int j=0;j<5;j++)
			{  
				for(int k=0;k<5;k++)
				{
					dObj3_s.at<TypeParam>(i,j,k) = cv::saturate_cast<TypeParam>(temp++);   //!< assigning unique value to each element of 3 dimensional Data Object dObj3_s
				}
			}
		}


	temp=0;

	// NOTE: How to assign values to 5d Data Objects.
	for(int i =0;i<10;i++) 
		{
			for(int j=0;j<12;j++)
			{  
				for(int k=0;k<16;k++)
				{
					for(int l=0;l<18;l++)
					{
						for(int m=0;m<10;m++)
						{
						//	dObj4_s.at<TypeParam>(temp_index[0,0,0,0,0]) = cv::saturate_cast<TypeParam>(temp++);   //!< assigning unique value to each element of 3 dimensional Data Object dObj3_s
						
						}
					}
				}
			}
		}

	std::cout<<"Data Object 4:"<<dObj4_s1<<std::endl;
	dObj1_sr = dObj1_s; 
	dObj2_sr = dObj2_s;
	dObj3_sr = dObj3_s;
	dObj4_sr = dObj4_s;
			
	int matLimits1d[] = {-4,-6};			//!< defining offsets for ROI 
	int matLimits2d_1[] = {-4,-4,-1,-4};		//!< defining offsets for ROI 
	int matLimits2d_2[] = {4,4,1,4};
	int matLimits3d_1[] = {-1,-1,-1,-1,-2,-1};
	int matLimits3d_2[]	= {1,1,1,1,2,1};
	int matLimits5d[] = {-4,-4,-1,-4,-2,-3,-1,-1,-2,-1};

	dObj1_sr.adjustROI(0,matLimits1d);			 //!< adjust ROI (shrinking because offset values are negative)
	dObj2_sr.adjustROI(2,matLimits2d_1);   //!< adjust ROI (shrinking because offset values are negative)
	dObj3_sr.adjustROI(3,matLimits3d_1);   //!< adjust ROI (shrinking because offset values are negative)
	dObj4_sr.adjustROI(5,matLimits5d);   //!< adjust ROI (shrinking because offset values are negative)
	
	//!< Testing functionality of copyTo() function for empty Data Object with regionOnly parameter = False.
	dObj1_s.copyTo(dObj1_d,false);					
	EXPECT_EQ(0,dObj1_d.getDims());					//!< Testing if dimension of dObj1_d is same as dimension of ROI of dObj1_s.
	EXPECT_EQ(dObj1_s.getSize(),dObj1_d.getSize() );	//!< Testing if size of dObj1_d is same as dimension of ROI of dObj1_s.

	//!< Testing functionality of copyTo() function for 2 dimensional Data Object with regionOnly parameter = False.
	dObj2_s.copyTo(dObj2_d,false);
	EXPECT_EQ(2,dObj2_d.getDims());
	EXPECT_EQ(dObj2_s.getSize(),dObj2_d.getSize() );

	//!< Testing functionality of copyTo() function for 3 dimensional Data Object with regionOnly parameter = False.
	dObj3_s.copyTo(dObj3_d,false);
	EXPECT_EQ(3,dObj3_d.getDims());
	EXPECT_EQ(dObj3_s.getSize(), dObj3_d.getSize() );    //!< Testing if the size of ROI of copied Data Object is same as the ROI of original Data Object.

	//!< Testing functionality of copyTo() function for 5 dimensional Data Object with regionOnly parameter = False.
	dObj4_s.copyTo(dObj4_d,false);
	EXPECT_EQ(5,dObj4_d.getDims());
	EXPECT_EQ(dObj4_s.getSize(), dObj4_d.getSize() );    //!< Testing if the size of ROI of copied Data Object is same as the ROI of original Data Object.

	//!< Testing functionality of copyTo() function for empty Data Object with regionOnly parameter = True.
	dObj1_sr.copyTo(dObj1_dr,false);
	EXPECT_EQ(0,dObj1_dr.getDims());
	EXPECT_EQ(dObj1_sr.getSize(), dObj1_dr.getSize() ); 

	//!< Testing functionality of copyTo() function for 2 dimensional Data Object with regionOnly parameter = False.
	dObj2_sr.copyTo(dObj2_dr,false);   //!< Copying only ROI from dObj2_sr to dObj2_dr.
	EXPECT_EQ(dObj2_sr.getDims() ,dObj2_dr.getDims() );	//!< Testing if the dimensions of ROI of copied Data Object are same as the ROI of original Data Object.
	EXPECT_EQ(dObj2_sr.getSize(), dObj2_dr.getSize() );    //!< Testing if the size of ROI of copied Data Object is same as the ROI of original Data Object.
	for(int i =0;i<2;i++) 
		{
			for(int j=0;j<5;j++)EXPECT_EQ(dObj2_dr.at<TypeParam>(i,j),cv::saturate_cast<TypeParam>(res1_str[i*5+j])); //!< Check if the values of elements in ROI are same as in the original				
		}

	dObj2_dr.locateROI(orgSize2d_dr,offsets2d_dr);
	dObj2_sr.locateROI(orgSize2d_sr,offsets2d_sr);
	for(int i=0;i<2;i++) EXPECT_EQ(orgSize2d_dr[i],orgSize2d_sr[i]);
	for(int i=0;i<2;i++) EXPECT_EQ(offsets2d_dr[i],offsets2d_sr[i]);
	
	dObj2_dr.adjustROI(2,matLimits2d_2); 
		for(int i =0;i<10;i++) 
		{
			for(int j=0;j<10;j++)
			{EXPECT_EQ( dObj2_s.at<TypeParam>(i,j),dObj2_dr.at<TypeParam>(i,j) );}   //!< checking if the original values of all elements retained after expanding ROI same as original Data Object dObj2_s.
		}

	//!< Testing functionality of copyTo() function for 3 dimensional Data Object with regionOnly parameter = False.
	dObj3_sr.copyTo(dObj3_dr,false);
	EXPECT_EQ(dObj3_sr.getDims() ,dObj3_dr.getDims() );	//!< Testing if the dimensions of ROI of copied Data Object are same as the ROI of original Data Object.
	EXPECT_EQ(dObj3_sr.getSize() ,dObj3_dr.getSize() );	 //!< Testing if the size of ROI of copied Data Object is same as the ROI of original Data Object.
	//std::cout<<"dObj3_dr"<<dObj3_dr<<std::endl;

	temp=0;
	for(int i =0;i<2;i++) 
		{
			for(int j=0;j<3;j++)
			{  
				for(int k=0;k<2;k++)
				{
					EXPECT_EQ(dObj3_dr.at<TypeParam>(i,j,k),cv::saturate_cast<TypeParam>(res2_str[temp++]) );   //!< Checking if elements of ROI of original Data Object dObj3_sr has been changed while copying it to dObj3_dr.
				}
			}
		}

	dObj3_dr.locateROI(orgSize3d_dr,offsets3d_dr);
	dObj3_sr.locateROI(orgSize3d_sr,offsets3d_sr);
	for(int i=0;i<3;i++) EXPECT_EQ(orgSize3d_dr[i],orgSize3d_sr[i]);
	for(int i=0;i<3;i++) EXPECT_EQ(offsets3d_dr[i],offsets3d_sr[i]);

	dObj3_dr.adjustROI(3,matLimits3d_2);   //!< adjust ROI back to original size of source Data Object dObj3_s. 

	for(int i =0;i<4;i++)		
		{
			for(int j=0;j<5;j++)
			{  
				for(int k=0;k<5;k++)
				{
					EXPECT_EQ(dObj3_dr.at<TypeParam>(i,j,k),dObj3_s.at<TypeParam>(i,j,k) );   //!< Checking if elements of ROI of original Data Object dObj3_dr has been changed while adusting the size of ROI back to original.
				}
			}
		}

	//!< Testing functionality of copyTo() function for 5 dimensional Data Object with regionOnly parameter = False.
	dObj4_sr.copyTo(dObj4_dr,false);
	EXPECT_EQ(dObj4_sr.getDims(),dObj4_dr.getDims() );	//!< Testing if the dimensions of ROI of copied Data Object are same as the ROI of original Data Object.
	EXPECT_EQ(dObj4_sr.getSize() ,dObj4_dr.getSize() );	//!< Testing if the size of ROI of copied Data Object is same as the ROI of original Data Object.
}

