#ifndef FUNCTIONS_ROITEST_H
#define FUNCTIONS_ROITEST_H
#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv/cv.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"


/*! \class ROITest
	\brief ROI methods test for real data types

	This test class checks functionality of different methods dealing with ROI for data objects.
*/
template <typename _Tp> class ROITest : public ::testing::Test 
	{ 
public:
	
	virtual void SetUp(void)
	{
		mat1_2d = ito::DataObject(12,13,ito::getDataType( (const _Tp *) NULL ));
		mat2_2d = ito::DataObject(10,15,ito::getDataType( (const _Tp *) NULL ));
		mat1_3d = ito::DataObject(13,12,14,ito::getDataType( (const _Tp *) NULL ));	
		mat2_3d = ito::DataObject(4,5,5,ito::getDataType( (const _Tp *) NULL ));

		dObj1 = ito::DataObject();
		dObj2 = ito::DataObject(10,10,ito::getDataType( (const _Tp *) NULL ));
		dObj3 = ito::DataObject(5,10,10,ito::getDataType( (const _Tp *) NULL ));
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

	ito::DataObject mat1_2d;
	ito::DataObject mat2_2d;
	ito::DataObject mat1_3d;
	ito::DataObject mat2_3d;
	ito::DataObject dObj1;
	ito::DataObject dObj2;
	ito::DataObject dObj3;
	ito::DataObject dObj4;
	ito::DataObject dObj5;
	ito::DataObject dObj6;
//	size_t *temp_size;
	};
	
TYPED_TEST_CASE(ROITest, ItomRealDataTypes);
//adjustROITest2d
/*!
	This test adjust the ROI of 2 dimensional matrices to check proper functionality of "adjustROI" method. It also checks "locateROI" method by comparing achieved offsets with original values.
*/
TYPED_TEST(ROITest, adjustROITest2d)
{
	int expSize2d[] = {12,13};  //expected size of mat1_2d after adjustROI
	int expOffsets2d[] = {4,1};  //expected offset of mat1_2d after adjustROI
	int orgSize2d[] = {0,0};
	int offsets2d[] = {0,0};
	unsigned int res1_str[] = {53,54,55,56,57,58,59,60,66,67,68,69,70,71,72,73,79,80,81,82,83,84,85,86,92,93,94,95,96,97,98,99}; // expected result array for testing elements of shrunk matrix mat1_2d after adjustROI
	const unsigned char matDims2d = 2;  
	int matLimits2d[] = {-4,-4,-1,-4}; //defining offsets for ROI of mat1_2d

	for(int i =0;i<12;i++) 
		{
			for(int j=0;j<13;j++)
			{mat1_2d.at<TypeParam>(i,j) = cv::saturate_cast<TypeParam>(13*i+j);}   //assigning unique value to each element of mat1_2d
		}

	//std::cout << mat1_2d << std::endl;

	mat1_2d.adjustROI(matDims2d,matLimits2d);   //adjust ROI of mat1_2d (shrinking because offset values are negative)
	mat1_2d.locateROI(orgSize2d,offsets2d);		//retrieve offset values of mat1_2d
	
	for(int i=0; i<2;i++) std::cout << orgSize2d[i] << std::endl;
	for(int i=0; i<2;i++) std::cout << offsets2d[i] << std::endl;

	for(int i=0; i<mat1_2d.getDims();i++) EXPECT_EQ(orgSize2d[i],expSize2d[i]);  //check retrieved size of matrix matches with original
	for(int i=0; i<mat1_2d.getDims();i++) EXPECT_EQ(offsets2d[i],expOffsets2d[i]);  //check retrieved offsets of ROI of mat1_2d matches with original
	//std::cout << mat1_2d << std::endl;

	for(int i =0;i<4;i++) 
		{
			for(int j=0;j<8;j++)EXPECT_EQ(mat1_2d.at<TypeParam>(i,j),cv::saturate_cast<TypeParam>(res1_str[i*8+j])); //check if the values of elements changed from the original				
		}

	mat1_2d.adjustROI(4,4,1,4);   //again expanding mat1_2d to original size
	mat1_2d.locateROI(orgSize2d,offsets2d);   //retrieve the dimensions and offset of ROI for expanded mat1_2d
	for(int i =0;i<2;i++) expOffsets2d[i] = 0;
	for(int i=0; i<2;i++) std::cout << orgSize2d[i] << std::endl;  
	for(int i=0; i<2;i++) std::cout << offsets2d[i] << std::endl;

	for(int i=0; i<mat1_2d.getDims();i++) EXPECT_EQ(orgSize2d[i],expSize2d[i]);  //check retrieved size of the matrix matches with original
	for(int i=0; i<mat1_2d.getDims();i++) EXPECT_EQ(offsets2d[i],expOffsets2d[i]); //check if the offsets of mat1_2d are 0 after expanding the matrix to its original size
	
    //std::cout << mat1_2d << std::endl;
	
	for(int i =0;i<12;i++) 
		{
			for(int j=0;j<13;j++)
			{EXPECT_EQ(mat1_2d.at<TypeParam>(i,j),cv::saturate_cast<TypeParam>(13*i+j));}  //check if the values of elements remained same as defined earlier after expanding it to original size
		}

	int expSize2d1[] = {10,15};
	int expOffsets2d1[] = {3,0};
	mat2_2d.adjustROI(-3,-2,3,3);
	mat2_2d.locateROI(orgSize2d,offsets2d);

	for(int i=0; i<3;i++) std::cout << orgSize2d[i] << std::endl;
	for(int i=0; i<3;i++) std::cout << offsets2d[i] << std::endl;

	for(int i=0; i<mat2_2d.getDims();i++) EXPECT_EQ(orgSize2d[i],expSize2d1[i]);
	for(int i=0; i<mat2_2d.getDims();i++) EXPECT_EQ(offsets2d[i],expOffsets2d1[i]);
}

//adjustROITest3d
/*!
	This test adjust the ROI of 3 dimensional matrices to check proper functionality of "adjustROI" method. It also checks "locateROI" method by comparing obtained offsets with original values.
*/
TYPED_TEST(ROITest, adjustROITest3d)
{
	int expSize3d[] = {13,12,14};	//expected size of 3d matrix mat1_3d
	int expOffsets3d[] = {4,1,2};	//expected offset sizes for 3d matrix mat1_3d
	int orgSize3d[] = {0,0,0};
	int offsets3d[] = {0,0,0};
	const unsigned char matDims3d = 3;
	int matLimits3d[] = {-4,-4,-1,-4,-2,-3};
	unsigned int res2_str[] = {26,27,28,31,32,33,36,37,38,41,42,43,51,52,53,56,57,58,61,62,63,66,67,68};
	mat1_3d.adjustROI(matDims3d,matLimits3d);  
	mat1_3d.locateROI(orgSize3d,offsets3d);

	//for(int i=0; i<3;i++) std::cout << orgSize3d[i] << std::endl;
	//for(int i=0; i<3;i++) std::cout << offsets3d[i] << std::endl;

	for(int i=0; i<mat1_3d.getDims();i++) EXPECT_EQ(orgSize3d[i],expSize3d[i]);		//checks if the size of matrix still sustain its original value as expected after adjusting ROI to desired location
	for(int i=0; i<mat1_3d.getDims();i++) EXPECT_EQ(offsets3d[i],expOffsets3d[i]);	//cheks if retrieved offset values for ROI after adjusting it are same as expected 

	int temp=0;
	for(int i =0;i<4;i++) 
		{
			for(int j=0;j<5;j++)
			{  
				for(int k=0;k<5;k++)
				{
					mat2_3d.at<TypeParam>(i,j,k) = cv::saturate_cast<TypeParam>(temp++);   //assigning unique value to each element of 3 dimensional matrix mat2_3d
				}
			}
		}

	//std::cout << mat2_3d << std::endl;

	int matLimits3d1[] = {-1,-1,0,-1,-1,-1};
	mat2_3d.adjustROI(matDims3d,matLimits3d1);  
	mat2_3d.locateROI(orgSize3d,offsets3d);

	//for(int i=0; i<3;i++) std::cout << orgSize3d[i] << std::endl;
	//for(int i=0; i<3;i++) std::cout << offsets3d[i] << std::endl;

	//std::cout << mat2_3d << std::endl;
	temp=0;
	for(int i =0;i<2;i++) 
		{
			for(int j=0;j<4;j++)
			{  
				for(int k=0;k<3;k++)
				{
					EXPECT_EQ(mat2_3d.at<TypeParam>(i,j,k),cv::saturate_cast<TypeParam>(res2_str[temp++]));   //checking value to each element of mat2_3d
				}
			}
		}

	int matLimits3d2[] = {1,0,0,1,0,1};
	mat2_3d.adjustROI(matDims3d,matLimits3d2);  
	mat2_3d.locateROI(orgSize3d,offsets3d);
	//for(int i=0; i<3;i++) std::cout << orgSize3d[i] << std::endl;
	//for(int i=0; i<3;i++) std::cout << offsets3d[i] << std::endl;

	unsigned int res3_str[] = {1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24,26,27,28,29,31,32,33,34,36,37,38,39,41,42,43,44,46,47,48,49,51,52,53,54,56,57,58,59,61,62,63,64,66,67,68,69,71,72,73,74};
	//std::cout << mat2_3d << std::endl;
	temp=0;
	for(int i =0;i<3;i++) 
		{
			for(int j=0;j<5;j++)
			{  
				for(int k=0;k<4;k++)
				{
					EXPECT_EQ(mat2_3d.at<TypeParam>(i,j,k),cv::saturate_cast<TypeParam>(res3_str[temp++]));   //checking value to each element of mat2_3d
				}
			}
		}
}

//adjustROITest3d
/*!
	This test adjust the ROI of 3 dimensional matrices to check proper functionality of "adjustROI" method. It also checks "locateROI" method by comparing obtained offsets with original values.
*/
TYPED_TEST(ROITest, seekMat_calcNumMats_Test)
{
	int i ;
	ito::Range *ranges1 = new ito::Range[3];
	ranges1[1]=ito::Range::all();
	ranges1[2]=ito::Range::all();
	ranges1[0]=ito::Range(2,4);

	dObj4= dObj3.at(ranges1);

	ito::Range *ranges2 = new ito::Range[5];
	ranges2[0]=ito::Range(1,3);
	ranges2[1]=ito::Range(2,3);
	ranges2[2]=ito::Range::all();
	ranges2[3]=ito::Range::all();
	ranges2[4]=ito::Range::all();

	dObj6= dObj5.at(ranges2);

	for(i=0;i<5;i++) EXPECT_EQ(0,dObj1.seekMat(i));			//!< Test for seekMat() function for one dimensional DataObject of Real DataTypes. 		
	for(i=0;i<5;i++) EXPECT_EQ(0, dObj2.seekMat(i));		//!< Test for seekMat() function for two dimensional DataObject of Real DataTypes.
	int test_str1[] = {0,1,2,3,4,0};
	for(i=0;i<6;i++) EXPECT_EQ(test_str1[i],dObj3.seekMat(i));	  //!< Test for seekMat() function for 3 dimensional DataObject of Real DataTypes.
	int test_str2[] = {2,3,0,0,0};
	for(i=0;i<5;i++) EXPECT_EQ(test_str2[i],dObj4.seekMat(i));	  //!< Test for seekMat() function for ROI of a previous 3 dimensional DataObject "dObj3".
	int test_str3[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,0,0,0,0,0,0,0};
	for(i=0;i<31;i++) EXPECT_EQ(test_str3[i],dObj5.seekMat(i));	  //!< Test for seekMat() function for 5 dimensional DataObject of Real DataTypes.
	int test_str4[] = {12,13,20,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	for(i=0;i<31;i++) EXPECT_EQ(test_str4[i],dObj6.seekMat(i));	  //!< Test for seekMat() function for ROI of a previous 5 dimensional DataObject "dObj5".
	EXPECT_EQ(0,dObj1.calcNumMats());						//!< Test for calcNumMats() function for 1 dimensional DataObject of Real DataTypes.
	EXPECT_EQ(1,dObj2.calcNumMats());						//!< Test for calcNumMats() function for 2 dimensional DataObject of Real DataTypes.
	EXPECT_EQ(5,dObj3.calcNumMats());						//!< Test for calcNumMats() function for 3 dimensional DataObject of Real DataTypes.
	EXPECT_EQ(2,dObj4.calcNumMats());						//!< Test for calcNumMats() function for ROI of a previous 3 dimensional DataObject "dObj3".
	EXPECT_EQ(24,dObj5.calcNumMats());						//!< Test for calcNumMats() function for 1 dimensional DataObject of Real DataTypes.	
	EXPECT_EQ(4,dObj6.calcNumMats());						//!< Test for calcNumMats() function for ROI of a previous 5 dimensional DataObject "dObj5".

	delete[] ranges1;
	delete[] ranges2;
}

#endif