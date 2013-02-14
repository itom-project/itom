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
template <typename _Tp> class adjustROI_func_test : public ::testing::Test 
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
	
TYPED_TEST_CASE(adjustROI_func_test, ItomRealDataTypes);


//!< This test checks the functionality of adjustROI function with four parameter implementation and general implementation.
TYPED_TEST(adjustROI_func_test, adjustROI_Test1)
{			
	int matLimits1d[] = {-4,-6};			//!< defining offsets for ROI 
	int matLimits2d[] = {-4,-4,-1,-3};		//!< defining offsets for ROI 
	int matLimits3d[] = {-1,-1,-1,-1,-2,-1};
	int matLimits5d[] = {-2,0,-1,-2,0,0,-1,-1,-2,0};
	int test_res[] = {21,22,23,24,25,26,27,31,32,33,34,35,36,37,41,42,43,44,45,46,47,51,52,53,54,55,56,57,61,62,63,64,65,66,67,71,72,73,74,75,76,77,81,82,83,84,85,86,87};

	int test_res2d[] = {41,42,43,44,45,46,51,52,53,54,55,56};
	int test_res3d[] = {32,33,37,38,42,43,57,58,62,63,67,68};
	//int test_res5d[] = {};

	for(int i = 0; i <10; i++)
	{
		for(int j = 0; j <10; j++)
		{
			dObj2.at<TypeParam>(i,j) = 10*i + j;
		}
	}

	//!< Initializing 3 dimensional data object dObj3.
	int temp=0;
	for(int i=0;i<4;i++)
	{
		for(int j=0;j<5;j++)
		{
			for(int k=0;k<5;k++)
			{
				dObj3.at<TypeParam>(i,j,k) = temp++;
			}
		}
	}

	temp=0;
	//!< Initializing 5 dimensional data object dObj5.
	std::size_t planeId; 
	TypeParam *rowPtr  = NULL; 
	std::size_t planeId_d; 
	TypeParam *rowPtr_d  = NULL;	
	std::size_t dim1 = dObj4.getSize(0);
	std::size_t dim2 = dObj4.getSize(1);
	std::size_t dim3 = dObj4.getSize(2);
	std::size_t dim4 = dObj4.getSize(3);
	std::size_t dim5 = dObj4.getSize(4);	
	size_t dataIdx = 0;
	size_t dataIdx_d = 0;
	for(int i=0; i<dim1; i++)
	{
		for(int j=0; j<dim2;j++)
		{
			for(int k=0; k<dim3;k++)
			{
				dataIdx = i*(dim1*dim2) + j*dim2 + k;
				planeId = dObj4.seekMat(dataIdx);

				for(int l=0; l<dim4;l++)
				{		
					rowPtr = (TypeParam*)dObj4.rowPtr(planeId,l);

					for(int m=0; m<dim5;m++)
					{
						rowPtr[m] = 10000 *i + 1000 * j+ 100 *k + 10 *l +m;			
					}
				}
			}
		}
	}

	dObj2.adjustROI(-2,-1,-1,-2);	//!< Adjusting the ROI of 2 dimensional data object dObj2 with four parameter implementation.
	temp=0;
	for(int i = 0; i <7; i++)
	{
		for(int j = 0; j <7; j++)
		{
			EXPECT_EQ(dObj2.at<TypeParam>(i,j), test_res[temp++]); //!< Testing if the ROI has been adjusted to desired position.
		}
	}
	
	dObj2.adjustROI(2,1,1,2);	//!< Adjusting back the ROI back to normal position.
	dObj2.adjustROI(2,matLimits2d);
	dObj3.adjustROI(3,matLimits3d);
	dObj4.adjustROI(5,matLimits5d);
	std::cout<< dObj4 << std::endl;

	//!< Checking values of 2 dimensional data object dObj2 after applying adjustROI().
	temp=0;
	for(int i = 0; i < 2; i++)
	{
		for(int j = 0; j < 6; j++)
		{
			EXPECT_EQ( cv::saturate_cast<TypeParam>(test_res2d[temp++]), dObj2.at<TypeParam>(i,j) );
		}
	}

	//!< Checking values of 3 dimensional data object dObj3 after applying adjustROI().
	temp=0;
	for(int i=0;i<2;i++)
	{
		for(int j=0;j<3;j++)
		{
			for(int k=0;k<2;k++)
			{
 				EXPECT_EQ( cv::saturate_cast<TypeParam>(test_res3d[temp++]), dObj3.at<TypeParam>(i,j,k) );
			}
		}
	}

	
	//!< Checking values of  5 dimensional data object dObj5 after applying adjustROI().

	TypeParam *rowPtr1= NULL; 
	TypeParam *rowPtr_d1= NULL;	
dim1 = dObj4.getSize(0);
dim2 = dObj4.getSize(1);
dim3 = dObj4.getSize(2);
dim4 = dObj4.getSize(3);
dim5 = dObj4.getSize(4);	
dataIdx = 0;
dataIdx_d = 0;
temp=0;
	for(int i=0; i<dim1; i++)
	{
		for(int j=0; j<dim2;j++)
		{
			for(int k=0; k<dim3;k++)
			{
				dataIdx = i*(dim2*dim3) + j*dim3 + k;
				//planeId = dObj4.seekMat(dataIdx);

				for(int l=0; l<dim4;l++)
				{		
					rowPtr1= (TypeParam*)dObj4.rowPtr(dataIdx,l);

					for(int m=0; m<dim5;m++)
					{
						rowPtr1[m] = cv::saturate_cast<TypeParam>(calcUniqueValue5D(i,j,k,l,m));			
					}
				}
			}
		}
	}

	unsigned int idx[] = {0,0,0,0,0};
	TypeParam v1;
	TypeParam v2;
	for(int i=0; i<dim1; i++)
	{
		idx[0] = i;
		for(int j=0; j<dim2;j++)
		{
			idx[1] = j;
			for(int k=0; k<dim3;k++)
			{
				idx[2] = k;
				for(int l=0; l<dim4;l++)
				{		
					idx[3] = l;
					for(int m=0; m<dim5;m++)
					{
						idx[4] = m;
						v1 = dObj4.at<TypeParam>(idx);
						v2 = cv::saturate_cast<TypeParam>(calcUniqueValue5D(i,j,k,l,m));
						EXPECT_EQ(v1,v2);			
					}
				}
			}
		}
	}


}

//!< This test checks the functionality of adjustROI function with wrong number of arguments to check if exception is raised or not.
TYPED_TEST(adjustROI_func_test, adjustROI_Test2)
{
	int matLimits2d[] = {-1,-2,-1,0,0,0};
	int matLimits3d[] = {-2,-3,-5,-1};
	int matLimits5d[] = {-1,-2,-1,0};
	//EXPECT_ANY_THROW(dObj1.adjustROI(-1,-3,-2,0));
	//EXPECT_ANY_THROW(dObj2.adjustROI(-1,-3,0,0));
	//EXPECT_ANY_THROW(dObj3.adjustROI(-1,-2,-2,-1));
	//EXPECT_ANY_THROW(dObj4.adjustROI(-1,0));

	//EXPECT_ANY_THROW(dObj1.adjustROI(0,matLimits2d));
	//EXPECT_ANY_THROW(dObj2.adjustROI(2,matLimits2d));
	//EXPECT_ANY_THROW(dObj3.adjustROI(3,matLimits3d));
	//EXPECT_ANY_THROW(dObj4.adjustROI(5,matLimits5d));
}

//!< This test checks the range of valid ROI.
TYPED_TEST(adjustROI_func_test, adjustROI_Test3)
{
//!< Defining ROI partially outside the valid matrix-region.
	int matLimits1d[] = {-4,1};			//!< defining offsets for ROI 
	int matLimits2d[] = {-4,1,-1,-3};		//!< defining offsets for ROI 
	int matLimits3d[] = {-6,4,1,-7,2,-9};
	int matLimits5d[] = {-2,0,-1,-2,1,0,1,-1,2,0};

	EXPECT_NO_THROW(dObj1.adjustROI(0,matLimits1d)); //is ok, since the dimension is 0
	EXPECT_ANY_THROW(dObj1.adjustROI(1,matLimits1d));
	EXPECT_ANY_THROW(dObj2.adjustROI(2,matLimits2d));
	EXPECT_ANY_THROW(dObj2.adjustROI(1,matLimits1d)); 
	EXPECT_ANY_THROW(dObj3.adjustROI(3,matLimits3d));
	EXPECT_ANY_THROW(dObj4.adjustROI(5,matLimits5d));
}

//!< This test checks the range of valid ROI.
TYPED_TEST(adjustROI_func_test, adjustROI_Test4)
{
//!< Defining ROI completely outside the valid matrix-region.
	int matLimits1d[] = {4,1};			//!< defining offsets for ROI 
	int matLimits2d[] = {-12,4,-13,5};		//!< defining offsets for ROI 
	int matLimits3d[] = {-1,1,-1,-1,2,-1};
	int matLimits5d[] = {2,-9,-10,15,1,-13,-12,17,2,-11};


	//EXPECT_NO_THROW(dObj1.adjustROI(0,matLimits1d)); //is ok, since the dimension is 0
	//EXPECT_ANY_THROW(dObj1.adjustROI(1,matLimits1d));
	EXPECT_ANY_THROW(dObj2.adjustROI(2,matLimits2d));
	//EXPECT_ANY_THROW(dObj2.adjustROI(1,matLimits1d)); 
	EXPECT_ANY_THROW(dObj3.adjustROI(3,matLimits3d));
	EXPECT_ANY_THROW(dObj4.adjustROI(5,matLimits5d));
}

//!< This test checks the functionality of at(ito::Range *ranges) function.
TYPED_TEST(adjustROI_func_test, adjustROI_Test5)
{
//!< Defining ROI completely outside the valid matrix-region.
	int matLimits1d[] = {4,1};			//!< defining offsets for ROI 
	int matLimits2d[] = {-1,4,-13,5};		//!< defining offsets for ROI 
	int matLimits3d[] = {-1,1,-1,-1,2,-1};
	int matLimits5d[] = {2,-9,-10,15,1,-13,-12,17,2,-11};

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
	ranges[0].end =3;
	ranges[1].start = 0;
	ranges[1].end =3;

	dObj2_test=dObj2.at(ranges);
	int test_dim= dObj2_test.getDims();
	size_t test_size = dObj2_test.getSize(0);
	std::cout<<dObj2_test<<std::endl;
	std::cout<<test_dim<<std::endl;
	std::cout<<test_size<<std::endl;
	
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