#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv/cv.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"


/*! \class at_func_test
	\brief at() methods test for data objects with real data types

	This test class checks functionality of different implementations of at() function on data objects of different sizes and different real data types.
*/
template <typename _Tp> class at_func_test2 : public ::testing::Test 
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
	
TYPED_TEST_CASE(at_func_test2, ItomRealDataTypes);

// at_Test4
/*!
	This test checks the functionality of at(ito::Range *ranges) function with 5 dimensional data objects using full ranges as parameter.
*/
TYPED_TEST(at_func_test2, at_Test4)
{	
TypeParam *rowPtr1= NULL;	//!< Row pointer for dObj4 to access each element in each row of dObj4
int dim1 = dObj4.getSize(0);	
int dim2 = dObj4.getSize(1);
int dim3 = dObj4.getSize(2);
int dim4 = dObj4.getSize(3);
int dim5 = dObj4.getSize(4);	
size_t dataIdx = 0;
int temp=0;
	for(int i=0; i<dim1; i++)
	{
		for(int j=0; j<dim2;j++)
		{
			for(int k=0; k<dim3;k++)
			{
				dataIdx = i*(dim2*dim3) + j*dim3 + k;

				for(int l=0; l<dim4;l++)
				{		
					rowPtr1= (TypeParam*)dObj4.rowPtr(dataIdx,l);

					for(int m=0; m<dim5;m++)
					{
						rowPtr1[m] = cv::saturate_cast<TypeParam>(calcUniqueValue5D(i,j,k,l,m));		//!< Assigning unique value to each element of dObj4.	
					}
				}
			}
		}
	}

ito::DataObject dObj4_test1;
ito::Range ranges1[5];
ranges1[0] = ito::Range::all();
ranges1[1] = ito::Range::all();
ranges1[2] = ito::Range::all();
ranges1[3] = ito::Range::all();
ranges1[4] = ito::Range::all();

dObj4_test1 = dObj4.at(ranges1);	//!< assigning the full data object dObj4 to dObj4_test1 using full range array defined with ranges1[]. 

	unsigned int idx[] = {0,0,0,0,0};
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
						EXPECT_EQ(dObj4.at<TypeParam>(idx),dObj4_test1.at<TypeParam>(idx));			//!< Checking values of  5 dimensional data object dObj4 after applying adjustROI().
					}
				}
			}
		}
	}

}

//at_Test5
/*!
	This test checks the functionality of at(ito::Range *ranges) function on 5 dimensional data object dObj4 using limited range as parameter.
*/
TYPED_TEST(at_func_test2, at_Test5)
{
	TypeParam *rowPtr1= NULL; 
	int dim1 = dObj4.getSize(0);
	int dim2 = dObj4.getSize(1);
	int dim3 = dObj4.getSize(2);
	int dim4 = dObj4.getSize(3);
	int dim5 = dObj4.getSize(4);	
	size_t dataIdx = 0;
	int temp=0;
	for(int i=0; i<dim1; i++)
	{
		for(int j=0; j<dim2;j++)
		{
			for(int k=0; k<dim3;k++)
			{
				dataIdx = i*(dim2*dim3) + j*dim3 + k;
				for(int l=0; l<dim4;l++)
				{		
					rowPtr1= (TypeParam*)dObj4.rowPtr(dataIdx,l);
					for(int m=0; m<dim5;m++)
					{
						rowPtr1[m] = cv::saturate_cast<TypeParam>(calcUniqueValue5D(i,j,k,l,m));		//!< Assigning unique value to each element of dObj4.	
					}
				}
			}
		}
	}

	/*!
	Assigning limits in ranges1[] for at() function to be applied on 5 dimensional data object dObj4.
	*/
	ito::DataObject dObj4_test1;
	ito::Range ranges1[5];
	ranges1[0].start = 0;
	ranges1[0].end = 1;
	ranges1[1].start = 0;
	ranges1[1].end = 1;
	ranges1[2].start = 0;
	ranges1[2].end = 4;
	ranges1[3].start = 1;
	ranges1[3].end = 3;
	ranges1[4].start = 2;
	ranges1[4].end = 3;
	int test_res5d[] = {12,22,112,122,212,222,312,322,412,422};		//!< Expected result vector after applying at() function on dObj4.
	dObj4_test1 = dObj4.at(ranges1);	//!< assigning the full data object dObj4 to dObj4_test1 using limited range array defined with ranges1[]. 
	dim1 = dObj4_test1.getSize(0);
	dim2 = dObj4_test1.getSize(1);
	dim3 = dObj4_test1.getSize(2);
	dim4 = dObj4_test1.getSize(3);
	dim5 = dObj4_test1.getSize(4);
	temp=0;
	TypeParam v1 ; 
	TypeParam v2 ;
	unsigned int idx[] = {0,0,0,0,0};
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
						v1 = dObj4_test1.at<TypeParam>(idx);
						v2 = cv::saturate_cast<TypeParam>(test_res5d[temp++]);
						EXPECT_EQ(v1,v2);		//!< Checking values of 5 dimensional data object dObj4 after applying adjustROI().
					}
				}
			}
		}
	}
}

//at_Test6
/*!
	This test checks the functionality of at(ito::Range *ranges) function on 5 dimensional data object dObj4 using wrong range limits as parameter.
*/
TYPED_TEST(at_func_test2, at_Test6)
{
	TypeParam *rowPtr1= NULL; 
	int dim1 = dObj4.getSize(0);
	int dim2 = dObj4.getSize(1);
	int dim3 = dObj4.getSize(2);
	int dim4 = dObj4.getSize(3);
	int dim5 = dObj4.getSize(4);	
	size_t dataIdx = 0;
	for(int i=0; i<dim1; i++)
	{
		for(int j=0; j<dim2;j++)
		{
			for(int k=0; k<dim3;k++)
			{
				dataIdx = i*(dim2*dim3) + j*dim3 + k;
				for(int l=0; l<dim4;l++)
				{		
					rowPtr1= (TypeParam*)dObj4.rowPtr(dataIdx,l);
					for(int m=0; m<dim5;m++)
					{
						rowPtr1[m] = cv::saturate_cast<TypeParam>(calcUniqueValue5D(i,j,k,l,m));		//!< Assigning unique value to each element of dObj4.	
					}
				}
			}
		}
	}

	/*!
	Assigning limits in ranges1[] for at() function to be applied on 4 dimensional data object dObj5. These ranges limits fall outside of size of dObj4 which should raise exception while test.
	*/
	ito::DataObject dObj4_test1;
	ito::Range ranges1[5];
	ranges1[0].start = 0;
	ranges1[0].end = 10;
	ranges1[1].start = 0;
	ranges1[1].end = 1;
	ranges1[2].start = 0;
	ranges1[2].end = 4;
	ranges1[3].start = 1;
	ranges1[3].end = 25;
	ranges1[4].start = 2;
	ranges1[4].end = 3;
	EXPECT_ANY_THROW(dObj4_test1 = dObj4.at(ranges1));	//!< Expect an exception to be raised because ranges are outside of original data object size
}