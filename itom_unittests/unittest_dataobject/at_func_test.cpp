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

// at_Test1
/*! 
	This test checks the functionality of at(ito::Range *ranges) function using limited range as parameter.
*/
TYPED_TEST(at_func_test, at_Test1)
{
	int check_arry1[] = {0,1,2,10,11,12,20,21,22,30,31,32}; //!< expected result array after applying at() function to dObj2

	for(int i =0; i<10;i++)
	{
		for(int j=0; j<10;j++)
		{
			dObj2.at<TypeParam>(i,j)= cv::saturate_cast<TypeParam>(10*i+j);	//!< Assigning unique value to each element of dObj2.
		}
	}

	//!< Test code with valid ranges less than dObj2
	ito::DataObject dObj2_test;
	ito::Range ranges[2];					//!< Range to be applied on 2 dimesional data object dObj2 with at() function
	ranges[0].start = 0;
	ranges[0].end =4;
	ranges[1].start = 0;
	ranges[1].end =3;
	dObj2_test=dObj2.at(ranges);			//!< assigning a part of data object dObj2 to dObj2_test bounded in a range defined in array ranges[] 
	EXPECT_EQ(dObj2.getDims(), dObj2_test.getDims());		//!< checks if the dimesions of dObj2_test is same as the original data object dObj2 after applying at() function.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(4),dObj2_test.getSize(0));	//!< checks expected size of 0th dimenstion of data object dObj2_test after applying at() function 
	EXPECT_EQ(cv::saturate_cast<TypeParam>(3),dObj2_test.getSize(1));	//!< checks expected size of 1st dimenstion of data object dObj2_test after applying at() function 
	int temp=0;
	for(int i=0;i<4;i++)
	{
		for(int j=0;j<3;j++)
		{
		EXPECT_EQ(dObj2_test.at<TypeParam>(i,j), check_arry1[temp++]);	//!< checks if the element values of dObj2_test, obtained with at() function applied on data object dObj2, is same as the original data object dObj2.
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
				dObj3.at<TypeParam>(i,j,k) = cv::saturate_cast<TypeParam>(temp++); //!< Assigning unique value to each element of dObj3.
			}
		}
	}
	ito::DataObject dObj3_test;
	ito::Range ranges_3d[3];	//!< Range to be applied on 3 dimesional data object dObj3 with at() function
	ranges_3d[0].start =1;
	ranges_3d[0].end =3;
	ranges_3d[1].start =0;
	ranges_3d[1].end =2;
	ranges_3d[2].start =0;
	ranges_3d[2].end =3;
	int check_arry2[] = {25,26,27,30,31,32,50,51,52,55,56,57};	//!< expected result array after applying at() function to dObj3

	dObj3_test = dObj3.at(ranges_3d);	//!< assigning a part of data object dObj3 to dObj3_test bounded in a range defined in array ranges_3d[] 
	EXPECT_EQ(dObj3.getDims(), dObj3_test.getDims());	//!< checks if the dimesions of dObj3_test is same as the original data object dObj3 after applying at() function.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(2),dObj3_test.getSize(0)); //!< checks expected size of 0th dimenstion of data object dObj3_test after applying at() function 
	EXPECT_EQ(cv::saturate_cast<TypeParam>(2),dObj3_test.getSize(1)); //!< checks expected size of 1st dimenstion of data object dObj3_test after applying at() function 
	EXPECT_EQ(cv::saturate_cast<TypeParam>(3),dObj3_test.getSize(2)); //!< checks expected size of 2nd dimenstion of data object dObj3_test after applying at() function 

	temp=0;
	for(int i=0;i<2; i++)
	{
		for(int j=0;j<2;j++)
		{
			for(int k=0;k<3;k++)
			{
				EXPECT_EQ(dObj3_test.at<TypeParam>(i,j,k),cv::saturate_cast<TypeParam>(check_arry2[temp++]));	//!< checks if the element values of dObj3_test, obtained with at() function applied on data object dObj3, is same as the original data object dObj3.
			}
		}
	}
}

//at_Test2
/*!
	This test checks the functionality of at(ito::Range *ranges) function using full ranges as parameter.
*/
TYPED_TEST(at_func_test, at_Test2)
{

	for(int i =0; i<10;i++)
	{
		for(int j=0; j<10;j++)
		{
			dObj2.at<TypeParam>(i,j)= cv::saturate_cast<TypeParam>(10*i+j);	//!< Assigning unique value to each element of dObj2.
		}
	}

	int	temp=0;
	for(int i=0;i<4; i++)
	{
		for(int j=0;j<5;j++)
		{
			for(int k=0;k<5;k++)
			{
				dObj3.at<TypeParam>(i,j,k) = cv::saturate_cast<TypeParam>(temp++);	//!< Assigning unique value to each element of dObj3.
			}
		}
	}

	//!< Test code with full ranges
	ito::DataObject dObj2_test1;
	ito::Range ranges1[2];			//!< Range to be applied on 2 dimesional data object dObj2 with at() function
	ranges1[0] = ito::Range::all();	//!< ranges1[0] element should point to full range of 0th dimension of the related data object
	ranges1[1] = ito::Range::all();	//!< ranges1[1] element should point to full range of 1st dimension of the related data object

	dObj2_test1 = dObj2.at(ranges1);	//!< assigning the full data object dObj2 to dObj2_test1 using full range array defined with range1[]. 

	EXPECT_EQ(dObj2.getDims(),dObj2_test1.getDims());	//!< dimension of both data objects should be same after applying at() function
	EXPECT_EQ(dObj2.getSize(0),dObj2_test1.getSize(0));	//!< checking if the size of 0th dimension is same as the original
	EXPECT_EQ(dObj2.getSize(1),dObj2_test1.getSize(1));	//!< checking if the size of 1st dimension is same as the original
	for(int i =0; i<10;i++)
	{
		for(int j=0; j<10;j++)
		{
			EXPECT_EQ(dObj2_test1.at<TypeParam>(i,j),cv::saturate_cast<TypeParam>(10*i+j));		//!< checks if the element values of dObj2_test1, obtained with at() function applied on data object dObj2, is same as the original data object dObj2.
		}
	}

/*!
	The following code deals with 3 dimensional data objects.
*/
	ito::DataObject dObj3_test1;		//!< declaring temporary 3 dimensional data object for test purpose.
	ito::Range ranges_3d[3];			//!< Range to be applied on 3 dimesional data object dObj3 with at() function.
	ranges_3d[0]= ito::Range::all();	//!< ranges_3d[0] element should point to full range of 0th dimension of the related data object.
	ranges_3d[1]= ito::Range::all();	//!< ranges_3d[1] element should point to full range of 1st dimension of the related data object.
	ranges_3d[2]= ito::Range::all();	//!< ranges_3d[2] element should point to full range of 2nd dimension of the related data object.
	 dObj3_test1 = dObj3.at(ranges_3d);	//!< assigning the full data object dObj3 to dObj3_test1 using full range array defined with range_3d[]. 

	 EXPECT_EQ(dObj3.getDims(),dObj3_test1.getDims());		//!< dimension of both data objects should be same after applying at() function
	 EXPECT_EQ(dObj3.getSize(0),dObj3_test1.getSize(0));	//!< checking if the size of 0th dimension is same as the original
	 EXPECT_EQ(dObj3.getSize(1),dObj3_test1.getSize(1));	//!< checking if the size of 1st dimension is same as the original
	 EXPECT_EQ(dObj3.getSize(2),dObj3_test1.getSize(2));	//!< checking if the size of 2nd dimension is same as the original

	temp=0;
	for(int i=0;i<4; i++)
	{
		for(int j=0;j<5;j++)
		{
			for(int k=0;k<5;k++)
			{
				EXPECT_EQ(dObj3_test1.at<TypeParam>(i,j,k),cv::saturate_cast<TypeParam>(temp++));	//!< checks if the element values of dObj3_test1, obtained with at() function applied on data object dObj3, is same as the original data object dObj3.
			}
		}
	}
}

// at_Test3
/*!
	This test checks the functionality of at(ito::Range *ranges) function using range parameter with wrong limits.
*/
TYPED_TEST(at_func_test, at_Test3)
{
	for(int i =0; i<10;i++)
	{
		for(int j=0; j<10;j++)
		{
			dObj2.at<TypeParam>(i,j)= cv::saturate_cast<TypeParam>(10*i+j);	//!< Assigning unique value to each element of dObj2.
		}
	}

	//!< Test code with valid ranges less than dObj3
	int	temp=0;
	for(int i=0;i<4; i++)
	{
		for(int j=0;j<5;j++)
		{
			for(int k=0;k<5;k++)
			{
				dObj3.at<TypeParam>(i,j,k) = cv::saturate_cast<TypeParam>(temp++);	//!< Assigning unique value to each element of dObj3.
			}
		}
	}
	//!< Test code with full ranges
	ito::DataObject dObj2_test1;
	ito::Range ranges1[2];		//!< Range variable to be used as parameter with at() function. In this test these ranges are going outside of original data object size which should raise exception while test.
	ranges1[0].start = 0;		//!< Start limit of ranges1[0]
	ranges1[0].end =15;			//!< End limit of ranges1[0]
	ranges1[1].start = 0;		//!< Start limit of ranges1[1]
	ranges1[1].end =3;			//!< End limit of ranges1[1]

	EXPECT_ANY_THROW(dObj2_test1 = dObj2.at(ranges1));			//!< Expect an exception to be raised because ranges are outside of original data object size
	
	//!< For 3 dimension Data Objects
	ito::DataObject dObj3_test1;
	ito::Range ranges_3d[3];	//!< Range variable to be used as parameter with at() function. In this test these ranges are going outside of original data object size which should raise exception while test.
	ranges_3d[0].start =1;	//!< Start limit of ranges_3d[0]
	ranges_3d[0].end =5;	//!< End limit of ranges_3d[0]
	ranges_3d[1].start =0;	//!< Start limit of ranges_3d[1]
	ranges_3d[1].end =2;	//!< End limit of ranges_3d[1]
	ranges_3d[2].start =1;	//!< Start limit of ranges_3d[2]
	ranges_3d[2].end =7;	//!< End limit of ranges_3d[2]
	EXPECT_ANY_THROW(dObj3_test1 = dObj3.at(ranges_3d));		//!< Expect an exception to be raised because ranges are outside of original data object size
}

// at_Test4
/*!
	This test checks the functionality of at(ito::Range *ranges) function with 5 dimensional data objects using full ranges as parameter.
*/
TYPED_TEST(at_func_test, at_Test4)
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
TYPED_TEST(at_func_test, at_Test5)
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
	Assigning limits in ranges1[] for at() function to be applied on 4 dimensional data object dObj5.
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
TYPED_TEST(at_func_test, at_Test6)
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