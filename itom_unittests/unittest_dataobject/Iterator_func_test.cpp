#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv/cv.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"


/*! \class iterator_test
	\brief iterator_test checks the functionality of DObjIterator for data objects of various dimensions.

	This test is a group of 3 tests for 2, 3 and 4 dimensional data objects. In each test, an iterator object of type DObjIterator is declared and data values of each element of data object is being fetched using this iterator. Later on test is performed such that data values fetched by iterators should match the original values given to each data object.
*/
template <typename _Tp> class iterator_test : public ::testing::Test 
	{ 
public:
	
	virtual void SetUp(void)
	{
		dObj2d = ito::DataObject(21,13,ito::getDataType( (const _Tp *) NULL ));
		dObj3d = ito::DataObject(5,10,10,ito::getDataType( (const _Tp *) NULL ));
		size_t *temp_size = new size_t[5];
		temp_size[0] = 3;
		temp_size[1] = 4;
		temp_size[2] = 2;
		temp_size[3] = 10;
		dObj4d = ito::DataObject(4,temp_size,ito::getDataType( (const _Tp *) NULL ));
	};
 
	virtual void TearDown(void) {};

	typedef _Tp valueType;	
	ito::DataObject dObj2d;
	ito::DataObject dObj3d;
	ito::DataObject dObj4d;
	};
	
TYPED_TEST_CASE(iterator_test, ItomRealDataTypes);

//iterator_test_2d
/*!
	This test checks the functionality of DObjIterator for 2 dimensional data objects.
*/
TYPED_TEST(iterator_test, iterator_test_2d)
{
	int temp = 0;		//!< Temporary variable for indexing some arrays used in this test.
	TypeParam objdata2d[273];	//!< This array holds the data of data object dObj2d.	
	TypeParam *dataptr2d; 
	dataptr2d = objdata2d; //!< Now pointer dataptr2d points to the array objdata2d.
	ito::DObjIterator it_2d;	//!< Declaration of DObjIterator
	
	int dim1_2d = dObj2d.getSize(0);
	int dim2_2d = dObj2d.getSize(1);
	
	for(int i=0;i<dim1_2d;i++)
	{
		for(int j=0;j<dim2_2d;j++)
		{
			dObj2d.at<TypeParam>(i,j) = cv::saturate_cast<TypeParam>(dim2_2d*i+j); //!< Assigning unique values to each element of 2 dimensional data object dObj2d.
			*dataptr2d=cv::saturate_cast<TypeParam>(dim2_2d*i+j);	//!< Defining the array with the same data values as in dObj2d for test purpose.
			dataptr2d++;
		}
	}
	temp=0;
	for(it_2d=dObj2d.begin();it_2d!=dObj2d.end();++it_2d)
	{
		// std::cout << *((TypeParam*)(*it)) << std::endl;
		EXPECT_EQ(cv::saturate_cast<TypeParam>(*((TypeParam*)(*it_2d))),cv::saturate_cast<TypeParam>(objdata2d[temp++])); //!< Testing the functionality of declared DObjIterator by comparing the values of each element of data object dObj2d with the same values stored in array objdata2d with this iterator it_2d .
	} 	
}

//iterator_test_3d
/*!
	This test checks the functionality of DObjIterator for 3 dimensional data objects.
*/
TYPED_TEST(iterator_test, iterator_test_3d)
{
	int temp;		//!< Temporary variable for indexing some arrays used in this test.
	TypeParam objdata3d[500];	//!< This array holds the data of data object dObj3d.
	TypeParam *dataptr3d;		
	dataptr3d = objdata3d;	//!< Now pointer dataptr3d points to the array objdata3d.

	ito::DObjIterator it_3d;	//!< Declaration of DObjIterator
	int dim1_3d = dObj3d.getSize(0);
	int dim2_3d = dObj3d.getSize(1);
	int dim3_3d = dObj3d.getSize(2);

	for(int i=0;i<dim1_3d;i++)
	{
		for(int j=0;j<dim2_3d;j++)
		{
			for(int k=0;k<dim3_3d;k++)
			{
				dObj3d.at<TypeParam>(i,j,k) = cv::saturate_cast<TypeParam>(dim1_3d*i*j+dim2_3d*j+k);	//!< Assigning unique values to each element of 3 dimensional data object dObj3d.
				*dataptr3d = cv::saturate_cast<TypeParam>(dim1_3d*i*j+dim2_3d*j+k);		//!< Defining the array with the same data values as in dObj3d for test purpose.
				dataptr3d++;
			}
		}
	}

	temp=0;
	for(it_3d=dObj3d.begin();it_3d!=dObj3d.end();++it_3d)
	{
		EXPECT_EQ(cv::saturate_cast<TypeParam>(*((TypeParam*)(*it_3d))),cv::saturate_cast<TypeParam>(objdata3d[temp++]));	//!< Testing the functionality of declared DObjIterator by comparing the values of each element of data object dObj3d with the same values stored in array objdata3d with this iterator it_3d.
	}
}

//iterator_test_4d
/*!
	This test checks the functionality of DObjIterator for 3 dimensional data objects.
*/
TYPED_TEST(iterator_test, iterator_test_4d)
{
	int temp;		//!< Temporary variable for indexing some arrays used in this test.
	TypeParam objdata4d[240];	//!< This array holds the data of data object dObj4d with size dim1 x dim2 x dim3 x dim4.
	TypeParam *dataptr4d;		
	dataptr4d = objdata4d;	//!< Now pointer dataptr3d points to the array objdata4d.

	ito::DObjIterator it_4d;	//!< Declaration of DObjIterator

	TypeParam *rowPtr1= NULL; 
	int dim1 = dObj4d.getSize(0);
	int dim2 = dObj4d.getSize(1);
	int dim3 = dObj4d.getSize(2);
	int dim4 = dObj4d.getSize(3);	
	size_t dataIdx = 0;
		for(int i=0; i<dim1;i++)
		{
			for(int j=0; j<dim2;j++)
			{
				dataIdx = i*dim2 + j;
				for(int k=0; k<dim3;k++)
				{		
					rowPtr1= (TypeParam*)dObj4d.rowPtr(dataIdx,k);
					for(int l=0; l<dim4;l++)
					{
						rowPtr1[l] = cv::saturate_cast<TypeParam>(dim1*i*j*k+dim2*j*k+dim3*k+l);		//!< Assigning unique value to each element of dObj4.	
						*dataptr4d = cv::saturate_cast<TypeParam>(dim1*i*j*k+dim2*j*k+dim3*k+l);		//!< Defining the array with the same data values as in dObj3d for test purpose.
						dataptr4d++;
					}
				}
			}
		}
	
	temp=0;
	for(it_4d=dObj4d.begin();it_4d!=dObj4d.end();++it_4d)
	{
		EXPECT_EQ(cv::saturate_cast<TypeParam>(*((TypeParam*)(*it_4d))),cv::saturate_cast<TypeParam>(objdata4d[temp++]));	//!< Testing the functionality of declared DObjIterator by comparing the values of each element of data object dObj3d with the same values stored in array objdata3d with this iterator it_3d.
	}
}