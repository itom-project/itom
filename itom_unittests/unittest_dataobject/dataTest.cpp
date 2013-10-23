
#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv/cv.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"

/*! \class dataTest
	\brief data test for all itom data types

	This test class checks functionality of different data test methods for data objects.
*/
template <typename _Tp> class dataTest : public ::testing::Test 
	{ 
public:

	
	 ito::DataObject matrixTest;
	
	 //this Structure is used by "checkZeros" test
	 struct MatrixContainer
	 {
	 public:
		 MatrixContainer(int nrOfDimensions, int dim1, int dim2, int dim3 = 0, int dim4 = 0) : m_dim1(dim1), m_dim2(dim2), m_dim3(dim3), m_dim4(dim4), m_nrOfDimensions(nrOfDimensions) 
		 {
			 matrix = ito::DataObject();
			switch(nrOfDimensions)
			{
			case 2:
				matrix.zeros(dim1, dim2, ito::getDataType( (const _Tp *) NULL ));
				break;
			case 3:
				matrix.zeros(dim1, dim2, dim3, ito::getDataType( (const _Tp *) NULL ));
				break;
			case 4:
				//...
				break;
			default:
				//throw error
				break;
			}
		 };

		 int m_dim1;
		 int m_dim2;
		 int m_dim3;
		 int m_dim4;
		 int m_nrOfDimensions;
		 ito::DataObject matrix;
		  
	 };

	 std::vector< MatrixContainer > matrices;
	  
	 //this Structure is used by "checkIdentity" test
	  struct MatrixContainer1
	 {
	 public:
		 MatrixContainer1(int eyeSize) : m_eyeSize(eyeSize) 
		 {
			 matrix = ito::DataObject();
			 matrix.eye(eyeSize, ito::getDataType( (const _Tp *) NULL ));
			
		 };
			 
		 int m_eyeSize;
		 ito::DataObject matrix;
		  
	 };

	 std::vector< MatrixContainer1 > matrices1;
	
	 //this Structure is used by "checkOnes" test
	  struct MatrixContainer2
	 {
	 public:
		 MatrixContainer2(int nrOfDimensions, int dim1, int dim2, int dim3 = 0, int dim4 = 0) : m_dim1(dim1), m_dim2(dim2), m_dim3(dim3), m_dim4(dim4), m_nrOfDimensions(nrOfDimensions) 
		 {
			 matrix = ito::DataObject();
			switch(nrOfDimensions)
			{
			case 2:
				matrix.ones(dim1, dim2, ito::getDataType( (const _Tp *) NULL ));
				break;
			case 3:
				matrix.ones(dim1, dim2, dim3, ito::getDataType( (const _Tp *) NULL ));
				break;
			case 4:
				//...
				break;
			default:
				//throw error
				break;
			}
		 };

		 int m_dim1;
		 int m_dim2;
		 int m_dim3;
		 int m_dim4;
		 int m_nrOfDimensions;
		 ito::DataObject matrix;
		  
	 };

	 std::vector< MatrixContainer2 > matrices2;


    virtual void SetUp(void)
    {
		//creating matrices for "checkZeros" test
		matrices.push_back( MatrixContainer(2,2,2) );
		matrices.push_back( MatrixContainer(2,4,4) );
		matrices.push_back( MatrixContainer(3,1,1,1) );
		
		//creating matrices for "checkIdentity" test
		matrices1.push_back( MatrixContainer1(1) );
		matrices1.push_back( MatrixContainer1(3) );
		matrices1.push_back( MatrixContainer1(2) );

		//creating matrices for "checkOnes" test
		matrices2.push_back( MatrixContainer2(2,2,2) ); 
		matrices2.push_back( MatrixContainer2(2,4,4) );
		matrices2.push_back( MatrixContainer2(3,1,1,1));
	};
	 virtual void TearDown(void) {};
	  typedef _Tp valueType;	
	};

TYPED_TEST_CASE(dataTest, ItomDataAllTypes);

//checkZeros
/*!
	This test checks functionality of "zeros" method for 1, 2 and 3 dimensional matrices by checking if the required matrix is zero matrix.
*/
TYPED_TEST(dataTest, checkZeros)
{
	MatrixContainer *temp;
	ito::DataObject tempDObj;
	
	for(size_t i=0 ; i< this->matrices.size() ; i++)
	{
		temp = &matrices[i];	
		temp->matrix.copyTo(tempDObj);
		EXPECT_EQ ( tempDObj.getDims() , temp->m_nrOfDimensions );

		switch(temp->m_nrOfDimensions)
		{
		case 2:
			for(int r=0; r<temp->m_dim1; r++)
			{
				for(int s=0; s<temp->m_dim2;s++)
				{
				EXPECT_EQ ( tempDObj.at<TypeParam>(r,s) , cv::saturate_cast<TypeParam>(0));
				}
			}
			//std::cout << tempDObj << std::endl;
			break;
		case 3:

			for(int r=0; r<temp->m_dim1; r++)
			{
				for(int s=0; s<temp->m_dim2;s++)
				{
					for(int t=0; t<temp->m_dim3;t++)
					{
					EXPECT_EQ ( tempDObj.at<TypeParam>(r,s,t) , cv::saturate_cast<TypeParam>(0));
					}
				}	
			}
			break;
		case 4:
			//...
			break;
		default:
			//error
			break;
		}
	}

}

//checkIdentity
/*!
	This test checks functionality of "eye" method for 2 dimentional matrices by checking if the required matrix is Identity matrix.
*/
TYPED_TEST(dataTest, checkIdentity)
{
	MatrixContainer1 *temp1;
	ito::DataObject tempDObj1;
	
	for(size_t i=0 ; i< this->matrices1.size() ; i++)
	{
		temp1 = &matrices1[i];
		temp1->matrix.copyTo(tempDObj1);

			for(int r=0; r<temp1->m_eyeSize; r++)
			{
				for(int s=0; s<temp1->m_eyeSize;s++)
				{
					if(r==s) 
					{EXPECT_EQ ( tempDObj1.at<TypeParam>(r,s) , cv::saturate_cast<TypeParam>(1));}
					else
					{EXPECT_EQ ( tempDObj1.at<TypeParam>(r,s) , cv::saturate_cast<TypeParam>(0));}
				}
			}
			//std::cout << tempDObj1 << std::endl;
	}
}

//checkOnes
/*!
	This test checks functionality of "ones" method for 2 and 3 dimensional matrices by checking if the required matrix is filled with Ones.
*/
TYPED_TEST(dataTest, checkOnes)
{
	MatrixContainer2 *temp2;
	ito::DataObject tempDObj2;

    TypeParam tarValue;

    if(typeid(valueType) == typeid(ito::Rgba32))
    {
        tarValue = (ito::uint32)0xFFFFFFFF;
    }
    else
    {
        tarValue = cv::saturate_cast<TypeParam>(1);
    }
	
	for(size_t i=0 ; i< this->matrices2.size() ; i++)
	{
		temp2 = &matrices2[i];
		temp2->matrix.copyTo(tempDObj2);
		EXPECT_EQ ( tempDObj2.getDims() , temp2->m_nrOfDimensions );

		switch(temp2->m_nrOfDimensions)
		{
		case 2:
            
            
			for(int r=0; r<temp2->m_dim1; r++)
			{
				for(int s=0; s<temp2->m_dim2;s++)
				{
				EXPECT_EQ ( tempDObj2.at<TypeParam>(r,s) , tarValue);
				}
			}
			//std::cout << tempDObj2 << std::endl;
			break;
		case 3:
			for(int r=0; r<temp2->m_dim1; r++)
			{
				for(int s=0; s<temp2->m_dim2;s++)
				{
					for(int t=0; t<temp2->m_dim3;t++)
					{
					EXPECT_EQ ( tempDObj2.at<TypeParam>(r,s,t) , tarValue);
					}
				}	
			}
			break;
		case 4:
			//...
			break;
		default:
			//error
			break;
		}
	}
}
