//#include <iostream>
//
//#include "../../Common/sharedStructures.h"
//
////opencv
//#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
//#pragma once
//#include "opencv/cv.h"
//#include "../../DataObject/dataobj.h"
//#include "gtest/gtest.h"
////#include "dataTest.h"
//#include "commonChannel.h"
//
//
////#ifdef _DEBUG
////    #pragma comment(lib, "../../Debug/DataObject.lib")
////    #pragma comment(lib, "../../gtest-1.6.0/msvc/itom/Debug/gtestd.lib")
////    #pragma comment(lib, "C:/OpenCV2.3/build/x86/vc10/lib/opencv_core230d.lib")
////#else
////    #pragma comment(lib, "../../Release/DataObject.lib")
////    pragma comment(lib, "../../gtest-1.6.0/msvc/itom/Debug/gtest.lib")
////    #pragma comment(lib, "C:/OpenCV2.3/build/x86/vc10/lib/opencv_core230.lib")
////#endif
//
//
//
////template<typename _Tp> int getTypeNumber() { return -1; };
////template<> int getTypeNumber<uint8>() { return ito::tUInt8; };
////template<> int getTypeNumber<int8>() { return ito::tInt8; };
////template<> int getTypeNumber<uint16>() { return ito::tUInt16; };
////template<> int getTypeNumber<int16>() { return ito::tInt16; };
////template<> int getTypeNumber<uint32>() { return ito::tUInt32; };
////template<> int getTypeNumber<int32>() { return ito::tInt32; };
////template<> int getTypeNumber<float32>() { return ito::tFloat32; };
////template<> int getTypeNumber<float64>() { return ito::tFloat64; };
////template<> int getTypeNumber<complex64>() { return ito::tComplex64; };
////template<> int getTypeNumber<complex128>() { return ito::tComplex128; };
////=== SATURATION TEST ===================================================================================================================
//
//template <typename _Tp> class SaturateTestReal : public ::testing::Test { };
//
//
//TYPED_TEST_CASE(SaturateTestReal, ItomRealDataTypes);
//
//TYPED_TEST(SaturateTestReal, checkSaturateBoundaries)
//{
//    TypeParam max = std::numeric_limits<TypeParam>::max();
//    TypeParam min = std::numeric_limits<TypeParam>::min();
//
//    if(std::numeric_limits<TypeParam>::is_exact)
//    {
//    
//        EXPECT_EQ(cv::saturate_cast<TypeParam>(max) , max);
//        EXPECT_EQ(cv::saturate_cast<TypeParam>(min) , min );
//        EXPECT_EQ(cv::saturate_cast<TypeParam>(max+1) , max );
//        EXPECT_EQ(cv::saturate_cast<TypeParam>(max-1), max - 1 );
//        EXPECT_EQ(cv::saturate_cast<TypeParam>(min+1), min + 1 );
//        EXPECT_EQ(cv::saturate_cast<TypeParam>(min-1) , min );
//    }
//    else
//    {
//        TypeParam epsilon = std::numeric_limits<TypeParam>::epsilon();
//        EXPECT_NEAR(cv::saturate_cast<TypeParam>(max) , max, epsilon);
//        EXPECT_NEAR(cv::saturate_cast<TypeParam>(min) , min, epsilon );
//        EXPECT_NEAR(cv::saturate_cast<TypeParam>(max+1) , max, epsilon );
//        EXPECT_NEAR(cv::saturate_cast<TypeParam>(max-1), max - 1 , epsilon);
//        EXPECT_NEAR(cv::saturate_cast<TypeParam>(min+1), min + 1, epsilon );
//        EXPECT_NEAR(cv::saturate_cast<TypeParam>(min-1) , min , epsilon);
//    }
//
//    EXPECT_FLOAT_EQ( cv::saturate_cast<float32>( (float64)std::numeric_limits<float64>::max() ), std::numeric_limits<float32>::max() );
//    EXPECT_FLOAT_EQ( cv::saturate_cast<float32>( (float64)std::numeric_limits<float64>::min() ), std::numeric_limits<float32>::min() );
//
//
//}
//
//
////=== ADDRESS TEST ===================================================================================================================
//template <typename _Tp> class AddressTest : public ::testing::Test 
//{ 
//public:
//    virtual void SetUp(void)
//    {
//        matrix1x1 = ito::DataObject(1,1,getTypeNumber<_Tp>());
//        matrix1x1.at<_Tp>(0,0) = cv::saturate_cast<_Tp>(11);
//
//        matrix1x2 = ito::DataObject(1,2,getTypeNumber<_Tp>());
//        matrix1x2.at<_Tp>(0,0) = cv::saturate_cast<_Tp>(11);
//        matrix1x2.at<_Tp>(0,1) = cv::saturate_cast<_Tp>(12);
//
//        matrix2x2 = ito::DataObject(2,2,getTypeNumber<_Tp>());
//        matrix2x2.at<_Tp>(0,0) = cv::saturate_cast<_Tp>(11);
//        matrix2x2.at<_Tp>(0,1) = cv::saturate_cast<_Tp>(12);
//        matrix2x2.at<_Tp>(1,0) = cv::saturate_cast<_Tp>(21);
//        matrix2x2.at<_Tp>(1,1) = cv::saturate_cast<_Tp>(22);
//
//        matrix1x1x1 = ito::DataObject(1,1,1,getTypeNumber<_Tp>());
//        matrix1x1x1.at<_Tp>(0,0,0) = 111;
//	    
//		
//	
//		//cv::Mat_<_Tp> *mat = (cv::Mat_<_Tp> *)(matrix1x1.get_mdata[0]);
//		//cv::MatIterator_<_Tp> iter = mat->begin();
//		//cv::MatConstIterator_<_Tp> iter_end = mat->end();
//
//		//int numberOfElements = 0;
//  //    for (; iter != iter_end; ++iter)
//  //    {
//		//  EXPECT_EQ(*iter,0);
//		//  numberOfElements++;
//  //       //(*lhsIt) *= factor2;
//  //    }
//
//	 // EXPECT_EQ(numberOfElements,12);
//
//    };
//
//    virtual void TearDown(void) {};
//
//    ito::DataObject matrix1x1;
//    ito::DataObject matrix1x2;
//    ito::DataObject matrix2x2;
//    ito::DataObject matrix1x1x1;
//	
//	
//   // ito::DataObject matrix2x1x2;
//
//    typedef _Tp valueType;
//};
//
//TYPED_TEST_CASE(AddressTest, ItomDataStandardTypes);
//
//TYPED_TEST(AddressTest, checkValues)
//{
//    int typeno = getTypeNumber<TypeParam>();
//    EXPECT_EQ ( this->matrix1x1.at<TypeParam>(0) , cv::saturate_cast<TypeParam>(11));
//    EXPECT_EQ ( this->matrix1x1.at<TypeParam>(0,0) , cv::saturate_cast<TypeParam>(11));
//
//    EXPECT_EQ ( this->matrix1x2.at<TypeParam>(0,0) , cv::saturate_cast<TypeParam>(11));
//    EXPECT_EQ ( this->matrix1x2.at<TypeParam>(0,1) , cv::saturate_cast<TypeParam>(12));
//
//    EXPECT_EQ ( this->matrix2x2.at<TypeParam>(0,0) , cv::saturate_cast<TypeParam>(11));
//    EXPECT_EQ ( this->matrix2x2.at<TypeParam>(0,1) , cv::saturate_cast<TypeParam>(12));
//    EXPECT_EQ ( this->matrix2x2.at<TypeParam>(1,0) , cv::saturate_cast<TypeParam>(21));
//    EXPECT_EQ ( this->matrix2x2.at<TypeParam>(1,1) , cv::saturate_cast<TypeParam>(22));
//
//    EXPECT_EQ ( this->matrix1x1x1.at<TypeParam>(0,0,0) , cv::saturate_cast<TypeParam>(111));
//    
//}
//
//TYPED_TEST(AddressTest, checkDim)
//{
//
//   // EXPECT_EQ ( this->matrix1x1.getDims<TypeParam>(), cv::saturate_cast<TypeParam>(2));
//   
//
//    EXPECT_EQ ( this->matrix1x2.getDims(),2);
//    EXPECT_EQ ( this->matrix1x2.getDims(),2);
//
//    EXPECT_EQ ( this->matrix2x2.getDims(),2);
//    EXPECT_EQ ( this->matrix2x2.getDims(),2);
//    EXPECT_EQ ( this->matrix2x2.getDims(),2);
//    EXPECT_EQ ( this->matrix2x2.getDims(),2);
//
//    EXPECT_EQ ( this->matrix1x1x1.getDims(),3);
//    
//}
//
////template <typename _Tp> class dataTest : public ::testing::Test 
////	{ 
////public:
////
////	
////	 ito::DataObject matrixTest;
////	
////	 //this Structure is used by "checkZeros" test
////	 struct MatrixContainer
////	 {
////	 public:
////		 MatrixContainer(int nrOfDimensions, int dim1, int dim2, int dim3 = 0, int dim4 = 0) : m_dim1(dim1), m_dim2(dim2), m_dim3(dim3), m_dim4(dim4), m_nrOfDimensions(nrOfDimensions) 
////		 {
////			 matrix = ito::DataObject();
////			switch(nrOfDimensions)
////			{
////			case 2:
////				matrix.zeros(dim1, dim2, getTypeNumber<_Tp>());
////				break;
////			case 3:
////				matrix.zeros(dim1, dim2, dim3, getTypeNumber<_Tp>());
////				break;
////			case 4:
////				//...
////				break;
////			default:
////				//throw error
////				break;
////			}
////		 };
////
////		 int m_dim1;
////		 int m_dim2;
////		 int m_dim3;
////		 int m_dim4;
////		 int m_nrOfDimensions;
////		 ito::DataObject matrix;
////		  
////	 };
////
////	 std::vector< MatrixContainer > matrices;
////	  
////	 //this Structure is used by "checkIdentity" test
////	  struct MatrixContainer1
////	 {
////	 public:
////		 MatrixContainer1(int eyeSize) : m_eyeSize(eyeSize) 
////		 {
////			 matrix = ito::DataObject();
////			 matrix.eye(eyeSize, getTypeNumber<_Tp>());
////			
////		 };
////			 
////		 int m_eyeSize;
////		 ito::DataObject matrix;
////		  
////	 };
////
////	 std::vector< MatrixContainer1 > matrices1;
////	
////	 //this Structure is used by "checkOnes" test
////	  struct MatrixContainer2
////	 {
////	 public:
////		 MatrixContainer2(int nrOfDimensions, int dim1, int dim2, int dim3 = 0, int dim4 = 0) : m_dim1(dim1), m_dim2(dim2), m_dim3(dim3), m_dim4(dim4), m_nrOfDimensions(nrOfDimensions) 
////		 {
////			 matrix = ito::DataObject();
////			switch(nrOfDimensions)
////			{
////			case 2:
////				matrix.ones(dim1, dim2, getTypeNumber<_Tp>());
////				break;
////			case 3:
////				matrix.ones(dim1, dim2, dim3, getTypeNumber<_Tp>());
////				break;
////			case 4:
////				//...
////				break;
////			default:
////				//throw error
////				break;
////			}
////		 };
////
////		 int m_dim1;
////		 int m_dim2;
////		 int m_dim3;
////		 int m_dim4;
////		 int m_nrOfDimensions;
////		 ito::DataObject matrix;
////		  
////	 };
////
////	 std::vector< MatrixContainer2 > matrices2;
////
////
////    virtual void SetUp(void)
////    {
////		/*matrixTest = ito::DataObject(2,2,getTypeNumber<_Tp>());
////		matrixTest.getDims();
////		matrixTest.getSize(0);
////		matrixTest.zeros(2,2,getTypeNumber<_Tp>());*/
////		//matrixTest.at<_Tp>(0,0) = cv::saturate_cast<_Tp>(0);
////
////		//creating matrices for "checkZeros" test
////		matrices.push_back( MatrixContainer(2,2,2) );
////		matrices.push_back( MatrixContainer(2,4,4) );
////		matrices.push_back( MatrixContainer(3,1,1,1) );
////		
////		//creating matrices for "checkIdentity" test
////		matrices1.push_back( MatrixContainer1(1) );
////		matrices1.push_back( MatrixContainer1(3) );
////		matrices1.push_back( MatrixContainer1(2) );
////
////		//creating matrices for "checkOnes" test
////		matrices2.push_back( MatrixContainer2(2,2,2) );
////		matrices2.push_back( MatrixContainer2(2,4,4) );
////		matrices2.push_back( MatrixContainer2(3,1,1,1));
////		
////	
////	};
////	 
////
////	 virtual void TearDown(void) {};
////	  typedef _Tp valueType;
////	
////	};
////
////TYPED_TEST_CASE(dataTest, ItomRealDataTypes);
////
////
////
//////This test checks if the required matrix is zero matrix.
////TYPED_TEST(dataTest, checkZeros)
////{
////	MatrixContainer *temp;
////
////	ito::DataObject tempDObj;
////
////	
////	for(int i=0 ; i< this->matrices.size() ; i++)
////	{
////		temp = &matrices[i];
////		
////		temp->matrix.copyTo(tempDObj);
////
////		EXPECT_EQ ( tempDObj.getDims() , temp->m_nrOfDimensions );
////
////		switch(temp->m_nrOfDimensions)
////		{
////		
////		case 2:
////			for(int r=0; r<temp->m_dim1; r++)
////			{
////				for(int s=0; s<temp->m_dim2;s++)
////				{
////				EXPECT_EQ ( tempDObj.at<TypeParam>(r,s) , cv::saturate_cast<TypeParam>(0));
////				}
////			}
////			//std::cout << tempDObj << std::endl;
////			break;
////		case 3:
////
////			for(int r=0; r<temp->m_dim1; r++)
////			{
////				for(int s=0; s<temp->m_dim2;s++)
////				{
////					for(int t=0; t<temp->m_dim3;t++)
////					{
////					EXPECT_EQ ( tempDObj.at<TypeParam>(r,s,t) , cv::saturate_cast<TypeParam>(0));
////					}
////				}	
////			}
////			break;
////		case 4:
////			//...
////			break;
////		default:
////			//error
////			break;
////		}
////	}
////
////}
////
////
//////This test checks if the required matrix is Identity matrix.
////TYPED_TEST(dataTest, checkIdentity)
////{
////	MatrixContainer1 *temp1;
////
////	ito::DataObject tempDObj1;
////	
////	
////	for(int i=0 ; i< this->matrices1.size() ; i++)
////	{
////	
////		temp1 = &matrices1[i];
////	
////		temp1->matrix.copyTo(tempDObj1);
////
////			for(int r=0; r<temp1->m_eyeSize; r++)
////			{
////				for(int s=0; s<temp1->m_eyeSize;s++)
////				{
////					if(r==s) 
////					{EXPECT_EQ ( tempDObj1.at<TypeParam>(r,s) , cv::saturate_cast<TypeParam>(1));}
////					else
////					{EXPECT_EQ ( tempDObj1.at<TypeParam>(r,s) , cv::saturate_cast<TypeParam>(0));}
////				}
////			}
////			//std::cout << tempDObj1 << std::endl;
////	}
////
////
////
////}
////
////
//////This test checks if the required matrix is filled with Ones.
////TYPED_TEST(dataTest, checkOnes)
////{
////	MatrixContainer2 *temp2;
////
////	ito::DataObject tempDObj2;
////
////	
////	for(int i=0 ; i< this->matrices2.size() ; i++)
////	{
////		temp2 = &matrices2[i];
////		
////		temp2->matrix.copyTo(tempDObj2);
////
////		EXPECT_EQ ( tempDObj2.getDims() , temp2->m_nrOfDimensions );
////
////		switch(temp2->m_nrOfDimensions)
////		{
////		
////		case 2:
////			for(int r=0; r<temp2->m_dim1; r++)
////			{
////				for(int s=0; s<temp2->m_dim2;s++)
////				{
////				EXPECT_EQ ( tempDObj2.at<TypeParam>(r,s) , cv::saturate_cast<TypeParam>(1));
////				}
////			}
////			//std::cout << tempDObj2 << std::endl;
////			break;
////		case 3:
////
////			for(int r=0; r<temp2->m_dim1; r++)
////			{
////				for(int s=0; s<temp2->m_dim2;s++)
////				{
////					for(int t=0; t<temp2->m_dim3;t++)
////					{
////					EXPECT_EQ ( tempDObj2.at<TypeParam>(r,s,t) , cv::saturate_cast<TypeParam>(1));
////					}
////				}	
////			}
////			break;
////		case 4:
////			//...
////			break;
////		default:
////			//error
////			break;
////		}
////	}
////
////
////}
//
//
//
////template <typename _Tp> class operatorTest : public ::testing::Test 
////	{ 
////public:
////
////
////	/*struct MatrixContainer3
////	 {
////	 public:
////		 MatrixContainer3(int nrOfDimensions, int dim1, int dim2, int dim3 = 0, int dim4 = 0) : m_dim1(dim1), m_dim2(dim2), m_dim3(dim3), m_dim4(dim4), m_nrOfDimensions(nrOfDimensions) 
////		 {
////			 matrix = ito::DataObject();
////			switch(nrOfDimensions)
////			{
////			case 2:
////				matrix = ito::DataObject(dim1, dim2,getTypeNumber<_Tp>());
////				break;
////			case 3:
////				matrix = ito::DataObject(dim1, dim2, dim3,getTypeNumber<_Tp>());
////				break;
////			case 4:
////				//...
////				break;
////			default:
////				//throw error
////				break;
////			}
////		 };
////
////		 int m_dim1;
////		 int m_dim2;
////		 int m_dim3;
////		 int m_dim4;
////		 int m_nrOfDimensions;
////		 ito::DataObject matrix;
////		  
////	 }; 
////	std::vector< MatrixContainer3 > matrices_add;
////	std::vector< MatrixContainer3 > matrices_sub;
////	std::vector< MatrixContainer3 > matrices_mul;
////	std::vector< MatrixContainer3 > matrices_addRes;
////	std::vector< MatrixContainer3 > matrices_subRes;
////	std::vector< MatrixContainer3 > matrices_divRes;*/
////
////	virtual void SetUp(void)
////    {
////
////	/*	matrices_add.push_back( MatrixContainer3(2,1,1) );
////		matrices_addRes.push_back( MatrixContainer3(2,1,1) );
////		matrices_add.push_back( MatrixContainer3(2,2,2) );
////		matrices_add.push_back( MatrixContainer3(3,2,2,2) );
////
////		matrices_add[0].matrix.at<_Tp>(0,0) = cv::saturate_cast<_Tp>(3);
////		matrices_add[1].matrix.at<_Tp>(0,0) = cv::saturate_cast<_Tp>(2);
////		matrices_addRes[0].matrix = matrices_add[0].matrix + matrices_add[1].matrix;
////		*/
////
////		
////
////		mat1_1d = ito::DataObject(1,1,getTypeNumber<_Tp>());
////		mat1_2d = ito::DataObject(4,5,getTypeNumber<_Tp>());
////		mat1_3d = ito::DataObject(3,2,4,getTypeNumber<_Tp>());
////
////		mat2_1d = ito::DataObject(1,1,getTypeNumber<_Tp>());
////		mat2_2d = ito::DataObject(4,5,getTypeNumber<_Tp>());
////		mat2_3d = ito::DataObject(3,2,4,getTypeNumber<_Tp>());
////		
////		add_mat3_1d = ito::DataObject(1,1,getTypeNumber<_Tp>());
////		add_mat3_2d = ito::DataObject(4,5,getTypeNumber<_Tp>());
////		add_mat3_3d = ito::DataObject(3,2,4,getTypeNumber<_Tp>());
////
////		sub_mat3_1d = ito::DataObject(1,1,getTypeNumber<_Tp>());
////		sub_mat3_2d = ito::DataObject(4,5,getTypeNumber<_Tp>());
////		sub_mat3_3d = ito::DataObject(3,2,4,getTypeNumber<_Tp>());
////
////		mul_mat3_1d = ito::DataObject(1,1,getTypeNumber<_Tp>());
////		mul_mat3_2d = ito::DataObject(4,5,getTypeNumber<_Tp>());
////		mul_mat3_3d = ito::DataObject(3,2,4,getTypeNumber<_Tp>()); 
////		
////	    mat1_1d.at<_Tp>(0,0) = cv::saturate_cast<_Tp>(3);
////        mat2_1d.at<_Tp>(0,0) = cv::saturate_cast<_Tp>(2);
////
////		mat1_2d.at<_Tp>(0,0) = cv::saturate_cast<_Tp>(2);
////		mat1_2d.at<_Tp>(0,1) = cv::saturate_cast<_Tp>(3);
////		mat1_2d.at<_Tp>(1,0) = cv::saturate_cast<_Tp>(4);
////		mat1_2d.at<_Tp>(1,1) = cv::saturate_cast<_Tp>(5);
////        mat2_2d.at<_Tp>(0,0) = cv::saturate_cast<_Tp>(1);
////		mat2_2d.at<_Tp>(0,1) = cv::saturate_cast<_Tp>(1);
////		mat2_2d.at<_Tp>(1,0) = cv::saturate_cast<_Tp>(2);
////		mat2_2d.at<_Tp>(1,1) = cv::saturate_cast<_Tp>(3);
////		
////		mat1_3d.at<_Tp>(0,0,3) = cv::saturate_cast<_Tp>(4);
////        mat2_3d.at<_Tp>(0,0,3) = cv::saturate_cast<_Tp>(1);
////		
////		add_mat3_1d = mat1_1d + mat2_1d;
////		add_mat3_2d = mat1_2d + mat2_2d;
////		add_mat3_3d = mat1_3d + mat2_3d;
////
////		sub_mat3_1d = mat1_1d - mat2_1d;
////		sub_mat3_2d = mat1_2d - mat2_2d;
////		sub_mat3_3d = mat1_3d - mat2_3d;
////
////		mul_mat3_1d = mat1_1d.mul(mat2_1d);
////		mul_mat3_2d = mat1_2d.mul(mat2_2d);
////		mul_mat3_3d = mat1_3d.mul(mat2_3d);
////
////	};
////	virtual void TearDown(void) {};
////	typedef _Tp valueType;
////	 ito::DataObject mat1_1d;
////	 ito::DataObject mat1_2d;
////	 ito::DataObject mat1_3d;
////
////	 ito::DataObject mat2_1d;
////	 ito::DataObject mat2_2d;
////	 ito::DataObject mat2_3d;
////
////	 ito::DataObject add_mat3_1d;
////	 ito::DataObject add_mat3_2d;
////	 ito::DataObject add_mat3_3d;
////
////	 ito::DataObject sub_mat3_1d;
////	 ito::DataObject sub_mat3_2d;
////	 ito::DataObject sub_mat3_3d;
////
//// 	 ito::DataObject mul_mat3_1d;
////	 ito::DataObject mul_mat3_2d;
////	 ito::DataObject mul_mat3_3d; 
////	};
////TYPED_TEST_CASE(operatorTest, ItomRealDataTypes);
////TYPED_TEST(operatorTest, AddTest)
////{
////	 EXPECT_EQ ( this->add_mat3_1d.at<TypeParam>(0,0) , cv::saturate_cast<TypeParam>(5));
////
////	 EXPECT_EQ ( this->add_mat3_2d.at<TypeParam>(0,0) , cv::saturate_cast<TypeParam>(3));
////	 EXPECT_EQ ( this->add_mat3_2d.at<TypeParam>(0,1) , cv::saturate_cast<TypeParam>(4));
////	 EXPECT_EQ ( this->add_mat3_2d.at<TypeParam>(1,0) , cv::saturate_cast<TypeParam>(6));
////	 EXPECT_EQ ( this->add_mat3_2d.at<TypeParam>(1,1) , cv::saturate_cast<TypeParam>(8));
////
////	 EXPECT_EQ ( this->add_mat3_3d.at<TypeParam>(0,0,3) , cv::saturate_cast<TypeParam>(5));
////	 
////}
////
////TYPED_TEST(operatorTest, SubTest)
////{
////
////	 EXPECT_EQ ( this->sub_mat3_1d.at<TypeParam>(0,0) , cv::saturate_cast<TypeParam>(1));
////
////	 EXPECT_EQ ( this->sub_mat3_2d.at<TypeParam>(0,0) , cv::saturate_cast<TypeParam>(1));
////	 EXPECT_EQ ( this->sub_mat3_2d.at<TypeParam>(0,1) , cv::saturate_cast<TypeParam>(2));
////	 EXPECT_EQ ( this->sub_mat3_2d.at<TypeParam>(1,0) , cv::saturate_cast<TypeParam>(2));
////	 EXPECT_EQ ( this->sub_mat3_2d.at<TypeParam>(1,1) , cv::saturate_cast<TypeParam>(2));
////
////	 EXPECT_EQ ( this->sub_mat3_3d.at<TypeParam>(0,0,3) , cv::saturate_cast<TypeParam>(3));
////	 
////}
////
////
////TYPED_TEST(operatorTest, MulTest)
////{
////
////	 EXPECT_EQ ( this->mul_mat3_1d.at<TypeParam>(0,0) , cv::saturate_cast<TypeParam>(6));
////
////	 EXPECT_EQ ( this->mul_mat3_2d.at<TypeParam>(0,0) , cv::saturate_cast<TypeParam>(2));
////	 EXPECT_EQ ( this->mul_mat3_2d.at<TypeParam>(0,1) , cv::saturate_cast<TypeParam>(3));
////	 EXPECT_EQ ( this->mul_mat3_2d.at<TypeParam>(1,0) , cv::saturate_cast<TypeParam>(8));
////	 EXPECT_EQ ( this->mul_mat3_2d.at<TypeParam>(1,1) , cv::saturate_cast<TypeParam>(15));
////
////	 EXPECT_EQ ( this->mul_mat3_3d.at<TypeParam>(0,0,3) , cv::saturate_cast<TypeParam>(4));
////	 
////}
//
//
//
//
//
//
//
//
//
//
//
//
//
//
