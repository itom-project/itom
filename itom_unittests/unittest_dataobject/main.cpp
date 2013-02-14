#include <iostream>
#include "gtest/gtest.h"

#include "dataTest.h"
#include "operatorTest.h"
#include "addressTest.h"
#include "saturateTest_real.h"
#include "saturateTest_real1.h"
#include "functions_ROITest.h"
#include "complexDataTest.h"
#include "dataObjectTag_Test.h"
#include "dataObjectTagType_Test.h"

////=======================================================================================================================================
//class FooTest : public ::testing::TestWithParam<int> {
//  // You can implement all the usual fixture class members here.
//  // To access the test parameter, call GetParam() from class
//  // TestWithParam<T>.
//};


int main(int argc, char* argv[])
{
	
	::testing::FLAGS_gtest_filter = "at_func_test/*.at_Test"; //To Perform perticular subtest check, give the Path of Perticular test. Comment this statement to perform whole test check.
	//::testing::FLAGS_gtest_filter = "dataTest/*checkIdentity*";
    ::testing::InitGoogleTest(&argc, argv);  //Initializing the google test.
	
	ito::DataObject obj(4,5,6,ito::tInt16);
	cv::Mat_<float> mat;

	RUN_ALL_TESTS();   // To start Test check

	//ito::DataObject mat1_1d = ito::DataObject(3,getTypeNumber<ito::int8>());
	//
	//int itrer=0;
	//bool valBool;
	//valBool = ito::isZeroValue<ito::int8>(itrer,0) ;
	//std::cout<<valBool<<std::endl;
	//cv::DataType<float>::type;
	std::system("pause");
	return 0;
}