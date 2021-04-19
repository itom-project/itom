#include "gtest/gtest.h"
#include <iostream>

////=======================================================================================================================================
// class FooTest : public ::testing::TestWithParam<int> {
//  // You can implement all the usual fixture class members here.
//  // To access the test parameter, call GetParam() from class
//  // TestWithParam<T>.
//};

int main(int argc, char *argv[])
{
    //::testing::FLAGS_gtest_filter = "iterator_test/*.iterator_test_2d"; //To Perform perticular subtest check, give
    //the Path of Perticular test. Comment this statement to perform whole test check.
    //::testing::FLAGS_gtest_filter = "const_iterator_test/*.*";
    //::testing::FLAGS_gtest_filter = "dataTest/*checkIdentity*";

    ::testing::InitGoogleTest(&argc, argv); // Initializing the google test.

    RUN_ALL_TESTS(); // To start Test check

    // ito::DataObject mat1_1d = ito::DataObject(3,getTypeNumber<ito::int8>());
    //
    // int itrer=0;
    // bool valBool;
    // valBool = ito::isZeroValue<ito::int8>(itrer,0) ;
    // std::cout<<valBool<<std::endl;
    // cv::DataType<float>::type;
    std::system("pause");
    return 0;
}
