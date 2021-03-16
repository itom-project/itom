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

    std::system("pause");
    return 0;
}
