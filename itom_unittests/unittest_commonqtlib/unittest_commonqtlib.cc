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

    bool executedByGoogleTestAdapter = false;

    for (int i = 0; i < argc; ++i)
    {
        const char* arg = argv[i];

        if (strcmp(arg, "-googletestadapter") == 0)
        {
            executedByGoogleTestAdapter = true;
            break;
        }
    }

    if (!executedByGoogleTestAdapter)
    {
        // execute this only, if the unittest is directly executed.
        // If it is executed by the VS Google Test Adapter extension,
        // no keyboard interaction must be implemented.
        std::system("pause");
    }

    return 0;
}
