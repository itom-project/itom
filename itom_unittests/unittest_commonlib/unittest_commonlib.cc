#include <iostream>
#include "gtest/gtest.h"

#include "retVal.h"
#include "sharedStructures.h"
#include "opencv/cv.h"


////=======================================================================================================================================
//class FooTest : public ::testing::TestWithParam<int> {
//  // You can implement all the usual fixture class members here.
//  // To access the test parameter, call GetParam() from class
//  // TestWithParam<T>.
//};


int main(int argc, char* argv[])
{
	//::testing::FLAGS_gtest_filter = "iterator_test/*.iterator_test_2d"; //To Perform perticular subtest check, give the Path of Perticular test. Comment this statement to perform whole test check.
	//::testing::FLAGS_gtest_filter = "const_iterator_test/*.*";
	//::testing::FLAGS_gtest_filter = "dataTest/*checkIdentity*";
    int64 t = cv::getTickCount();
    int m = 1000000;

    t = cv::getTickCount();
    for (int i = 0; i <m ;++i)
    {
        ito::RetVal retval;
        retval += ito::retWarning;
        for (int j = 0; j < 100;++j)
        {
            ito::RetVal retval3;
            retval += retval3;
        }
        retval += ito::RetVal(ito::retError,0,"testeritis");
        ito::RetVal retval2(retval);
        retval = ito::RetVal::format(ito::retError,0, "sdf %s %s", "sdf", "dfg");
        retval.containsError();
    }
    t = cv::getTickCount() - t;
    std::cout << t/cv::getTickFrequency() << "\n" << std::endl;

    t = cv::getTickCount();
    for (int i = 0; i <m ;++i)
    {
        ito::RetValDeprecated retval;
        retval += ito::retWarning;
        for (int j = 0; j < 100;++j)
        {
            ito::RetValDeprecated retval3;
            retval += retval3;
        }
        retval += ito::RetValDeprecated(ito::retError,0,"testeritis");
        ito::RetValDeprecated retval2(retval);
        retval = ito::RetValDeprecated::format(ito::retError,0, "sdf %s %s", "sdf", "dfg");
        retval.containsError();
    }
    t = cv::getTickCount() - t;
    std::cout << t/cv::getTickFrequency() << "\n" << std::endl;
    
    

	::testing::InitGoogleTest(&argc, argv);  //Initializing the google test.
	


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