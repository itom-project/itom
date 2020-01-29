
#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv2\opencv.hpp"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"

template <typename _Tp> class DataObjectFreeCreate_Test : public ::testing::Test 
    { 
public:

    virtual void SetUp(void)
    {
        
    }
    
    virtual void TearDown(void){}
};
    

TYPED_TEST_CASE(DataObjectFreeCreate_Test, ItomDataAllTypes);

//shallowCopyConvertCrash_Test
/*!
    This test causes a crash for dataObject version <= 1.5.0.0 since the
    reference counter has not been set to zero if the data object was freed
    but not the last one (therefore the ref counter was decremented but not
    deleted, nevertheless it has to be set to zero within this instance of
    dataObject)
*/
TYPED_TEST(DataObjectFreeCreate_Test, shallowCopyConvertCrash_Test)
{
    ito::DataObject a(600, 600, ito::tComplex128);
    ito::DataObject b = ito::DataObject(a);
    ito::DataObject c(600, 600, ito::tFloat64);
    c.convertTo(b, ito::tComplex128);
};
