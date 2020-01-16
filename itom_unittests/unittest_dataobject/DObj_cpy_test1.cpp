#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv2\opencv.hpp"
#include "../../DataObject/dataobj.h"
#include "../../common/typeDefs.h"
#include "gtest/gtest.h"
//#include "test_global.h"
#include "commonChannel.h"


/*! \class copyTests1
    \brief test for deepCopyPartial(..) method.

    This test class checks functionality of deepCopyPartial(..) method on different data objects of different sizes and types.
*/
template <typename _Tp> class copyTests1 : public ::testing::Test 
    { 
public:
    
    virtual void SetUp(void)
    {
        int temp_size1[] = {10,10};
        int temp_size[] = {4,4,4,4,4};
        int temp_size2[] = {1,1,2,1,1};
        int temp_size3[] = {5,1,1};
        int temp_size4[] = {1,5};
        int temp_size5[] = {1,4,1,2};
        int temp_size6[] = {4,2};

        ito::tDataType type = ito::getDataType2<_Tp*>();

        dObj1_s = ito::DataObject(0,temp_size1,type);
        dObj2_s = ito::DataObject(2,temp_size1,type);
        dObj3_s = ito::DataObject(4,5,5,type);
        dObj4_s = ito::DataObject(5,temp_size,type);
        dObj4_s1 = ito::DataObject(5,temp_size2,type);
        dObj1_d = ito::DataObject(0,temp_size1,type);
        dObj2_d = ito::DataObject(2,temp_size1,type);
        dObj3_d = ito::DataObject(4,5,5,type);
        dObj4_d = ito::DataObject(5,temp_size,type);


        dObj5x1x1 = ito::DataObject(3, temp_size3, type);
        dObj5x1x1.setTo(0);

        dObj1x5 = ito::DataObject(2, temp_size4, type);
        for (int i = 0; i < 5; ++i)
        {
            dObj1x5.at<_Tp>(0,i) = i;
        }

        dObj1x4x1x2 = ito::DataObject(4, temp_size5, type);
        dObj1x4x1x2.setTo(0);

        dObj4x2 = ito::DataObject(2, temp_size6, type);
        for (int r = 0; r < 4; ++r)
        {
            for (int c = 0; c < 2; ++c)
            {
                dObj4x2.at<_Tp>(r,c) = r*2+c;
            }
        }
    };
 
    virtual void TearDown(void) {};

    typedef _Tp valueType;    
    ito::DataObject dObj1_s;
    ito::DataObject dObj2_s;
    ito::DataObject dObj3_s;
    ito::DataObject dObj4_s;
    ito::DataObject dObj4_s1;

    ito::DataObject dObj1_sr;
    ito::DataObject dObj2_sr;
    ito::DataObject dObj3_sr;
    ito::DataObject dObj4_sr;

    ito::DataObject dObj1_d;
    ito::DataObject dObj2_d;
    ito::DataObject dObj3_d;
    ito::DataObject dObj4_d;

    ito::DataObject dObj1_dr;
    ito::DataObject dObj2_dr;
    ito::DataObject dObj3_dr;
    ito::DataObject dObj4_dr;

    ito::DataObject dObj1_dr1;
    ito::DataObject dObj2_dr1;
    ito::DataObject dObj3_dr1;
    ito::DataObject dObj4_dr1;

    ito::DataObject dObj1_dr2;
    ito::DataObject dObj2_dr2;
    ito::DataObject dObj3_dr2;
    ito::DataObject dObj4_dr2;

    ito::DataObject dObj5x1x1;
    ito::DataObject dObj1x5;
    ito::DataObject dObj1x4x1x2;
    ito::DataObject dObj4x2;
    };
    
TYPED_TEST_CASE(copyTests1, ItomRealDataTypes);


//deepCopyPartial_Test
/*!
    deepCopyPartial(..) function must raise an exception whenever the size (of ROI) of source and destination data objects are unequal. This test varifies this functionality.
*/
TYPED_TEST(copyTests1, deepCopyPartial_Test)
{
    dObj1_sr = dObj1_s; 
    dObj2_sr = dObj2_s;
    dObj3_sr = dObj3_s;
    dObj4_sr = dObj4_s;
            
    int matLimits1d_1[] = {-4,-6};            //!< defining offsets for ROI 
    int matLimits2d_1[] = {-4,-4,-1,-4};        //!< defining offsets for ROI 
    int matLimits3d_1[] = {-1,-1,-1,-1,-2,-1};
    int matLimits5d_1[] = {-1,-1,-1,0,-2,-1,-1,-1,-2,-1};

    dObj1_sr.adjustROI(0,matLimits1d_1);             //!< adjust ROI (shrinking because offset values are negative)
    dObj2_sr.adjustROI(2,matLimits2d_1);   //!< adjust ROI (shrinking because offset values are negative)
    dObj3_sr.adjustROI(3,matLimits3d_1);   //!< adjust ROI (shrinking because offset values are negative)
    dObj4_sr.adjustROI(5,matLimits5d_1);   //!< adjust ROI (shrinking because offset values are negative)

    dObj1_dr = dObj1_sr; 
    dObj2_dr = dObj2_sr;
    dObj3_dr = dObj3_sr;
    dObj4_dr = dObj4_sr;

    dObj1_dr1 = dObj1_d; 
    dObj2_dr1 = dObj2_d;
    dObj3_dr1 = dObj3_d;
    dObj4_dr1 = dObj4_d;

    int matLimits1d_2[] = {-3,-7};                //!< defining offsets for ROI 
    int matLimits2d_2[] = {-3,-3,-2,-5};        //!< defining offsets for ROI 
    int matLimits3d_2[] = {-1,-1,-1,-1,-1,-2};
    int matLimits5d_2[] = {-1,-1,-1,-1,-1,-1,-1,-1,-2,-1};

    dObj1_dr1.adjustROI(0,matLimits1d_2);    //!< adjust ROI (shrinking because offset values are negative)
    dObj2_dr1.adjustROI(2,matLimits2d_2);   //!< adjust ROI (shrinking because offset values are negative)
    dObj3_dr1.adjustROI(3,matLimits3d_2);   //!< adjust ROI (shrinking because offset values are negative)
    dObj4_dr1.adjustROI(5,matLimits5d_2);   //!< adjust ROI (shrinking because offset values are negative)

    dObj1_dr2 = dObj1_d; 
    dObj2_dr2 = dObj2_d;
    dObj3_dr2 = dObj3_d;
    dObj4_dr2 = dObj4_d;

    int matLimits1d_3[] = {-2,-2};                //!< defining offsets for ROI 
    int matLimits2d_3[] = {-2,-2,-2,-5};        //!< defining offsets for ROI 
    int matLimits3d_3[] = {-1,0,-1,-1,-1,-2};
    int matLimits5d_3[] = {0,0,-2,0,-2,0,-1,0,-2,-1};

    dObj1_dr2.adjustROI(0,matLimits1d_3);    //!< adjust ROI (shrinking because offset values are negative)
    dObj2_dr2.adjustROI(2,matLimits2d_3);   //!< adjust ROI (shrinking because offset values are negative)
    dObj3_dr2.adjustROI(3,matLimits3d_3);   //!< adjust ROI (shrinking because offset values are negative)
    dObj4_dr2.adjustROI(5,matLimits5d_3);   //!< adjust ROI (shrinking because offset values are negative)
    
    //!< Test for different sizes of ROI must raise exception.
    EXPECT_ANY_THROW(dObj2_sr.deepCopyPartial(dObj2_dr2));
    EXPECT_ANY_THROW(dObj3_sr.deepCopyPartial(dObj3_dr2));
    EXPECT_ANY_THROW(dObj4_sr.deepCopyPartial(dObj4_dr2));


}

//deepCopyPartial_Test1
/*!
    deepCopyPartial(..) function must raise an exception whenever the type of source and destination data objects are unequal. This test varifies this functionality.
*/
TYPED_TEST(copyTests1, deepCopyPartial_Test1)
{
    //!< Test for different types of Data Objects should raise exception.
    dObj2_s = ito::DataObject(2,2,ito::tUInt16);
    dObj2_d = ito::DataObject(2,2,ito::tUInt8);
    EXPECT_ANY_THROW(dObj2_s.deepCopyPartial(dObj2_d));
}

//deepCopyPartial_Test2
/*!
    Values of ROI of coursce must be equal to values in ROI of destination after copying. 
    Also values outside of ROI of destination must remain equal if checked after adjusting ROI back to original size and position.
    The above two conditions are checked in this test for 2 and 3 dimensional data objects.
*/
TYPED_TEST(copyTests1, deepCopyPartial_Test2)
{
    
    int temp=0;
    for(int i =0;i<10;i++) 
        {
            for(int j=0;j<10;j++)
            {dObj2_s.at<TypeParam>(i,j) = cv::saturate_cast<TypeParam>(10*i+j);}   //!< assigning unique value to each element of Data Object dObj2_s.
        }
    temp=0;
    for(int i =0;i<4;i++) 
        {
            for(int j=0;j<5;j++)
            {  
                for(int k=0;k<5;k++)
                {
                    dObj3_s.at<TypeParam>(i,j,k) = cv::saturate_cast<TypeParam>(temp++);   //!< assigning unique value to each element of 3 dimensional Data Object dObj3_s
                }
            }
        }


    dObj2_s.deepCopyPartial(dObj2_d);
    dObj3_s.deepCopyPartial(dObj3_d);

    for(int i =0;i<10;i++) 
        {
            for(int j=0;j<10;j++)
            {EXPECT_EQ(dObj2_s.at<TypeParam>(i,j),dObj2_d.at<TypeParam>(i,j));}   //!< assigning unique value to each element of Data Object dObj2_s.
        }

    for(int i =0;i<4;i++) 
        {
            for(int j=0;j<5;j++)
            {  
                for(int k=0;k<5;k++)
                {
                    EXPECT_EQ(dObj3_s.at<TypeParam>(i,j,k),dObj3_d.at<TypeParam>(i,j,k));   //!< assigning unique value to each element of 3 dimensional Data Object dObj3_s
                }
            }
        }

    dObj1_sr = dObj1_s; 
    dObj2_sr = dObj2_s;
    dObj3_sr = dObj3_s;
    
    int matLimits1d_1[] = {-4,-6};            //!< defining offsets for ROI 
    int matLimits2d_1[] = {-4,-4,-1,-4};        //!< defining offsets for ROI 
    int matLimits3d_1[] = {-1,-1,-1,-1,-2,-1};

    dObj1_sr.adjustROI(0,matLimits1d_1);             //!< adjust ROI (shrinking because offset values are negative)
    dObj2_sr.adjustROI(2,matLimits2d_1);   //!< adjust ROI (shrinking because offset values are negative)
    dObj3_sr.adjustROI(3,matLimits3d_1);   //!< adjust ROI (shrinking because offset values are negative)

    dObj1_dr = dObj1_d; 
    dObj2_dr = dObj2_d;
    dObj3_dr = dObj3_d;

    dObj1_dr.adjustROI(0,matLimits1d_1);             //!< adjust ROI (shrinking because offset values are negative)
    dObj2_dr.adjustROI(2,matLimits2d_1);   //!< adjust ROI (shrinking because offset values are negative)
    dObj3_dr.adjustROI(3,matLimits3d_1);   //!< adjust ROI (shrinking because offset values are negative)

    dObj2_sr.deepCopyPartial(dObj2_dr);
    dObj3_sr.deepCopyPartial(dObj3_dr);

    for(int i =0;i<2;i++) 
    {
        for(int j=0;j<5;j++)EXPECT_EQ(dObj2_sr.at<TypeParam>(i,j),dObj2_dr.at<TypeParam>(i,j)); //!< Check if the values of elements in ROI are same as in the original                
    }

    for(int i =0;i<2;i++) 
    {
        for(int j=0;j<3;j++)
        {  
            for(int k=0;k<2;k++)
            {
                EXPECT_EQ(dObj3_sr.at<TypeParam>(i,j,k),dObj3_dr.at<TypeParam>(i,j,k) );   //!< Checking if elements of ROI of original Data Object dObj3_sr has been changed while copying it to dObj3_dr.
            }
        }
    }

    int matLimits1d_2[] = {4,6};            //!< defining offsets for ROI 
    int matLimits2d_2[] = {4,4,1,4};        //!< defining offsets for ROI 
    int matLimits3d_2[] = {1,1,1,1,2,1};
    int matLimits5d_2[] = {4,4,1,4,2,3,1,1,2,1};

    dObj1_dr.adjustROI(0,matLimits1d_2);             //!< adjust ROI (expanding because offset values are negative)
    dObj2_dr.adjustROI(2,matLimits2d_2);   //!< adjust ROI (expanding because offset values are negative)
    dObj3_dr.adjustROI(3,matLimits3d_2);   //!< adjust ROI (expanding because offset values are negative)

    //!< Values of dObj1_dr can not be verified again after adjusting ROI because it is an empty Dataobject.

    //!< Checking values of dObj2_dr back with source values after adjusting ROI back equal to source ROI.
    for(int i =0;i<10;i++) 
    {
        for(int j=0;j<10;j++)EXPECT_EQ(dObj2_s.at<TypeParam>(i,j),dObj2_dr.at<TypeParam>(i,j)); //!< Check if the values of elements in ROI are same as in the original                
    }

    //!< Checking values of dObj3_dr back with source values after adjusting ROI back equal to source ROI.
    for(int i =0;i<4;i++) 
    {
        for(int j=0;j<5;j++)
        {  
            for(int k=0;k<5;k++)
            {
                EXPECT_EQ(dObj3_s.at<TypeParam>(i,j,k),dObj3_dr.at<TypeParam>(i,j,k) );   //!< Checking if elements of ROI of original Data Object dObj3_sr has been changed while copying it to dObj3_dr.
            }
        }
    }
}

// deepCopyPartial_Test3
/*!
    This test is basically the extension of deepCopyPartial_Test2 adding 5 dimensional data objects under test.
    So this test checks same functionality of deepCopyPartial() method on 5 dimensional data objects
*/
TYPED_TEST(copyTests1, deepCopyPartial_Test3)
{
    int planeId; 
    TypeParam *rowPtr  = NULL;    //!< Row pointer to locate each element of 5 dimensional data object dObj4_s at each row
    int planeId_d; 
    TypeParam *rowPtr_d  = NULL;    //!< Row pointer to locate each element of 5 dimensional data object dObj4_d at each row
    
    int dim1 = dObj4_s.getSize(0);
    int dim2 = dObj4_s.getSize(1);
    int dim3 = dObj4_s.getSize(2);
    int dim4 = dObj4_s.getSize(3);
    int dim5 = dObj4_s.getSize(4);
    
    int dataIdx = 0;
    int dataIdx_d = 0;

    for(int i=0; i<dim1; i++)
    {
        for(int j=0; j<dim2;j++)
        {
            for(int k=0; k<dim3;k++)
            {
                dataIdx = i*(dim1*dim2) + j*dim2 + k;
                planeId = dObj4_s.seekMat(dataIdx);

                for(int l=0; l<dim4;l++)
                {        
                    rowPtr = (TypeParam*)dObj4_s.rowPtr(planeId,l);

                    for(int m=0; m<dim5;m++)
                    {
                        rowPtr[m] = cv::saturate_cast<TypeParam>( i*10000 + j*1000 + k*100 + l*10 + m );    //!< Assigning unique value to each element of dObj4_s    
                    }
                }
            }
        }
    }


    dObj4_s.deepCopyPartial(dObj4_d);        //!< Creating a deep copy of dObj4_s into dObj4_d using deepCopyPartial() funtion

    int dim1_d = dObj4_d.getSize(0);
    int dim2_d = dObj4_d.getSize(1);
    int dim3_d = dObj4_d.getSize(2);
    int dim4_d = dObj4_d.getSize(3);
    int dim5_d = dObj4_d.getSize(4);

    for(int i=0; i<dim1_d; i++)
    {
        for(int j=0; j<dim2_d;j++)
        {
            for(int k=0; k<dim3_d;k++)
            {
                dataIdx_d = i*(dim1_d*dim2_d) + j*dim2_d + k;
                planeId_d = dObj4_d.seekMat(dataIdx_d);

                for(int l=0; l<dim4_d;l++)
                {        
                    rowPtr_d = (TypeParam*)dObj4_d.rowPtr(planeId_d,l);

                    for(int m=0; m<dim5_d;m++)
                    {
                        EXPECT_EQ( rowPtr_d[m],cv::saturate_cast<TypeParam>(i*10000 + j*1000 + k*100 + l*10 + m) );    //!< checking if the values of elements are same as original data object after applying deepCopyPartial() function.
                    }
                }
            }
        }
    }

}

//deepCopyPartial_Test4
/*!
    In this test, we define a 5 dimension data object, make a deep copy from it using deepCopyPartial() function , adjust the ROI of copied data object to smaller ROI, adjust the ROI back to original size of source data object and then compare the values of the elements if the values have been changed.
*/
TYPED_TEST(copyTests1, deepCopyPartial_Test4)
{
    int matLimits5d_1[] = {-1,-2,-1,-1,-2,0,-1,-1,-2,-1};
    dObj4_dr = dObj4_d;
    int matLimits5d_2[] = {1,2,1,1,2,0,1,1,2,1};
    int planeId; 
    TypeParam *rowPtr  = NULL; 
    int planeId_d; 
    TypeParam *rowPtr_d  = NULL;    
    int dim1 = dObj4_s.getSize(0);
    int dim2 = dObj4_s.getSize(1);
    int dim3 = dObj4_s.getSize(2);
    int dim4 = dObj4_s.getSize(3);
    int dim5 = dObj4_s.getSize(4);    
    int dataIdx = 0;
    int dataIdx_d = 0;
    for(int i=0; i<dim1; i++)
    {
        for(int j=0; j<dim2;j++)
        {
            for(int k=0; k<dim3;k++)
            {
                dataIdx = i*(dim1*dim2) + j*dim2 + k;
                planeId = dObj4_s.seekMat(dataIdx);

                for(int l=0; l<dim4;l++)
                {        
                    rowPtr = (TypeParam*)dObj4_s.rowPtr(planeId,l);

                    for(int m=0; m<dim5;m++)
                    {
                        rowPtr[m] = cv::saturate_cast<TypeParam>( i*10000 + j*1000 + k*100 + l*10 + m );    //!< Assigning unique value to each element of data object dObj4_s        
                    }
                }
            }
        }
    }

    dObj4_sr = dObj4_s;
    dObj4_s.deepCopyPartial(dObj4_d);
    dObj4_dr = dObj4_d;
    dObj4_dr.adjustROI(5,matLimits5d_1);   //!< adjust ROI (shrinking because offset values are negative)    
    dObj4_sr.adjustROI(5,matLimits5d_1);   //!< adjust ROI (shrinking because offset values are negative)
    dObj4_sr.deepCopyPartial(dObj4_dr);       //!< creating a deep copy of dObj4_sr into dObj4_dr
    dObj4_dr.adjustROI(5,matLimits5d_2);   //!< adjust ROI (shrinking because offset values are negative)
    int dim1_d = dObj4_d.getSize(0);
    int dim2_d = dObj4_d.getSize(1);
    int dim3_d = dObj4_d.getSize(2);
    int dim4_d = dObj4_d.getSize(3);
    int dim5_d = dObj4_d.getSize(4);

    //!< To test the values of dObj4_dr are same as those of origianal after adjusting the ROI back to original size.
    for(int i=0; i<dim1_d; i++)
    {
        for(int j=0; j<dim2_d;j++)
        {
            for(int k=0; k<dim3_d;k++)
            {
                dataIdx_d = i*(dim1_d*dim2_d) + j*dim2_d + k;
                planeId_d = dObj4_d.seekMat(dataIdx_d);
                for(int l=0; l<dim4_d;l++)
                {        
                    rowPtr_d = (TypeParam*)dObj4_dr.rowPtr(planeId_d,l);
                    for(int m=0; m<dim5_d;m++)
                    {
                        EXPECT_EQ( rowPtr_d[m],cv::saturate_cast<TypeParam>(i*10000 + j*1000 + k*100 + l*10 + m) ); //!< checking the element values of dObj4_dr are same as the original data object after adjusting ROI back to original size same as original data object dObj4_s
                    }
                }
            }
        }
    }

}


//deepCopyPartial_Test5
/*!
    This test checks that it is possible to copy the content of a 1x5 array to a 5x1x1 array (same type and same size if all one-dimensions are omitted)

*/
TYPED_TEST(copyTests1, deepCopyPartial_Test5)
{
    EXPECT_TRUE(dObj1x5.deepCopyPartial(dObj5x1x1) == ito::retOk);
    ito::DObjConstIterator it = dObj5x1x1.constBegin();
    TypeParam counter = 0;
    while (it != dObj5x1x1.constEnd())
    {
        EXPECT_EQ(cv::saturate_cast<TypeParam>(counter), *((TypeParam*)*it));
        counter++;
        it++;
    }

    EXPECT_TRUE(dObj4x2.deepCopyPartial(dObj1x4x1x2) == ito::retOk);
    it = dObj1x4x1x2.constBegin();
    counter = 0;
    while (it != dObj1x4x1x2.constEnd())
    {
        EXPECT_EQ(cv::saturate_cast<TypeParam>(counter), *((TypeParam*)*it));
        counter++;
        it++;
    }

    EXPECT_THROW(dObj1x5.deepCopyPartial(dObj1x4x1x2), cv::Exception);
    EXPECT_THROW(dObj1x5.deepCopyPartial(dObj4x2), cv::Exception);

    EXPECT_THROW(dObj1x4x1x2.deepCopyPartial(dObj1x5), cv::Exception);
    EXPECT_THROW(dObj1x4x1x2.deepCopyPartial(dObj5x1x1), cv::Exception);

    EXPECT_THROW(dObj4x2.deepCopyPartial(dObj5x1x1), cv::Exception);
    EXPECT_THROW(dObj4x2.deepCopyPartial(dObj1x5), cv::Exception);

    EXPECT_THROW(dObj5x1x1.deepCopyPartial(dObj1x4x1x2), cv::Exception);
    EXPECT_THROW(dObj5x1x1.deepCopyPartial(dObj4x2), cv::Exception);
}