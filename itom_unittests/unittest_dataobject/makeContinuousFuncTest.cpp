#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv2\opencv.hpp"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"


/*! \class makeContinuousFunc_test
    \brief Address test for all real data types declared as "ItomRealDataTypes"

    This Test checks functionality of makeContinuous() function.
*/
template <typename _Tp> class makeContinuousFunc_test : public ::testing::Test 
{ 
public:

    virtual void SetUp(void)
    {
        dObj_3d = ito::DataObject(4,5,5,ito::getDataType2<_Tp*>());
        int *temp_size = new int[4];
        temp_size[0] = 4;
        temp_size[1] = 5;
        temp_size[2] = 2;
        temp_size[3] = 3;
        dObj_4d = ito::DataObject(4,temp_size,ito::getDataType2<_Tp*>());
        //dObj_4d_con = ito::DataObject(4,temp_size,ito::getDataType2<_Tp*>(),1);
        //dObj_4dres = dObj_4d;
    };

    virtual void TearDown(void) {};

    typedef _Tp valueType;
    ito::DataObject dObj_3d; 
    ito::DataObject dObj_4d;
    ito::DataObject dObj_3dres;
    ito::DataObject dObj_4dres;
};


TYPED_TEST_CASE(makeContinuousFunc_test, ItomRealDataTypes);

//nonContTest_3d
/*!
    This test class checks the functionality and vulnerability of the makeContinuous() of 3 dimensional data objects.
*/
TYPED_TEST(makeContinuousFunc_test, nonContTest_3d)
{
    int dim1 = dObj_3d.getSize(0);
    int dim2 = dObj_3d.getSize(1);
    int dim3 = dObj_3d.getSize(2);
    int test_res3d[] = {36,37,38,41,42,43,61,62,63,66,67,68};    //!< Expected result vector for dObj3 after adjustROI method using 2 parameter (general) implementation
    int matLimits3d[] = {-1,-1,-2,-1,-1,-1};    //!< defining ROI offsets for 3 Dimensional Data Object dObj3
    int temp=0;

    for(int i=0; i<dim1; i++)
    {
        for(int j=0; j<dim2; j++)
        {    
            for(int k=0; k<dim3; k++)
            {
                dObj_3d.at<TypeParam>(i,j,k) = cv::saturate_cast<TypeParam>(temp++);            //!< Assigning each element of the data object dObj_3d with a unique integer value.
            }
        }
    }

    
    
    dObj_3d.adjustROI(3,matLimits3d);    //!< adjusting ROI of dObj3 with general 3 parameter adjustROI method to desired position
    
    //std::cout << dObj_3d << std::endl;

    //this must not crash!
    ito::DataObject *newObj = new ito::DataObject( ito::makeContinuous(dObj_3d) );
    delete newObj;

    dObj_3dres = ito::makeContinuous(dObj_3d);                    //!< Converting a non-continous data object dObj_3d into a continous one and storing the result into dObj_3dres data object. 

    

    //compare inner size of first plane in ROI with last two dimensions of dataObject
    int planeID;
    cv::Mat* plane;

    //std::cout << dObj_3dres << std::endl;
    dim1 = dObj_3dres.getSize(0);
    dim2 = dObj_3dres.getSize(1);
    dim3 = dObj_3dres.getSize(2);
    
    temp=0;
    TypeParam *linePtr;
    for(int i=0; i<dim1; i++)
    {
        planeID = dObj_3dres.seekMat(i);
        plane = (cv::Mat*)dObj_3dres.get_mdata()[planeID];
        //EXPECT_EQ(plane->cols, dObj_3dres.getSize(2) );
        //EXPECT_EQ(plane->rows, dObj_3dres.getSize(1) );

        for(int j=0; j<dim2; j++)
        {    
            linePtr = (TypeParam*)plane->ptr(j);
            for(int k=0; k<dim3; k++)
            {
                EXPECT_EQ(linePtr[k], cv::saturate_cast<TypeParam>(test_res3d[temp]));
                EXPECT_EQ(dObj_3dres.at<TypeParam>(i,j,k),cv::saturate_cast<TypeParam>(test_res3d[temp++]));    //!< Testing each element value of resulted data object dObj_3dres with respective element value of source data object dObj_3d after being applied to makeContinuous() function.
            }
        }
    }

}

TYPED_TEST(makeContinuousFunc_test, nonContTest_big3d)
{
    ito::DataObject bigDataObj(200,100,100,ito::getDataType( (const TypeParam *) NULL ) );
    int limits[] = {0,0,-50,-49,-50,-49};
    bigDataObj.adjustROI(3, limits );
    ito::DataObject *roiBased = new ito::DataObject(ito::makeContinuous(bigDataObj));        // ---------------------------> 21.05.2013 NOTE: TEST Fails here (Should dissapear after updating the files)
    delete roiBased;
}

//nonContTest_4d
/*!
    This test class checks the functionality and vulnerability of the makeContinuous() of 4 dimensional data objects.
*/
TYPED_TEST(makeContinuousFunc_test, nonContTest_4d)
{
    int dim1 = dObj_4d.getSize(0);
    int dim2 = dObj_4d.getSize(1);
    int dim3 = dObj_4d.getSize(2);
    int dim4 = dObj_4d.getSize(3);
    int matLimits4d[] = {-1,-1,0,-1,0,0,-1,-1}; //!< defining ROI offsets for 4 Dimensional Data Object dObj_4d
    int test_res4d[] = {31,34,37,40,43,46,49,52,61,64,67,70,73,76,79,82};    //!< Expected result vector for dObj3 after adjustROI method using 2 parameter (general) implementation
    int temp=0;
    TypeParam *rowPtr1= NULL; 
    int dataIdx = 0;        
    temp=0;

    for(int j=0; j<dim1;j++)
    {
        for(int k=0; k<dim2;k++)
        {
            dataIdx = j*dim2 + k;

            for(int l=0; l<dim3;l++)
            {        
                rowPtr1= (TypeParam*)dObj_4d.rowPtr(dataIdx,l);

                for(int m=0; m<dim4;m++)
                {
                    rowPtr1[m] = cv::saturate_cast<TypeParam>(temp++);    //!< assign unique value to each element of data object dObj4    
                }
            }
        }
    }
    //std::cout << dObj_4d << std::endl;
    dObj_4d.adjustROI(4,matLimits4d);                    //!< adjusting ROI of dObj_4d with general 2 parameter adjustROI method to desired position
    dObj_4dres = ito::makeContinuous(dObj_4d);            //!< Converting a non-continous data object dObj_4d into a continous one and storing the result into dObj_4dres data object. 
    //std::cout << dObj_4d << std::endl;
    dim1 = dObj_4d.getSize(0);
    dim2 = dObj_4d.getSize(1);
    dim3 = dObj_4d.getSize(2);
    dim4 = dObj_4d.getSize(3);
    unsigned int idx[] = {0,0,0,0};
    TypeParam v1;
    TypeParam v2;
    temp=0;
    for(int i=0; i<dim1; i++)
    {
        idx[0] = i;
        for(int j=0; j<dim2;j++)
        {
            idx[1] = j;
            for(int k=0; k<dim3;k++)
            {
                idx[2] = k;
                for(int l=0; l<dim4;l++)
                {        
                    idx[3] = l;                    
                    v1 = dObj_4dres.at<TypeParam>(idx);
                    v2 = cv::saturate_cast<TypeParam>(test_res4d[temp++]);
                    EXPECT_EQ(v1,v2);            //!< Testing each element value of resulted data object dObj_4dres with respective element value of source data object dObj_4d after being applied to makeContinuous() function.
                }
            }
        }
    }


}

TYPED_TEST(makeContinuousFunc_test, nonContTest_big4d)
{
    int *temp_size = new int[4];
    temp_size[0] = 50;
    temp_size[1] = 50;
    temp_size[2] = 100;
    temp_size[3] = 130;
    ito::DataObject bigDataObj(4,temp_size,ito::getDataType( (const TypeParam *) NULL ) );
    int limits[] = {0,0,-20,-20,-31,-49,-60,-10};
    bigDataObj.adjustROI(4, limits );
    ito::DataObject *roiBased = new ito::DataObject(ito::makeContinuous(bigDataObj));        // ---------------------------> 24.05.2013 NOTE: TEST Fails here (Should dissapear after updating the files)
    delete roiBased;
    //delete temp_size;
}

TYPED_TEST(makeContinuousFunc_test, ContTest_3d)
{    
    ito::DataObject dObj_3d_con(4,5,5,ito::getDataType( (const TypeParam *) NULL ),1);
    int dim1 = dObj_3d_con.getSize(0);
    int dim2 = dObj_3d_con.getSize(1);
    int dim3 = dObj_3d_con.getSize(2);
    int test_res3d[] = {32,33,37,38,42,43,57,58,62,63,67,68};    //!< Expected result vector for dObj3 after adjustROI method using 2 parameter (general) implementation
    //int matLimits3d[] = {-1,-1,-1,-1,-2,-1};    //!< defining ROI offsets for 3 Dimensional Data Object dObj3
    int matLimits3d[] = {0,0,0,0,-1,-1};
    int temp=0;

    for(int i=0; i<dim1; i++)
    {
        for(int j=0; j<dim2; j++)
        {    
            for(int k=0; k<dim3; k++)
            {
                dObj_3d_con.at<TypeParam>(i,j,k) = cv::saturate_cast<TypeParam>(temp++);            //!< Assigning each element of the data object dObj_3d with a unique integer value.
            }
        }
    }
    
    dObj_3d_con.adjustROI(3,matLimits3d);    //!< adjusting ROI of dObj3 with general 3 parameter adjustROI method to desired position
    //std::cout << dObj_3d_con << std::endl;
    //temp=0;
    //for(int i=0; i<4; i++)
    //{
    //    for(int j=0; j<3; j++)
    //    {    
    //        for(int k=0; k<2; k++)
    //        {
    //            EXPECT_EQ(dObj_3d_con.at<TypeParam>(i,j,k),cv::saturate_cast<TypeParam>(test_res3d[temp++]));    //!< Testing each element value of resulted data object dObj_3dres with respective element value of source data object dObj_3d after being applied to makeContinuous() function.
    //        }
    //    }
    //}
}


