#include <iostream>

#include "../../common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.

#include "opencv2/opencv.hpp"
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

        this->dObj1_s = ito::DataObject(0,temp_size1,type);
        this->dObj2_s = ito::DataObject(2,temp_size1,type);
        this->dObj3_s = ito::DataObject(4,5,5,type);
        this->dObj4_s = ito::DataObject(5,temp_size,type);
        this->dObj4_s1 = ito::DataObject(5,temp_size2,type);
        this->dObj1_d = ito::DataObject(0,temp_size1,type);
        this->dObj2_d = ito::DataObject(2,temp_size1,type);
        this->dObj3_d = ito::DataObject(4,5,5,type);
        this->dObj4_d = ito::DataObject(5,temp_size,type);


        this->dObj5x1x1 = ito::DataObject(3, temp_size3, type);
        this->dObj5x1x1.setTo(0);

        this->dObj1x5 = ito::DataObject(2, temp_size4, type);
        for (int i = 0; i < 5; ++i)
        {
            this->dObj1x5.at<_Tp>(0,i) = i;
        }

        this->dObj1x4x1x2 = ito::DataObject(4, temp_size5, type);
        this->dObj1x4x1x2.setTo(0);

        this->dObj4x2 = ito::DataObject(2, temp_size6, type);
        for (int r = 0; r < 4; ++r)
        {
            for (int c = 0; c < 2; ++c)
            {
                this->dObj4x2.at<_Tp>(r,c) = r*2+c;
            }
        }
    };
 
    virtual void TearDown(void) {};

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
    this->dObj1_sr = this->dObj1_s;
    this->dObj2_sr = this->dObj2_s;
    this->dObj3_sr = this->dObj3_s;
    this->dObj4_sr = this->dObj4_s;
            
    int matLimits1d_1[] = {-4,-6};            //!< defining offsets for ROI 
    int matLimits2d_1[] = {-4,-4,-1,-4};        //!< defining offsets for ROI 
    int matLimits3d_1[] = {-1,-1,-1,-1,-2,-1};
    int matLimits5d_1[] = {-1,-1,-1,0,-2,-1,-1,-1,-2,-1};

    this->dObj1_sr.adjustROI(0,matLimits1d_1);             //!< adjust ROI (shrinking because offset values are negative)
    this->dObj2_sr.adjustROI(2,matLimits2d_1);   //!< adjust ROI (shrinking because offset values are negative)
    this->dObj3_sr.adjustROI(3,matLimits3d_1);   //!< adjust ROI (shrinking because offset values are negative)
    this->dObj4_sr.adjustROI(5,matLimits5d_1);   //!< adjust ROI (shrinking because offset values are negative)

    this->dObj1_dr = this->dObj1_sr;
    this->dObj2_dr = this->dObj2_sr;
    this->dObj3_dr = this->dObj3_sr;
    this->dObj4_dr = this->dObj4_sr;

    this->dObj1_dr1 = this->dObj1_d;
    this->dObj2_dr1 = this->dObj2_d;
    this->dObj3_dr1 = this->dObj3_d;
    this->dObj4_dr1 = this->dObj4_d;

    int matLimits1d_2[] = {-3,-7};                //!< defining offsets for ROI 
    int matLimits2d_2[] = {-3,-3,-2,-5};        //!< defining offsets for ROI 
    int matLimits3d_2[] = {-1,-1,-1,-1,-1,-2};
    int matLimits5d_2[] = {-1,-1,-1,-1,-1,-1,-1,-1,-2,-1};

    this->dObj1_dr1.adjustROI(0,matLimits1d_2);    //!< adjust ROI (shrinking because offset values are negative)
    this->dObj2_dr1.adjustROI(2,matLimits2d_2);   //!< adjust ROI (shrinking because offset values are negative)
    this->dObj3_dr1.adjustROI(3,matLimits3d_2);   //!< adjust ROI (shrinking because offset values are negative)
    this->dObj4_dr1.adjustROI(5,matLimits5d_2);   //!< adjust ROI (shrinking because offset values are negative)

    this->dObj1_dr2 = this->dObj1_d;
    this->dObj2_dr2 = this->dObj2_d;
    this->dObj3_dr2 = this->dObj3_d;
    this->dObj4_dr2 = this->dObj4_d;

    int matLimits1d_3[] = {-2,-2};                //!< defining offsets for ROI 
    int matLimits2d_3[] = {-2,-2,-2,-5};        //!< defining offsets for ROI 
    int matLimits3d_3[] = {-1,0,-1,-1,-1,-2};
    int matLimits5d_3[] = {0,0,-2,0,-2,0,-1,0,-2,-1};

    this->dObj1_dr2.adjustROI(0,matLimits1d_3);    //!< adjust ROI (shrinking because offset values are negative)
    this->dObj2_dr2.adjustROI(2,matLimits2d_3);   //!< adjust ROI (shrinking because offset values are negative)
    this->dObj3_dr2.adjustROI(3,matLimits3d_3);   //!< adjust ROI (shrinking because offset values are negative)
    this->dObj4_dr2.adjustROI(5,matLimits5d_3);   //!< adjust ROI (shrinking because offset values are negative)
    
    //!< Test for different sizes of ROI must raise exception.
    EXPECT_ANY_THROW(this->dObj2_sr.deepCopyPartial(this->dObj2_dr2));
    EXPECT_ANY_THROW(this->dObj3_sr.deepCopyPartial(this->dObj3_dr2));
    EXPECT_ANY_THROW(this->dObj4_sr.deepCopyPartial(this->dObj4_dr2));


}

//deepCopyPartial_Test1
/*!
    deepCopyPartial(..) function must raise an exception whenever the type of source and destination data objects are unequal. This test varifies this functionality.
*/
TYPED_TEST(copyTests1, deepCopyPartial_Test1)
{
    //!< Test for different types of Data Objects should raise exception.
    this->dObj2_s = ito::DataObject(2,2,ito::tUInt16);
    this->dObj2_d = ito::DataObject(2,2,ito::tUInt8);
    EXPECT_ANY_THROW(this->dObj2_s.deepCopyPartial(this->dObj2_d));
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
            {this->dObj2_s.template at<TypeParam>(i,j) = cv::saturate_cast<TypeParam>(10*i+j);}   //!< assigning unique value to each element of Data Object this->dObj2_s.
        }
    temp=0;
    for(int i =0;i<4;i++) 
        {
            for(int j=0;j<5;j++)
            {  
                for(int k=0;k<5;k++)
                {
                    this->dObj3_s.template at<TypeParam>(i,j,k) = cv::saturate_cast<TypeParam>(temp++);   //!< assigning unique value to each element of 3 dimensional Data Object this->dObj3_s
                }
            }
        }


    this->dObj2_s.deepCopyPartial(this->dObj2_d);
    this->dObj3_s.deepCopyPartial(this->dObj3_d);

    for(int i =0;i<10;i++) 
        {
            for(int j=0;j<10;j++)
            {EXPECT_EQ(this->dObj2_s.template at<TypeParam>(i,j),this->dObj2_d.template at<TypeParam>(i,j));}   //!< assigning unique value to each element of Data Object this->dObj2_s.
        }

    for(int i =0;i<4;i++) 
        {
            for(int j=0;j<5;j++)
            {  
                for(int k=0;k<5;k++)
                {
                    EXPECT_EQ(this->dObj3_s.template at<TypeParam>(i,j,k),this->dObj3_d.template at<TypeParam>(i,j,k));   //!< assigning unique value to each element of 3 dimensional Data Object this->dObj3_s
                }
            }
        }

    this->dObj1_sr = this->dObj1_s;
    this->dObj2_sr = this->dObj2_s;
    this->dObj3_sr = this->dObj3_s;
    
    int matLimits1d_1[] = {-4,-6};            //!< defining offsets for ROI 
    int matLimits2d_1[] = {-4,-4,-1,-4};        //!< defining offsets for ROI 
    int matLimits3d_1[] = {-1,-1,-1,-1,-2,-1};

    this->dObj1_sr.adjustROI(0,matLimits1d_1);             //!< adjust ROI (shrinking because offset values are negative)
    this->dObj2_sr.adjustROI(2,matLimits2d_1);   //!< adjust ROI (shrinking because offset values are negative)
    this->dObj3_sr.adjustROI(3,matLimits3d_1);   //!< adjust ROI (shrinking because offset values are negative)

    this->dObj1_dr = this->dObj1_d;
    this->dObj2_dr = this->dObj2_d;
    this->dObj3_dr = this->dObj3_d;

    this->dObj1_dr.adjustROI(0,matLimits1d_1);             //!< adjust ROI (shrinking because offset values are negative)
    this->dObj2_dr.adjustROI(2,matLimits2d_1);   //!< adjust ROI (shrinking because offset values are negative)
    this->dObj3_dr.adjustROI(3,matLimits3d_1);   //!< adjust ROI (shrinking because offset values are negative)

    this->dObj2_sr.deepCopyPartial(this->dObj2_dr);
    this->dObj3_sr.deepCopyPartial(this->dObj3_dr);

    for(int i =0;i<2;i++) 
    {
        for(int j=0;j<5;j++)EXPECT_EQ(this->dObj2_sr.template at<TypeParam>(i,j),this->dObj2_dr.template at<TypeParam>(i,j)); //!< Check if the values of elements in ROI are same as in the original
    }

    for(int i =0;i<2;i++) 
    {
        for(int j=0;j<3;j++)
        {  
            for(int k=0;k<2;k++)
            {
                EXPECT_EQ(this->dObj3_sr.template at<TypeParam>(i,j,k),this->dObj3_dr.template at<TypeParam>(i,j,k) );   //!< Checking if elements of ROI of original Data Object this->dObj3_sr has been changed while copying it to this->dObj3_dr.
            }
        }
    }

    int matLimits1d_2[] = {4,6};            //!< defining offsets for ROI 
    int matLimits2d_2[] = {4,4,1,4};        //!< defining offsets for ROI 
    int matLimits3d_2[] = {1,1,1,1,2,1};
    int matLimits5d_2[] = {4,4,1,4,2,3,1,1,2,1};

    this->dObj1_dr.adjustROI(0,matLimits1d_2);             //!< adjust ROI (expanding because offset values are negative)
    this->dObj2_dr.adjustROI(2,matLimits2d_2);   //!< adjust ROI (expanding because offset values are negative)
    this->dObj3_dr.adjustROI(3,matLimits3d_2);   //!< adjust ROI (expanding because offset values are negative)

    //!< Values of this->dObj1_dr can not be verified again after adjusting ROI because it is an empty Dataobject.

    //!< Checking values of this->dObj2_dr back with source values after adjusting ROI back equal to source ROI.
    for(int i =0;i<10;i++) 
    {
        for(int j=0;j<10;j++)EXPECT_EQ(this->dObj2_s.template at<TypeParam>(i,j),this->dObj2_dr.template at<TypeParam>(i,j)); //!< Check if the values of elements in ROI are same as in the original
    }

    //!< Checking values of this->dObj3_dr back with source values after adjusting ROI back equal to source ROI.
    for(int i =0;i<4;i++) 
    {
        for(int j=0;j<5;j++)
        {  
            for(int k=0;k<5;k++)
            {
                EXPECT_EQ(this->dObj3_s.template at<TypeParam>(i,j,k),this->dObj3_dr.template at<TypeParam>(i,j,k) );   //!< Checking if elements of ROI of original Data Object this->dObj3_sr has been changed while copying it to this->dObj3_dr.
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
    TypeParam *rowPtr  = NULL;    //!< Row pointer to locate each element of 5 dimensional data object this->dObj4_s at each row
    int planeId_d; 
    TypeParam *rowPtr_d  = NULL;    //!< Row pointer to locate each element of 5 dimensional data object this->dObj4_d at each row
    
    int dim1 = this->dObj4_s.getSize(0);
    int dim2 = this->dObj4_s.getSize(1);
    int dim3 = this->dObj4_s.getSize(2);
    int dim4 = this->dObj4_s.getSize(3);
    int dim5 = this->dObj4_s.getSize(4);
    
    int dataIdx = 0;
    int dataIdx_d = 0;

    for(int i=0; i<dim1; i++)
    {
        for(int j=0; j<dim2;j++)
        {
            for(int k=0; k<dim3;k++)
            {
                dataIdx = i*(dim1*dim2) + j*dim2 + k;
                planeId = this->dObj4_s.seekMat(dataIdx);

                for(int l=0; l<dim4;l++)
                {        
                    rowPtr = (TypeParam*)this->dObj4_s.rowPtr(planeId,l);

                    for(int m=0; m<dim5;m++)
                    {
                        rowPtr[m] = cv::saturate_cast<TypeParam>( i*10000 + j*1000 + k*100 + l*10 + m );    //!< Assigning unique value to each element of this->dObj4_s
                    }
                }
            }
        }
    }


    this->dObj4_s.deepCopyPartial(this->dObj4_d);        //!< Creating a deep copy of this->dObj4_s into this->dObj4_d using deepCopyPartial() funtion

    int dim1_d = this->dObj4_d.getSize(0);
    int dim2_d = this->dObj4_d.getSize(1);
    int dim3_d = this->dObj4_d.getSize(2);
    int dim4_d = this->dObj4_d.getSize(3);
    int dim5_d = this->dObj4_d.getSize(4);

    for(int i=0; i<dim1_d; i++)
    {
        for(int j=0; j<dim2_d;j++)
        {
            for(int k=0; k<dim3_d;k++)
            {
                dataIdx_d = i*(dim1_d*dim2_d) + j*dim2_d + k;
                planeId_d = this->dObj4_d.seekMat(dataIdx_d);

                for(int l=0; l<dim4_d;l++)
                {        
                    rowPtr_d = (TypeParam*)this->dObj4_d.rowPtr(planeId_d,l);

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
    this->dObj4_dr = this->dObj4_d;
    int matLimits5d_2[] = {1,2,1,1,2,0,1,1,2,1};
    int planeId; 
    TypeParam *rowPtr  = NULL; 
    int planeId_d; 
    TypeParam *rowPtr_d  = NULL;    
    int dim1 = this->dObj4_s.getSize(0);
    int dim2 = this->dObj4_s.getSize(1);
    int dim3 = this->dObj4_s.getSize(2);
    int dim4 = this->dObj4_s.getSize(3);
    int dim5 = this->dObj4_s.getSize(4);
    int dataIdx = 0;
    int dataIdx_d = 0;
    for(int i=0; i<dim1; i++)
    {
        for(int j=0; j<dim2;j++)
        {
            for(int k=0; k<dim3;k++)
            {
                dataIdx = i*(dim1*dim2) + j*dim2 + k;
                planeId = this->dObj4_s.seekMat(dataIdx);

                for(int l=0; l<dim4;l++)
                {        
                    rowPtr = (TypeParam*)this->dObj4_s.rowPtr(planeId,l);

                    for(int m=0; m<dim5;m++)
                    {
                        rowPtr[m] = cv::saturate_cast<TypeParam>( i*10000 + j*1000 + k*100 + l*10 + m );    //!< Assigning unique value to each element of data object this->dObj4_s
                    }
                }
            }
        }
    }

    this->dObj4_sr = this->dObj4_s;
    this->dObj4_s.deepCopyPartial(this->dObj4_d);
    this->dObj4_dr = this->dObj4_d;
    this->dObj4_dr.adjustROI(5,matLimits5d_1);   //!< adjust ROI (shrinking because offset values are negative)
    this->dObj4_sr.adjustROI(5,matLimits5d_1);   //!< adjust ROI (shrinking because offset values are negative)
    this->dObj4_sr.deepCopyPartial(this->dObj4_dr);       //!< creating a deep copy of this->dObj4_sr into this->dObj4_dr
    this->dObj4_dr.adjustROI(5,matLimits5d_2);   //!< adjust ROI (shrinking because offset values are negative)
    int dim1_d = this->dObj4_d.getSize(0);
    int dim2_d = this->dObj4_d.getSize(1);
    int dim3_d = this->dObj4_d.getSize(2);
    int dim4_d = this->dObj4_d.getSize(3);
    int dim5_d = this->dObj4_d.getSize(4);

    //!< To test the values of this->dObj4_dr are same as those of origianal after adjusting the ROI back to original size.
    for(int i=0; i<dim1_d; i++)
    {
        for(int j=0; j<dim2_d;j++)
        {
            for(int k=0; k<dim3_d;k++)
            {
                dataIdx_d = i*(dim1_d*dim2_d) + j*dim2_d + k;
                planeId_d = this->dObj4_d.seekMat(dataIdx_d);
                for(int l=0; l<dim4_d;l++)
                {        
                    rowPtr_d = (TypeParam*)this->dObj4_dr.rowPtr(planeId_d,l);
                    for(int m=0; m<dim5_d;m++)
                    {
                        EXPECT_EQ( rowPtr_d[m],cv::saturate_cast<TypeParam>(i*10000 + j*1000 + k*100 + l*10 + m) ); //!< checking the element values of this->dObj4_dr are same as the original data object after adjusting ROI back to original size same as original data object this->dObj4_s
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
    EXPECT_TRUE(this->dObj1x5.deepCopyPartial(this->dObj5x1x1) == ito::retOk);
    ito::DObjConstIterator it = this->dObj5x1x1.constBegin();
    TypeParam counter = 0;
    while (it != this->dObj5x1x1.constEnd())
    {
        EXPECT_EQ(cv::saturate_cast<TypeParam>(counter), *((TypeParam*)*it));
        counter++;
        it++;
    }

    EXPECT_TRUE(this->dObj4x2.deepCopyPartial(this->dObj1x4x1x2) == ito::retOk);
    it = this->dObj1x4x1x2.constBegin();
    counter = 0;
    while (it != this->dObj1x4x1x2.constEnd())
    {
        EXPECT_EQ(cv::saturate_cast<TypeParam>(counter), *((TypeParam*)*it));
        counter++;
        it++;
    }

    EXPECT_THROW(this->dObj1x5.deepCopyPartial(this->dObj1x4x1x2), cv::Exception);
    EXPECT_THROW(this->dObj1x5.deepCopyPartial(this->dObj4x2), cv::Exception);

    EXPECT_THROW(this->dObj1x4x1x2.deepCopyPartial(this->dObj1x5), cv::Exception);
    EXPECT_THROW(this->dObj1x4x1x2.deepCopyPartial(this->dObj5x1x1), cv::Exception);

    EXPECT_THROW(this->dObj4x2.deepCopyPartial(this->dObj5x1x1), cv::Exception);
    EXPECT_THROW(this->dObj4x2.deepCopyPartial(this->dObj1x5), cv::Exception);

    EXPECT_THROW(this->dObj5x1x1.deepCopyPartial(this->dObj1x4x1x2), cv::Exception);
    EXPECT_THROW(this->dObj5x1x1.deepCopyPartial(this->dObj4x2), cv::Exception);
}
