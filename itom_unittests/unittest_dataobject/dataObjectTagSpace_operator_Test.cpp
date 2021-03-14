
#include "../../common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.

#include "opencv2/opencv.hpp"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"

/*! \class dataObjectTag_Test
    \brief Test for DataObjectTag class and functions for all itom data types

    This test class checks functionality of different fuctions on data objects Tags.
*/

template <typename _Tp> class dataObjectTagSpace_operator_Test : public ::testing::Test 
    { 
public:

    virtual void SetUp(void)
    {
        //Creating 1,2 and 3 dimension DataObjects for this Perticular Test class.

        rotMat[0] =  4.0;
        rotMat[1] =  3.0;
        rotMat[2] = 11.0;
        rotMat[3] =  2.0;
        rotMat[4] =  0.5;
        rotMat[5] = 99.0;
        rotMat[6] = 22.0;
        rotMat[7] = 7.0;
        rotMat[8] = 35.0;

        this->mat1_2d = ito::DataObject(3,4,ito::getDataType2<_Tp*>());
        this->mat1_2d.setTag("testTag1", "test");
        this->mat1_2d.setTag("testTag2", 0.0);
        this->mat1_2d.setTag("testTag3", 1.0);
        
        this->mat1_2d.setAxisDescription(0, "y");
        this->mat1_2d.setAxisDescription(1, "x");

        this->mat1_2d.setAxisUnit(0, "mm");
        this->mat1_2d.setAxisUnit(1, "%");

        this->mat1_2d.setAxisOffset(0, 0.0);
        this->mat1_2d.setAxisOffset(1, 1.0);

        this->mat1_2d.setAxisScale(0, 2.0);
        this->mat1_2d.setAxisScale(1, 0.5);

        this->mat1_2d.setValueUnit("mm");
        this->mat1_2d.setValueDescription("val");
        this->mat1_2d.setXYRotationalMatrix(rotMat[0], rotMat[1], rotMat[2], rotMat[3], rotMat[4], rotMat[5], rotMat[6], rotMat[7], rotMat[8]);
        

        this->mat2_2d = ito::DataObject(3,3,ito::getDataType2<_Tp*>());
        this->mat2_2d.setTag("testTag1", "test");
        this->mat2_2d.setTag("testTag2", 0.0);
        this->mat2_2d.setTag("testTag3", 1.0);
        this->mat2_2d.setAxisDescription(0, "y");
        this->mat2_2d.setAxisDescription(1, "x");

        this->mat2_2d.setAxisUnit(0, "mm");
        this->mat2_2d.setAxisUnit(1, "%");

        this->mat2_2d.setAxisOffset(0, 2.0);
        this->mat2_2d.setAxisOffset(1, 3.0);

        this->mat2_2d.setAxisScale(0, 1.0);
        this->mat2_2d.setAxisScale(1, 2.0);
        this->mat2_2d.setValueUnit("mm");
        this->mat2_2d.setValueDescription("val");
        this->mat2_2d.setXYRotationalMatrix(rotMat[0], rotMat[1], rotMat[2], rotMat[3], rotMat[4], rotMat[5], rotMat[6], rotMat[7], rotMat[8]);

        this->mat1_3d = ito::DataObject(3,3,3,ito::getDataType2<_Tp*>());
        this->mat1_3d.setTag("testTag1", "test");
        this->mat1_3d.setTag("testTag2", 0.0);
        this->mat1_3d.setTag("testTag3", 1.0);
        
        this->mat1_3d.setAxisDescription(0, "z");
        this->mat1_3d.setAxisDescription(1, "y");
        this->mat1_3d.setAxisDescription(2, "x");

        this->mat1_3d.setAxisUnit(0, "s");
        this->mat1_3d.setAxisUnit(1, "%");
        this->mat1_3d.setAxisUnit(2, "mm");

        this->mat1_3d.setAxisOffset(0, 0.0);
        this->mat1_3d.setAxisOffset(1, 1.0);
        this->mat1_3d.setAxisOffset(2, 1.0);

        this->mat1_3d.setAxisScale(0, 2.0);
        this->mat1_3d.setAxisScale(1, 0.5);
        this->mat1_3d.setAxisScale(2, 5.0);
        this->mat1_3d.setValueUnit("mm");
        this->mat1_3d.setValueDescription("val");
        this->mat1_3d.setXYRotationalMatrix(rotMat[0], rotMat[1], rotMat[2], rotMat[3], rotMat[4], rotMat[5], rotMat[6], rotMat[7], rotMat[8]);


        this->mat2_3d = ito::DataObject(1,3,3,ito::getDataType2<_Tp*>());
        this->mat2_3d.setTag("testTag1", "test");
        this->mat2_3d.setTag("testTag2", 0.0);
        this->mat2_3d.setTag("testTag3", 1.0);
        
        this->mat2_3d.setAxisDescription(0, "z");
        this->mat2_3d.setAxisDescription(1, "y");
        this->mat2_3d.setAxisDescription(2, "x");

        this->mat2_3d.setAxisUnit(0, "s");
        this->mat2_3d.setAxisUnit(1, "%");
        this->mat2_3d.setAxisUnit(2, "mm");

        this->mat2_3d.setAxisOffset(0, 0.0);
        this->mat2_3d.setAxisOffset(1, 1.0);
        this->mat2_3d.setAxisOffset(2, 1.0);

        this->mat2_3d.setAxisScale(0, 2.0);
        this->mat2_3d.setAxisScale(1, 0.5);
        this->mat2_3d.setAxisScale(2, 5.0);
        this->mat2_3d.setValueUnit("mm");
        this->mat2_3d.setValueDescription("val");
        this->mat2_3d.setXYRotationalMatrix(rotMat[0], rotMat[1], rotMat[2], rotMat[3], rotMat[4], rotMat[5], rotMat[6], rotMat[7], rotMat[8]);
    }

    virtual void validTagSpace(const ito::DataObject &base, const ito::DataObject &temp)
    {
        double rotTempMat[9];
        bool vop1;
        bool vop2;
        bool vop3;
        std::string key1 = temp.getTagKey(0,vop1);
        std::string key2 = temp.getTagKey(1,vop2);
        std::string key3 = temp.getTagKey(2,vop3);
        EXPECT_EQ(key1,"testTag1");                        //checks if the key1 is same as the one assigned by "setTag" function.
        EXPECT_EQ(key2,"testTag2");                        //checks if the key2 is same as the one assigned by "setTag" function.
        EXPECT_EQ(key3,"testTag3");                        //checks if the key3 is same as the one assigned by "setTag" function.
        EXPECT_TRUE(vop1);                                //checks if the above operation is valid for 1st Tag 
        EXPECT_TRUE(vop2);                                //checks if the above operation is valid for 2nd Tag 
        EXPECT_TRUE(vop3);                                //checks if the above operation is valid for 3rd Tag

        ito::DataObjectTagType tag1 = temp.getTag("testTag1", vop1);
        ito::DataObjectTagType tag2 = temp.getTag("testTag2", vop2);
        ito::DataObjectTagType tag3 = temp.getTag("testTag3", vop3);

        EXPECT_EQ(tag1.getType(), ito::DataObjectTagType::typeString);    
        EXPECT_EQ(tag2.getType(), ito::DataObjectTagType::typeDouble);
        EXPECT_EQ(tag3.getType(), ito::DataObjectTagType::typeDouble);

        EXPECT_EQ(tag1.getVal_ToString(), "test");                        //checks if the key1 is same as the one assigned by "setTag" function.
        EXPECT_DOUBLE_EQ(tag2.getVal_ToDouble(), 0.0);                        //checks if the key2 is same as the one assigned by "setTag" function.
        EXPECT_DOUBLE_EQ(tag3.getVal_ToDouble(), 1.0);                        //checks if the key3 is same as the one assigned by "setTag" function.

        EXPECT_TRUE(vop1);                                //checks if the above operation is valid for 1st Tag 
        EXPECT_TRUE(vop2);                                //checks if the above operation is valid for 2nd Tag 
        EXPECT_TRUE(vop3);                                //checks if the above operation is valid for 3rd Tag

        EXPECT_EQ(base.getValueUnit(),          temp.getValueUnit());
        EXPECT_EQ(base.getValueDescription(),   temp.getValueDescription());
    
        temp.getXYRotationalMatrix(rotTempMat[0], rotTempMat[1], rotTempMat[2], rotTempMat[3], rotTempMat[4], rotTempMat[5], rotTempMat[6], rotTempMat[7], rotTempMat[8]);
        for(int i = 0; i < 9; i++)
        {
            EXPECT_DOUBLE_EQ(rotMat[i], rotTempMat[i]);
        }   
    }

    virtual void validAxisTags(const ito::DataObject &base, const ito::DataObject &temp, bool invertLastDims = false)
    {
        bool vop1;
        bool vop2;
        bool vop3;

        if(invertLastDims)
        {

            int axisNumTemp= temp.getDims()-3;
            int yTemp = temp.getDims() - 2;
            int xTemp = temp.getDims() - 1;
            int yBase = base.getDims() - 2;
            int xBase = base.getDims() - 1;
            if(axisNumTemp < 0)
            {
                for(int axisNumBase = base.getDims()-3; axisNumBase > -1 ; axisNumBase--)
                {
                    EXPECT_EQ(base.getAxisDescription(axisNumBase, vop1), temp.getAxisDescription(axisNumTemp, vop1));
                    EXPECT_EQ(base.getAxisUnit(axisNumBase, vop1),        temp.getAxisUnit(axisNumTemp, vop1));
                    EXPECT_DOUBLE_EQ(base.getAxisOffset(axisNumBase),     temp.getAxisOffset(axisNumTemp));
                    EXPECT_DOUBLE_EQ(base.getAxisScale(axisNumBase),      temp.getAxisScale(axisNumTemp));

                    axisNumTemp--;
                    if(axisNumTemp < 0)
                    {
                        break;
                    }
                }
            }
            
            EXPECT_EQ(base.getAxisDescription(xBase, vop1), temp.getAxisDescription(yTemp, vop1));
            EXPECT_EQ(base.getAxisUnit(xBase, vop1),        temp.getAxisUnit(yTemp, vop1));
            EXPECT_DOUBLE_EQ(base.getAxisOffset(xBase),     temp.getAxisOffset(yTemp));
            EXPECT_DOUBLE_EQ(base.getAxisScale(xBase),      temp.getAxisScale(yTemp));

            EXPECT_EQ(base.getAxisDescription(yBase, vop1), temp.getAxisDescription(xTemp, vop1));
            EXPECT_EQ(base.getAxisUnit(yBase, vop1),        temp.getAxisUnit(xTemp, vop1));
            EXPECT_DOUBLE_EQ(base.getAxisOffset(yBase),     temp.getAxisOffset(xTemp));
            EXPECT_DOUBLE_EQ(base.getAxisScale(yBase),      temp.getAxisScale(xTemp));     
            
        }
        else
        {
            int axisNumTemp= temp.getDims()-1;

            for(int axisNumBase = base.getDims()-1; axisNumBase > -1 ; axisNumBase--)
            {
                EXPECT_EQ(base.getAxisDescription(axisNumBase, vop1), temp.getAxisDescription(axisNumTemp, vop1));
                EXPECT_EQ(base.getAxisUnit(axisNumBase, vop1),        temp.getAxisUnit(axisNumTemp, vop1));
                EXPECT_DOUBLE_EQ(base.getAxisOffset(axisNumBase),     temp.getAxisOffset(axisNumTemp));
                EXPECT_DOUBLE_EQ(base.getAxisScale(axisNumBase),      temp.getAxisScale(axisNumTemp));

                axisNumTemp--;
                if(axisNumTemp < 0)
                {
                    break;
                }
            }        
        }
    }

    virtual void invalid(const ito::DataObject &base, const ito::DataObject &temp)
    {
        double rotTempMat[9];
        double rotTempMat2[9];

        EXPECT_NE(base.getTagListSize(), temp.getTagListSize());
        EXPECT_NE(base.getValueUnit(), temp.getValueUnit());
        EXPECT_NE(base.getValueDescription(), temp.getValueDescription());

        base.getXYRotationalMatrix(rotTempMat[0], rotTempMat[1], rotTempMat[2], rotTempMat[3], rotTempMat[4], rotTempMat[5], rotTempMat[6], rotTempMat[7], rotTempMat[8]);
        temp.getXYRotationalMatrix(rotTempMat2[0], rotTempMat2[1], rotTempMat2[2], rotTempMat2[3], rotTempMat2[4], rotTempMat2[5], rotTempMat2[6], rotTempMat2[7], rotTempMat2[8]);
        for(int i = 0; i < 9; i++)
        {
            EXPECT_NE(rotTempMat2[i], rotTempMat[i]);
        }   
    }

    virtual void TearDown(void){}
    typedef _Tp valueType;

    ito::DataObject mat1_2d;
    ito::DataObject mat2_2d;

    ito::DataObject mat1_3d;
    ito::DataObject mat2_3d;

    double rotMat[9];
};

TYPED_TEST_CASE(dataObjectTagSpace_operator_Test,ItomDataAllTypes);

//Check if operant supportes copy of tag space

//tagSpace_copyTo_Test
/*!
    This test checks functionality of "copyTo" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_copyTo_Test)
{
    ito::DataObject dTempTest;

    this->mat1_3d.copyTo(dTempTest);
    this->validTagSpace(this->mat1_3d, dTempTest);
    this->validAxisTags(this->mat1_3d, dTempTest);

    this->mat1_2d.copyTo(dTempTest);
    this->validTagSpace(this->mat1_2d, dTempTest);
    this->validAxisTags(this->mat1_2d, dTempTest);

};

//tagSpace_copyTagSpaces_Test
/*!
    This test checks functionality of "copyTagMap" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_copyTagSpaces_Test)
{
    ito::DataObject dTempTest(this->mat1_3d.getDims(), this->mat1_3d.getSize(), this->mat1_3d.getType());

    this->mat1_3d.copyTagMapTo(dTempTest);
    this->mat1_3d.copyAxisTagsTo(dTempTest);
    this->validTagSpace(this->mat1_3d, dTempTest);
    this->validAxisTags(this->mat1_3d, dTempTest);

    dTempTest = ito::DataObject(this->mat1_2d.getDims(), this->mat1_2d.getSize(), this->mat1_2d.getType());

    this->mat1_2d.copyTagMapTo(dTempTest);
    this->mat1_2d.copyAxisTagsTo(dTempTest);
    this->validTagSpace(this->mat1_2d, dTempTest);
    this->validAxisTags(this->mat1_2d, dTempTest);

    dTempTest = ito::DataObject(this->mat1_2d.getDims(), this->mat1_2d.getSize(), this->mat1_2d.getType());

    this->mat1_3d.copyTagMapTo(dTempTest);
    this->mat1_3d.copyAxisTagsTo(dTempTest);
    this->validTagSpace(this->mat1_3d, dTempTest);
    this->validAxisTags(this->mat1_3d, dTempTest);

};

//tagSpace_convertTo_Test
/*!
    This test checks functionality of "converTo"/astype of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_convertTo_Test)
{
    ito::DataObject dTempTest1;
    
    this->mat1_2d.convertTo(dTempTest1, this->mat1_2d.getType());
    this->validTagSpace(this->mat1_2d, dTempTest1);
    this->validAxisTags(this->mat1_2d, dTempTest1);

    ito::DataObject dTempTest2;
    this->mat1_2d.convertTo(dTempTest2, this->mat1_2d.getType(), 2.0, 1.0);
    this->validTagSpace(this->mat1_2d, dTempTest2);
    this->validAxisTags(this->mat1_2d, dTempTest2);

};

//tagSpace_copySallow_Test
/*!
    This test checks functionality of "shallow copy" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_copySallow_Test)
{
    ito::DataObject dTempTest1 = this->mat1_2d;
    
    this->validTagSpace(this->mat1_2d, dTempTest1);
    this->validAxisTags(this->mat1_2d, dTempTest1);

    ito::DataObject dTempTest2(this->mat1_2d);
    this->validTagSpace(this->mat1_2d, dTempTest2);
    this->validAxisTags(this->mat1_2d, dTempTest2);

};

//tagSpace_assignedScalar_Test
/*!
    This test checks functionality of "operator assignScalar" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_assignedScalar_Test)
{
    ito::DataObject dTempTest1;
    
    this->mat1_2d.copyTo(dTempTest1);

    dTempTest1 = 11;

    this->validTagSpace(this->mat1_2d, dTempTest1);
    this->validAxisTags(this->mat1_2d, dTempTest1);

};

//tagSpace_add_Test
/*!
    This test checks functionality of "addScalar" and "add" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_add_Test)
{
    ito::DataObject dTempTest1;
    
    this->mat1_2d.copyTo(dTempTest1);

    dTempTest1 += 11;

    this->validTagSpace(this->mat1_2d, dTempTest1);
    this->validAxisTags(this->mat1_2d, dTempTest1);

    ito::DataObject dTempTest2 = this->mat1_2d + 10;
    this->validTagSpace(this->mat1_2d, dTempTest2);
    this->validAxisTags(this->mat1_2d, dTempTest2);

    dTempTest1 = ito::DataObject();
    dTempTest1.ones(this->mat1_2d.getDims(), this->mat1_2d.getSize(), this->mat1_2d.getType());

    ito::DataObject dTempTest3 = this->mat1_2d + dTempTest1;
    this->validTagSpace(this->mat1_2d, dTempTest3);
    this->validAxisTags(this->mat1_2d, dTempTest3);
    this->invalid(dTempTest1, dTempTest3);

};

//tagSpace_sub_Test
/*!
    This test checks functionality of "subScalar" and "sub" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_sub_Test)
{
    ito::DataObject dTempTest1;
    
    this->mat1_2d.copyTo(dTempTest1);

    dTempTest1 -= 11;

    this->validTagSpace(this->mat1_2d, dTempTest1);
    this->validAxisTags(this->mat1_2d, dTempTest1);

    ito::DataObject dTempTest2 = this->mat1_2d - 10;
    this->validTagSpace(this->mat1_2d, dTempTest2);
    this->validAxisTags(this->mat1_2d, dTempTest2);

    dTempTest1 = ito::DataObject();
    dTempTest1.ones(this->mat1_2d.getDims(), this->mat1_2d.getSize(), this->mat1_2d.getType());

    ito::DataObject dTempTest3 = this->mat1_2d - dTempTest1;
    this->validTagSpace(this->mat1_2d, dTempTest3);
    this->validAxisTags(this->mat1_2d, dTempTest3);
    this->invalid(dTempTest1, dTempTest3);

};

//tagSpace_mul_Test
/*!
    This test checks functionality of "mulScalar" and "mul" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_mul_Test)
{
    ito::DataObject dTempTest1;
    
    this->mat2_2d.copyTo(dTempTest1);

    dTempTest1 *= 11;

    this->validTagSpace(this->mat2_2d, dTempTest1);
    this->validAxisTags(this->mat2_2d, dTempTest1);

    ito::DataObject dTempTest2 = this->mat2_2d * 10;
    this->validTagSpace(this->mat2_2d, dTempTest2);
    this->validAxisTags(this->mat2_2d, dTempTest2);

    if(this->mat2_2d.getType() == ito::tFloat32 || this->mat2_2d.getType() == ito::tFloat64)
    {
        dTempTest1 = ito::DataObject();
        dTempTest1.ones(this->mat2_2d.getDims(), this->mat2_2d.getSize(), this->mat2_2d.getType());
        this->mat2_2d = 2;
        ito::DataObject dTempTest3 = this->mat2_2d * dTempTest1;
        this->validTagSpace(this->mat2_2d, dTempTest3);
        this->validAxisTags(this->mat2_2d, dTempTest3);
        this->invalid(dTempTest1, dTempTest3);

    }
    else
    {
        ito::DataObject dTempTest3;
        dTempTest1 = ito::DataObject();
        dTempTest1.ones(this->mat2_2d.getDims(), this->mat2_2d.getSize(), this->mat2_2d.getType());
        this->mat2_2d = 2;
        EXPECT_ANY_THROW(dTempTest3 = this->mat2_2d * dTempTest1);
        
    }
    
    dTempTest2 = ito::DataObject();
    dTempTest1 = ito::DataObject();
    dTempTest1.ones(this->mat1_2d.getDims(), this->mat1_2d.getSize(), this->mat1_2d.getType());

    dTempTest2 = this->mat1_2d.mul(dTempTest1);
    this->validTagSpace(this->mat1_2d, dTempTest2);
    this->validAxisTags(this->mat1_2d, dTempTest2);
    this->invalid(dTempTest1, dTempTest2);

    dTempTest2 = dTempTest1.mul(this->mat1_2d);
    this->invalid(this->mat1_2d, dTempTest2);
    
};

//tagSpace_div_Test
/*!
    This test checks functionality of "divFunc" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_div_Test)
{
    ito::DataObject dTempTest1;
    ito::DataObject dTempTest2;
    ito::DataObject dTempTest3;

    dTempTest1 = ito::DataObject();
    dTempTest1.ones(this->mat1_2d.getDims(), this->mat1_2d.getSize(), this->mat1_2d.getType());

    this->mat1_2d.copyTo(dTempTest2);
    dTempTest2 = 1;

    dTempTest3 = dTempTest2.mul(dTempTest1);
    this->validTagSpace(dTempTest2, dTempTest3);
    this->validAxisTags(dTempTest2, dTempTest3);
    this->invalid(dTempTest1, dTempTest3);

    dTempTest3 = dTempTest1.mul(dTempTest2);
    this->invalid(dTempTest2, dTempTest3);

};

//tagSpace_comp_Test
/*!
    This test checks functionality of "elementwise comparision" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_comp_Test)
{
    ito::DataObject dTempTest1;

    dTempTest1.ones(this->mat1_2d.getDims(), this->mat1_2d.getSize(), this->mat1_2d.getType());

    ito::DataObject dTempTest3 = this->mat1_2d == dTempTest1;
    this->validTagSpace(this->mat1_2d, dTempTest3);
    this->validAxisTags(this->mat1_2d, dTempTest3);
    this->invalid(dTempTest1, dTempTest3);

    dTempTest3 = this->mat1_2d < dTempTest1;
    this->validTagSpace(this->mat1_2d, dTempTest3);
    this->validAxisTags(this->mat1_2d, dTempTest3);
    this->invalid(dTempTest1, dTempTest3);

    dTempTest3 = this->mat1_2d > dTempTest1;
    this->validTagSpace(this->mat1_2d, dTempTest3);
    this->validAxisTags(this->mat1_2d, dTempTest3);
    this->invalid(dTempTest1, dTempTest3);

    dTempTest3 = this->mat1_2d >= dTempTest1;
    this->validTagSpace(this->mat1_2d, dTempTest3);
    this->validAxisTags(this->mat1_2d, dTempTest3);
    this->invalid(dTempTest1, dTempTest3);

    dTempTest3 = this->mat1_2d <= dTempTest1;
    this->validTagSpace(this->mat1_2d, dTempTest3);
    this->validAxisTags(this->mat1_2d, dTempTest3);
    this->invalid(dTempTest1, dTempTest3);

    dTempTest3 = this->mat1_2d != dTempTest1;
    this->validTagSpace(this->mat1_2d, dTempTest3);
    this->validAxisTags(this->mat1_2d, dTempTest3);
    this->invalid(dTempTest1, dTempTest3);

};

//tagSpace_shift_Test
/*!
    This test checks functionality of "subScalar" and "sub" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_shift_Test)
{
    ito::DataObject dTempTest1;

    dTempTest1.ones(this->mat1_2d.getDims(), this->mat1_2d.getSize(), this->mat1_2d.getType());
    this->mat1_2d.copyAxisTagsTo(dTempTest1);
    this->mat1_2d.copyTagMapTo(dTempTest1);

    if(this->mat1_2d.getType() == ito::tInt8   ||
       this->mat1_2d.getType() == ito::tUInt8  ||
       this->mat1_2d.getType() == ito::tInt16  ||
       this->mat1_2d.getType() == ito::tUInt16 ||
       this->mat1_2d.getType() == ito::tInt32  )
    {
        dTempTest1 <<= 1;
        this->validTagSpace(this->mat1_2d, dTempTest1);
        this->validAxisTags(this->mat1_2d, dTempTest1);


        dTempTest1.ones(this->mat1_2d.getDims(), this->mat1_2d.getSize(), this->mat1_2d.getType());
        this->mat1_2d.copyAxisTagsTo(dTempTest1);
        this->mat1_2d.copyTagMapTo(dTempTest1);

        dTempTest1 >>= 1;
        this->validTagSpace(this->mat1_2d, dTempTest1);
        this->validAxisTags(this->mat1_2d, dTempTest1);

        dTempTest1 = ito::DataObject();
        dTempTest1 = this->mat1_2d >> 1;
        this->validTagSpace(this->mat1_2d, dTempTest1);
        this->validAxisTags(this->mat1_2d, dTempTest1);

        dTempTest1 = ito::DataObject();
        dTempTest1 = this->mat1_2d << 1;
        this->validTagSpace(this->mat1_2d, dTempTest1);
        this->validAxisTags(this->mat1_2d, dTempTest1);
    }
    else
    {
        EXPECT_ANY_THROW(dTempTest1 <<= 1;);
        EXPECT_ANY_THROW(dTempTest1 >>= 1;);
        EXPECT_ANY_THROW(dTempTest1 = this->mat1_2d >> 1;);
        EXPECT_ANY_THROW(dTempTest1 = this->mat1_2d << 1;);
    }
};

//tagSpace_bitwiseCompare_Test
/*!
    This test checks functionality of "subScalar" and "sub" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_bitwiseCompare_Test)
{
    ito::DataObject dTempTest1;
    dTempTest1.ones(this->mat1_2d.getDims(), this->mat1_2d.getSize(), this->mat1_2d.getType());

    ito::DataObject dTempTest2;
    dTempTest2.ones(this->mat1_2d.getDims(), this->mat1_2d.getSize(), this->mat1_2d.getType());

    this->mat1_2d.copyAxisTagsTo(dTempTest1);
    this->mat1_2d.copyTagMapTo(dTempTest1);

    if(this->mat1_2d.getType() == ito::tInt8   ||
       this->mat1_2d.getType() == ito::tUInt8  ||
       this->mat1_2d.getType() == ito::tInt16  ||
       this->mat1_2d.getType() == ito::tUInt16 ||
       this->mat1_2d.getType() == ito::tInt32  )
    {
        dTempTest1 &= dTempTest2;
        this->validTagSpace(this->mat1_2d, dTempTest1);
        this->validAxisTags(this->mat1_2d, dTempTest1);
        this->invalid(dTempTest1, dTempTest2);

        dTempTest1.ones(this->mat1_2d.getDims(), this->mat1_2d.getSize(), this->mat1_2d.getType());
        this->mat1_2d.copyAxisTagsTo(dTempTest1);
        this->mat1_2d.copyTagMapTo(dTempTest1);
        dTempTest1 |= dTempTest2;
        this->validTagSpace(this->mat1_2d, dTempTest1);
        this->validAxisTags(this->mat1_2d, dTempTest1);
        this->invalid(dTempTest1, dTempTest2);

        dTempTest1.ones(this->mat1_2d.getDims(), this->mat1_2d.getSize(), this->mat1_2d.getType());
        this->mat1_2d.copyAxisTagsTo(dTempTest1);
        this->mat1_2d.copyTagMapTo(dTempTest1);
        dTempTest1 ^= dTempTest2;
        this->validTagSpace(this->mat1_2d, dTempTest1);
        this->validAxisTags(this->mat1_2d, dTempTest1);
        this->invalid(dTempTest1, dTempTest2);

        dTempTest1 = ito::DataObject();
        dTempTest1 = this->mat1_2d & dTempTest2;
        this->validTagSpace(this->mat1_2d, dTempTest1);
        this->validAxisTags(this->mat1_2d, dTempTest1);
        this->invalid(dTempTest1, dTempTest2);

        dTempTest1 = ito::DataObject();
        dTempTest1 = this->mat1_2d | dTempTest2;
        this->validTagSpace(this->mat1_2d, dTempTest1);
        this->validAxisTags(this->mat1_2d, dTempTest1);
        this->invalid(dTempTest1, dTempTest2);

        dTempTest1 = ito::DataObject();
        dTempTest1 = this->mat1_2d ^ dTempTest2;
        this->validTagSpace(this->mat1_2d, dTempTest1);
        this->validAxisTags(this->mat1_2d, dTempTest1);
        this->invalid(dTempTest1, dTempTest2);
    }
    else
    {
        EXPECT_ANY_THROW(dTempTest1 &= dTempTest2;);
        EXPECT_ANY_THROW(dTempTest1 |= dTempTest2;);
        EXPECT_ANY_THROW(dTempTest1 ^= dTempTest2;);
        EXPECT_ANY_THROW(dTempTest1 = this->mat1_2d & dTempTest2;);
        EXPECT_ANY_THROW(dTempTest1 = this->mat1_2d | dTempTest2;);
        EXPECT_ANY_THROW(dTempTest1 = this->mat1_2d ^ dTempTest2;);
    }
};

//tagSpace_conj_Test
/*!
    This test checks functionality of "conj" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_conj_Test)
{
    ito::DataObject dTempTest1;
    this->mat1_2d.copyTo(dTempTest1);
    if(this->mat1_2d.getType() == ito::tComplex64 || this->mat1_2d.getType() == ito::tComplex128)
    {
        dTempTest1.conj();
        this->validTagSpace(this->mat1_2d, dTempTest1);
        this->validAxisTags(this->mat1_2d, dTempTest1);

    }
    else
    {
        EXPECT_ANY_THROW(dTempTest1.conj());
    }  
};

//tagSpace_adj_Test
/*!
    This test checks functionality of "adj" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_adj_Test)
{
    ito::DataObject dTempTest1;
    ito::DataObject dTempTest2;
    this->mat1_2d.copyTo(dTempTest1);
    if(this->mat1_2d.getType() == ito::tComplex64 || this->mat1_2d.getType() == ito::tComplex128)
    {
        dTempTest2 = dTempTest1.adj();
        this->validTagSpace(this->mat1_2d, dTempTest2);
        this->validAxisTags(this->mat1_2d, dTempTest2, true);

    }
    else
    {
        EXPECT_ANY_THROW(dTempTest1.adj());
    }  
};

//tagSpace_adj_Test
/*!
    This test checks functionality of "trans" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_trans_Test)
{
    ito::DataObject dTempTest1;
    ito::DataObject dTempTest2;
    this->mat1_2d.copyTo(dTempTest1);

    dTempTest2 = dTempTest1.trans();
    this->validTagSpace(this->mat1_2d, dTempTest2);
    this->validAxisTags(this->mat1_2d, dTempTest2, true);


};

//tagSpace_squeeze_Test
/*!
    This test checks functionality of "squeeze" if a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_squeeze_Test)
{
    ito::DataObject dTempTest1;
    ito::DataObject dTempTest2;
    this->mat2_3d.copyTo(dTempTest1);

    dTempTest2 = dTempTest1.squeeze();
    this->validTagSpace(this->mat2_3d, dTempTest2);
    this->validAxisTags(this->mat2_3d, dTempTest2);

};

//tagSpace_toGray_Test
/*!
    This test checks functionality of "toGray" if a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_toGray_Test)
{
    ito::DataObject dTempTest1(this->mat1_2d.getDims(), this->mat1_2d.getSize(), ito::tRGBA32);
    
    this->mat1_2d.copyTagMapTo(dTempTest1);
    this->mat1_2d.copyAxisTagsTo(dTempTest1);

    if(this->mat1_2d.getType() == ito::tComplex64 || this->mat1_2d.getType() == ito::tComplex128 || this->mat1_2d.getType() == ito::tRGBA32)
    {
        EXPECT_ANY_THROW(dTempTest1.toGray(this->mat1_2d.getType()););
    }
    else
    {
        ito::DataObject dTempTest2 = dTempTest1.toGray(this->mat1_2d.getType());
        this->validTagSpace(this->mat1_2d, dTempTest2);
        this->validAxisTags(this->mat1_2d, dTempTest2);
    }


};

//tagSpace_makeContinous_Test
/*!
    This test checks functionality of "makeContinous_Test" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_makeContinous_Test)
{
    ito::DataObject dTempTest;

    this->mat1_3d.copyTo(dTempTest);

    EXPECT_EQ(this->mat1_3d.getContinuous(), false);

    ito::makeContinuous(dTempTest);

    this->validTagSpace(this->mat1_3d, dTempTest);
    this->validAxisTags(this->mat1_3d, dTempTest);

};

//tagSpace_real_Test
/*!
    This test checks functionality of "real" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_real_Test)
{
    ito::DataObject dTempTest2;
    ito::DataObject dTempTest1;

    this->mat1_2d.copyTo(dTempTest1);
    if(this->mat1_2d.getType() == ito::tComplex64 || this->mat1_2d.getType() == ito::tComplex128)
    {
        dTempTest2 = ito::real(dTempTest1);
        this->validTagSpace(this->mat1_2d, dTempTest2);
        this->validAxisTags(this->mat1_2d, dTempTest2);

    }
    else
    {
        EXPECT_ANY_THROW(ito::real(dTempTest1););
    }     
};

//tagSpace_imag_Test
/*!
    This test checks functionality of "imag" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_imag_Test)
{
    ito::DataObject dTempTest2;
    ito::DataObject dTempTest1;

    this->mat1_2d.copyTo(dTempTest1);
    if(this->mat1_2d.getType() == ito::tComplex64 || this->mat1_2d.getType() == ito::tComplex128)
    {
        dTempTest2 = ito::imag(dTempTest1);
        this->validTagSpace(this->mat1_2d, dTempTest2);
        this->validAxisTags(this->mat1_2d, dTempTest2);

    }
    else
    {
        EXPECT_ANY_THROW(ito::imag(dTempTest1););
    }     
};

//tagSpace_arg_Test
/*!
    This test checks functionality of "arg" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_arg_Test)
{
    ito::DataObject dTempTest2;
    ito::DataObject dTempTest1;

    this->mat1_2d.copyTo(dTempTest1);
    if(this->mat1_2d.getType() == ito::tComplex64 || this->mat1_2d.getType() == ito::tComplex128)
    {
        dTempTest2 = ito::arg(dTempTest1);
        this->validTagSpace(this->mat1_2d, dTempTest2);
        this->validAxisTags(this->mat1_2d, dTempTest2);

    }
    else
    {
        EXPECT_ANY_THROW(ito::imag(dTempTest1););
    }     
};

//tagSpace_abs_Test
/*!
    This test checks functionality of "arg" of a DataObject is compatible with tagSpace copy.
*/
TYPED_TEST(dataObjectTagSpace_operator_Test, tagSpace_abs_Test)
{
    ito::DataObject dTempTest2;
    ito::DataObject dTempTest1;

    this->mat1_2d.copyTo(dTempTest1);
    if(this->mat1_2d.getType() == ito::tRGBA32)
    {
        EXPECT_ANY_THROW(ito::abs(dTempTest1););
    }
    else
    {
        dTempTest2 = ito::abs(dTempTest1);
        this->validTagSpace(this->mat1_2d, dTempTest2);
        this->validAxisTags(this->mat1_2d, dTempTest2);
    }


};
