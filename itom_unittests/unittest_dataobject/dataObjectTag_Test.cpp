
#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv/cv.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"

/*! \class dataObjectTag_Test
	\brief Test for DataObjectTag class and functions for all itom data types

	This test class checks functionality of different fuctions on data objects Tags.
*/

template <typename _Tp> class dataObjectTag_Test : public ::testing::Test 
	{ 
public:

	virtual void SetUp(void)
	{
		//Creating 1,2 and 3 dimension DataObjects for this Perticular Test class.

		mat1_1d = ito::DataObject(3,ito::getDataType( (const _Tp *) NULL ) );
		mat2_1d = ito::DataObject(3,ito::getDataType( (const _Tp *) NULL ));
		mat3_1d = ito::DataObject(3,ito::getDataType( (const _Tp *) NULL ));

		mat1_2d = ito::DataObject(3,4,ito::getDataType( (const _Tp *) NULL ));
		mat2_2d = ito::DataObject(3,4,ito::getDataType( (const _Tp *) NULL ));
		mat3_2d = ito::DataObject(3,4,ito::getDataType( (const _Tp *) NULL ));

		mat1_3d = ito::DataObject(3,3,3,ito::getDataType( (const _Tp *) NULL ));
		mat2_3d = ito::DataObject(3,3,3,ito::getDataType( (const _Tp *) NULL ));
		mat3_3d = ito::DataObject(3,3,3,ito::getDataType( (const _Tp *) NULL ));

		};
	virtual void TearDown(void){};
	typedef _Tp valueType;
	ito::DataObject mat1_1d;
	ito::DataObject mat2_1d;
	ito::DataObject mat3_1d;

	ito::DataObject mat1_2d;
	ito::DataObject mat2_2d;
	ito::DataObject mat3_2d;

	ito::DataObject mat1_3d;
	ito::DataObject mat2_3d;
	ito::DataObject mat3_3d;

	ito::DataObject matEmptyTest;
};

TYPED_TEST_CASE(dataObjectTag_Test,ItomDataStandardTypes);
//setTag_Test
/*!
	This test checks functionality of "setTag" and "getTag" functions of DataObject for 1 dimensional matrices
*/
TYPED_TEST(dataObjectTag_Test,set_getTag_Test)
{
	bool vop1;
	bool vop2;
	mat1_1d.setTag("creator1", "Shishir");
    ito::DataObjectTagType Tag1_ = mat1_1d.getTag("creator1",vop1);
    std::string Tag1 = Tag1_.getVal_ToString();

    ito::DataObjectTagType Tag2_ = mat1_1d.getTag("creator2",vop2);
    std::string Tag2 = Tag2_.getVal_ToString();
    
	EXPECT_EQ(Tag1,"Shishir");
	EXPECT_TRUE(vop1);
	EXPECT_FALSE(vop2);

	//Test for empty matrix
	matEmptyTest.setTag("creator1", "Shishir");
	Tag1 = matEmptyTest.getTag("creator1",vop1).getVal_ToString();
	EXPECT_EQ(Tag1,"");
	EXPECT_FALSE(vop1);							//shows that no Tag can be created for empty matrix.

};

//set_getValueUnit_Test
/*!
	This test checks functionality of "setValueUnit" and "getValueUnit" functions of DataObject.
*/
TYPED_TEST(dataObjectTag_Test,set_getValueUnit_Test)
{
	std::string TempUnit;
	mat1_1d.setValueUnit("mm");  //Setting Value Unit to mat1_1d
	TempUnit = mat1_1d.getValueUnit();   //Retrieving the Value Unit of mat1_1d
	EXPECT_EQ(TempUnit,"mm");  //check if retrieved value is same as original

	//Test for empty matrix
	matEmptyTest.setValueUnit("mm");
	TempUnit = matEmptyTest.getValueUnit();
	EXPECT_EQ(TempUnit,"");					//shows empty matrix can not be assigned any Unit Value.
};

//set_getValueDescription_Test
/*!
	This test checks functionality of "setValueDescription" and "getValueDescription" functions of a DataObject. 
*/
TYPED_TEST(dataObjectTag_Test,set_getValueDescription_Test)
{
	std::string str1 = "It is a test value.";
	std::string chkStr;
	mat1_1d.setValueDescription(str1);
	chkStr = mat1_1d.getValueDescription();
	EXPECT_EQ(chkStr,str1);  //check if the retrieved Value Description of mat1_1d is same as the original one.

	//Test for empty matrix
	int chk;
	chk = matEmptyTest.setValueDescription(str1);
	chkStr = matEmptyTest.getValueDescription();
	EXPECT_EQ(chkStr,"");
	EXPECT_EQ(chk,cv::saturate_cast<int>(1));
};

//set_getAxisOffset_Test1d
/*!
	This test checks functionality of "setAxisOffset" and "getAxisOffset" functions of DataObject for 1 dimensional matrices.
*/
TYPED_TEST(dataObjectTag_Test,set_getAxisOffset_Test1d)
{
	mat1_1d.setAxisOffset(0,-2);
	double AxisOffset =mat1_1d.getAxisOffset(0);
	EXPECT_EQ(AxisOffset,cv::saturate_cast<double>(-2));  //checks if the retrieved Axis Offset value of mat1_1d is same as original one.
};

//set_getAxisOffset_Test2d
/*!
	This test checks functionality of "setAxisOffset" and "getAxisOffset" functions of DataObject for 2 dimensional matrices.
*/
TYPED_TEST(dataObjectTag_Test,set_getAxisOffset_Test2d)
{
	mat1_2d.setAxisOffset(0,-2);
	mat1_2d.setAxisOffset(1,3);
	double AxisOffset1 =mat1_2d.getAxisOffset(0);
	double AxisOffset2 =mat1_2d.getAxisOffset(1);
	EXPECT_EQ(AxisOffset1,cv::saturate_cast<double>(-2));  //checks if the retrieved 1st Axis Offset value of mat1_2d matches with  original one.
	EXPECT_EQ(AxisOffset2,cv::saturate_cast<double>(3));   //checks if the retrieved 2nd Axis Offset value of mat1_2d matches with  original one.
};

//set_getAxisOffset_Test3d
/*!
	This test checks functionality of "setAxisOffset" and "getAxisOffset" functions of DataObject for 3 dimensional matrices.
*/
TYPED_TEST(dataObjectTag_Test,set_getAxisOffset_Test3d)
{
	mat1_3d.setAxisOffset(0,5);
	mat1_3d.setAxisOffset(1,-0.5);
	mat1_3d.setAxisOffset(2,3.24);
	double AxisOffset1 =mat1_3d.getAxisOffset(0);
	double AxisOffset2 =mat1_3d.getAxisOffset(1);
	double AxisOffset3 =mat1_3d.getAxisOffset(2);
	EXPECT_EQ(AxisOffset1,cv::saturate_cast<double>(5));	//checks if the retrieved 1st Axis Offset value of mat1_3d matches with  original one.
	EXPECT_EQ(AxisOffset2,cv::saturate_cast<double>(-0.5));	//checks if the retrieved 2nd Axis Offset value of mat1_3d matches with  original one.
	EXPECT_EQ(AxisOffset3,cv::saturate_cast<double>(3.24));	//checks if the retrieved 3rd Axis Offset value of mat1_3d matches with  original one.
};

//set_getAxisScales_Test1d
/*!
	This test checks functionality of "setAxisScales" and "getAxisScales" functions of DataObject for 1 dimensional matrices.
*/
TYPED_TEST(dataObjectTag_Test,set_getAxisScales_Test1d)
{
	mat1_1d.setAxisScale(0,2.5);						//Scale the axis0 with "setAxisScale" function
	double AxisScale =mat1_1d.getAxisScale(0);			//Get the axis0 scalling with "getAxisScales" function
	EXPECT_EQ(AxisScale,cv::saturate_cast<double>(2.5)); //checks if the retrieved Axis scale is same as the one set by "setAxisScale" function
};

//set_getAxisScales_Test2d
/*!
	This test checks functionality of "setAxisScales" and "getAxisScales" functions of DataObject for 2 dimensional matrices.
*/
TYPED_TEST(dataObjectTag_Test,set_getAxisScales_Test2d)
{
	mat1_2d.setAxisScale(0,1);
	mat1_2d.setAxisScale(1,10);
	double AxisScale1 =mat1_2d.getAxisScale(0);
	double AxisScale2 =mat1_2d.getAxisScale(1);
	EXPECT_EQ(AxisScale1,cv::saturate_cast<double>(1));	 //checks if the retrieved Axis0 scalling value is same as the one set by "setAxisScale" function 
	EXPECT_EQ(AxisScale2,cv::saturate_cast<double>(10)); //checks if the retrieved Axis1 scalling value is same as the one set by "setAxisScale" function
};

//set_getAxisScales_Test3d
/*!
	This test checks functionality of "setAxisScales" and "getAxisScales" functions of DataObject for 3 dimensional matrices.
*/
TYPED_TEST(dataObjectTag_Test,set_getAxisScales_Test3d)
{
	mat1_3d.setAxisScale(0,5);
	mat1_3d.setAxisScale(1,-0.5);
	mat1_3d.setAxisScale(2,3.24);
	double AxisScale1 =mat1_3d.getAxisScale(0);
	double AxisScale2 =mat1_3d.getAxisScale(1);
	double AxisScale3 =mat1_3d.getAxisScale(2);
	EXPECT_EQ(AxisScale1,cv::saturate_cast<double>(5));		//checks if the retrieved Axis0 scalling is same as the one set by "setAxisScale" function
	EXPECT_EQ(AxisScale2,cv::saturate_cast<double>(-0.5));	//checks if the retrieved Axis1 scalling is same as the one set by "setAxisScale" function
	EXPECT_EQ(AxisScale3,cv::saturate_cast<double>(3.24));	//checks if the retrieved Axis2 scalling is same as the one set by "setAxisScale" function
};

//getValueOffset_Test
/*!
	This test checks functionality of "getValueOffset" function of a DataObject.
*/
TYPED_TEST(dataObjectTag_Test,getValueOffset_Test)
{
	double ValueOffset;
	ValueOffset=mat1_1d.getValueOffset();
	EXPECT_EQ(ValueOffset,cv::saturate_cast<double>(0));	//checks if the Value Offset of mat1_1d matrix is 0 (default value)
};

//getValueScale_Test
/*!
	This test checks functionality of "getValueScale" function of a DataObject.
*/
TYPED_TEST(dataObjectTag_Test,getValueScale_Test)
{
	double ValueScale;
	ValueScale=mat1_1d.getValueScale();
	EXPECT_EQ(ValueScale,cv::saturate_cast<double>(1));		//checks if the scale value of the mat1_1d matrix is 1 (default value)
};

//set_getAxisUnit_Test1d
/*!
	This test checks functionality of "setAxisUnit" and "getAxisUnit" functions of DataObject for 1 dimensional matrices.
*/
TYPED_TEST(dataObjectTag_Test,set_getAxisUnit_Test1d)
{
	mat1_1d.setAxisUnit(0,"cm");
	bool vop;
	std::string AxisUnit =mat1_1d.getAxisUnit(0,vop);
	EXPECT_EQ(AxisUnit,"cm");							//checks if the retrieved Axis Unit by "getAxisUnit" function is same as the one assigned by "setAxisUnit" function.
	EXPECT_TRUE(vop);									//checks if the operations performed with these functions were valid or not.

	//Test for empty matrix
	matEmptyTest.setAxisUnit(0,"cm");
	EXPECT_ANY_THROW(AxisUnit =matEmptyTest.getAxisUnit(0,vop));
};

//set_getAxisUnit_Test2d
/*!
	This test checks functionality of "setAxisUnit" and "getAxisUnit" functions of DataObject for 2 dimensional matrices.
*/
TYPED_TEST(dataObjectTag_Test,set_getAxisUnit_Test2d)
{
	mat1_2d.setAxisUnit(0,"µm");
	mat1_2d.setAxisUnit(1,"cm");
	bool vop1;
	bool vop2;
	std::string AxisUnit1 =mat1_2d.getAxisUnit(0,vop1);
	std::string AxisUnit2 =mat1_2d.getAxisUnit(1,vop2);
	EXPECT_EQ(AxisUnit1,"µm");							//checks if the retrieved Axis0 Unit by "getAxisUnit" function is same as the one assigned by "setAxisUnit" function.
	EXPECT_EQ(AxisUnit2,"cm");							//checks if the retrieved Axis1 Unit by "getAxisUnit" function is same as the one assigned by "setAxisUnit" function.
	EXPECT_TRUE(vop1);									//checks if the operations performed for Axis0 with these functions were valid or not.
	EXPECT_TRUE(vop2);									//checks if the operations performed for Axis1 with these functions were valid or not.
};

//set_getAxisUnit_Test3d
/*!
	This test checks functionality of "setAxisUnit" and "getAxisUnit" functions of DataObject for 3 dimensional matrices.
*/
TYPED_TEST(dataObjectTag_Test,set_getAxisUnit_Test3d)
{
	EXPECT_GT(mat1_3d.setAxisUnit(-1,"nm"),0);
	mat1_3d.setAxisUnit(1,"mm");
	mat1_3d.setAxisUnit(2,"");
	bool vop1=true;
	bool vop2=false;
	bool vop3=false;
	std::string AxisUnit1;
	EXPECT_ANY_THROW(AxisUnit1 =mat1_3d.getAxisUnit(-1,vop1));
	std::string AxisUnit2 =mat1_3d.getAxisUnit(1,vop2);
	std::string AxisUnit3 =mat1_3d.getAxisUnit(2,vop3);
	EXPECT_NE(AxisUnit1,"nm");							//checks if the retrieved Axis0 Unit by "getAxisUnit" function is same as the one assigned by "setAxisUnit" function.
	EXPECT_EQ(AxisUnit2,"mm");							//checks if the retrieved Axis1 Unit by "getAxisUnit" function is same as the one assigned by "setAxisUnit" function.
	EXPECT_EQ(AxisUnit3,"");							//checks if the retrieved Axis2 Unit by "getAxisUnit" function is same as the one assigned by "setAxisUnit" function.		
	EXPECT_FALSE(vop1);									//checks if the operations performed for Axis0 with these functions were valid or not.
	EXPECT_TRUE(vop2);									//checks if the operations performed for Axis1 with these functions were valid or not.
	EXPECT_TRUE(vop3);									//checks if the operations performed for Axis2 with these functions were valid or not.
};

//set_getAxisDescription_Test1d
/*!
	This test checks functionality of "setAxisDescription" and "getAxisDescription" functions of DataObject for 1 dimensional matrices.
*/
TYPED_TEST(dataObjectTag_Test,set_getAxisDescription_Test1d)
{
	mat1_1d.setAxisDescription(0,"X-Axis ---> Height");			//set any relevant axis description to Axis0 of mat1_1d
	bool vop;
	std::string AxisDescrip =mat1_1d.getAxisDescription(0,vop); //retrive the given axis description by "getAxisDescription" function for Axis0
	EXPECT_EQ(AxisDescrip,"X-Axis ---> Height");				//checks if the retrieved axis description is same as the one given 
	EXPECT_TRUE(vop);											//checks if the above operation is valid for this matrix

	//Test for empty matrix
	int chk = matEmptyTest.setAxisDescription(0,"X-Axis ---> Height");
	EXPECT_ANY_THROW(	AxisDescrip =matEmptyTest.getAxisDescription(0,vop));
//	EXPECT_EQ(AxisDescrip,"");					//for empty martix, there can not be any Axis description.
	EXPECT_EQ(chk,cv::saturate_cast<int>(1));	
//	EXPECT_FALSE(vop);							//for empty matrix the above operation is false as expected
};

//set_getAxisDescription_Test2d
/*!
	This test checks functionality of "setAxisDescription" and "getAxisDescription" functions of DataObject for 2 dimensional matrices.
*/
TYPED_TEST(dataObjectTag_Test,set_getAxisDescription_Test2d)
{
	mat1_2d.setAxisDescription(0,"X-Axis ---> Height");
	mat1_2d.setAxisDescription(1,"Y-Axis ---> Time");
	bool vop1;
	bool vop2;
	std::string AxisDescrip1 =mat1_2d.getAxisDescription(0,vop1); 
	std::string AxisDescrip2 =mat1_2d.getAxisDescription(1,vop2);
	EXPECT_EQ(AxisDescrip1,"X-Axis ---> Height");					//checks if the retrieved axis description for Axis0 is same as the one given 
	EXPECT_EQ(AxisDescrip2,"Y-Axis ---> Time");						//checks if the retrieved axis description for Axis1 is same as the one given 
	EXPECT_TRUE(vop1);												//checks if the above operation is valid for Axis0 of this matrix
	EXPECT_TRUE(vop2);												//checks if the above operation is valid for Axis1 of this matrix
};

//set_getAxisDescription_Test3d
/*!
	This test checks functionality of "setAxisDescription" and "getAxisDescription" functions of DataObject for 3 dimensional matrices.
*/
TYPED_TEST(dataObjectTag_Test,set_getAxisDescription_Test3d)
{
	EXPECT_GT(mat1_3d.setAxisDescription(-1,"X-Axis ---> Height"),0);				//Here the Axis index value is given negative which is wrong.
	EXPECT_EQ(mat1_3d.setAxisDescription(1,"Y-Axis ---> Time"),0);
	EXPECT_EQ(mat1_3d.setAxisDescription(2,""),0);
	std::string AxisDescrip1;
	bool vop1 = true;
	bool vop2 = false;
	bool vop3 = false;
	EXPECT_ANY_THROW( AxisDescrip1 = mat1_3d.getAxisDescription(-1, vop1) ) ; //vop1 is undefined
	std::string AxisDescrip2 =mat1_3d.getAxisDescription(1, vop2);
	std::string AxisDescrip3 =mat1_3d.getAxisDescription(2, vop3);
	EXPECT_NE(AxisDescrip1,"X-Axis ---> Height");					//Here Axis Description can not be as expected as Axis Index was negative.
	EXPECT_EQ(AxisDescrip2,"Y-Axis ---> Time");						//checks if the retrieved axis description for Axis1 is same as the one given 
	EXPECT_EQ(AxisDescrip3,"");										//checks if the retrieved axis description for Axis2 is same as the one given 
	EXPECT_FALSE(vop1);
	EXPECT_TRUE(vop2);
	EXPECT_TRUE(vop3);
};

//getTagbyIndex_Test
/*!
	This test checks functionality of "getTagbyIndex" of a DataObject.
*/
TYPED_TEST(dataObjectTag_Test,getTagbyIndex_Test)
{	
	mat1_1d.setTag("creator1","Shishir1");					//set the key1 and its correspondent value1 for mat1_1d
	mat1_1d.setTag("creator2","Shishir2");					//set the key2 and its correspondent value2 for mat1_1d
	std::string key1, key2, value1,value2;
    ito::DataObjectTagType value1_, value2_;
	bool test1 = mat1_1d.getTagByIndex(0,key1,value1_);		//get the key1 and value1 of mat1_1d by index value using "getTagByIndex" function
	bool test2 = mat1_1d.getTagByIndex(1,key2,value2_);		//get the key2 and value2 of mat1_1d by index value using "getTagByIndex" function
	EXPECT_EQ(key1,"creator1");								//checks if the retrieved tag's key1 has same contents.
	EXPECT_EQ(key2,"creator2");								//checks if the retrieved tag's key2 has same contents.
    value1 = value1_.getVal_ToString();
    value2 = value2_.getVal_ToString();
	EXPECT_EQ(value1,"Shishir1");							//checks if the retrieved key1's value has same contents.
	EXPECT_EQ(value2,"Shishir2");							//checks if the retrieved key2's value has same contents.
	EXPECT_TRUE(test1);										//checks if the above operations were successful or not for matrix index0.
	EXPECT_TRUE(test2);										//checks if the above operations were successful or not for matrix index1.

	//Test for empty matrix
	matEmptyTest.setTag("creator1","Shishir1");
	test1 = matEmptyTest.getTagByIndex(0,key1,value1_);
    value1 = value1_.getVal_ToString();
	EXPECT_EQ(key1,"");
	EXPECT_EQ(value1,"");
	EXPECT_FALSE(test1);								//for empty matrix the test fails.
};

//getTagKey_Test
/*!
	This test checks functionality of "getTagKey" of a DataObject.
*/
TYPED_TEST(dataObjectTag_Test,getTagKey_Test)
{	
	mat1_1d.setTag("creator1","Shishir3");
	mat1_1d.setTag("creator2","Shishir4");
	mat1_1d.setTag("creator3","Shishir5");
	bool vop1;
	bool vop2;
	bool vop3;
	std::string key1 = mat1_1d.getTagKey(0,vop1);
	std::string key2 = mat1_1d.getTagKey(1,vop2);
	std::string key3 = mat1_1d.getTagKey(2,vop3);
	EXPECT_EQ(key1,"creator1");						//checks if the key1 is same as the one assigned by "setTag" function.
	EXPECT_EQ(key2,"creator2");						//checks if the key2 is same as the one assigned by "setTag" function.
	EXPECT_EQ(key3,"creator3");						//checks if the key3 is same as the one assigned by "setTag" function.
	EXPECT_TRUE(vop1);								//checks if the above operation is valid for 1st Tag 
	EXPECT_TRUE(vop2);								//checks if the above operation is valid for 2nd Tag 
	EXPECT_TRUE(vop3);								//checks if the above operation is valid for 3rd Tag 

	//Test for empty matrix
	matEmptyTest.setTag("creator1","Shishir3");
	key1 = matEmptyTest.getTagKey(0,vop1);
	EXPECT_EQ(key1,"");								//Tag key and Tag value can not be assigned to an empty matrix.
	EXPECT_FALSE(vop1);								//shows the above operation for empty tag fails.
};

//getTagListSize_Test
/*!
	This test checks functionality of "getTagListSize" of a DataObject.
*/
TYPED_TEST(dataObjectTag_Test,getTagListSize_Test)
{	
	mat1_1d.setTag("creator1","Shishir3");
	mat1_1d.setTag("creator2","Shishir4");
	mat1_1d.setTag("creator3","Shishir5");
	int TaglistSize = mat1_1d.getTagListSize();
	EXPECT_EQ(TaglistSize,cv::saturate_cast<int>(3));	//checks the retrieved size of Tag List is same as expected.
};

//getTagListSize_Test
/*!
	This test checks functionality of "getTagListSize" of a DataObject.
*/
TYPED_TEST(dataObjectTag_Test,delete_existTag_Test)
{	
	mat1_1d.setTag("creator1","Shishir1");
	mat1_1d.setTag("creator2","Shishir2");
	mat1_1d.setTag("creator3","Shishir3");
	bool chk1 = mat1_1d.existTag("creator1");
	bool chk2 = mat1_1d.existTag("creator2");
	bool chk3 = mat1_1d.existTag("creator3");
	bool chk4 = mat1_1d.existTag("creator4");

	bool Delete1 = mat1_1d.deleteTag("creator1");	
	bool Delete2 = mat1_1d.deleteTag("creator2");
	bool Delete3 = mat1_1d.deleteTag("creator3");
	bool Delete4 = mat1_1d.deleteTag("creator4");

	EXPECT_TRUE(chk1);								//value of chk1 should be True as there exists Tag for index0
	EXPECT_TRUE(chk2);								//value of chk2 should be True as there exists Tag for index1
	EXPECT_TRUE(chk3);								//value of chk3 should be True as there exists Tag for index2
	EXPECT_FALSE(chk4);								//value of chk4 should be False as there does not exist Tag for index3
	EXPECT_TRUE(Delete1);							//it shows "deleteTag" function for key1 "creator1" is performed sucessfully.
	EXPECT_TRUE(Delete2);							//it shows "deleteTag" function for key2 "creator2" is performed sucessfully.
	EXPECT_TRUE(Delete3);							//it shows "deleteTag" function for key3 "creator3" is performed sucessfully.
	EXPECT_FALSE(Delete4);							//because there was no existance of key4 "creator4", the function "deleteTag" was unsuccessful and the value of Delete4 should be expected False.

	chk1 = mat1_1d.existTag("creator1");
	chk2 = mat1_1d.existTag("creator2");
	chk3 = mat1_1d.existTag("creator3");
	chk4 = mat1_1d.existTag("creator4");

	//because all Tags has been deleted for mat1_1d now, the check for existTag will show False.
	EXPECT_FALSE(chk1);							
	EXPECT_FALSE(chk2);
	EXPECT_FALSE(chk3);
	EXPECT_FALSE(chk4);

	mat1_1d.setTag("creator1","Shishir1");
	mat1_1d.setTag("creator2","Shishir2");
	mat1_1d.setTag("creator3","Shishir3");

	chk1 = mat1_1d.deleteAllTags();
	EXPECT_TRUE(chk1);						//shows deleteAllTags() function performed successfully.

	//Test for empty matrix
	matEmptyTest.setTag("creator1","Shishir3");		
	chk1 = mat1_1d.existTag("creator1");
	EXPECT_FALSE(chk1);						//shows that no tags can be created for empty matrix.
	Delete1 = mat1_1d.deleteTag("creator1");
	EXPECT_FALSE(Delete1);					//because there was no tag created for empty matrix, value of Delete1 is False as expected.
};
