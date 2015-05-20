#ifndef DATAOBJECTTAGTYPE_TEST_H
#define DATAOBJECTTAGTYPE_TEST_H

#include <iostream>

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

template <typename _Tp> class dataObjectTagType_Test : public ::testing::Test 
    { 
public:

    virtual void SetUp(void)
    {
    ito::DataObjectTagType EmptyObj1 =  ito::DataObjectTagType();        /*!< Declaring empty DataObjectTagType. */
    ito::DataObjectTagType StrObj1 = ito::DataObjectTagType();            /*!< Declaring DataObjectTagType Object with string. */
    ito::DataObjectTagType Obj1 =  ito::DataObjectTagType();                                        /*!< Declaring DataObjectTagType with temporary double value.*/
    ito::DataObjectTagType Obj2 =  ito::DataObjectTagType();        /*!< Declaring DataObjectTagType with std::numeric_limits<double>::quiet_NaN().*/
    ito::DataObjectTagType Obj3 =  ito::DataObjectTagType();    /*!< Declaring DataObjectTagType with std::numeric_limits<double>::signaling_NaN().*/
    ito::DataObjectTagType Obj4 =  ito::DataObjectTagType();            /*!< Declaring DataObjectTagType with std::numeric_limits<double>::infinity().*/

    //!< Creating Copied Objects using DataObjectTagType Copy Constructor.
    ito::DataObjectTagType CpyStrObj1 = ito::DataObjectTagType();        /*!< Copying the String Object StrObj1 into CpyStrObj1 using Copy Constructor */
    ito::DataObjectTagType CpyEmptyObj1 = ito::DataObjectTagType();    /*!< Copying the Empty Object EmptyObj1 into CpyEmptyObj1 using Copy Constructor */
    ito::DataObjectTagType CpyObj1 = ito::DataObjectTagType();                /*!< Copying the Object Obj1 into CpyObj1 using Copy Constructor */
    ito::DataObjectTagType CpyObj2 = ito::DataObjectTagType();                /*!< Copying the Object Obj2 into CpyObj2 using Copy Constructor */
    ito::DataObjectTagType CpyObj3 = ito::DataObjectTagType();                /*!< Copying the Object Obj3 into CpyObj3 using Copy Constructor */
    ito::DataObjectTagType CpyObj4 = ito::DataObjectTagType();                /*!< Copying the Object Obj4 into CpyObj4 using Copy Constructor */

    //!< Creating Objects for copy with assignment operator "=".
    ito::DataObjectTagType AsgnStrObj1 = ito::DataObjectTagType();
    ito::DataObjectTagType AsgnEmptyObj1 = ito::DataObjectTagType(); 
    ito::DataObjectTagType AsgnObj1 = ito::DataObjectTagType(); 
    ito::DataObjectTagType AsgnObj2 = ito::DataObjectTagType(); 
    ito::DataObjectTagType AsgnObj3 = ito::DataObjectTagType(); 
    ito::DataObjectTagType AsgnObj4 = ito::DataObjectTagType();
        
    };

    virtual void TearDown(void){};
    typedef _Tp valueType;
    ito::DataObjectTagType EmptyObj1;
    ito::DataObjectTagType StrObj1;
    ito::DataObjectTagType Obj1;
    ito::DataObjectTagType Obj2;
    ito::DataObjectTagType Obj3;
    ito::DataObjectTagType Obj4;

    ito::DataObjectTagType CpyEmptyObj1;
    ito::DataObjectTagType CpyStrObj1;
    ito::DataObjectTagType CpyObj1;
    ito::DataObjectTagType CpyObj2;
    ito::DataObjectTagType CpyObj3;
    ito::DataObjectTagType CpyObj4;

    ito::DataObjectTagType AsgnEmptyObj1;
    ito::DataObjectTagType AsgnStrObj1;
    ito::DataObjectTagType AsgnObj1;
    ito::DataObjectTagType AsgnObj2;
    ito::DataObjectTagType AsgnObj3;
    ito::DataObjectTagType AsgnObj4;
};

TYPED_TEST_CASE(dataObjectTagType_Test,ItomDataTypes);

//setTag_Test
/*!
    This test checks functionality of "getType()" function of empty DataObjects
*/
TYPED_TEST(dataObjectTagType_Test,getType_Test)
{
    double tempVal3=24;
    double val1= std::numeric_limits<double>::signaling_NaN();
    double val2= std::numeric_limits<double>::quiet_NaN();
    double val3= std::numeric_limits<double>::infinity();
    double valDouble;
    std::ostringstream s4;
    std::string str4;
    StrObj1 = ito::DataObjectTagType("Test String");            /*!< Declaring DataObjectTagType Object with string. */

    Obj1 =  ito::DataObjectTagType(tempVal3);                                        /*!< Declaring DataObjectTagType with temporary double value.*/
    Obj2 =  ito::DataObjectTagType(std::numeric_limits<double>::quiet_NaN());        /*!< Declaring DataObjectTagType with std::numeric_limits<double>::quiet_NaN().*/
    Obj3 =  ito::DataObjectTagType(std::numeric_limits<double>::signaling_NaN());    /*!< Declaring DataObjectTagType with std::numeric_limits<double>::signaling_NaN().*/
    Obj4 =  ito::DataObjectTagType(std::numeric_limits<double>::infinity());            /*!< Declaring DataObjectTagType with std::numeric_limits<double>::infinity().*/

    //!< Creating Copied Objects using DataObjectTagType Copy Constructor.
    CpyStrObj1 = ito::DataObjectTagType(StrObj1);        /*!< Copying the String Object StrObj1 into CpyStrObj1 using Copy Constructor */
    CpyEmptyObj1 = ito::DataObjectTagType(EmptyObj1);    /*!< Copying the Empty Object EmptyObj1 into CpyEmptyObj1 using Copy Constructor */
    CpyObj1 = ito::DataObjectTagType(Obj1);                /*!< Copying the Object Obj1 into CpyObj1 using Copy Constructor */
    CpyObj2 = ito::DataObjectTagType(Obj2);                /*!< Copying the Object Obj2 into CpyObj2 using Copy Constructor */
    CpyObj3 = ito::DataObjectTagType(Obj3);                /*!< Copying the Object Obj3 into CpyObj3 using Copy Constructor */
    CpyObj4 = ito::DataObjectTagType(Obj4);                /*!< Copying the Object Obj4 into CpyObj4 using Copy Constructor */

    //!< Copying the Obj1,Obj2,Obj3 and Obj4 using assigned Operator "="
    AsgnStrObj1 = StrObj1;
    AsgnEmptyObj1 = EmptyObj1;
    AsgnObj1 = Obj1;
    AsgnObj2 = Obj2;
    AsgnObj3 = Obj3;
    AsgnObj4 = Obj4;

    //!< Test for getType() function
    EXPECT_EQ( ito::DataObjectTagType::typeString,StrObj1.getType() );            /*!< Test for getType() function with String Object of DataObjectTagType. */
    EXPECT_EQ( ito::DataObjectTagType::typeInvalid,EmptyObj1.getType() );        /*!< Test for getType() function with empty object of DataObjectTagType. */
    EXPECT_EQ( ito::DataObjectTagType::typeDouble,Obj1.getType() );            
    EXPECT_EQ( ito::DataObjectTagType::typeDouble,Obj2.getType() );
    EXPECT_EQ( ito::DataObjectTagType::typeDouble,Obj3.getType() );
    EXPECT_EQ( ito::DataObjectTagType::typeDouble,Obj4.getType() );

    //!< Test for getType() function with copied objects.
    EXPECT_EQ( ito::DataObjectTagType::typeString,CpyStrObj1.getType() );            /*!< Test for getType() function with Copied String Object using Copy Constructor of DataObjectTagType. */
    EXPECT_EQ( ito::DataObjectTagType::typeInvalid,CpyEmptyObj1.getType() );        /*!< Test for getType() function with empty object of DataObjectTagType. */
    EXPECT_EQ( ito::DataObjectTagType::typeDouble,CpyObj1.getType() );
    EXPECT_EQ( ito::DataObjectTagType::typeDouble,CpyObj2.getType() );
    EXPECT_EQ( ito::DataObjectTagType::typeDouble,CpyObj3.getType() );
    EXPECT_EQ( ito::DataObjectTagType::typeDouble,CpyObj4.getType() );

    //!< Test for getType() function with copied objects using Assignment Operator "=".
    EXPECT_EQ( ito::DataObjectTagType::typeString,AsgnStrObj1.getType() );            /*!< Test for getType() function with Copied String Object using Assignment Operator "=" of DataObjectTagType. */
    EXPECT_EQ( ito::DataObjectTagType::typeInvalid,AsgnEmptyObj1.getType() );        /*!< Test for getType() function with empty object of DataObjectTagType. */
    EXPECT_EQ(ito::DataObjectTagType::typeDouble,AsgnObj1.getType());
    EXPECT_EQ(ito::DataObjectTagType::typeDouble,AsgnObj2.getType());
    EXPECT_EQ(ito::DataObjectTagType::typeDouble,AsgnObj3.getType());
    EXPECT_EQ(ito::DataObjectTagType::typeDouble,AsgnObj4.getType());

}

//setTag_Test
/*!
    This test checks functionality of "isValid()" function of empty DataObjects
*/
TYPED_TEST(dataObjectTagType_Test,isValid_Test)
{
    double tempVal3=24;
    double val1= std::numeric_limits<double>::signaling_NaN();
    double val2= std::numeric_limits<double>::quiet_NaN();
    double val3= std::numeric_limits<double>::infinity();
    double valDouble;
    std::ostringstream s4;
    std::string str4;
    StrObj1 = ito::DataObjectTagType("Test String");            /*!< Declaring DataObjectTagType Object with string. */

    Obj1 =  ito::DataObjectTagType(tempVal3);                                        /*!< Declaring DataObjectTagType with temporary double value.*/
    Obj2 =  ito::DataObjectTagType(std::numeric_limits<double>::quiet_NaN());        /*!< Declaring DataObjectTagType with std::numeric_limits<double>::quiet_NaN().*/
    Obj3 =  ito::DataObjectTagType(std::numeric_limits<double>::signaling_NaN());    /*!< Declaring DataObjectTagType with std::numeric_limits<double>::signaling_NaN().*/
    Obj4 =  ito::DataObjectTagType(std::numeric_limits<double>::infinity());            /*!< Declaring DataObjectTagType with std::numeric_limits<double>::infinity().*/

    //!< Creating Copied Objects using DataObjectTagType Copy Constructor.
    CpyStrObj1 = ito::DataObjectTagType(StrObj1);        /*!< Copying the String Object StrObj1 into CpyStrObj1 using Copy Constructor */
    CpyEmptyObj1 = ito::DataObjectTagType(EmptyObj1);    /*!< Copying the Empty Object EmptyObj1 into CpyEmptyObj1 using Copy Constructor */
    CpyObj1 = ito::DataObjectTagType(Obj1);                /*!< Copying the Object Obj1 into CpyObj1 using Copy Constructor */
    CpyObj2 = ito::DataObjectTagType(Obj2);                /*!< Copying the Object Obj2 into CpyObj2 using Copy Constructor */
    CpyObj3 = ito::DataObjectTagType(Obj3);                /*!< Copying the Object Obj3 into CpyObj3 using Copy Constructor */
    CpyObj4 = ito::DataObjectTagType(Obj4);                /*!< Copying the Object Obj4 into CpyObj4 using Copy Constructor */

    //!< Copying the Obj1,Obj2,Obj3 and Obj4 using assigned Operator "="
    AsgnStrObj1 = StrObj1;
    AsgnEmptyObj1 = EmptyObj1;
    AsgnObj1 = Obj1;
    AsgnObj2 = Obj2;
    AsgnObj3 = Obj3;
    AsgnObj4 = Obj4;

    //!< Test for isValid() function
    EXPECT_TRUE( StrObj1.isValid() );                /*!< Test for isValid() function with empty objects of DataObjectTagType.  */
    EXPECT_FALSE( EmptyObj1.isValid() );            /*!< Test for isValid() function with empty objects of DataObjectTagType.  */
    EXPECT_TRUE( Obj1.isValid() );                    /*!< Test for isValid() function with double value object of DataObjectTagType. */
    EXPECT_TRUE( Obj2.isValid() );                    /*!< Test for isValid() function with quiet_NaN value object of DataObjectTagType. */
    EXPECT_TRUE( Obj3.isValid() );                    /*!< Test for isValid() function with signaling_NaN value object of DataObjectTagType. */
    EXPECT_TRUE( Obj4.isValid() );                    /*!< Test for isValid() function with infinity value object of DataObjectTagType. */

    //!< Test for isValid() function with Copied Objects with Copy Constructor of DataObjectTagType.
    EXPECT_TRUE( CpyStrObj1.isValid() );            /*!< Test for isValid() function with empty objects of DataObjectTagType. */
    EXPECT_FALSE( CpyEmptyObj1.isValid() );            /*!< Test for isValid() function with empty objects of DataObjectTagType.  */
    EXPECT_TRUE( CpyObj1.isValid() );                /*!< Test for isValid() function with double value object of DataObjectTagType. */
    EXPECT_TRUE( CpyObj2.isValid() );                /*!< Test for isValid() function with quiet_NaN value object of DataObjectTagType. */
    EXPECT_TRUE( CpyObj3.isValid() );                /*!< Test for isValid() function with signaling_NaN value object of DataObjectTagType. */
    EXPECT_TRUE( CpyObj4.isValid() );                /*!< Test for isValid() function with infinity value object of DataObjectTagType. */

    //!< Test for isValid() function with Copied Objects with Assignment Operator "=" of DataObjectTagType.
    EXPECT_TRUE( AsgnStrObj1.isValid() );            /*!< Test for isValid() function with empty objects of DataObjectTagType. */
    EXPECT_FALSE( AsgnEmptyObj1.isValid() );        /*!< Test for isValid() function with empty objects of DataObjectTagType.  */
    EXPECT_TRUE( AsgnObj1.isValid() );                /*!< Test for isValid() function with double value object of DataObjectTagType. */
    EXPECT_TRUE( AsgnObj2.isValid() );                /*!< Test for isValid() function with quiet_NaN value object of DataObjectTagType. */
    EXPECT_TRUE( AsgnObj3.isValid() );                /*!< Test for isValid() function with signaling_NaN value object of DataObjectTagType. */
    EXPECT_TRUE( AsgnObj4.isValid() );                /*!< Test for isValid() function with infinity value object of DataObjectTagType. */
}

//setTag_Test
/*!
    This test checks functionality of "getVal_ToDouble()" function of empty DataObjects
*/
TYPED_TEST(dataObjectTagType_Test,getVal_ToDouble_Test)
{
    double tempVal3=24;
    double val1= std::numeric_limits<double>::signaling_NaN();
    double val2= std::numeric_limits<double>::quiet_NaN();
    double val3= std::numeric_limits<double>::infinity();
    double valDouble;
    std::ostringstream s4;
    std::string str4;
    StrObj1 = ito::DataObjectTagType("Test String");            /*!< Declaring DataObjectTagType Object with string. */

    Obj1 =  ito::DataObjectTagType(tempVal3);                                        /*!< Declaring DataObjectTagType with temporary double value.*/
    Obj2 =  ito::DataObjectTagType(std::numeric_limits<double>::quiet_NaN());        /*!< Declaring DataObjectTagType with std::numeric_limits<double>::quiet_NaN().*/
    Obj3 =  ito::DataObjectTagType(std::numeric_limits<double>::signaling_NaN());    /*!< Declaring DataObjectTagType with std::numeric_limits<double>::signaling_NaN().*/
    Obj4 =  ito::DataObjectTagType(std::numeric_limits<double>::infinity());            /*!< Declaring DataObjectTagType with std::numeric_limits<double>::infinity().*/

    //!< Creating Copied Objects using DataObjectTagType Copy Constructor.
    CpyStrObj1 = ito::DataObjectTagType(StrObj1);        /*!< Copying the String Object StrObj1 into CpyStrObj1 using Copy Constructor */
    CpyEmptyObj1 = ito::DataObjectTagType(EmptyObj1);    /*!< Copying the Empty Object EmptyObj1 into CpyEmptyObj1 using Copy Constructor */
    CpyObj1 = ito::DataObjectTagType(Obj1);                /*!< Copying the Object Obj1 into CpyObj1 using Copy Constructor */
    CpyObj2 = ito::DataObjectTagType(Obj2);                /*!< Copying the Object Obj2 into CpyObj2 using Copy Constructor */
    CpyObj3 = ito::DataObjectTagType(Obj3);                /*!< Copying the Object Obj3 into CpyObj3 using Copy Constructor */
    CpyObj4 = ito::DataObjectTagType(Obj4);                /*!< Copying the Object Obj4 into CpyObj4 using Copy Constructor */

    //!< Copying the Obj1,Obj2,Obj3 and Obj4 using assigned Operator "="
    AsgnStrObj1 = StrObj1;
    AsgnEmptyObj1 = EmptyObj1;
    AsgnObj1 = Obj1;
    AsgnObj2 = Obj2;
    AsgnObj3 = Obj3;
    AsgnObj4 = Obj4;


    //!< Test for getVal_ToDouble() function.
    valDouble=StrObj1.getVal_ToDouble();
    EXPECT_EQ( 0,std::memcmp(&val1,&valDouble,sizeof(val1) ) );                /*!< valDouble contains the output of getVal_ToDouble() function for String Object of DataObjectTagType. */
    EXPECT_DOUBLE_EQ( tempVal3,Obj1.getVal_ToDouble() );                    /*!< Test for getVal_ToDouble function for Object with Double value of DataObjectTagType. */
    valDouble=EmptyObj1.getVal_ToDouble();                                    /*!< val2 contains the output of getVal_ToDouble() function for Empty Object of DataObjectTagType. */
    EXPECT_EQ( 0,std::memcmp(&val1,&valDouble,sizeof(val1) ) );                /*!< Test for getVal_ToDouble() function with empty objects of DataObjectTagType. */
    valDouble=Obj2.getVal_ToDouble();
    EXPECT_EQ( 0,std::memcmp(&val2,&valDouble,sizeof(val2) ) );
    valDouble=Obj3.getVal_ToDouble();
    EXPECT_EQ( 0,std::memcmp(&val1,&valDouble,sizeof(val1) ) );
    valDouble=Obj4.getVal_ToDouble();
    EXPECT_EQ( 0,std::memcmp(&val3,&valDouble,sizeof(val3) ) );

    //!< The above Test for getVal_ToDouble() function could also be accomplished by following code. 
        /*!<    
            EXPECT_TRUE( cvIsNaN( Obj2.getVal_ToDouble() ) );
            EXPECT_TRUE( cvIsNaN( Obj3.getVal_ToDouble() ) );
            EXPECT_TRUE( cvIsInf( Obj4.getVal_ToDouble() ) ); 
        */
    
    //!< Test for getVal_ToDouble() function with copied objects.
    valDouble=CpyStrObj1.getVal_ToDouble();
    EXPECT_EQ( 0,std::memcmp(&val1,&valDouble,sizeof(val1) ) );                /*!< valDouble contains the output of getVal_ToDouble() function for Copied String Object with Copy Constructor of DataObjectTagType. */
    EXPECT_DOUBLE_EQ( tempVal3,CpyObj1.getVal_ToDouble() );                    /*!< Test for getVal_ToDouble function for Object with Double value of DataObjectTagType. */
    valDouble=CpyEmptyObj1.getVal_ToDouble();                                /*!< val2 contains the output of getVal_ToDouble() function for Empty Object of DataObjectTagType. */
    EXPECT_EQ( 0,std::memcmp(&val1,&valDouble,sizeof(val1) ) );                /*!< Test for getVal_ToDouble() function with empty objects of DataObjectTagType. */
    valDouble=CpyObj2.getVal_ToDouble();
    EXPECT_EQ( 0,std::memcmp(&val2,&valDouble,sizeof(val2) ) );
    valDouble=CpyObj3.getVal_ToDouble();
    EXPECT_EQ( 0,std::memcmp(&val1,&valDouble,sizeof(val1) ) );
    valDouble=CpyObj4.getVal_ToDouble();
    EXPECT_EQ( 0,std::memcmp(&val3,&valDouble,sizeof(val3) ) );

    //!< Test for getVal_ToDouble() function with copied objects using Assignment Operator "=".
    valDouble=AsgnStrObj1.getVal_ToDouble();
    EXPECT_EQ( 0,std::memcmp(&val1,&valDouble,sizeof(val1) ) );                /*!< val2 contains the output of getVal_ToDouble() function for String Object of DataObjectTagType. */
    EXPECT_DOUBLE_EQ( tempVal3,AsgnObj1.getVal_ToDouble() );                /*!< Test for getVal_ToDouble function for Object with Double value of DataObjectTagType. */
    valDouble=AsgnEmptyObj1.getVal_ToDouble();                                /*!< val2 contains the output of getVal_ToDouble() function for Empty Object of DataObjectTagType. */
    EXPECT_EQ( 0,std::memcmp(&val1,&valDouble,sizeof(val1) ) );                /*!< Test for getVal_ToDouble() function with empty objects of DataObjectTagType. */
    valDouble=AsgnObj2.getVal_ToDouble();
    EXPECT_EQ( 0,std::memcmp(&val2,&valDouble,sizeof(val2) ) );
    valDouble=AsgnObj3.getVal_ToDouble();
    EXPECT_EQ( 0,std::memcmp(&val1,&valDouble,sizeof(val1) ) );
    valDouble=AsgnObj4.getVal_ToDouble();
    EXPECT_EQ( 0,std::memcmp(&val3,&valDouble,sizeof(val3) ) );
}

//setTag_Test
/*!
    This test checks functionality of "getVal_ToString()" function of empty DataObjects
*/
TYPED_TEST(dataObjectTagType_Test,getVal_ToString_Test)
{
    double tempVal3=24;
    double val1= std::numeric_limits<double>::signaling_NaN();
    double val2= std::numeric_limits<double>::quiet_NaN();
    double val3= std::numeric_limits<double>::infinity();
    double valDouble;
    std::ostringstream s4;
    std::string str4;
    StrObj1 = ito::DataObjectTagType("Test String");            /*!< Declaring DataObjectTagType Object with string. */

    Obj1 =  ito::DataObjectTagType(tempVal3);                                        /*!< Declaring DataObjectTagType with temporary double value.*/
    Obj2 =  ito::DataObjectTagType(std::numeric_limits<double>::quiet_NaN());        /*!< Declaring DataObjectTagType with std::numeric_limits<double>::quiet_NaN().*/
    Obj3 =  ito::DataObjectTagType(std::numeric_limits<double>::signaling_NaN());    /*!< Declaring DataObjectTagType with std::numeric_limits<double>::signaling_NaN().*/
    Obj4 =  ito::DataObjectTagType(std::numeric_limits<double>::infinity());            /*!< Declaring DataObjectTagType with std::numeric_limits<double>::infinity().*/

    //!< Creating Copied Objects using DataObjectTagType Copy Constructor.
    CpyStrObj1 = ito::DataObjectTagType(StrObj1);        /*!< Copying the String Object StrObj1 into CpyStrObj1 using Copy Constructor */
    CpyEmptyObj1 = ito::DataObjectTagType(EmptyObj1);    /*!< Copying the Empty Object EmptyObj1 into CpyEmptyObj1 using Copy Constructor */
    CpyObj1 = ito::DataObjectTagType(Obj1);                /*!< Copying the Object Obj1 into CpyObj1 using Copy Constructor */
    CpyObj2 = ito::DataObjectTagType(Obj2);                /*!< Copying the Object Obj2 into CpyObj2 using Copy Constructor */
    CpyObj3 = ito::DataObjectTagType(Obj3);                /*!< Copying the Object Obj3 into CpyObj3 using Copy Constructor */
    CpyObj4 = ito::DataObjectTagType(Obj4);                /*!< Copying the Object Obj4 into CpyObj4 using Copy Constructor */

    //!< Copying the Obj1,Obj2,Obj3 and Obj4 using assigned Operator "="
    AsgnStrObj1 = StrObj1;
    AsgnEmptyObj1 = EmptyObj1;
    AsgnObj1 = Obj1;
    AsgnObj2 = Obj2;
    AsgnObj3 = Obj3;
    AsgnObj4 = Obj4;

    //!< Test for getVal_ToString() function. 
    EXPECT_EQ("Test String",StrObj1.getVal_ToString() );                    //!< Test for getVal_ToString() with String Objects of DataObjectTagType
    EXPECT_EQ("",EmptyObj1.getVal_ToString());                                //!< Test for getVal_ToString() with empty objects of DataObjectTagType
    s4 << tempVal3;        //!< Creating StreamString representation of Double value for further comparision.
    str4=s4.str();      //!< Converting StreamString into String for further comparision.
    EXPECT_EQ(str4,Obj1.getVal_ToString());
    EXPECT_EQ("1.#QNAN",Obj2.getVal_ToString());
    EXPECT_EQ("1.#QNAN",Obj3.getVal_ToString());
    EXPECT_EQ("Inf",Obj4.getVal_ToString());

    //!< Test for getVal_ToString() function with copied objects.
    EXPECT_EQ("Test String",CpyStrObj1.getVal_ToString() );                    //!< Test for getVal_ToString() with Copied String Objects with Copy Constructor of DataObjectTagType
    EXPECT_EQ(str4,CpyObj1.getVal_ToString());
    EXPECT_EQ("1.#QNAN",CpyObj2.getVal_ToString());
    EXPECT_EQ("1.#QNAN",CpyObj3.getVal_ToString());
    EXPECT_EQ("Inf",CpyObj4.getVal_ToString());

    //!< Test for getVal_ToString() function with copied objects using Assignment Operator "=".
    EXPECT_EQ("Test String",AsgnStrObj1.getVal_ToString() );                    //!< Test for getVal_ToString() with Copied String Objects with Assignment Operator "=" of DataObjectTagType
    EXPECT_EQ(str4,AsgnObj1.getVal_ToString());
    EXPECT_EQ("1.#QNAN",AsgnObj2.getVal_ToString());
    EXPECT_EQ("1.#QNAN",AsgnObj3.getVal_ToString());
    EXPECT_EQ("Inf",AsgnObj4.getVal_ToString());
}

#endif