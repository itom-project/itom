
#include "../../common/sharedStructures.h"

// opencv
#pragma warning(disable : 4996) // C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This
                                // function or variable may be unsafe. Consider using fopen_s instead.

#include "../../DataObject/dataobj.h"
#include "commonChannel.h"
#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"

/*! \class dataObjectTag_Test
    \brief Test for DataObjectTag class and functions for all itom data types

    This test class checks functionality of different fuctions on data objects Tags.
*/

template <typename _Tp> class dataObjectTagType_Test : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
        ito::DataObjectTagType EmptyObj1 = ito::DataObjectTagType(); /*!< Declaring empty DataObjectTagType. */
        ito::DataObjectTagType StrObj1 =
            ito::DataObjectTagType(); /*!< Declaring DataObjectTagType Object with string. */
        ito::DataObjectTagType Obj1 =
            ito::DataObjectTagType(); /*!< Declaring DataObjectTagType with temporary double value.*/
        ito::DataObjectTagType Obj2 =
            ito::DataObjectTagType(); /*!< Declaring DataObjectTagType with std::numeric_limits<double>::quiet_NaN().*/
        ito::DataObjectTagType Obj3 = ito::DataObjectTagType(); /*!< Declaring DataObjectTagType with
                                                                   std::numeric_limits<double>::signaling_NaN().*/
        ito::DataObjectTagType Obj4 =
            ito::DataObjectTagType(); /*!< Declaring DataObjectTagType with std::numeric_limits<double>::infinity().*/

        //!< Creating Copied Objects using DataObjectTagType Copy Constructor.
        ito::DataObjectTagType CpyStrObj1 =
            ito::DataObjectTagType(); /*!< Copying the String Object StrObj1 into CpyStrObj1 using Copy Constructor */
        ito::DataObjectTagType CpyEmptyObj1 = ito::DataObjectTagType(); /*!< Copying the Empty Object EmptyObj1 into
                                                                           CpyEmptyObj1 using Copy Constructor */
        ito::DataObjectTagType CpyObj1 =
            ito::DataObjectTagType(); /*!< Copying the Object Obj1 into CpyObj1 using Copy Constructor */
        ito::DataObjectTagType CpyObj2 =
            ito::DataObjectTagType(); /*!< Copying the Object Obj2 into CpyObj2 using Copy Constructor */
        ito::DataObjectTagType CpyObj3 =
            ito::DataObjectTagType(); /*!< Copying the Object Obj3 into CpyObj3 using Copy Constructor */
        ito::DataObjectTagType CpyObj4 =
            ito::DataObjectTagType(); /*!< Copying the Object Obj4 into CpyObj4 using Copy Constructor */

        //!< Creating Objects for copy with assignment operator "=".
        ito::DataObjectTagType AsgnStrObj1 = ito::DataObjectTagType();
        ito::DataObjectTagType AsgnEmptyObj1 = ito::DataObjectTagType();
        ito::DataObjectTagType AsgnObj1 = ito::DataObjectTagType();
        ito::DataObjectTagType AsgnObj2 = ito::DataObjectTagType();
        ito::DataObjectTagType AsgnObj3 = ito::DataObjectTagType();
        ito::DataObjectTagType AsgnObj4 = ito::DataObjectTagType();
    };

    virtual void TearDown(void){};

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

TYPED_TEST_CASE(dataObjectTagType_Test, ItomDataAllTypes);

// setTag_Test
/*!
    This test checks functionality of "getType()" function of empty DataObjects
*/
TYPED_TEST(dataObjectTagType_Test, getType_Test)
{
    double tempVal3 = 24;
    double val1 = std::numeric_limits<double>::signaling_NaN();
    double val2 = std::numeric_limits<double>::quiet_NaN();
    double val3 = std::numeric_limits<double>::infinity();
    double valDouble;
    std::ostringstream s4;
    std::string str4;
    this->StrObj1 = ito::DataObjectTagType("Test String"); /*!< Declaring DataObjectTagType Object with string. */

    this->Obj1 = ito::DataObjectTagType(tempVal3); /*!< Declaring DataObjectTagType with temporary double value.*/
    this->Obj2 =
        ito::DataObjectTagType(std::numeric_limits<double>::quiet_NaN()); /*!< Declaring DataObjectTagType with
                                                                             std::numeric_limits<double>::quiet_NaN().*/
    this->Obj3 = ito::DataObjectTagType(
        std::numeric_limits<double>::signaling_NaN()); /*!< Declaring DataObjectTagType with
                                                          std::numeric_limits<double>::signaling_NaN().*/
    this->Obj4 =
        ito::DataObjectTagType(std::numeric_limits<double>::infinity()); /*!< Declaring DataObjectTagType with
                                                                            std::numeric_limits<double>::infinity().*/

    //!< Creating Copied Objects using DataObjectTagType Copy Constructor.
    this->CpyStrObj1 = ito::DataObjectTagType(
        this->StrObj1); /*!< Copying the String Object StrObj1 into CpyStrObj1 using Copy Constructor */
    this->CpyEmptyObj1 = ito::DataObjectTagType(
        this->EmptyObj1); /*!< Copying the Empty Object EmptyObj1 into CpyEmptyObj1 using Copy Constructor */
    this->CpyObj1 =
        ito::DataObjectTagType(this->Obj1); /*!< Copying the Object Obj1 into CpyObj1 using Copy Constructor */
    this->CpyObj2 =
        ito::DataObjectTagType(this->Obj2); /*!< Copying the Object Obj2 into CpyObj2 using Copy Constructor */
    this->CpyObj3 =
        ito::DataObjectTagType(this->Obj3); /*!< Copying the Object Obj3 into CpyObj3 using Copy Constructor */
    this->CpyObj4 =
        ito::DataObjectTagType(this->Obj4); /*!< Copying the Object Obj4 into CpyObj4 using Copy Constructor */

    //!< Copying the Obj1,Obj2,Obj3 and Obj4 using assigned Operator "="
    this->AsgnStrObj1 = this->StrObj1;
    this->AsgnEmptyObj1 = this->EmptyObj1;
    this->AsgnObj1 = this->Obj1;
    this->AsgnObj2 = this->Obj2;
    this->AsgnObj3 = this->Obj3;
    this->AsgnObj4 = this->Obj4;

    //!< Test for getType() function
    EXPECT_EQ(ito::DataObjectTagType::typeString,
              this->StrObj1.getType()); /*!< Test for getType() function with String Object of DataObjectTagType. */
    EXPECT_EQ(ito::DataObjectTagType::typeInvalid,
              this->EmptyObj1.getType()); /*!< Test for getType() function with empty object of DataObjectTagType. */
    EXPECT_EQ(ito::DataObjectTagType::typeDouble, this->Obj1.getType());
    EXPECT_EQ(ito::DataObjectTagType::typeDouble, this->Obj2.getType());
    EXPECT_EQ(ito::DataObjectTagType::typeDouble, this->Obj3.getType());
    EXPECT_EQ(ito::DataObjectTagType::typeDouble, this->Obj4.getType());

    //!< Test for getType() function with copied objects.
    EXPECT_EQ(ito::DataObjectTagType::typeString,
              this->CpyStrObj1.getType()); /*!< Test for getType() function with Copied String Object using Copy
                                              Constructor of DataObjectTagType. */
    EXPECT_EQ(ito::DataObjectTagType::typeInvalid,
              this->CpyEmptyObj1.getType()); /*!< Test for getType() function with empty object of DataObjectTagType. */
    EXPECT_EQ(ito::DataObjectTagType::typeDouble, this->CpyObj1.getType());
    EXPECT_EQ(ito::DataObjectTagType::typeDouble, this->CpyObj2.getType());
    EXPECT_EQ(ito::DataObjectTagType::typeDouble, this->CpyObj3.getType());
    EXPECT_EQ(ito::DataObjectTagType::typeDouble, this->CpyObj4.getType());

    //!< Test for getType() function with copied objects using Assignment Operator "=".
    EXPECT_EQ(ito::DataObjectTagType::typeString,
              this->AsgnStrObj1.getType()); /*!< Test for getType() function with Copied String Object using Assignment
                                               Operator "=" of DataObjectTagType. */
    EXPECT_EQ(
        ito::DataObjectTagType::typeInvalid,
        this->AsgnEmptyObj1.getType()); /*!< Test for getType() function with empty object of DataObjectTagType. */
    EXPECT_EQ(ito::DataObjectTagType::typeDouble, this->AsgnObj1.getType());
    EXPECT_EQ(ito::DataObjectTagType::typeDouble, this->AsgnObj2.getType());
    EXPECT_EQ(ito::DataObjectTagType::typeDouble, this->AsgnObj3.getType());
    EXPECT_EQ(ito::DataObjectTagType::typeDouble, this->AsgnObj4.getType());
}

// setTag_Test
/*!
    This test checks functionality of "isValid()" function of empty DataObjects
*/
TYPED_TEST(dataObjectTagType_Test, isValid_Test)
{
    double tempVal3 = 24;
    double val1 = std::numeric_limits<double>::signaling_NaN();
    double val2 = std::numeric_limits<double>::quiet_NaN();
    double val3 = std::numeric_limits<double>::infinity();
    double valDouble;
    std::ostringstream s4;
    std::string str4;
    this->StrObj1 = ito::DataObjectTagType("Test String"); /*!< Declaring DataObjectTagType Object with string. */

    this->Obj1 = ito::DataObjectTagType(tempVal3); /*!< Declaring DataObjectTagType with temporary double value.*/
    this->Obj2 =
        ito::DataObjectTagType(std::numeric_limits<double>::quiet_NaN()); /*!< Declaring DataObjectTagType with
                                                                             std::numeric_limits<double>::quiet_NaN().*/
    this->Obj3 = ito::DataObjectTagType(
        std::numeric_limits<double>::signaling_NaN()); /*!< Declaring DataObjectTagType with
                                                          std::numeric_limits<double>::signaling_NaN().*/
    this->Obj4 =
        ito::DataObjectTagType(std::numeric_limits<double>::infinity()); /*!< Declaring DataObjectTagType with
                                                                            std::numeric_limits<double>::infinity().*/

    //!< Creating Copied Objects using DataObjectTagType Copy Constructor.
    this->CpyStrObj1 = ito::DataObjectTagType(
        this->StrObj1); /*!< Copying the String Object StrObj1 into CpyStrObj1 using Copy Constructor */
    this->CpyEmptyObj1 = ito::DataObjectTagType(
        this->EmptyObj1); /*!< Copying the Empty Object EmptyObj1 into CpyEmptyObj1 using Copy Constructor */
    this->CpyObj1 =
        ito::DataObjectTagType(this->Obj1); /*!< Copying the Object Obj1 into CpyObj1 using Copy Constructor */
    this->CpyObj2 =
        ito::DataObjectTagType(this->Obj2); /*!< Copying the Object Obj2 into CpyObj2 using Copy Constructor */
    this->CpyObj3 =
        ito::DataObjectTagType(this->Obj3); /*!< Copying the Object Obj3 into CpyObj3 using Copy Constructor */
    this->CpyObj4 =
        ito::DataObjectTagType(this->Obj4); /*!< Copying the Object Obj4 into CpyObj4 using Copy Constructor */

    //!< Copying the Obj1,Obj2,Obj3 and Obj4 using assigned Operator "="
    this->AsgnStrObj1 = this->StrObj1;
    this->AsgnEmptyObj1 = this->EmptyObj1;
    this->AsgnObj1 = this->Obj1;
    this->AsgnObj2 = this->Obj2;
    this->AsgnObj3 = this->Obj3;
    this->AsgnObj4 = this->Obj4;

    //!< Test for isValid() function
    EXPECT_TRUE(this->StrObj1.isValid()); /*!< Test for isValid() function with empty objects of DataObjectTagType.  */
    EXPECT_FALSE(
        this->EmptyObj1.isValid()); /*!< Test for isValid() function with empty objects of DataObjectTagType.  */
    EXPECT_TRUE(
        this->Obj1.isValid()); /*!< Test for isValid() function with double value object of DataObjectTagType. */
    EXPECT_TRUE(
        this->Obj2.isValid()); /*!< Test for isValid() function with quiet_NaN value object of DataObjectTagType. */
    EXPECT_TRUE(
        this->Obj3.isValid()); /*!< Test for isValid() function with signaling_NaN value object of DataObjectTagType. */
    EXPECT_TRUE(
        this->Obj4.isValid()); /*!< Test for isValid() function with infinity value object of DataObjectTagType. */

    //!< Test for isValid() function with Copied Objects with Copy Constructor of DataObjectTagType.
    EXPECT_TRUE(
        this->CpyStrObj1.isValid()); /*!< Test for isValid() function with empty objects of DataObjectTagType. */
    EXPECT_FALSE(
        this->CpyEmptyObj1.isValid()); /*!< Test for isValid() function with empty objects of DataObjectTagType.  */
    EXPECT_TRUE(
        this->CpyObj1.isValid()); /*!< Test for isValid() function with double value object of DataObjectTagType. */
    EXPECT_TRUE(
        this->CpyObj2.isValid()); /*!< Test for isValid() function with quiet_NaN value object of DataObjectTagType. */
    EXPECT_TRUE(
        this->CpyObj3
            .isValid()); /*!< Test for isValid() function with signaling_NaN value object of DataObjectTagType. */
    EXPECT_TRUE(
        this->CpyObj4.isValid()); /*!< Test for isValid() function with infinity value object of DataObjectTagType. */

    //!< Test for isValid() function with Copied Objects with Assignment Operator "=" of DataObjectTagType.
    EXPECT_TRUE(
        this->AsgnStrObj1.isValid()); /*!< Test for isValid() function with empty objects of DataObjectTagType. */
    EXPECT_FALSE(
        this->AsgnEmptyObj1.isValid()); /*!< Test for isValid() function with empty objects of DataObjectTagType.  */
    EXPECT_TRUE(
        this->AsgnObj1.isValid()); /*!< Test for isValid() function with double value object of DataObjectTagType. */
    EXPECT_TRUE(
        this->AsgnObj2.isValid()); /*!< Test for isValid() function with quiet_NaN value object of DataObjectTagType. */
    EXPECT_TRUE(
        this->AsgnObj3
            .isValid()); /*!< Test for isValid() function with signaling_NaN value object of DataObjectTagType. */
    EXPECT_TRUE(
        this->AsgnObj4.isValid()); /*!< Test for isValid() function with infinity value object of DataObjectTagType. */
}

// setTag_Test
/*!
    This test checks functionality of "getVal_ToDouble()" function of empty DataObjects
*/
TYPED_TEST(dataObjectTagType_Test, getVal_ToDouble_Test)
{
    double tempVal3 = 24;
    double sigNaN = std::numeric_limits<double>::signaling_NaN();
    double quietNaN = std::numeric_limits<double>::quiet_NaN();
    double inf = std::numeric_limits<double>::infinity();
    double valDouble;
    std::ostringstream s4;
    std::string str4;
    this->StrObj1 = ito::DataObjectTagType("Test String"); /*!< Declaring DataObjectTagType Object with string. */

    this->Obj1 = ito::DataObjectTagType(tempVal3); /*!< Declaring DataObjectTagType with temporary double value.*/
    this->Obj2 =
        ito::DataObjectTagType(std::numeric_limits<double>::quiet_NaN()); /*!< Declaring DataObjectTagType with
                                                                             std::numeric_limits<double>::quiet_NaN().*/
    this->Obj3 = ito::DataObjectTagType(
        std::numeric_limits<double>::signaling_NaN()); /*!< Declaring DataObjectTagType with
                                                          std::numeric_limits<double>::signaling_NaN().*/
    this->Obj4 =
        ito::DataObjectTagType(std::numeric_limits<double>::infinity()); /*!< Declaring DataObjectTagType with
                                                                            std::numeric_limits<double>::infinity().*/

    //!< Creating Copied Objects using DataObjectTagType Copy Constructor.
    this->CpyStrObj1 = ito::DataObjectTagType(
        this->StrObj1); /*!< Copying the String Object StrObj1 into CpyStrObj1 using Copy Constructor */
    this->CpyEmptyObj1 = ito::DataObjectTagType(
        this->EmptyObj1); /*!< Copying the Empty Object EmptyObj1 into CpyEmptyObj1 using Copy Constructor */
    this->CpyObj1 =
        ito::DataObjectTagType(this->Obj1); /*!< Copying the Object Obj1 into CpyObj1 using Copy Constructor */
    this->CpyObj2 =
        ito::DataObjectTagType(this->Obj2); /*!< Copying the Object Obj2 into CpyObj2 using Copy Constructor */
    this->CpyObj3 =
        ito::DataObjectTagType(this->Obj3); /*!< Copying the Object Obj3 into CpyObj3 using Copy Constructor */
    this->CpyObj4 =
        ito::DataObjectTagType(this->Obj4); /*!< Copying the Object Obj4 into CpyObj4 using Copy Constructor */

    //!< Copying the Obj1,Obj2,Obj3 and Obj4 using assigned Operator "="
    this->AsgnStrObj1 = this->StrObj1;
    this->AsgnEmptyObj1 = this->EmptyObj1;
    this->AsgnObj1 = this->Obj1;
    this->AsgnObj2 = this->Obj2;
    this->AsgnObj3 = this->Obj3;
    this->AsgnObj4 = this->Obj4;

    //!< Test for getVal_ToDouble() function.
    valDouble = this->StrObj1.getVal_ToDouble();
    EXPECT_EQ(0, std::memcmp(&quietNaN, &valDouble,
                             sizeof(quietNaN))); /*!< valDouble contains the output of getVal_ToDouble() function for
                                                    String Object of DataObjectTagType. */
    EXPECT_DOUBLE_EQ(tempVal3, this->Obj1.getVal_ToDouble()); /*!< Test for getVal_ToDouble function for Object with
                                                                 Double value of DataObjectTagType. */
    valDouble = this->EmptyObj1.getVal_ToDouble(); /*!< val2 contains the output of getVal_ToDouble() function for Empty
                                                      Object of DataObjectTagType. */
    EXPECT_EQ(
        0, std::memcmp(
               &quietNaN, &valDouble,
               sizeof(quietNaN))); /*!< Test for getVal_ToDouble() function with empty objects of DataObjectTagType. */
    valDouble = this->Obj2.getVal_ToDouble();
    EXPECT_EQ(0, std::memcmp(&quietNaN, &valDouble, sizeof(quietNaN)));
    valDouble = this->Obj3.getVal_ToDouble();
    EXPECT_EQ(0, std::memcmp(&sigNaN, &valDouble, sizeof(sigNaN)));
    valDouble = this->Obj4.getVal_ToDouble();
    EXPECT_EQ(0, std::memcmp(&inf, &valDouble, sizeof(inf)));

    //!< The above Test for getVal_ToDouble() function could also be accomplished by following code.
    /*!<
        EXPECT_TRUE( cvIsNaN( Obj2.getVal_ToDouble() ) );
        EXPECT_TRUE( cvIsNaN( Obj3.getVal_ToDouble() ) );
        EXPECT_TRUE( cvIsInf( Obj4.getVal_ToDouble() ) );
    */

    //!< Test for getVal_ToDouble() function with copied objects.
    valDouble = this->CpyStrObj1.getVal_ToDouble();
    EXPECT_EQ(0, std::memcmp(&quietNaN, &valDouble,
                             sizeof(quietNaN))); /*!< valDouble contains the output of getVal_ToDouble() function for
                                                    Copied String Object with Copy Constructor of DataObjectTagType. */
    EXPECT_DOUBLE_EQ(tempVal3, this->CpyObj1.getVal_ToDouble()); /*!< Test for getVal_ToDouble function for Object with
                                                                    Double value of DataObjectTagType. */
    valDouble = this->CpyEmptyObj1.getVal_ToDouble(); /*!< val2 contains the output of getVal_ToDouble() function for
                                                         Empty Object of DataObjectTagType. */
    EXPECT_EQ(
        0, std::memcmp(
               &quietNaN, &valDouble,
               sizeof(quietNaN))); /*!< Test for getVal_ToDouble() function with empty objects of DataObjectTagType. */
    valDouble = this->CpyObj2.getVal_ToDouble();
    EXPECT_EQ(0, std::memcmp(&quietNaN, &valDouble, sizeof(quietNaN)));
    valDouble = this->CpyObj3.getVal_ToDouble();
    EXPECT_EQ(0, std::memcmp(&sigNaN, &valDouble, sizeof(sigNaN)));
    valDouble = this->CpyObj4.getVal_ToDouble();
    EXPECT_EQ(0, std::memcmp(&inf, &valDouble, sizeof(inf)));

    //!< Test for getVal_ToDouble() function with copied objects using Assignment Operator "=".
    valDouble = this->AsgnStrObj1.getVal_ToDouble();
    EXPECT_EQ(
        0, std::memcmp(&quietNaN, &valDouble, sizeof(quietNaN))); /*!< val2 contains the output of getVal_ToDouble()
                                                                     function for String Object of DataObjectTagType. */
    EXPECT_DOUBLE_EQ(tempVal3, this->AsgnObj1.getVal_ToDouble()); /*!< Test for getVal_ToDouble function for Object with
                                                                     Double value of DataObjectTagType. */
    valDouble = this->AsgnEmptyObj1.getVal_ToDouble(); /*!< val2 contains the output of getVal_ToDouble() function for
                                                          Empty Object of DataObjectTagType. */
    EXPECT_EQ(
        0, std::memcmp(
               &quietNaN, &valDouble,
               sizeof(quietNaN))); /*!< Test for getVal_ToDouble() function with empty objects of DataObjectTagType. */
    valDouble = this->AsgnObj2.getVal_ToDouble();
    EXPECT_EQ(0, std::memcmp(&quietNaN, &valDouble, sizeof(quietNaN)));
    valDouble = this->AsgnObj3.getVal_ToDouble();
    EXPECT_EQ(0, std::memcmp(&sigNaN, &valDouble, sizeof(sigNaN)));
    valDouble = this->AsgnObj4.getVal_ToDouble();
    EXPECT_EQ(0, std::memcmp(&inf, &valDouble, sizeof(inf)));
}

// setTag_Test
/*!
    This test checks functionality of "getVal_ToString()" function of empty DataObjects
*/
TYPED_TEST(dataObjectTagType_Test, getVal_ToString_Test)
{
    double tempVal3 = 24;
    double val1 = std::numeric_limits<double>::signaling_NaN();
    double val2 = std::numeric_limits<double>::quiet_NaN();
    double val3 = std::numeric_limits<double>::infinity();
    double valDouble;
    std::ostringstream s4;
    std::string str4;
    this->StrObj1 = ito::DataObjectTagType("Test String"); /*!< Declaring DataObjectTagType Object with string. */

    this->Obj1 = ito::DataObjectTagType(tempVal3); /*!< Declaring DataObjectTagType with temporary double value.*/
    this->Obj2 =
        ito::DataObjectTagType(std::numeric_limits<double>::quiet_NaN()); /*!< Declaring DataObjectTagType with
                                                                             std::numeric_limits<double>::quiet_NaN().*/
    this->Obj3 = ito::DataObjectTagType(
        std::numeric_limits<double>::signaling_NaN()); /*!< Declaring DataObjectTagType with
                                                          std::numeric_limits<double>::signaling_NaN().*/
    this->Obj4 =
        ito::DataObjectTagType(std::numeric_limits<double>::infinity()); /*!< Declaring DataObjectTagType with
                                                                            std::numeric_limits<double>::infinity().*/

    //!< Creating Copied Objects using DataObjectTagType Copy Constructor.
    this->CpyStrObj1 = ito::DataObjectTagType(
        this->StrObj1); /*!< Copying the String Object StrObj1 into CpyStrObj1 using Copy Constructor */
    this->CpyEmptyObj1 = ito::DataObjectTagType(
        this->EmptyObj1); /*!< Copying the Empty Object EmptyObj1 into CpyEmptyObj1 using Copy Constructor */
    this->CpyObj1 =
        ito::DataObjectTagType(this->Obj1); /*!< Copying the Object Obj1 into CpyObj1 using Copy Constructor */
    this->CpyObj2 =
        ito::DataObjectTagType(this->Obj2); /*!< Copying the Object Obj2 into CpyObj2 using Copy Constructor */
    this->CpyObj3 =
        ito::DataObjectTagType(this->Obj3); /*!< Copying the Object Obj3 into CpyObj3 using Copy Constructor */
    this->CpyObj4 =
        ito::DataObjectTagType(this->Obj4); /*!< Copying the Object Obj4 into CpyObj4 using Copy Constructor */

    //!< Copying the Obj1,Obj2,Obj3 and Obj4 using assigned Operator "="
    this->AsgnStrObj1 = this->StrObj1;
    this->AsgnEmptyObj1 = this->EmptyObj1;
    this->AsgnObj1 = this->Obj1;
    this->AsgnObj2 = this->Obj2;
    this->AsgnObj3 = this->Obj3;
    this->AsgnObj4 = this->Obj4;

    //!< Test for getVal_ToString() function.
    EXPECT_EQ(this->StrObj1.getVal_ToString(),
              "Test String"); //!< Test for getVal_ToString() with String Objects of DataObjectTagType
    EXPECT_EQ(this->EmptyObj1.getVal_ToString(),
              "");   //!< Test for getVal_ToString() with empty objects of DataObjectTagType
    s4 << tempVal3;  //!< Creating StreamString representation of Double value for further comparision.
    str4 = s4.str(); //!< Converting StreamString into String for further comparision.
    EXPECT_EQ(str4.data(), this->Obj1.getVal_ToString());
    EXPECT_EQ(this->Obj2.getVal_ToString(), "NaN");
    EXPECT_EQ(this->Obj3.getVal_ToString(), "NaN");
    EXPECT_EQ(this->Obj4.getVal_ToString(), "Inf");

    //!< Test for getVal_ToString() function with copied objects.
    EXPECT_EQ(this->CpyStrObj1.getVal_ToString(),
              "Test String"); //!< Test for getVal_ToString() with Copied String Objects with Copy Constructor of
                              //!< DataObjectTagType
    EXPECT_EQ(this->CpyObj1.getVal_ToString(), str4.data());
    EXPECT_EQ(this->CpyObj2.getVal_ToString(), "NaN");
    EXPECT_EQ(this->CpyObj3.getVal_ToString(), "NaN");
    EXPECT_EQ(this->CpyObj4.getVal_ToString(), "Inf");

    //!< Test for getVal_ToString() function with copied objects using Assignment Operator "=".
    EXPECT_EQ(this->AsgnStrObj1.getVal_ToString(),
              "Test String"); //!< Test for getVal_ToString() with Copied String Objects with Assignment Operator "=" of
                              //!< DataObjectTagType
    EXPECT_EQ(this->AsgnObj1.getVal_ToString(), str4.data());
    EXPECT_EQ(this->AsgnObj2.getVal_ToString(), "NaN");
    EXPECT_EQ(this->AsgnObj3.getVal_ToString(), "NaN");
    EXPECT_EQ(this->AsgnObj4.getVal_ToString(), "Inf");
}
