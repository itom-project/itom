#include "param.h"
#include "byteArray.h"
#include "gtest/gtest.h"
#include <chrono>

using namespace ito;

void checkParamBase(
    ParamBase pb,
    const char* name,
    const ParamBase::Type type,
    const bool ckName,
    const bool ckType)
{
    if (ckName)
    {
        EXPECT_STREQ(pb.getName(), name);
    }

    if (ckType)
    {
        EXPECT_EQ(pb.getType(), type);
    }
}

void checkInt(
    ParamBase pb,
    const char* name,
    const ParamBase::Type type,
    const int32 value,
    const bool ckName,
    const bool ckType,
    const bool ckValue)
{
    checkParamBase(pb, name, type, ckName, ckType);

    if (ckValue)
    {
        EXPECT_EQ(pb.getVal<int32>(), value);

        int32 addIntValue = 1;
        int32 testVal = value + addIntValue;
        int32 pbVal = pb.getVal<int32>() + addIntValue;
        pb.setVal<int32>(pbVal);
        EXPECT_EQ(pb.getVal<int32>(), testVal);
    }
}

void checkDouble(
    ParamBase pb,
    const char* name,
    const ParamBase::Type type,
    const float64 value,
    const bool ckName,
    const bool ckType,
    const bool ckValue)
{
    checkParamBase(pb, name, type, ckName, ckType);

    if (ckValue)
    {
        EXPECT_DOUBLE_EQ(pb.getVal<float64>(), value);

        float64 addDoubleValue = 1.112;
        float64 testVal = value + addDoubleValue;
        float64 pbVal = pb.getVal<float64>() + addDoubleValue;
        pb.setVal<float64>(pbVal);
        EXPECT_DOUBLE_EQ(pb.getVal<float64>(), testVal);
    }
}

void checkDouble(
    ParamBase pb,
    const char* name,
    const ParamBase::Type type,
    const complex128 value,
    const bool ckName,
    const bool ckType,
    const bool ckValue)
{
    checkParamBase(pb, name, type, ckName, ckType);

    if (ckValue)
    {
        EXPECT_DOUBLE_EQ(pb.getVal<complex128>().real(), value.real());
        EXPECT_DOUBLE_EQ(pb.getVal<complex128>().imag(), value.imag());

        complex128 addDoubleValue = complex128(1.112, -7.546);
        complex128 testVal = value + addDoubleValue;
        complex128 pbVal = pb.getVal<complex128>() + addDoubleValue;
        pb.setVal<complex128>(pbVal);
        EXPECT_DOUBLE_EQ(pb.getVal<complex128>().real(), testVal.real());
        EXPECT_DOUBLE_EQ(pb.getVal<complex128>().imag(), testVal.imag());
    }
}

void checkChar(
    ParamBase pb,
    const char* name,
    const ParamBase::Type type,
    const char value,
    const bool ckName,
    const bool ckType,
    const bool ckValue)
{
    checkParamBase(pb, name, type, ckName, ckType);

    if (ckValue)
    {
        EXPECT_EQ(pb.getVal<char>(), value);

        int addIntValue = 1;
        int testVal = value + addIntValue;
        int pbVal = pb.getVal<int>() + addIntValue;
        pb.setVal<char>(pbVal);
        EXPECT_EQ(pb.getVal<char>(), testVal);
    }
}

TEST(ParamTest, Constructor)
{
    ByteArray paramBaInt("ByteArray");

    // integer
    const char* paramName = "pbInt";
    ParamBase::Type paramType = ParamBase::Int;
    const int32 paramValueInt = 42;

    ParamBase pbI1(paramName);
    checkInt(pbI1, paramName, paramType, paramValueInt, true, false, false);

    ParamBase pbI2(paramName, paramType);
    checkInt(pbI2, paramName, paramType, paramValueInt, true, true, false);

    ParamBase pbI3(paramName, paramType, paramValueInt);
    checkInt(pbI3, paramName, paramType, paramValueInt, true, true, true);

    //    ParamBase pbI4(paramBaInt, ParamBase::IntArray, paramValueInt);
    //    checkInt(pbI4, paramName, paramType, paramValueInt, true, true, true);

    // double
    paramName = "pbDouble";
    paramType = ParamBase::Double;
    const float64 paramValueDouble = 1.321;

    ParamBase pbD1(paramName);
    checkDouble(pbD1, paramName, paramType, paramValueDouble, true, false, false);

    ParamBase pbD2(paramName, paramType);
    checkDouble(pbD2, paramName, paramType, paramValueDouble, true, true, false);

    ParamBase pbD3(paramName, paramType, paramValueDouble);
    checkDouble(pbD3, paramName, paramType, paramValueDouble, true, true, true);

    //    ParamBase pbD4(paramBaInt, ParamBase::DoubleArray, paramValueDouble);
    //    checkDouble(pbD4, paramName, paramType, paramValueDouble, true, true, true);

    // complex
    paramName = "pbComplex";
    paramType = ParamBase::Complex;
    const complex128 paramValueComplex(1.321, 5.999);

    ParamBase pbCm1(paramName);
    checkDouble(pbCm1, paramName, paramType, paramValueComplex, true, false, false);

    ParamBase pbCm2(paramName, paramType);
    checkDouble(pbCm2, paramName, paramType, paramValueComplex, true, true, false);

    ParamBase pbCm3(paramName, paramType, paramValueComplex);
    checkDouble(pbCm3, paramName, paramType, paramValueComplex, true, true, true);

    //    ParamBase pbD4(paramBaInt, ParamBase::DoubleArray, paramValueDouble);
    //    checkDouble(pbD4, paramName, paramType, paramValueDouble, true, true, true);

    // char
    paramName = "pbChar";
    paramType = ParamBase::Char;
    const char paramValueChar = 42;

    ParamBase pbC1(paramName);
    checkChar(pbC1, paramName, paramType, paramValueChar, true, false, false);

    ParamBase pbC2(paramName, paramType);
    checkChar(pbC2, paramName, paramType, paramValueChar, true, true, false);

    ParamBase pbC3(paramName, paramType, paramValueChar);
    checkChar(pbC3, paramName, paramType, paramValueChar, true, true, true);
}

TEST(ParamTest, EqualOperator)
{
    ParamBase p1;
    ParamBase p2;
    ParamBase p3("p3", ParamBase::Int, 5);
    ParamBase p4("p4", ParamBase::Int, 5);
    int values1[] = {2, 3};
    ParamBase p5("p5", ParamBase::IntArray, 2, values1);
    int values2[] = {2, 3, 4};
    ParamBase p6("p6", ParamBase::IntArray, 3, values2);
    EXPECT_TRUE(p1 == p2);
    EXPECT_TRUE(p1 == p1);
    EXPECT_FALSE(p1 == p3);
    EXPECT_TRUE(p3 == p3);
    EXPECT_TRUE(p3 == p4);
    EXPECT_TRUE(p5 == p5);
    EXPECT_FALSE(p5 == p4);
    EXPECT_FALSE(p4 == p5);
    EXPECT_TRUE(p6 == p6);
    EXPECT_FALSE(p5 == p6);
    EXPECT_FALSE(p6 == p5);

    ParamBase p7("p7", ParamBase::String, "test");
    ParamBase p8("p8", ParamBase::String, "test");
    ParamBase p9("p9", ParamBase::String, "test_");
    EXPECT_TRUE(p7 == p8);
    EXPECT_FALSE(p7 == p9);
    EXPECT_TRUE(p9 == p9);

    ParamBase p3d("p3d", ParamBase::Double, 5.0);
    ParamBase p4d("p4d", ParamBase::Double, 5.0);
    float64 values1d[] = {2.0, 3.0};
    ParamBase p5d("p5d", ParamBase::DoubleArray, 2, values1d);
    float64 values2d[] = {2.0, 3.0, 4.0};
    ParamBase p6d("p6d", ParamBase::DoubleArray, 3, values2d);
    ParamBase p7d("p7d", ParamBase::DoubleArray, 2, values2d);
    EXPECT_TRUE(p3d == p4d);
    EXPECT_FALSE(p3d == p3);
    EXPECT_TRUE(p5d == p5d);
    EXPECT_FALSE(p5d == p6d);
    EXPECT_TRUE(p5d == p7d);

    ParamBase p3c("p3c", ParamBase::Complex, 5.0);
    ParamBase p4c("p4c", ParamBase::Complex, 5.0);
    std::complex<double> values1c[] = {
        std::complex<double>(2, 3.123579567945), std::complex<double>(3, -5.5)};
    ParamBase p5c("p5c", ParamBase::ComplexArray, 2, values1c);
    std::complex<double> values2c[] = {
        std::complex<double>(2, 3.123579567945),
        std::complex<double>(3, -5.5),
        std::complex<double>(0, 0)};
    ParamBase p6c("p6c", ParamBase::ComplexArray, 3, values2c);
    ParamBase p7c("p7c", ParamBase::ComplexArray, 2, values2c);
    ParamBase p8c("p8c", ParamBase::ComplexArray);
    EXPECT_TRUE(p3c == p4c);
    EXPECT_FALSE(p3c == p3);
    EXPECT_TRUE(p5c == p5c);
    EXPECT_FALSE(p5c == p6c);
    EXPECT_TRUE(p5c == p7c);
    EXPECT_FALSE(p1 == p8c);
    EXPECT_TRUE(p8c == p8c);

    EXPECT_FALSE(p3d == p3);
    EXPECT_FALSE(p3d == p8);
}

TEST(ParamTest, UnequalOperator)
{
    ParamBase p1;
    ParamBase p2;
    ParamBase p3("p3", ParamBase::Int, 5);
    ParamBase p4("p4", ParamBase::Int, 5);
    int values1[] = {2, 3};
    ParamBase p5("p5", ParamBase::IntArray, 2, values1);
    int values2[] = {2, 3, 4};
    ParamBase p6("p6", ParamBase::IntArray, 3, values2);
    EXPECT_FALSE(p1 != p2);
    EXPECT_FALSE(p1 != p1);
    EXPECT_TRUE(p1 != p3);
    EXPECT_FALSE(p3 != p3);
    EXPECT_FALSE(p3 != p4);
    EXPECT_FALSE(p5 != p5);
    EXPECT_TRUE(p5 != p4);
    EXPECT_TRUE(p4 != p5);
    EXPECT_FALSE(p6 != p6);
    EXPECT_TRUE(p5 != p6);
    EXPECT_TRUE(p6 != p5);
}

TEST(ParamTest, NumericTest)
{
    EXPECT_TRUE(ParamBase("p1", ParamBase::Char).isNumeric());
    EXPECT_TRUE(ParamBase("p1", ParamBase::Int).isNumeric());
    EXPECT_TRUE(ParamBase("p1", ParamBase::Double).isNumeric());
    EXPECT_TRUE(ParamBase("p1", ParamBase::Complex).isNumeric());
    EXPECT_FALSE(ParamBase("p1", ParamBase::CharArray).isNumeric());
    EXPECT_FALSE(ParamBase("p1", ParamBase::IntArray).isNumeric());
    EXPECT_FALSE(ParamBase("p1", ParamBase::DoubleArray).isNumeric());
    EXPECT_FALSE(ParamBase("p1", ParamBase::ComplexArray).isNumeric());
    EXPECT_FALSE(ParamBase("p1", ParamBase::String).isNumeric());
    EXPECT_FALSE(ParamBase("p1", ParamBase::HWRef).isNumeric());
    EXPECT_FALSE(ParamBase("p1", ParamBase::DObjPtr).isNumeric());
    EXPECT_FALSE(ParamBase("p1", ParamBase::PointCloudPtr).isNumeric());

    EXPECT_FALSE(ParamBase("p1", ParamBase::Char).isNumericArray());
    EXPECT_FALSE(ParamBase("p1", ParamBase::Int).isNumericArray());
    EXPECT_FALSE(ParamBase("p1", ParamBase::Double).isNumericArray());
    EXPECT_FALSE(ParamBase("p1", ParamBase::Complex).isNumericArray());
    EXPECT_TRUE(ParamBase("p1", ParamBase::CharArray).isNumericArray());
    EXPECT_TRUE(ParamBase("p1", ParamBase::IntArray).isNumericArray());
    EXPECT_TRUE(ParamBase("p1", ParamBase::DoubleArray).isNumericArray());
    EXPECT_TRUE(ParamBase("p1", ParamBase::ComplexArray).isNumericArray());
    EXPECT_FALSE(ParamBase("p1", ParamBase::String).isNumericArray());
    EXPECT_FALSE(ParamBase("p1", ParamBase::HWRef).isNumericArray());
    EXPECT_FALSE(ParamBase("p1", ParamBase::DObjPtr).isNumericArray());
    EXPECT_FALSE(ParamBase("p1", ParamBase::PointCloudPtr).isNumericArray());
}

TEST(ParamTest, ParamBaseScalarComplexTest)
{
    complex128 val(-2.2, 30.5);
    ParamBase scalarComplex("scalarComplex", ParamBase::Complex, val);
    EXPECT_TRUE(scalarComplex.isValid());
    EXPECT_STREQ(scalarComplex.getName(), "scalarComplex");
    EXPECT_EQ(scalarComplex.getLen(), 1);

    EXPECT_DOUBLE_EQ(scalarComplex.getVal<float64>(), val.real());
    EXPECT_EQ(scalarComplex.getVal<int32>(), (int32)val.real());
    auto val2 = scalarComplex.getVal<complex128>();
    EXPECT_DOUBLE_EQ(val.real(), val2.real());
    EXPECT_DOUBLE_EQ(val.imag(), val2.imag());

    scalarComplex.setVal<float64>(-3.2);
    EXPECT_DOUBLE_EQ(scalarComplex.getVal<float64>(), -3.2);

    complex128 newVal(1.0, -2.6);
    scalarComplex.setVal<complex128>(newVal);
    val2 = scalarComplex.getVal<complex128>();
    EXPECT_DOUBLE_EQ(newVal.real(), val2.real());
    EXPECT_DOUBLE_EQ(newVal.imag(), val2.imag());
}

TEST(ParamTest, ParamBaseIndexingOperator)
{
    // scalar types
    int scalarType[] = {
        ParamBase::Char, ParamBase::Int, ParamBase::Double, ParamBase::Complex, ParamBase::String};
    for (int t : scalarType)
    {
        ParamBase intParam("scalar", t);
        EXPECT_FALSE(intParam[0].isValid());
    }

    // array types
    int32 intArr[] = {5, 6, 7};
    ParamBase intArrayParam("array", ParamBase::IntArray, 3, intArr);
    float64 dblArr[] = {5.0, 5.6, 7.0};
    ParamBase dblArrayParam("array", ParamBase::DoubleArray, 3, dblArr);
    complex128 cmplxArr[] = {complex128(5.0, 1.7), complex128(-4.5, 6.7), complex128(7.0, 5.6)};
    ParamBase cmplxArrayParam("array", ParamBase::ComplexArray, 3, cmplxArr);
    ByteArray strList[] = {ByteArray("hello"), ByteArray("no 123"), ByteArray("-123.5")};
    ParamBase strListParam("stringList", ParamBase::StringList, 3, strList);

    ParamBase params[] = {intArrayParam, dblArrayParam, cmplxArrayParam};

    for (const auto& p : params)
    {
        ParamBase p0 = p[0];
        EXPECT_TRUE(p0.isValid());
        EXPECT_STREQ(p0.getName(), "array[0]");
        EXPECT_DOUBLE_EQ(p0.getVal<float64>(), 5);

        ParamBase p2 = p[2];
        EXPECT_TRUE(p2.isValid());
        EXPECT_STREQ(p2.getName(), "array[2]");
        EXPECT_DOUBLE_EQ(p2.getVal<float64>(), 7);

        ParamBase pm1 = p[-1];
        EXPECT_FALSE(pm1.isValid());

        ParamBase p3 = p[3];
        EXPECT_FALSE(p3.isValid());
    }

    // string list
    {
        ParamBase p0 = strListParam[0];
        EXPECT_TRUE(p0.isValid());
        EXPECT_STREQ(p0.getName(), "stringList[0]");
        EXPECT_STREQ(p0.getVal<const char*>(), strList[0].data());

        ParamBase p2 = strListParam[2];
        EXPECT_TRUE(p2.isValid());
        EXPECT_STREQ(p2.getName(), "stringList[2]");
        EXPECT_STREQ(p2.getVal<const char*>(), strList[2].data());

        ParamBase pm1 = strListParam[-1];
        EXPECT_FALSE(pm1.isValid());

        ParamBase p3 = strListParam[3];
        EXPECT_FALSE(p3.isValid());
    }
}

TEST(ParamTest, ParamCopyOperator)
{
    char charArr[] = {-127, 0, 127};
    int32 intArr[] = {5, -6, 7};
    float64 dblArr[] = {5.0, 5.6, -7.0};
    complex128 cmplxArr[] = {complex128(5.0, 1.7), complex128(-4.5, 6.7), complex128(7.0, 5.6)};
    ByteArray strList[] = {ByteArray("hello"), ByteArray("no 123"), ByteArray("-123.5")};

    ParamBase params[] = {
        ParamBase("char", ParamBase::Char, 5),
        ParamBase("int", ParamBase::Int, -5),
        ParamBase("double", ParamBase::Double, -5.78),
        ParamBase("complex", ParamBase::Complex, complex128(5, -3.2)),
        ParamBase("charArray", ParamBase::CharArray, 3, charArr),
        ParamBase("intArray", ParamBase::IntArray, 3, intArr),
        ParamBase("doubleArray", ParamBase::DoubleArray, 3, dblArr),
        ParamBase("complexArray", ParamBase::ComplexArray, 3, cmplxArr),
        ParamBase("string", ParamBase::String, "test string 123"),
        ParamBase("stringList", ParamBase::StringList, 3, strList),
        ParamBase("hwref", ParamBase::HWRef, nullptr)};

    for (const auto& p1 : params)
    {
        ParamBase p2(p1);

        EXPECT_TRUE(p1 == p2) << p1.getName();

        ParamBase p3;
        p3 = p1;

        EXPECT_TRUE(p1 == p3) << p1.getName();
    }
}

TEST(ParamTest, ParamAssignmentOperator)
{
    char charArr[] = {-127, 0, 127};
    int32 intArr[] = {5, -6, 7};
    float64 dblArr[] = {5.0, 5.6, -7.0};
    complex128 cmplxArr[] = {complex128(5.0, 1.7), complex128(-4.5, 6.7), complex128(7.0, 5.6)};
    ByteArray strList[] = {ByteArray("hello"), ByteArray("no 123"), ByteArray("-123.5")};

    ParamBase param("stringList", ParamBase::StringList, 3, strList);
    EXPECT_EQ(param.getVal<const ByteArray*>()[1], strList[1]);
    int len = 0;
    param.getVal<const ByteArray*>(len);
    EXPECT_EQ(len, 3);

    param = ParamBase("intArray", ParamBase::IntArray, 3, intArr);
    EXPECT_EQ(param.getVal<const int32*>()[2], intArr[2]);
    param.getVal<const int32*>(len);
    EXPECT_EQ(len, 3);

    param = ParamBase("string", ParamBase::String, "hello");
    EXPECT_STREQ(param.getVal<const char*>(), "hello");
    param.getVal<const char*>(len);
    EXPECT_EQ(len, 5);
}


TEST(ParamTest, ParamIndexingOperator)
{
    // scalar types
    int scalarType[] = {
        ParamBase::Char, ParamBase::Int, ParamBase::Double, ParamBase::Complex, ParamBase::String};
    for (int t : scalarType)
    {
        Param intParam("scalar", t);
        EXPECT_FALSE(intParam[0].isValid());
    }

    // array types
    int32 intArr[] = {5, 6, 7};
    Param intArrayParam("array", ParamBase::IntArray, 3, intArr, "info");
    float64 dblArr[] = {5.0, 5.6, 7.0};
    Param dblArrayParam("array", ParamBase::DoubleArray, 3, dblArr, "info");
    complex128 cmplxArr[] = {complex128(5.0, 1.7), complex128(-4.5, 6.7), complex128(7.0, 5.6)};
    Param cmplxArrayParam("array", ParamBase::ComplexArray, 3, cmplxArr, "info");

    Param params[] = {intArrayParam, dblArrayParam, cmplxArrayParam};

    for (auto p : params)
    {
        Param p0 = p[0];
        EXPECT_TRUE(p0.isValid());
        EXPECT_STREQ(p0.getName(), "array[0]");
        EXPECT_DOUBLE_EQ(p0.getVal<float64>(), 5);
        EXPECT_STREQ(p0.getInfo(), "info");
        EXPECT_EQ(p0.getMeta(), nullptr);

        Param p2 = p[2];
        EXPECT_TRUE(p2.isValid());
        EXPECT_STREQ(p2.getName(), "array[2]");
        EXPECT_DOUBLE_EQ(p2.getVal<float64>(), 7);
        EXPECT_STREQ(p2.getInfo(), "info");
        EXPECT_EQ(p2.getMeta(), nullptr);

        Param pm1 = p[-1];
        EXPECT_FALSE(pm1.isValid());

        Param p3 = p[3];
        EXPECT_FALSE(p3.isValid());
    }

    intArrayParam.setMeta(new IntArrayMeta(0, 100, 2, "category"), true);
    Param p1 = intArrayParam[1];
    EXPECT_TRUE(p1.getMeta() != nullptr);
    IntMeta* m = p1.getMetaT<IntMeta>();
    EXPECT_TRUE(m != nullptr);
    EXPECT_TRUE(m->getType() == ParamMeta::rttiIntMeta);
    EXPECT_TRUE(m->getMax() == 100);

    dblArrayParam.setMeta(new DoubleArrayMeta(0.0, 100.0, 2.0, "category"), true);
    Param p2 = dblArrayParam[1];
    EXPECT_TRUE(p2.getMeta() != nullptr);
    DoubleMeta* m2 = p2.getMetaT<DoubleMeta>();
    EXPECT_TRUE(m2 != nullptr);
    EXPECT_TRUE(m2->getType() == ParamMeta::rttiDoubleMeta);
    EXPECT_DOUBLE_EQ(m2->getMax(), 100.0);

    ByteArray stringlist[] = {"test1", "test2", "test3"};
    Param stringlistParam("stringlist", ParamBase::StringList, 3, stringlist, "info");
    auto p3 = stringlistParam[1];
    EXPECT_STREQ(p3.getVal<const char*>(), "test2");
    EXPECT_TRUE(p3.getType() == ParamBase::String);
    Param stringlistParam2 = stringlistParam;
    ByteArray* ba2 = stringlistParam2.getVal<ByteArray*>();
    ba2[1] = "xxx";
    EXPECT_STREQ(stringlistParam.getVal<const ByteArray*>()[1].data(), "test2");
    EXPECT_STREQ(stringlistParam2.getVal<const ByteArray*>()[1].data(), "xxx");
}

TEST(ParamTest, ParamImplicitSharing)
{
    IntMeta* intMeta = new IntMeta(0, 10, 2);
    Param pInt1("pInt1", ParamBase::Int, 5, intMeta, "description");

    EXPECT_EQ(pInt1.getMin(), 0);
    EXPECT_EQ(pInt1.getMax(), 10);
    EXPECT_EQ(pInt1.getMetaT<IntMeta>()->getStepSize(), 2);
    EXPECT_STREQ(pInt1.getInfo(), "description");

    Param pInt1_1 = pInt1;
    EXPECT_EQ(pInt1_1.getMin(), 0);
    EXPECT_EQ(pInt1_1.getMax(), 10);
    EXPECT_EQ(pInt1_1.getConstMetaT<IntMeta>()->getStepSize(), 2);

    Param pInt1_2(pInt1);
    EXPECT_EQ(pInt1_2.getMin(), 0);
    EXPECT_EQ(pInt1_2.getMax(), 10);
    EXPECT_EQ(pInt1_2.getConstMetaT<IntMeta>()->getStepSize(), 2);
    pInt1_2.setInfo("newinfo");
    EXPECT_STREQ(pInt1_2.getInfo(), "newinfo");

    // modify pInt1_1 and check that the meta of pInt1 and pInt1_2 are not changed
    IntMeta* intMeta1 = static_cast<IntMeta*>(pInt1_1.getMeta());
    intMeta1->setMin(2);
    intMeta1->setMax(12);
    intMeta1->setStepSize(4);

    EXPECT_EQ(pInt1.getMin(), 0);
    EXPECT_EQ(pInt1.getMax(), 10);
    EXPECT_EQ(pInt1.getConstMetaT<IntMeta>()->getStepSize(), 2);

    EXPECT_EQ(pInt1_1.getMin(), 2);
    EXPECT_EQ(pInt1_1.getMax(), 12);
    EXPECT_EQ(pInt1_1.getConstMetaT<IntMeta>()->getStepSize(), 4);

    EXPECT_EQ(pInt1_2.getMin(), 0);
    EXPECT_EQ(pInt1_2.getMax(), 10);
    EXPECT_EQ(pInt1_2.getConstMetaT<IntMeta>()->getStepSize(), 2);

    pInt1.setMeta(new ito::IntMeta(3, 33, 10, "category"), true);

    EXPECT_EQ(pInt1.getMin(), 3);
    EXPECT_EQ(pInt1.getMax(), 33);
    EXPECT_EQ(pInt1.getConstMetaT<IntMeta>()->getStepSize(), 10);
    EXPECT_EQ(pInt1.getConstMetaT<IntMeta>()->getCategory(), "category");

    EXPECT_EQ(pInt1_1.getMin(), 2);
    EXPECT_EQ(pInt1_1.getMax(), 12);
    EXPECT_EQ(pInt1_1.getConstMetaT<IntMeta>()->getStepSize(), 4);

    EXPECT_EQ(pInt1_2.getMin(), 0);
    EXPECT_EQ(pInt1_2.getMax(), 10);
    EXPECT_EQ(pInt1_2.getConstMetaT<IntMeta>()->getStepSize(), 2);
}

TEST(ParamTest, ParamImplicitSharing2)
{
    float64 values[] = {10, 20};
    Param pDoubleArr = Param("p", ParamBase::DoubleArray, 2, values, "info");
    DoubleArrayMeta* m = new DoubleArrayMeta(-0.5, 22.5, 0.5, 0, 10, 1);
    pDoubleArr.setMeta(m, true);

    Param pDoubleArr2 = pDoubleArr;

    const auto constDblMeta = pDoubleArr.getConstMetaT<DoubleArrayMeta>();
    EXPECT_DOUBLE_EQ(constDblMeta->getMax(), 22.5);

    const auto constDblMeta2 = pDoubleArr2.getConstMetaT<DoubleArrayMeta>();
    EXPECT_DOUBLE_EQ(constDblMeta2->getMax(), 22.5);

    pDoubleArr2.setMeta(new DoubleIntervalMeta(-5.0, 25.0, 1.0), true);

    const auto constDblMeta3 = pDoubleArr.getConstMetaT<DoubleArrayMeta>();
    EXPECT_DOUBLE_EQ(constDblMeta3->getMax(), 22.5);

    const auto constDblMeta4 = pDoubleArr2.getConstMetaT<DoubleIntervalMeta>();
    EXPECT_DOUBLE_EQ(constDblMeta4->getMax(), 25.0);

    pDoubleArr = Param("p2", ParamBase::Int, 3, nullptr, "");
    EXPECT_EQ(pDoubleArr.getConstMeta(), nullptr);
    EXPECT_NE(pDoubleArr2.getConstMeta(), nullptr);
}


TEST(ParamTest, FlagTest)
{
    ParamBase p1("name", ParamBase::Char | ParamBase::Readonly, 0);
    EXPECT_TRUE(p1.getFlags() & ParamBase::In);
    EXPECT_FALSE(p1.getFlags() & ParamBase::NoAutosave);
    EXPECT_TRUE(p1.getFlags() & ParamBase::Readonly);

    ParamBase p2 = p1;
    EXPECT_TRUE(p2.getFlags() & ParamBase::In);
    EXPECT_FALSE(p2.getFlags() & ParamBase::NoAutosave);
    EXPECT_TRUE(p2.getFlags() & ParamBase::Readonly);

    ParamBase p3(p1);
    EXPECT_TRUE(p3.getFlags() & ParamBase::In);
    EXPECT_FALSE(p3.getFlags() & ParamBase::NoAutosave);
    EXPECT_TRUE(p3.getFlags() & ParamBase::Readonly);

    // check the "no autosave" types
    uint32 noAutosaveTypes[] = {
        ParamBase::DObjPtr,
        ParamBase::HWRef,
        ParamBase::PointCloudPtr,
        ParamBase::PointPtr,
        ParamBase::PolygonMeshPtr};

    for (auto t : noAutosaveTypes)
    {
        ParamBase p1("name", t);
        EXPECT_TRUE(p1.getFlags() & ParamBase::NoAutosave);
    }

    // check if flags are also copied for indexing operator
    const char charArray[] = {1, 2, 3, 4, 5};
    p1 = ParamBase(
        "name", ParamBase::CharArray | ParamBase::Out | ParamBase::Readonly, 5, charArray);
    auto charArray2 = p1[2];
    EXPECT_TRUE(charArray2.getFlags() & ParamBase::Readonly);
    EXPECT_TRUE(charArray2.getFlags() & ParamBase::Out);
    EXPECT_FALSE(charArray2.getFlags() & ParamBase::In);
}

TEST(ParamTest, ConstGetValTest)
{
    ByteArray ba[] = {"blub", "blob"};
    ParamBase aa("name", ParamBase::StringList, 2, ba);

    double* dbl = new double[256];

    ParamBase pp("name", ParamBase::DoubleArray, 256, dbl);
    ParamBase pp2 = pp;
    pp.setVal<double*>(dbl, 5);
    EXPECT_EQ(pp.getLen(), 5);
    EXPECT_EQ(pp2.getLen(), 256);
    const double* d = pp.getVal<const double*>();

    /*auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100000; ++i)
    {
        ParamBase p("name", ParamBase::Double, 45.4);
        auto p2(p);
        auto p3 = p2;
        double xx = p3.getVal<double>();
    }

    auto end_time1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100000; ++i)
    {
        ParamBase p("name", ParamBase::DoubleArray, 256, dbl);
        auto p2(p);
        auto p3 = p2;
        const double* xx = p3.getVal<const double*>();
    }

    auto end_time2 = std::chrono::high_resolution_clock::now();*/

    delete[] dbl;

    //EXPECT_TRUE(false) << "time:" << (end_time1 - start_time) / std::chrono::milliseconds(1)
    //                   << ", dbl-array:" << (end_time2 - end_time1) / std::chrono::milliseconds(1);
}
