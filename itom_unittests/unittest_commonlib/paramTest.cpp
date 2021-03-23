#include "param.h"
#include "byteArray.h"
#include "gtest/gtest.h"

void checkParamBase(ito::ParamBase pb, const char *name, const ito::ParamBase::Type type, const bool ckName,
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

void checkInt(ito::ParamBase pb, const char *name, const ito::ParamBase::Type type, const ito::int32 value,
              const bool ckName, const bool ckType, const bool ckValue)
{
    checkParamBase(pb, name, type, ckName, ckType);

    if (ckValue)
    {
        EXPECT_EQ(pb.getVal<ito::int32>(), value);

        ito::int32 addIntValue = 1;
        ito::int32 testVal = value + addIntValue;
        ito::int32 pbVal = pb.getVal<ito::int32>() + addIntValue;
        pb.setVal<ito::int32>(pbVal);
        EXPECT_EQ(pb.getVal<ito::int32>(), testVal);
    }
}

void checkDouble(ito::ParamBase pb, const char *name, const ito::ParamBase::Type type, const ito::float64 value,
                 const bool ckName, const bool ckType, const bool ckValue)
{
    checkParamBase(pb, name, type, ckName, ckType);

    if (ckValue)
    {
        EXPECT_DOUBLE_EQ(pb.getVal<ito::float64>(), value);

        ito::float64 addDoubleValue = 1.112;
        ito::float64 testVal = value + addDoubleValue;
        ito::float64 pbVal = pb.getVal<ito::float64>() + addDoubleValue;
        pb.setVal<ito::float64>(pbVal);
        EXPECT_DOUBLE_EQ(pb.getVal<ito::float64>(), testVal);
    }
}

void checkDouble(ito::ParamBase pb, const char *name, const ito::ParamBase::Type type, const ito::complex128 value,
                 const bool ckName, const bool ckType, const bool ckValue)
{
    checkParamBase(pb, name, type, ckName, ckType);

    if (ckValue)
    {
        EXPECT_DOUBLE_EQ(pb.getVal<ito::complex128>().real(), value.real());
        EXPECT_DOUBLE_EQ(pb.getVal<ito::complex128>().imag(), value.imag());

        ito::complex128 addDoubleValue = ito::complex128(1.112, -7.546);
        ito::complex128 testVal = value + addDoubleValue;
        ito::complex128 pbVal = pb.getVal<ito::complex128>() + addDoubleValue;
        pb.setVal<ito::complex128>(pbVal);
        EXPECT_DOUBLE_EQ(pb.getVal<ito::complex128>().real(), testVal.real());
        EXPECT_DOUBLE_EQ(pb.getVal<ito::complex128>().imag(), testVal.imag());
    }
}

void checkChar(ito::ParamBase pb, const char *name, const ito::ParamBase::Type type, const char value,
               const bool ckName, const bool ckType, const bool ckValue)
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
    ito::ByteArray paramBaInt("ByteArray");

    // integer
    const char *paramName = "pbInt";
    ito::ParamBase::Type paramType = ito::ParamBase::Int;
    const ito::int32 paramValueInt = 42;

    ito::ParamBase pbI1(paramName);
    checkInt(pbI1, paramName, paramType, paramValueInt, true, false, false);

    ito::ParamBase pbI2(paramName, paramType);
    checkInt(pbI2, paramName, paramType, paramValueInt, true, true, false);

    ito::ParamBase pbI3(paramName, paramType, paramValueInt);
    checkInt(pbI3, paramName, paramType, paramValueInt, true, true, true);

    //    ito::ParamBase pbI4(paramBaInt, ito::ParamBase::IntArray, paramValueInt);
    //    checkInt(pbI4, paramName, paramType, paramValueInt, true, true, true);

    // double
    paramName = "pbDouble";
    paramType = ito::ParamBase::Double;
    const ito::float64 paramValueDouble = 1.321;

    ito::ParamBase pbD1(paramName);
    checkDouble(pbD1, paramName, paramType, paramValueDouble, true, false, false);

    ito::ParamBase pbD2(paramName, paramType);
    checkDouble(pbD2, paramName, paramType, paramValueDouble, true, true, false);

    ito::ParamBase pbD3(paramName, paramType, paramValueDouble);
    checkDouble(pbD3, paramName, paramType, paramValueDouble, true, true, true);

    //    ito::ParamBase pbD4(paramBaInt, ito::ParamBase::DoubleArray, paramValueDouble);
    //    checkDouble(pbD4, paramName, paramType, paramValueDouble, true, true, true);

    // complex
    paramName = "pbComplex";
    paramType = ito::ParamBase::Complex;
    const ito::complex128 paramValueComplex(1.321, 5.999);

    ito::ParamBase pbCm1(paramName);
    checkDouble(pbCm1, paramName, paramType, paramValueComplex, true, false, false);

    ito::ParamBase pbCm2(paramName, paramType);
    checkDouble(pbCm2, paramName, paramType, paramValueComplex, true, true, false);

    ito::ParamBase pbCm3(paramName, paramType, paramValueComplex);
    checkDouble(pbCm3, paramName, paramType, paramValueComplex, true, true, true);

    //    ito::ParamBase pbD4(paramBaInt, ito::ParamBase::DoubleArray, paramValueDouble);
    //    checkDouble(pbD4, paramName, paramType, paramValueDouble, true, true, true);

    // char
    paramName = "pbChar";
    paramType = ito::ParamBase::Char;
    const char paramValueChar = 42;

    ito::ParamBase pbC1(paramName);
    checkChar(pbC1, paramName, paramType, paramValueChar, true, false, false);

    ito::ParamBase pbC2(paramName, paramType);
    checkChar(pbC2, paramName, paramType, paramValueChar, true, true, false);

    ito::ParamBase pbC3(paramName, paramType, paramValueChar);
    checkChar(pbC3, paramName, paramType, paramValueChar, true, true, true);
}

TEST(ParamTest, EqualOperator)
{
    ito::ParamBase p1;
    ito::ParamBase p2;
    ito::ParamBase p3("p3", ito::ParamBase::Int, 5);
    ito::ParamBase p4("p4", ito::ParamBase::Int, 5);
    int values1[] = {2, 3};
    ito::ParamBase p5("p5", ito::ParamBase::IntArray, 2, values1);
    int values2[] = {2, 3, 4};
    ito::ParamBase p6("p6", ito::ParamBase::IntArray, 3, values2);
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

    ito::ParamBase p7("p7", ito::ParamBase::String, "test");
    ito::ParamBase p8("p8", ito::ParamBase::String, "test");
    ito::ParamBase p9("p9", ito::ParamBase::String, "test_");
    EXPECT_TRUE(p7 == p8);
    EXPECT_FALSE(p7 == p9);
    EXPECT_TRUE(p9 == p9);

    ito::ParamBase p3d("p3d", ito::ParamBase::Double, 5.0);
    ito::ParamBase p4d("p4d", ito::ParamBase::Double, 5.0);
    ito::float64 values1d[] = {2.0, 3.0};
    ito::ParamBase p5d("p5d", ito::ParamBase::DoubleArray, 2, values1d);
    ito::float64 values2d[] = {2.0, 3.0, 4.0};
    ito::ParamBase p6d("p6d", ito::ParamBase::DoubleArray, 3, values2d);
    ito::ParamBase p7d("p7d", ito::ParamBase::DoubleArray, 2, values2d);
    EXPECT_TRUE(p3d == p4d);
    EXPECT_FALSE(p3d == p3);
    EXPECT_TRUE(p5d == p5d);
    EXPECT_FALSE(p5d == p6d);
    EXPECT_TRUE(p5d == p7d);

    ito::ParamBase p3c("p3c", ito::ParamBase::Complex, 5.0);
    ito::ParamBase p4c("p4c", ito::ParamBase::Complex, 5.0);
    std::complex<double> values1c[] = {std::complex<double>(2, 3.123579567945), std::complex<double>(3, -5.5)};
    ito::ParamBase p5c("p5c", ito::ParamBase::ComplexArray, 2, values1c);
    std::complex<double> values2c[] = {std::complex<double>(2, 3.123579567945), std::complex<double>(3, -5.5),
                                       std::complex<double>(0, 0)};
    ito::ParamBase p6c("p6c", ito::ParamBase::ComplexArray, 3, values2c);
    ito::ParamBase p7c("p7c", ito::ParamBase::ComplexArray, 2, values2c);
    ito::ParamBase p8c("p8c", ito::ParamBase::ComplexArray);
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
    ito::ParamBase p1;
    ito::ParamBase p2;
    ito::ParamBase p3("p3", ito::ParamBase::Int, 5);
    ito::ParamBase p4("p4", ito::ParamBase::Int, 5);
    int values1[] = {2, 3};
    ito::ParamBase p5("p5", ito::ParamBase::IntArray, 2, values1);
    int values2[] = {2, 3, 4};
    ito::ParamBase p6("p6", ito::ParamBase::IntArray, 3, values2);
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
    EXPECT_TRUE(ito::ParamBase("p1", ito::ParamBase::Char).isNumeric());
    EXPECT_TRUE(ito::ParamBase("p1", ito::ParamBase::Int).isNumeric());
    EXPECT_TRUE(ito::ParamBase("p1", ito::ParamBase::Double).isNumeric());
    EXPECT_TRUE(ito::ParamBase("p1", ito::ParamBase::Complex).isNumeric());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::CharArray).isNumeric());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::IntArray).isNumeric());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::DoubleArray).isNumeric());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::ComplexArray).isNumeric());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::String).isNumeric());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::HWRef).isNumeric());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::DObjPtr).isNumeric());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::PointCloudPtr).isNumeric());

    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::Char).isNumericArray());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::Int).isNumericArray());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::Double).isNumericArray());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::Complex).isNumericArray());
    EXPECT_TRUE(ito::ParamBase("p1", ito::ParamBase::CharArray).isNumericArray());
    EXPECT_TRUE(ito::ParamBase("p1", ito::ParamBase::IntArray).isNumericArray());
    EXPECT_TRUE(ito::ParamBase("p1", ito::ParamBase::DoubleArray).isNumericArray());
    EXPECT_TRUE(ito::ParamBase("p1", ito::ParamBase::ComplexArray).isNumericArray());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::String).isNumericArray());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::HWRef).isNumericArray());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::DObjPtr).isNumericArray());
    EXPECT_FALSE(ito::ParamBase("p1", ito::ParamBase::PointCloudPtr).isNumericArray());
}

TEST(ParamTest, ParamBaseScalarComplexTest)
{
    ito::complex128 val(-2.2, 30.5);
    ito::ParamBase scalarComplex("scalarComplex", ito::ParamBase::Complex, val);
    EXPECT_TRUE(scalarComplex.isValid());
    EXPECT_STREQ(scalarComplex.getName(), "scalarComplex");
    EXPECT_EQ(scalarComplex.getLen(), 1);

    EXPECT_DOUBLE_EQ(scalarComplex.getVal<ito::float64>(), val.real());
    EXPECT_EQ(scalarComplex.getVal<ito::int32>(), (ito::int32)val.real());
    auto val2 = scalarComplex.getVal<ito::complex128>();
    EXPECT_DOUBLE_EQ(val.real(), val2.real());
    EXPECT_DOUBLE_EQ(val.imag(), val2.imag());

    scalarComplex.setVal<ito::float64>(-3.2);
    EXPECT_DOUBLE_EQ(scalarComplex.getVal<ito::float64>(), -3.2);

    ito::complex128 newVal(1.0, -2.6);
    scalarComplex.setVal<ito::complex128>(newVal);
    val2 = scalarComplex.getVal<ito::complex128>();
    EXPECT_DOUBLE_EQ(newVal.real(), val2.real());
    EXPECT_DOUBLE_EQ(newVal.imag(), val2.imag());
}

TEST(ParamTest, ParamBaseIndexingOperator)
{
    // scalar types
    int scalarType[] = {ito::ParamBase::Char, ito::ParamBase::Int, ito::ParamBase::Double, ito::ParamBase::Complex,
                        ito::ParamBase::String};
    for (int t : scalarType)
    {
        ito::ParamBase intParam("scalar", t);
        EXPECT_FALSE(intParam[0].isValid());
    }

    // array types
    ito::int32 intArr[] = { 5, 6, 7 };
    ito::ParamBase intArrayParam("array", ito::ParamBase::IntArray, 3, intArr);
    ito::float64 dblArr[] = { 5.0, 5.6, 7.0 };
    ito::ParamBase dblArrayParam("array", ito::ParamBase::DoubleArray, 3, dblArr);
    ito::complex128 cmplxArr[] = { ito::complex128(5.0, 1.7), ito::complex128(-4.5, 6.7), ito::complex128(7.0, 5.6) };
    ito::ParamBase cmplxArrayParam("array", ito::ParamBase::ComplexArray, 3, cmplxArr);
    ito::ByteArray strList[] = { ito::ByteArray("hello"), ito::ByteArray("no 123"), ito::ByteArray("-123.5") };
    ito::ParamBase strListParam("stringList", ito::ParamBase::StringList, 3, strList);

    ito::ParamBase params[] = { intArrayParam, dblArrayParam, cmplxArrayParam };

    for (const auto &p : params)
    {
        ito::ParamBase p0 = p[0];
        EXPECT_TRUE(p0.isValid());
        EXPECT_STREQ(p0.getName(), "array[0]");
        EXPECT_DOUBLE_EQ(p0.getVal<ito::float64>(), 5);

        ito::ParamBase p2 = p[2];
        EXPECT_TRUE(p2.isValid());
        EXPECT_STREQ(p2.getName(), "array[2]");
        EXPECT_DOUBLE_EQ(p2.getVal<ito::float64>(), 7);

        ito::ParamBase pm1 = p[-1];
        EXPECT_FALSE(pm1.isValid());

        ito::ParamBase p3 = p[3];
        EXPECT_FALSE(p3.isValid());
    }

    // string list
    {
        ito::ParamBase p0 = strListParam[0];
        EXPECT_TRUE(p0.isValid());
        EXPECT_STREQ(p0.getName(), "stringList[0]");
        EXPECT_STREQ(p0.getVal<const char*>(), strList[0].data());

        ito::ParamBase p2 = strListParam[2];
        EXPECT_TRUE(p2.isValid());
        EXPECT_STREQ(p2.getName(), "stringList[2]");
        EXPECT_STREQ(p2.getVal<const char*>(), strList[2].data());

        ito::ParamBase pm1 = strListParam[-1];
        EXPECT_FALSE(pm1.isValid());

        ito::ParamBase p3 = strListParam[3];
        EXPECT_FALSE(p3.isValid());
    }
}

TEST(ParamTest, ParamCopyOperator)
{
    char charArr[] = { -127 , 0, 127 };
    ito::int32 intArr[] = { 5, -6, 7 };
    ito::float64 dblArr[] = { 5.0, 5.6, -7.0 };
    ito::complex128 cmplxArr[] = { ito::complex128(5.0, 1.7), ito::complex128(-4.5, 6.7), ito::complex128(7.0, 5.6) };
    ito::ByteArray strList[] = { ito::ByteArray("hello"), ito::ByteArray("no 123"), ito::ByteArray("-123.5") };

    ito::ParamBase params[] = {
        ito::ParamBase("char", ito::ParamBase::Char, 5),
        ito::ParamBase("int", ito::ParamBase::Int, -5),
        ito::ParamBase("double", ito::ParamBase::Double, -5.78),
        ito::ParamBase("complex", ito::ParamBase::Complex, ito::complex128(5, -3.2)),
        ito::ParamBase("charArray", ito::ParamBase::CharArray, 3, charArr),
        ito::ParamBase("intArray", ito::ParamBase::IntArray, 3, intArr),
        ito::ParamBase("doubleArray", ito::ParamBase::DoubleArray, 3, dblArr),
        ito::ParamBase("complexArray", ito::ParamBase::ComplexArray, 3, cmplxArr),
        ito::ParamBase("string", ito::ParamBase::String, "test string 123"),
        ito::ParamBase("stringList", ito::ParamBase::StringList, 3, strList),
        ito::ParamBase("hwref", ito::ParamBase::HWRef, nullptr)
    };

    for (const auto &p1 : params)
    {
        ito::ParamBase p2(p1);

        EXPECT_TRUE(p1 == p2) << p1.getName();

        ito::ParamBase p3;
        p3 = p1;

        EXPECT_TRUE(p1 == p3) << p1.getName();
    }
}

TEST(ParamTest, ParamAssignmentOperator)
{
    char charArr[] = { -127 , 0, 127 };
    ito::int32 intArr[] = { 5, -6, 7 };
    ito::float64 dblArr[] = { 5.0, 5.6, -7.0 };
    ito::complex128 cmplxArr[] = { ito::complex128(5.0, 1.7), ito::complex128(-4.5, 6.7), ito::complex128(7.0, 5.6) };
    ito::ByteArray strList[] = { ito::ByteArray("hello"), ito::ByteArray("no 123"), ito::ByteArray("-123.5") };

    ito::ParamBase param("stringList", ito::ParamBase::StringList, 3, strList);
    EXPECT_EQ(param.getVal<const ito::ByteArray*>()[1], strList[1]);

    param = ito::ParamBase("intArray", ito::ParamBase::IntArray, 3, intArr);
    EXPECT_EQ(param.getVal<const ito::int32*>()[2], intArr[2]);

    param = ito::ParamBase("string", ito::ParamBase::String, "hello");
    EXPECT_STREQ(param.getVal<const char*>(), "hello");
}


TEST(ParamTest, ParamIndexingOperator)
{
    // scalar types
    int scalarType[] = { ito::ParamBase::Char, ito::ParamBase::Int, ito::ParamBase::Double, ito::ParamBase::Complex,
                        ito::ParamBase::String };
    for (int t : scalarType)
    {
        ito::Param intParam("scalar", t);
        EXPECT_FALSE(intParam[0].isValid());
    }

    // array types
    ito::int32 intArr[] = { 5, 6, 7 };
    ito::Param intArrayParam("array", ito::ParamBase::IntArray, 3, intArr, "info");
    ito::float64 dblArr[] = { 5.0, 5.6, 7.0 };
    ito::Param dblArrayParam("array", ito::ParamBase::DoubleArray, 3, dblArr, "info");
    ito::complex128 cmplxArr[] = { ito::complex128(5.0, 1.7), ito::complex128(-4.5, 6.7), ito::complex128(7.0, 5.6) };
    ito::Param cmplxArrayParam("array", ito::ParamBase::ComplexArray, 3, cmplxArr, "info");

    ito::Param params[] = { intArrayParam, dblArrayParam, cmplxArrayParam };

    for (auto p : params)
    {
        ito::Param p0 = p[0];
        EXPECT_TRUE(p0.isValid());
        EXPECT_STREQ(p0.getName(), "array[0]");
        EXPECT_DOUBLE_EQ(p0.getVal<ito::float64>(), 5);
        EXPECT_STREQ(p0.getInfo(), "info");
        EXPECT_EQ(p0.getMeta(), nullptr);

        ito::Param p2 = p[2];
        EXPECT_TRUE(p2.isValid());
        EXPECT_STREQ(p2.getName(), "array[2]");
        EXPECT_DOUBLE_EQ(p2.getVal<ito::float64>(), 7);
        EXPECT_STREQ(p2.getInfo(), "info");
        EXPECT_EQ(p2.getMeta(), nullptr);

        ito::Param pm1 = p[-1];
        EXPECT_FALSE(pm1.isValid());

        ito::Param p3 = p[3];
        EXPECT_FALSE(p3.isValid());

    }

    intArrayParam.setMeta(new ito::IntArrayMeta(0, 100, 2, "category"), true);
    ito::Param p1 = intArrayParam[1];
    EXPECT_TRUE(p1.getMeta() != nullptr);
    ito::IntMeta *m = p1.getMetaT<ito::IntMeta>();
    EXPECT_TRUE(m != nullptr);
    EXPECT_TRUE(m->getType() == ito::ParamMeta::rttiIntMeta);
    EXPECT_TRUE(m->getMax() == 100);

    dblArrayParam.setMeta(new ito::DoubleArrayMeta(0.0, 100.0, 2.0, "category"), true);
    ito::Param p2 = dblArrayParam[1];
    EXPECT_TRUE(p2.getMeta() != nullptr);
    ito::DoubleMeta *m2 = p2.getMetaT<ito::DoubleMeta>();
    EXPECT_TRUE(m2 != nullptr);
    EXPECT_TRUE(m2->getType() == ito::ParamMeta::rttiDoubleMeta);
    EXPECT_DOUBLE_EQ(m2->getMax(), 100.0);
}
