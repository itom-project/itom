#include "param.h"
#include "byteArray.h"
#include "gtest/gtest.h"

void checkParamBase(ito::ParamBase pb, const char* name, const ito::ParamBase::Type type, const bool ckName, const bool ckType)
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

void checkInt(ito::ParamBase pb, const char* name, const ito::ParamBase::Type type, const int value, const bool ckName, const bool ckType, const bool ckValue)
{
    checkParamBase(pb, name, type, ckName, ckType);

    if (ckValue)
    {
        EXPECT_EQ(pb.getVal<int>(), value);

        int addIntValue = 1;
        int testVal = value + addIntValue;
        int pbVal = pb.getVal<int>() + addIntValue;
        pb.setVal<int>(pbVal);
        EXPECT_EQ(pb.getVal<int>(), testVal);
    }
}

void checkDouble(ito::ParamBase pb, const char* name, const ito::ParamBase::Type type, const double value, const bool ckName, const bool ckType, const bool ckValue)
{
    checkParamBase(pb, name, type, ckName, ckType);

    if (ckValue)
    {
        EXPECT_EQ(pb.getVal<double>(), value);

        double addDoubleValue = 1.112;
        double testVal = value + addDoubleValue;
        double pbVal = pb.getVal<double>() + addDoubleValue;
        pb.setVal<int>(pbVal);
        EXPECT_EQ(pb.getVal<double>(), testVal);
    }
}


TEST(ParamTest, Constructor)
{
    ito::ByteArray paramBaInt("ByteArray");

    // integer
    const char *paramName = "pbInt";
    ito::ParamBase::Type paramType = ito::ParamBase::Int;
    int paramValueInt = 42;

    ito::ParamBase pbI1(paramName);
    checkInt(pbI1, paramName, paramType, paramValueInt, true, false, false);

    ito::ParamBase pbI2(paramName, paramType);
    checkInt(pbI2, paramName, paramType, paramValueInt, true, true, false);

    ito::ParamBase pbI3(paramName, paramType, paramValueInt);
    checkInt(pbI3, paramName, paramType, paramValueInt, true, true, true);

    ito::ParamBase pbI4(paramBaInt, ito::ParamBase::IntArray, paramValueInt);
    checkInt(pbI4, paramName, paramType, paramValueInt, true, true, true);

    // double
    paramName = "pbDouble";
    paramType = ito::ParamBase::Double;
    int paramValueDouble = 1.321;

    ito::ParamBase pbD1(paramName);
    checkDouble(pbD1, paramName, paramType, paramValueDouble, true, false, false);

    ito::ParamBase pbD2(paramName, paramType);
    checkDouble(pbD2, paramName, paramType, paramValueDouble, true, true, false);

    ito::ParamBase pbD3(paramName, paramType, paramValueDouble);
    checkDouble(pbD3, paramName, paramType, paramValueDouble, true, true, true);

    ito::ParamBase pbD4(paramBaInt, ito::ParamBase::DoubleArray, paramValueDouble);
    checkDouble(pbD4, paramName, paramType, paramValueDouble, true, true, true);

/*    ito::ParamBase pb02("pbInt02", );

        ParamBase(const char *name);                                                               // type-less ParamBase with name only
        ParamBase(const char *name, const uint32 type);                                               // constructor with type and name
        ParamBase(const char *name, const uint32 type, const char *val);                              // constructor with name and type, char val
        ParamBase(const char *name, const uint32 type, const double val);                             // constructor with name and type, double val and optional info
        ParamBase(const char *name, const uint32 type, const int val);                                // constructor with name and type, int val and optional info
        ParamBase(const ByteArray &name, const uint32 type, const char *val);                              // constructor with name and type, char val
        ParamBase(const ByteArray &name, const uint32 type, const double val);                             // constructor with name and type, double val and optional info
        ParamBase(const ByteArray &name, const uint32 type, const int val);                                // constructor with name and type, int val and optional info
        ParamBase(const char *name, const uint32 type, const unsigned int size, const char *values);  // array constructor with name and type, size and array
        ParamBase(const char *name, const uint32 type, const unsigned int size, const int *values);   // array constructor with name and type, size and array
        ParamBase(const char *name, const uint32 type, const unsigned int size, const double *values);// array constructor with name and type, size and array
    EXPECT_EQ(ba1, helloWorldStr);
    EXPECT_EQ(ba1.size(), strlen(helloWorldStr));
    EXPECT_EQ(ba1.length(), strlen(helloWorldStr));
    EXPECT_STREQ(ba1.data(), helloWorldStr);
    EXPECT_STREQ(helloWorldStr, ba1.data());
    EXPECT_STRNE("no", ba1.data());
    EXPECT_STRNE(ba1.data(), "no");

    ito::ByteArray ba2;
    EXPECT_EQ(ba2.size(),0);
    EXPECT_EQ(ba2.length(),0);
    EXPECT_STREQ(ba2.data(), "\0");
    EXPECT_STREQ("\0", ba2.data());
    EXPECT_STRNE("no", ba2.data());
    EXPECT_STRNE(ba2.data(), "no");

    EXPECT_NE(ba1,ba2);
    EXPECT_EQ(ba1.empty(),false);
    EXPECT_EQ(ba2.empty(),true);*/
}

/*
TEST(ByteArrayTest, CopyConstructor)
{
    ito::ByteArray ba1("hello world");
    ito::ByteArray ba2(ba1);
    ito::ByteArray *ba3 = new ito::ByteArray(ba1);

    EXPECT_EQ(ba1, ba2);
    EXPECT_EQ(ba1.size(), ba2.size());
    EXPECT_EQ(ba1.length(), ba2.length());
    EXPECT_EQ(ba1.empty(),false);
    EXPECT_EQ(ba2.empty(),false);

    EXPECT_EQ(ba1, *ba3);
    EXPECT_EQ(ba1.size(), ba3->size());
    EXPECT_EQ(ba1.length(), ba3->length());
    EXPECT_EQ(ba3->empty(),false);

    delete ba3;
}

TEST(ByteArrayTest, Assignment)
{
    ito::ByteArray ba1("hello world");
    ito::ByteArray ba2 = ba1;
    ito::ByteArray ba3 = "test";

    EXPECT_EQ(ba1, ba2);
    EXPECT_EQ(ba1.size(), ba2.size());
    EXPECT_EQ(ba1.length(), ba2.length());
    EXPECT_EQ(ba1.empty(),false);
    EXPECT_EQ(ba2.empty(),false);
    EXPECT_EQ(ba3, "test");
    EXPECT_STREQ(ba3.data(), "test");
    EXPECT_EQ(ba3.size(), 4);
}

TEST(ByteArrayTest, Accessing)
{
    ito::ByteArray ba1("hello world");

    EXPECT_EQ(ba1[0], 'h');
    EXPECT_EQ(ba1[2], 'l');

    int i = 0;
    unsigned j = 0;

    EXPECT_EQ(ba1[i], 'h');
    EXPECT_EQ(ba1[j], 'h');
}

TEST(ByteArrayTest, Append)
{
    ito::ByteArray ba1("hello world");
    ba1.append(".");
    ito::ByteArray ba2(ba1);
    ba2.append(",");

    EXPECT_EQ(ba1, "hello world.");
    EXPECT_EQ(ba2, "hello world.,");

    //check short appends (realloc without new starting pointer)
    for (int i = 0; i < 10; i++)
    {
        ba2.append("x");
        EXPECT_EQ(ba2.size(), ba1.size() + 2 + i);
    }

    ba2 = "start";
    const char *a = "hello world hello world";
    //check longer appends (realloc with possibly new starting pointer)
    for (int i = 0; i < 100; i++)
    {
        ba2.append(a);
        EXPECT_EQ(ba2.size(), 5 + (i+1)*strlen(a));
    }

    EXPECT_EQ(ba2[5], 'h');
}*/
