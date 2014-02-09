#include "byteArray.h"
#include "gtest/gtest.h"


TEST(ByteArrayTest, Constructor)
{
    const char *helloWorldStr = "hello world";
    ito::ByteArray ba1("hello world");

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
    EXPECT_EQ(ba2.empty(),true);
}

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
}
