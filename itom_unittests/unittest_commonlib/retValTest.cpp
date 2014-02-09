#include "retVal.h"
#include "gtest/gtest.h"


TEST(RetValTest, Constructor)
{
    ito::RetVal ret1;
    EXPECT_EQ(ret1.containsError(),0);
    EXPECT_EQ(ret1.containsWarning(),0);
    EXPECT_EQ(ret1.containsWarningOrError(),0);
    EXPECT_TRUE(ret1 == ito::retOk);
    EXPECT_EQ(ret1.errorCode(),0);
    EXPECT_STREQ(ret1.errorMessage(), "");

    ito::RetVal ret2(ito::retOk);
    EXPECT_EQ(ret2.containsError(),0);
    EXPECT_EQ(ret2.containsWarning(),0);
    EXPECT_EQ(ret2.containsWarningOrError(),0);
    EXPECT_TRUE(ret2 == ito::retOk);
    EXPECT_EQ(ret2.errorCode(),0);
    EXPECT_STREQ(ret2.errorMessage(), "");

    ito::RetVal ret3(ito::retWarning);
    EXPECT_EQ(ret3.containsError(),0);
    EXPECT_GE(ret3.containsWarning(),1);
    EXPECT_GE(ret3.containsWarningOrError(),1);
    EXPECT_TRUE(ret3 == ito::retWarning);
    EXPECT_EQ(ret3.errorCode(),0);
    EXPECT_STREQ(ret3.errorMessage(), "");

    ito::RetVal ret4(ito::retError);
    EXPECT_GE(ret4.containsError(),1);
    EXPECT_EQ(ret4.containsWarning(),0);
    EXPECT_GE(ret4.containsWarningOrError(),1);
    EXPECT_TRUE(ret4 == ito::retError);
    EXPECT_EQ(ret4.errorCode(),0);
    EXPECT_STREQ(ret4.errorMessage(), "");

    ito::RetVal ret5(ito::retError, 5, "test");
    EXPECT_GE(ret5.containsError(),1);
    EXPECT_EQ(ret5.containsWarning(),0);
    EXPECT_GE(ret5.containsWarningOrError(),1);
    EXPECT_TRUE(ret5 == ito::retError);
    EXPECT_EQ(ret5.errorCode(),5);
    EXPECT_STREQ(ret5.errorMessage(), "test");
}

