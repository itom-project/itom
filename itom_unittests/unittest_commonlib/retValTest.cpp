#include "retVal.h"
#include "gtest/gtest.h"

TEST(RetValTest, Constructor)
{
    ito::RetVal ret1;
    EXPECT_EQ(ret1.containsError(), 0);
    EXPECT_EQ(ret1.containsWarning(), 0);
    EXPECT_EQ(ret1.containsWarningOrError(), 0);
    EXPECT_GT(ret1 == ito::retOk, 0);
    EXPECT_EQ(ret1.errorCode(), 0);
    EXPECT_STREQ(ret1.errorMessage(), "");

    ito::RetVal ret2(ito::retOk);
    EXPECT_EQ(ret2.containsError(), 0);
    EXPECT_EQ(ret2.containsWarning(), 0);
    EXPECT_EQ(ret2.containsWarningOrError(), 0);
    EXPECT_GT(ret2 == ito::retOk, 0);
    EXPECT_EQ(ret2.errorCode(), 0);
    EXPECT_STREQ(ret2.errorMessage(), "");

    ito::RetVal ret3(ito::retWarning);
    EXPECT_EQ(ret3.containsError(), 0);
    EXPECT_GE(ret3.containsWarning(), 1);
    EXPECT_GE(ret3.containsWarningOrError(), 1);
    EXPECT_GT(ret3 == ito::retWarning, 0);
    EXPECT_EQ(ret3.errorCode(), 0);
    EXPECT_STREQ(ret3.errorMessage(), "");

    ito::RetVal ret4(ito::retError);
    EXPECT_GE(ret4.containsError(), 1);
    EXPECT_EQ(ret4.containsWarning(), 0);
    EXPECT_GE(ret4.containsWarningOrError(), 1);
    EXPECT_GT(ret4 == ito::retError, 0);
    EXPECT_EQ(ret4.errorCode(), 0);
    EXPECT_STREQ(ret4.errorMessage(), "");

    ito::RetVal ret5(ito::retError, 5, "test");
    EXPECT_GE(ret5.containsError(), 1);
    EXPECT_EQ(ret5.containsWarning(), 0);
    EXPECT_GE(ret5.containsWarningOrError(), 1);
    EXPECT_GT(ret5 == ito::retError, 0);
    EXPECT_EQ(ret5.errorCode(), 5);
    EXPECT_STREQ(ret5.errorMessage(), "test");
}

TEST(RetValTest, Assignment)
{
    ito::RetVal ret;

    ret = ito::retOk;
    EXPECT_EQ(ret.containsError(), 0);
    EXPECT_EQ(ret.containsWarning(), 0);
    EXPECT_EQ(ret.containsWarningOrError(), 0);
    EXPECT_GT(ret == ito::retOk, 0);
    EXPECT_EQ(ret.errorCode(), 0);
    EXPECT_STREQ(ret.errorMessage(), "");

    ret = ito::RetVal(ito::retOk);
    EXPECT_EQ(ret.containsError(), 0);
    EXPECT_EQ(ret.containsWarning(), 0);
    EXPECT_EQ(ret.containsWarningOrError(), 0);
    EXPECT_GT(ret == ito::retOk, 0);
    EXPECT_EQ(ret.errorCode(), 0);
    EXPECT_STREQ(ret.errorMessage(), "");

    ret = ito::RetVal(ito::retWarning);
    EXPECT_EQ(ret.containsError(), 0);
    EXPECT_GE(ret.containsWarning(), 1);
    EXPECT_GE(ret.containsWarningOrError(), 1);
    EXPECT_GT(ret == ito::retWarning, 0);
    EXPECT_EQ(ret.errorCode(), 0);
    EXPECT_STREQ(ret.errorMessage(), "");

    ret = ito::RetVal(ito::retError);
    EXPECT_GE(ret.containsError(), 1);
    EXPECT_EQ(ret.containsWarning(), 0);
    EXPECT_GE(ret.containsWarningOrError(), 1);
    EXPECT_GT(ret == ito::retError, 0);
    EXPECT_EQ(ret.errorCode(), 0);
    EXPECT_STREQ(ret.errorMessage(), "");

    ret = ito::RetVal((int)ito::retError);
    EXPECT_GE(ret.containsError(), 1);
    EXPECT_EQ(ret.containsWarning(), 0);
    EXPECT_GE(ret.containsWarningOrError(), 1);
    EXPECT_GT(ret == ito::retError, 0);
    EXPECT_EQ(ret.errorCode(), 0);
    EXPECT_STREQ(ret.errorMessage(), "");

    ret = ito::RetVal(ito::retError, 5, "test");
    EXPECT_GE(ret.containsError(), 1);
    EXPECT_EQ(ret.containsWarning(), 0);
    EXPECT_GE(ret.containsWarningOrError(), 1);
    EXPECT_GT(ret == ito::retError, 0);
    EXPECT_EQ(ret.errorCode(), 5);
    EXPECT_STREQ(ret.errorMessage(), "test");
}

TEST(RetValTest, Message)
{
    ito::RetVal ret;
    EXPECT_STREQ(ret.errorMessage(), "");
    EXPECT_FALSE(ret.hasErrorMessage());

    ret.appendRetMessage("appended");
    EXPECT_STREQ(ret.errorMessage(), "appended");
    EXPECT_TRUE(ret.hasErrorMessage());

    ret = ito::RetVal(ito::retOk, 0, "test");
    EXPECT_STREQ(ret.errorMessage(), "test");
    EXPECT_TRUE(ret.hasErrorMessage());

    ret = ito::RetVal(ito::retError, 0, "error");
    EXPECT_STREQ(ret.errorMessage(), "error");
    EXPECT_TRUE(ret.hasErrorMessage());

    ret.appendRetMessage(" appended");
    EXPECT_STREQ(ret.errorMessage(), "error appended");
    EXPECT_TRUE(ret.hasErrorMessage());
}

TEST(RetValTest, Format)
{
    ito::RetVal ret = ito::RetVal::format(ito::retWarning, 0, "message '%s' contains %i numbers", "test", 0);
    EXPECT_STREQ(ret.errorMessage(), "message 'test' contains 0 numbers");

    ret = ito::RetVal::format(ito::retWarning, 0, "no message");
    EXPECT_STREQ(ret.errorMessage(), "no message");
}

TEST(RetValTest, PlusOperation)
{
    ito::RetVal ret;
    EXPECT_EQ(ret.containsWarningOrError(), 0);

    ret += ito::RetVal(ito::retWarning, 1, "");
    EXPECT_GE(ret.containsWarning(), 1);
    EXPECT_EQ(ret.errorCode(), 1);

    ret += ito::RetVal(ito::retWarning, 2, "warning");
    EXPECT_GE(ret.containsWarning(), 1);
    EXPECT_EQ(ret.errorCode(), 1); // first of same level wins

    ret += ito::RetVal(ito::retError, 3, "error");
    EXPECT_EQ(ret.containsWarning(), 0);
    EXPECT_GE(ret.containsError(), 1);
    EXPECT_EQ(ret.errorCode(), 3);

    ret = ito::RetVal(ito::retWarning, 3, "warning") + ito::RetVal(ito::retError, 4, "error");
    EXPECT_EQ(ret.containsWarning(), 0);
    EXPECT_GE(ret.containsError(), 1);
    EXPECT_EQ(ret.errorCode(), 4);
    EXPECT_STREQ(ret.errorMessage(), "error");

    ret = ito::RetVal(ito::retError, 4, "error") + ito::RetVal(ito::retWarning, 3, "warning");
    EXPECT_EQ(ret.containsWarning(), 0);
    EXPECT_GE(ret.containsError(), 1);
    EXPECT_EQ(ret.errorCode(), 4);
    EXPECT_STREQ(ret.errorMessage(), "error");

    ret += ito::retOk;
    EXPECT_EQ(ret.containsWarning(), 0);
    EXPECT_GE(ret.containsError(), 1);
    EXPECT_EQ(ret.errorCode(), 4);
    EXPECT_STREQ(ret.errorMessage(), "error");
}

TEST(RetValTest, Comparison)
{
    ito::RetVal ret1(ito::retOk);
    ito::RetVal ret2(ito::retWarning);
    ito::RetVal ret3(ito::retWarning);
    ito::RetVal ret4(ito::retError, 0, "error1");
    ito::RetVal ret5(ito::retError, 1, "error2");

    EXPECT_GT(ret1 != ret2, 0);
    EXPECT_GT(ret1 != ret3, 0);
    EXPECT_GT(ret1 != ret4, 0);
    EXPECT_GT(ret1 != ret5, 0);

    EXPECT_GT(ret2 != ret1, 0);
    EXPECT_GT(ret2 == ret3, 0);
    EXPECT_GT(ret2 != ret4, 0);
    EXPECT_GT(ret2 != ret5, 0);

    EXPECT_GT(ret3 != ret1, 0);
    EXPECT_GT(ret3 == ret2, 0);
    EXPECT_GT(ret3 != ret4, 0);
    EXPECT_GT(ret3 != ret5, 0);

    EXPECT_GT(ret4 != ret1, 0);
    EXPECT_GT(ret4 != ret2, 0);
    EXPECT_GT(ret4 != ret3, 0);
    EXPECT_GT(ret4 == ret5, 0);

    EXPECT_GT(ret5 != ret1, 0);
    EXPECT_GT(ret5 != ret2, 0);
    EXPECT_GT(ret5 != ret3, 0);
    EXPECT_GT(ret5 == ret4, 0);
}
