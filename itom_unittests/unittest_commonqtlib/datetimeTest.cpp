#include "numeric.h"
#include "gtest/gtest.h"

#include "helperDatetime.h"
#include "typeDefs.h"

using namespace ito;

class DatetimeTest : public testing::Test
{
protected:
    // You can define per-test set-up logic as usual.
    void SetUp() override
    {
    }

};

TEST_F(DatetimeTest, TimeDeltaToDSU)
{
    {
        int64 days_ = 9999;
        int64 seconds_ = 3600 * 23;
        int64 useconds_ = 222 + 777 * 1000;
        TimeDelta td(useconds_ + (seconds_ + days_ * 24L * 3600L) * 1000000L);

        int d, s, u;
        timedelta::toDSU(td, d, s, u);
        EXPECT_EQ(d, days_);
        EXPECT_EQ(s, seconds_);
        EXPECT_EQ(u, useconds_);
    }

    {
        int64 days_ = -1;
        int64 seconds_ = -2;
        int64 useconds_ = -3;
        TimeDelta td(useconds_ + (seconds_ + days_ * 24L * 3600L) * 1000000L);

        int d, s, u;
        timedelta::toDSU(td, d, s, u);
        EXPECT_EQ(d, days_);
        EXPECT_EQ(s, seconds_);
        EXPECT_EQ(u, useconds_);
    }
}


