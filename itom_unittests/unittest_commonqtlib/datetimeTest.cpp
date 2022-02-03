#include "numeric.h"
#include "gtest/gtest.h"

#include "helperDatetime.h"
#include "typeDefs.h"

#include <qdatetime.h>

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

TEST_F(DatetimeTest, DateTimeFromTo)
{
    int components[7];
    const int numRows = 2;

    int desired[numRows][8] = {
        // each row is year, month, day, hour, minute, second, usecond, utcoffset
        {1930, 1, 1, 10, 11, 12, 13,-3600},
        {2023, 2, 5, 0, 0, 0, 1, 3600}
    };

    for (int i = 0; i < numRows; ++i)
    {
        ito::DateTime date1 = datetime::fromYMDHMSU(
            desired[i][0],
            desired[i][1],
            desired[i][2],
            desired[i][3],
            desired[i][4],
            desired[i][5],
            desired[i][6],
            desired[i][7]);

        datetime::toYMDHMSU(
            date1,
            components[0],
            components[1],
            components[2],
            components[3],
            components[4],
            components[5],
            components[6]);

        for (int j = 0; j < 7; ++j)
        {
            EXPECT_EQ(components[j], desired[i][j]);
        }
        
        EXPECT_EQ(desired[i][7], date1.utcOffset);
    }
}
