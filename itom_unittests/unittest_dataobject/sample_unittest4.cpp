
#include "gtest/gtest.h"
#include "sample4.h"

// Tests the Increment() method.
TEST(Counter, Increment) {
  Counter c;

  // EXPECT_EQ() evaluates its arguments exactly once, so they
  // can have side effects.

  EXPECT_EQ(0, c.Increment());
  EXPECT_EQ(1, c.Increment());
  EXPECT_EQ(2, c.Increment());
}
