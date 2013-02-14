// This sample teaches how to reuse a test fixture in multiple test
// cases by deriving sub-fixtures from it.
//
// When you define a test fixture, you specify the name of the test
// case that will use this fixture.  Therefore, a test fixture can
// be used by only one test case.
//
// Sometimes, more than one test cases may want to use the same or
// slightly different test fixtures.  For example, you may want to
// make sure that all tests for a GUI library don't leak important
// system resources like fonts and brushes.  In Google Test, you do
// this by putting the shared logic in a super (as in "super class")
// test fixture, and then have each test case use a fixture derived
// from this super fixture.

#include <limits.h>
#include <time.h>
#include "sample3-inl.h"
#include "gtest/gtest.h"
#include "sample1.h"

// In this sample, we want to ensure that every test finishes within
// ~5 seconds.  If a test takes longer to run, we consider it a
// failure.
//
// We put the code for timing a test in a test fixture called
// "QuickTest".  QuickTest is intended to be the super fixture that
// other fixtures derive from, therefore there is no test case with
// the name "QuickTest".  This is OK.
//
// Later, we will derive multiple test fixtures from QuickTest.
class QuickTest : public testing::Test {
 protected:
  // Remember that SetUp() is run immediately before a test starts.
  // This is a good place to record the start time.
  virtual void SetUp() {
    start_time_ = time(NULL);
  }

  // TearDown() is invoked immediately after a test finishes.  Here we
  // check if the test was too slow.
  virtual void TearDown() {
    // Gets the time when the test finishes
    const time_t end_time = time(NULL);

    // Asserts that the test took no more than ~5 seconds.  Did you
    // know that you can use assertions in SetUp() and TearDown() as
    // well?
    EXPECT_TRUE(end_time - start_time_ <= 0) << "The test took too long.";
  }

  // The UTC time (in seconds) when the test starts
  time_t start_time_;
};


// We derive a fixture named IntegerFunctionTest from the QuickTest
// fixture.  All tests using this fixture will be automatically
// required to be quick.
class IntegerFunctionTest : public QuickTest {
  // We don't need any more logic than already in the QuickTest fixture.
  // Therefore the body is empty.
};


// Now we can write tests in the IntegerFunctionTest test case.

// Tests Factorial()
TEST_F(IntegerFunctionTest, Factorial) {
  // Tests factorial of negative numbers.
  EXPECT_EQ(1, Factorial(-5));
  EXPECT_EQ(1, Factorial(-1));
  EXPECT_GT(Factorial(-10), 0);

  // Tests factorial of 0.
  EXPECT_EQ(1, Factorial(0));

  // Tests factorial of positive numbers.
  EXPECT_EQ(1, Factorial(1));
  EXPECT_EQ(2, Factorial(2));
  EXPECT_EQ(6, Factorial(3));
  EXPECT_EQ(40320, Factorial(8));
}


// Tests IsPrime()
TEST_F(IntegerFunctionTest, IsPrime) {
  // Tests negative input.
  EXPECT_FALSE(IsPrime(-1));
  EXPECT_FALSE(IsPrime(-2));
  EXPECT_FALSE(IsPrime(INT_MIN));

  // Tests some trivial cases.
  EXPECT_FALSE(IsPrime(0));
  EXPECT_FALSE(IsPrime(1));
  EXPECT_TRUE(IsPrime(2));
  EXPECT_TRUE(IsPrime(3));

  // Tests positive input.
  EXPECT_FALSE(IsPrime(4));
  EXPECT_TRUE(IsPrime(5));
  EXPECT_FALSE(IsPrime(6));
  EXPECT_TRUE(IsPrime(23));
}


// The next test case (named "QueueTest") also needs to be quick, so
// we derive another fixture from QuickTest.
//
// The QueueTest test fixture has some logic and shared objects in
// addition to what's in QuickTest already.  We define the additional
// stuff inside the body of the test fixture, as usual.
class QueueTest : public QuickTest {
 protected:
  virtual void SetUp() {
    // First, we need to set up the super fixture (QuickTest).
    QuickTest::SetUp();

    // Second, some additional setup for this fixture.
    q1_.Enqueue(1);
    q2_.Enqueue(2);
    q2_.Enqueue(3);
  }

  // By default, TearDown() inherits the behavior of
  // QuickTest::TearDown().  As we have no additional cleaning work
  // for QueueTest, we omit it here.
  //
  // virtual void TearDown() {
  //   QuickTest::TearDown();
  // }

  Queue<int> q0_;
  Queue<int> q1_;
  Queue<int> q2_;
};


// Now, let's write tests using the QueueTest fixture.

// Tests the default constructor.
TEST_F(QueueTest, DefaultConstructor) {
  EXPECT_EQ(0u, q0_.Size());
}

// Tests Dequeue().
TEST_F(QueueTest, Dequeue) {
  int* n = q0_.Dequeue();
  EXPECT_TRUE(n == NULL);

  n = q1_.Dequeue();
  EXPECT_TRUE(n != NULL);
  EXPECT_EQ(1, *n);
  EXPECT_EQ(0u, q1_.Size());
  delete n;

  n = q2_.Dequeue();
  EXPECT_TRUE(n != NULL);
  EXPECT_EQ(2, *n);
  EXPECT_EQ(1u, q2_.Size());
  delete n;
}
