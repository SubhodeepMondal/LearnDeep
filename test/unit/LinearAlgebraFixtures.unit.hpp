#ifndef LINEAR_ALGEBRA_FIXTURES_UNIT_HPP
#define LINEAR_ALGEBRA_FIXTURES_UNIT_HPP
#include <gtest/gtest.h>
#include <tensor.h>

class MathTest : public testing::Test {

protected:
  virtual void SetUp() override;

  virtual void TearDown() override;
};

#endif // LINEAR_ALGEBRA_FIXTURES_UNIT_HPP