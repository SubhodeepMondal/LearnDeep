#ifndef LINEAR_ALGEBRA_FIXTURES_UNIT_HPP
#define LINEAR_ALGEBRA_FIXTURES_UNIT_HPP
#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

class MathTest : public testing::Test {

protected:
  virtual void SetUp() override;

  virtual void TearDown() override;
};

class FrameworkTest : public testing::Test {
protected:
  virtual void SetUp() override {};
  virtual void TearDown() override {};
};
#endif // LINEAR_ALGEBRA_FIXTURES_UNIT_HPP