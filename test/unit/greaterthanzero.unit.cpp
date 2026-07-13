#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, GreaterThanZero_1) {

  tf::tensor A, C;
  A.tf_create(tf_float64, 78, 89);

  A.tensor_of(
      load_bin("test/data/GreaterThanZero_Test_1_input_sample.bin", 78 * 89)
          .data());

  C = A.greater_than_zero();

  auto output =
      load_bin("test/data/GraterThanZeor_Test_1_output_sample.bin", 78 * 89);
  for (int i = 0; i < 78 * 89; i++) {
    EXPECT_NEAR(C.getData()[i], output[i], 1e-6) << "at: " << i;
  }
}