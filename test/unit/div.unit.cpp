#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Division_Test_1) {
  tf::tensor A, B, C;
  A.tf_create(tf_float64, 32, 11);
  B.tf_create(tf_float64, 32, 11);

  A.tensor_of(
      load_bin("test/data/MatrixDiv_Test_1_input.bin", 11 * 32).data());
  B.tensor_of(load_bin("test/data/MatrixDiv_Test_1_input_b.bin", 11 * 32).data());
  auto c = load_bin("test/data/MatrixDiv_Test_1_output.bin", 11 * 32);

  C = A.div(B);
  for (int i = 0; i < 11 * 32; i++) {
    EXPECT_NEAR(C.getData()[i], c[i], 1e-6) << "at: " <<  i;
  }
}

TEST_F(MathTest, Division_Test_2) {
  tf::tensor A, B, C;
  A.tf_create(tf_float64, 37, 13, 32, 11);
  B.tf_create(tf_float64, 37, 1, 32, 11);

  A.tensor_of(load_bin("test/data/MatrixDiv_Test_2_input.bin", 37 * 13 * 32 * 11).data());
  B.tensor_of(
      load_bin("test/data/MatrixDiv_Test_2_input_b.bin", 37 * 32 * 11)
          .data());
  auto c = load_bin("test/data/MatrixDiv_Test_2_output.bin", 37 * 13 * 32 * 11);

  C = A.div(B);
  for (int i = 0; i < 37 * 13 * 32 * 11; i++) {
    EXPECT_NEAR(C.getData()[i], c[i], 1e-6) << "at: " << i;
  }
}