#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Logarithm_Test_1) {
  tf::tensor A, C;
  A.tf_create(tf_float64, 18, 67);

  A.tensor_of(load_bin("test/data/MatrixLog_Test_1_input.bin", 67 * 18).data());
  auto c = load_bin("test/data/MatrixLog_Test_1_output.bin", 67 * 18);

  C = A.log();
  for (int i = 0; i < 18 * 67; i++) {
    EXPECT_NEAR(C.getData()[i], c[i], 1e-6) << "at: " << i;
  }
}

TEST_F(MathTest, Logarithm_Test_2) {
  tf::tensor A, B, C;
  A.tf_create(tf_float64, 13, 21, 10, 8, 9);

  A.tensor_of(
      load_bin("test/data/MatrixLog_Test_2_input.bin", 9 * 8 * 10 * 21 * 13)
          .data());
  auto c =
      load_bin("test/data/MatrixLog_Test_2_output.bin", 9 * 8 * 10 * 21 * 13);
  {
    tf::graph_context ctx;
    C = A.log();

    ctx.run();

    for (int i = 0; i < 9 * 8 * 10 * 21 * 13; i++) {
      EXPECT_NEAR(C.getData()[i], c[i], 1e-6) << "at: " << i;
    }
  }
}