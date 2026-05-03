#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Relu_Test_1) {

  tf::tensor A, C;
  A.tf_create(tf_float64, 63, 32);

  A.tensor_of(
      load_bin("test/data/Relu_Test_1_input_sample.bin", 63 * 32).data());

  C = A.relu();

  auto output = load_bin("test/data/Relu_Test_1_output_sample.bin", 63 * 32);
  for (int i = 0; i < 63 * 32; i++) {
    EXPECT_NEAR(C.getData()[i], output[i], 1e-6) << "at: " << i;
  }
}

TEST_F(MathTest, Relu_Test_2) {

  tf::tensor A, C;
  A.tf_create(tf_float64, 51, 48, 36, 72);

  A.tensor_of(
      load_bin("test/data/Relu_Test_2_input_sample.bin", 51 * 48 * 36 * 72)
          .data());

  auto output =
      load_bin("test/data/Relu_Test_2_output_sample.bin", 51 * 48 * 36 * 72);
  {
    tf::graph_context ctx;

    C = A.relu();

    ctx.run();

    auto output =
        load_bin("test/data/Relu_Test_2_output_sample.bin", 51 * 48 * 36 * 72);
    for (int i = 0; i < 51 * 48 * 36; i++) {
      EXPECT_NEAR(C.getData()[i], output[i], 1e-6);
    }
  }
}
