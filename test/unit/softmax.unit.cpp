#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include "LinearAlgebraFixtures.unit.hpp"

namespace {

void expect_near_tensor(const tf::tensor &actual,
                        const std::vector<std::float64_t> &expected) {
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(actual.getData()[i], expected[i], 1e-6) << "at: " << i;
  }
}

} // namespace

TEST_F(MathTest, Softmax_Test_1_Eager_2D_Axis0) {
  constexpr size_t tensor_size = 167 * 544;

  tf::tensor A, C;
  A.tf_create(tf_float64, 167, 544);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_1_input.bin", tensor_size).data());

  C = A.softmax(0);

  const std::vector<std::float64_t> output =
      load_bin("test/data/Softmax_Test_1_output.bin", tensor_size);
  expect_near_tensor(C, output);
}

TEST_F(MathTest, Softmax_Test_1_Graph_2D_Axis0) {
  constexpr size_t tensor_size = 167 * 544;

  tf::tensor A, C;
  A.tf_create(tf_float64, 167, 544);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_1_input.bin", tensor_size).data());

  {
    tf::graph_context ctx;

    C = A.softmax(0);
    ctx.run();

    const std::vector<std::float64_t> output =
        load_bin("test/data/Softmax_Test_1_output.bin", tensor_size);
    expect_near_tensor(C, output);
  }
}

TEST_F(MathTest, Softmax_Test_2_Eager_3D_Axis1) {
  constexpr size_t tensor_size = 78 * 37 * 19;

  tf::tensor A, C;
  A.tf_create(tf_float64, 78, 37, 19);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_2_input.bin", tensor_size).data());

  C = A.softmax(1);

  const std::vector<std::float64_t> output =
      load_bin("test/data/Softmax_Test_2_output.bin", tensor_size);
  expect_near_tensor(C, output);
}

TEST_F(MathTest, Softmax_Test_2_Graph_3D_Axis1) {
  constexpr size_t tensor_size = 78 * 37 * 19;

  tf::tensor A, C;
  A.tf_create(tf_float64, 78, 37, 19);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_2_input.bin", tensor_size).data());

  {
    tf::graph_context ctx;

    C = A.softmax(1);
    ctx.run();

    const std::vector<std::float64_t> output =
        load_bin("test/data/Softmax_Test_2_output.bin", tensor_size);
    expect_near_tensor(C, output);
  }
}

TEST_F(MathTest, Softmax_Test_3_Eager_5D_Axis4) {
  constexpr size_t tensor_size = 7 * 11 * 112 * 5 * 6;

  tf::tensor A, C;
  A.tf_create(tf_float64, 7, 11, 112, 5, 6);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_3_input.bin", tensor_size).data());

  C = A.softmax(4);

  const std::vector<std::float64_t> output =
      load_bin("test/data/Softmax_Test_3_output.bin", tensor_size);
  expect_near_tensor(C, output);
}

TEST_F(MathTest, Softmax_Test_3_Graph_5D_Axis4) {
  constexpr size_t tensor_size = 7 * 11 * 112 * 5 * 6;

  tf::tensor A, C;
  A.tf_create(tf_float64, 7, 11, 112, 5, 6);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_3_input.bin", tensor_size).data());

  {
    tf::graph_context ctx;

    C = A.softmax(4);
    ctx.run();

    const std::vector<std::float64_t> output =
        load_bin("test/data/Softmax_Test_3_output.bin", tensor_size);
    expect_near_tensor(C, output);
  }
}
