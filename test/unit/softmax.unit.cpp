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
  A.tf_create(tf_float64, 544, 167);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_1_input.bin", tensor_size).data());

  C = A.softmax(1);

  const std::vector<std::float64_t> output =
      load_bin("test/data/Softmax_Test_1_output.bin", tensor_size);
  expect_near_tensor(C, output);
}

TEST_F(MathTest, Softmax_Test_1_Graph_2D_Axis0) {
  constexpr size_t tensor_size = 167 * 544;

  tf::tensor A, C;
  A.tf_create(tf_float64, 544, 167);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_1_input.bin", tensor_size).data());

  {
    tf::graph_context ctx;

    C = A.softmax(1);
    ctx.run();

    const std::vector<std::float64_t> output =
        load_bin("test/data/Softmax_Test_1_output.bin", tensor_size);
    expect_near_tensor(C, output);
  }
}

TEST_F(MathTest, Softmax_Test_2_Eager_3D_Axis1) {
  constexpr size_t tensor_size = 78 * 37 * 19;

  tf::tensor A, C;
  A.tf_create(tf_float64, 19, 37, 78);
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
  A.tf_create(tf_float64, 19, 37, 78);
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
  A.tf_create(tf_float64, 6, 5, 112, 11, 7);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_3_input.bin", tensor_size).data());

  C = A.softmax(0);

  const std::vector<std::float64_t> output =
      load_bin("test/data/Softmax_Test_3_output.bin", tensor_size);
  expect_near_tensor(C, output);
}

TEST_F(MathTest, Softmax_Test_3_Graph_5D_Axis4) {
  constexpr size_t tensor_size = 7 * 11 * 112 * 5 * 6;

  tf::tensor A, C;
  A.tf_create(tf_float64, 6, 5, 112, 11, 7);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_3_input.bin", tensor_size).data());

  {
    tf::graph_context ctx;

    C = A.softmax(0);
    ctx.run();

    const std::vector<std::float64_t> output =
        load_bin("test/data/Softmax_Test_3_output.bin", tensor_size);
    expect_near_tensor(C, output);
  }
}

TEST_F(MathTest, Softmax_Test_4_Eager_1D_Axis0_Size8000) {
  constexpr size_t tensor_size = 8000;

  tf::tensor A, C;
  A.tf_create(tf_float64, 8000);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_4_input.bin", tensor_size).data());

  C = A.softmax(0);

  const std::vector<std::float64_t> output =
      load_bin("test/data/Softmax_Test_4_output.bin", tensor_size);
  expect_near_tensor(C, output);
}

TEST_F(MathTest, Softmax_Test_5_Graph_1D_Axis0_Size16000) {
  constexpr size_t tensor_size = 16000;

  tf::tensor A, C;
  A.tf_create(tf_float64, 16000);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_5_input.bin", tensor_size).data());

  {
    tf::graph_context ctx;

    C = A.softmax(0);
    ctx.run();

    const std::vector<std::float64_t> output =
        load_bin("test/data/Softmax_Test_5_output.bin", tensor_size);
    expect_near_tensor(C, output);
  }
}

TEST_F(MathTest, Softmax_Test_6_Eager_2D_Axis0_Size8000x2) {
  constexpr size_t tensor_size = 8000 * 2;

  tf::tensor A, C;
  A.tf_create(tf_float64, 2, 8000);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_6_input.bin", tensor_size).data());

  C = A.softmax(1);

  const std::vector<std::float64_t> output =
      load_bin("test/data/Softmax_Test_6_output.bin", tensor_size);
  expect_near_tensor(C, output);
}

TEST_F(MathTest, Softmax_Test_7_Graph_2D_Axis1_Size2x8000) {
  constexpr size_t tensor_size = 2 * 8000;

  tf::tensor A, C;
  A.tf_create(tf_float64, 8000, 2);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_7_input.bin", tensor_size).data());

  {
    tf::graph_context ctx;

    C = A.softmax(0);
    ctx.run();

    const std::vector<std::float64_t> output =
        load_bin("test/data/Softmax_Test_7_output.bin", tensor_size);
    expect_near_tensor(C, output);
  }
}

TEST_F(MathTest, Softmax_Test_8_Eager_1D_Axis0_Size13) {
  constexpr size_t tensor_size = 13;

  tf::tensor A, C;
  A.tf_create(tf_float64, 13);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_8_input.bin", tensor_size).data());

  C = A.softmax(0);

  const std::vector<std::float64_t> output =
      load_bin("test/data/Softmax_Test_8_output.bin", tensor_size);
  expect_near_tensor(C, output);
}

TEST_F(MathTest, Softmax_Test_9_Graph_1D_Axis0_Size180) {
  constexpr size_t tensor_size = 180;

  tf::tensor A, C;
  A.tf_create(tf_float64, 180);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_9_input.bin", tensor_size).data());

  {
    tf::graph_context ctx;

    C = A.softmax(0);
    ctx.run();

    const std::vector<std::float64_t> output =
        load_bin("test/data/Softmax_Test_9_output.bin", tensor_size);
    expect_near_tensor(C, output);
  }
}

TEST_F(MathTest, Softmax_Test_10_Eager_2D_Axis0_Size13x43) {
  constexpr size_t tensor_size = 13 * 43;

  tf::tensor A, C;
  A.tf_create(tf_float64, 43, 13);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_10_input.bin", tensor_size).data());

  C = A.softmax(1);

  const std::vector<std::float64_t> output =
      load_bin("test/data/Softmax_Test_10_output.bin", tensor_size);
  expect_near_tensor(C, output);
}

TEST_F(MathTest, Softmax_Test_11_Graph_2D_Axis1_Size43x13) {
  constexpr size_t tensor_size = 43 * 13;

  tf::tensor A, C;
  A.tf_create(tf_float64, 13, 43);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_11_input.bin", tensor_size).data());

  {
    tf::graph_context ctx;

    C = A.softmax(0);
    ctx.run();

    const std::vector<std::float64_t> output =
        load_bin("test/data/Softmax_Test_11_output.bin", tensor_size);
    expect_near_tensor(C, output);
  }
}

TEST_F(MathTest, Softmax_Test_12_Eager_3D_Axis1_Size13x180x43) {
  constexpr size_t tensor_size = 13 * 180 * 43;

  tf::tensor A, C;
  A.tf_create(tf_float64, 43, 180, 13);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_12_input.bin", tensor_size).data());

  C = A.softmax(1);

  const std::vector<std::float64_t> output =
      load_bin("test/data/Softmax_Test_12_output.bin", tensor_size);
  expect_near_tensor(C, output);
}

TEST_F(MathTest, Softmax_Test_13_Graph_3D_Axis2_Size13x43x180) {
  constexpr size_t tensor_size = 13 * 43 * 180;

  tf::tensor A, C;
  A.tf_create(tf_float64, 180, 43, 13);
  A.tensor_of(
      load_bin("test/data/Softmax_Test_13_input.bin", tensor_size).data());

  {
    tf::graph_context ctx;

    C = A.softmax(0);
    ctx.run();

    const std::vector<std::float64_t> output =
        load_bin("test/data/Softmax_Test_13_output.bin", tensor_size);
    expect_near_tensor(C, output);
  }
}
