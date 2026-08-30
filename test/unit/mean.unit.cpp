#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, MatrixMean_Test_1) {
  tf::tensor A, C;
  A.tf_create(tf_float64, 11, 67);

  A.tensor_of(
      load_bin("test/data/MatrixMean_Test_1_input.bin", 11 * 67).data());

  C = A.mean(1);

  std::vector<std::float64_t> output =
      load_bin("test/data/MatrixMean_Test_1_output.bin", 11);
  for (int i = 0; i < 11; i++) {
    EXPECT_NEAR(C.getData()[i], output[i], 1e-6) << "at i: " << i;
  }
}

TEST_F(MathTest, MatrixMean_Test_2) {

  tf::tensor A, C;
  A.tf_create(tf_float64, 8, 20, 11, 37);
  A.tensor_of(
      load_bin("test/data/MatrixMean_Test_2_input.bin", 8 * 20 * 11 * 37)
          .data());

  {
    tf::graph_context ctx;

    C = A.mean(1);
    ctx.run();

    std::vector<std::float64_t> output =
        load_bin("test/data/MatrixMean_Test_2_output.bin", 37 * 11 * 8);

    for (int i = 0; i < 37 * 11 * 8; i++) {
      EXPECT_NEAR(C.getData()[i], output[i], 1e-6) << "at i: " << i;
    }
  }
}

TEST_F(MathTest, MatrixMean_Test_3) {

  std::float64_t a[] = {0.37454012, 0.95071431, 0.73199394, 0.59865848,
                        0.15601864, 0.15599452, 0.05808361, 0.86617615,
                        0.60111501, 0.70807258, 0.02058449, 0.96990985};

  std::float64_t c_mean[] = {0.4910291, 0.50678013, 0.24812175, 0.65453725};

  tf::tensor A, B, C, D;
  A.tf_create(tf_float64, 4, 3);
  B.tf_create(tf_float64, 4);
  C.tf_create(tf_float64, 4);
  D.tf_create(tf_float64, 4);

  A.tensor_of(a);
  C.tensor_of(c_mean);
  {
    tf::graph_context ctx;

    B = A.mean(1);
    D = B.mul(C);

    ctx.run();

    // for (int i = 0; i < 4; i++) {
    //   EXPECT_NEAR(B.getData()[i], c_mean[i], 0.0001);
    // }

    // B.print_data();
    ctx.initialize_gradient();
    ctx.compute_gradient();
    tf::tensor A_grad = ctx.get_gradient(A);
  }
}