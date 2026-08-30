#include "LinearAlgebraFixtures.unit.hpp"
#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

TEST_F(MathTest, MatrixReductionSum_Test_1) {

  tf::tensor A, C;

  A.tf_create(tf_float64, 33, 73);
  A.tensor_of(load_bin("test/data/MatrixReductionSum_Test_1_input.bin", 33 * 73)
                  .data());

  // In numpy this concept reverses index stared from right to left
  C = A.reducesum({1});

  std::vector<std::float64_t> output =
      load_bin("test/data/MatrixReductionSum_Test_1_output.bin", 33);
  for (int i = 0; i < 33; i++) {
    EXPECT_NEAR(C.getData()[i], output[i], 1e-6) << "at: " << i;
  }
}

TEST_F(MathTest, MatrixReductionSum_Test_2) {

  std::float64_t c_reducesum[] = {2.39105, 1.66398, 2.27044, 1.49826};

  tf::tensor A, C;
  A.tf_create(tf_float64, 65, 78, 13);

  A.tensor_of(
      load_bin("test/data/MatrixReductionSum_Test_2_input.bin", 65 * 78 * 13)
          .data());

  {
    tf::graph_context ctx;

    C = A.reducesum({0, 2});

    ctx.run();
    std::vector<std::float64_t> output =
        load_bin("test/data/MatrixReductionSum_Test_2_output.bin", 78);
    for (int i = 0; i < 78; i++) {
      EXPECT_NEAR(C.getData()[i], output[i], 1e-6) << "at: " << i;
    }
  }
}

TEST_F(MathTest, MatrixReductionSum_Test_3) {

  std::float64_t a[] = {0.42602198, 0.51120308, 0.66381781, 0.79000792,
                        0.73980886, 0.1366799,  0.3818528,  0.40564105,
                        0.79132994, 0.1810338,  0.52634304, 0.7717289,
                        0.18137833, 0.35597476, 0.79365669, 0.16725214};

  std::float64_t c_reducesum[] = {2.39105, 1.66398, 2.27044, 1.49826};

  tf::tensor A, B, C, D;
  A.tf_create(tf_float64, 4, 4);
  B.tf_create(tf_float64, 4);
  C.tf_create(tf_float64, 4);
  D.tf_create(tf_float64, 4);

  A.tensor_of(a);
  D.tensor_of(c_reducesum);

  {
    tf::graph_context ctx;

    B = A.reducesum({0});
    C = D.mul(B);

    ctx.run();

    for (int i = 0; i < 4; i++) {
      EXPECT_NEAR(B.getData()[i], c_reducesum[i], 0.0001);
    }

    ctx.initialize_gradient();
    ctx.compute_gradient();

    tf::tensor A_grad = ctx.get_gradient(A);
  }
}
