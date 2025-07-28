#include "LinearAlgebraFixtures.unit.hpp"
#include <gtest/gtest.h>
#include <tensor.h>

TEST_F(MathTest, GraphCreation_test) {
  std::float64_t a[] = {0.00805433, 0.71604533, 0.53269858, 0.34661127,
                        0.56137041, 0.14270995, 0.62471964, 0.31898735,
                        0.32166846, 0.11043887, 0.43579798, 0.68246876,
                        0.91418964, 0.23475763, 0.45345491, 0.52145146};

  std::float64_t b[] = {0.49704563, 0.07919775, 0.28764169, 0.48468005,
                        0.55964598, 0.51628139, 0.81488106, 0.51503196,
                        0.2614797,  0.78048695, 0.91069071, 0.08636421,
                        0.19244527, 0.8450769,  0.84808829, 0.6912999};
  tf::tensor A, B, C, D;;
  A.tf_create(tf_float64, 4, 4);
  B.tf_create(tf_float64, 4, 4);
  C.tf_create(tf_float64, 4, 4);

  A.tensor_of(a);
  B.tensor_of(b);
  tf::graph g;
  g.tf_create_graph();
  ASSERT_TRUE(g.ptr != nullptr);
  // ASSERT_STRING_EQ(typeid(g).name(), "class tf::graph");

  g.graph_start_recording_session();
  ASSERT_TRUE(g.isSessionActive);

  C = A.add(B);
  ASSERT_TRUE(C.ptr != nullptr);
  EXPECT_TRUE(C.dt_type == tf_float64);

  g.graph_end_recording_session();
  g.graph_travarse_data_node();
  g.graph_execute();
  g.graph_travarse_data_node();
}
