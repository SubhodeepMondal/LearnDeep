#include "LinearAlgebraFixtures.unit.hpp"
#include <gtest/gtest.h>
#include <tensor.h>

TEST_F(MathTest, GraphCreation_test) {
  std::float64_t a[] = {
      0.42602198, 0.51120308, 0.66381781, 0.79000792, 0.73980886, 0.1366799,
      0.3818528,  0.40564105, 0.79132994, 0.1810338,  0.52634304, 0.7717289,
      0.18137833, 0.35597476, 0.79365669, 0.16725214,
  };

  std::float64_t b[] = {
      0.09328713, 0.4622637,  0.78946444, 0.14203406, 0.84087653, 0.19860089,
      0.85865357, 0.1971424,  0.00456894, 0.05930002, 0.91838192, 0.94089892,
      0.47830435, 0.49946382, 0.6011153,  0.97711538,
  };
  tf::tensor A, B, C;
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

  C = A.add(g, B);
  ASSERT_TRUE(C.ptr != nullptr);
  EXPECT_TRUE(C.dt_type == tf_float64);

  g.graph_end_recording_session();
  // g.graph_travarse_data_node();
  g.graph_execute();
  // g.graph_travarse_data_node();
  g.graph_clear();
}
