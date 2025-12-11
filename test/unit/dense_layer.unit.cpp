#include "LinearAlgebraFixtures.unit.hpp"
#include <gtest/gtest.h>
#include <tensor.h>

TEST_F(FrameworkTest, DenseLayer) {
  tf::tensor x;
  tf::tensor input, output;
  x.tf_create(tf_float64, 64, 512);
  input.tf_create(tf_float64, 64, 4096);
  output.tf_create(tf_float64, 24, 4096);

  auto dense_1 = tf::layer::dense(24)({x});

  tf::model mymodel({x}, dense_1);
  mymodel.fit({input}, {output});
}