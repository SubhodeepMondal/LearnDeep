#include "../LinearAlgebraFixtures.unit.hpp"
#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include <iostream>
#include <stdfloat>

TEST_F(FrameworkTest, SoftmaxLayer_Test_1) {
  tf::tensor x;
  tf::tensor weight, bias;
  tf::tensor input, output;

  unsigned batch_size = 128;
  unsigned input_features = 128;
  unsigned dense_feature = 56;
  unsigned epochs = 1;

  x.tf_create(tf_float64, input_features, batch_size);
  input.tf_create(tf_float64, input_features, batch_size);
  weight.tf_create(tf_float64, dense_feature, input_features);
  bias.tf_create(tf_float64, dense_feature, 1);
  output.tf_create(tf_float64, dense_feature, batch_size);

  input.tensor_of(load_bin("test/data/SoftmaxLayer_Test_1_input.bin",
                           input_features * batch_size)
                      .data());
  weight.tensor_of(load_bin("test/data/SoftmaxLayer_Test_1_weights.bin",
                            input_features * dense_feature)
                       .data());
  bias.tensor_of(
      load_bin("test/data/SoftmaxLayer_Test_1_bias.bin", dense_feature).data());
  output.tensor_of(load_bin("test/data/SoftmaxLayer_Test_1_output.bin",
                            dense_feature * batch_size)
                       .data());

  auto dense = tf::layer::dense(dense_feature);
  auto softmax = tf::layer::softmax();

  tf::callback::trace call_back;

  call_back.record_parameter_on_batch_end(
      softmax.getLayerPtr(), Layer_Parameter::softmax_training_output);

  auto dense_output = dense({x});
  auto softmax_output = softmax({dense_output});

  dense.set_weight(weight);
  dense.set_bias(bias);

  tf::model mymodel({x}, softmax_output);
  mymodel.shuffle(false);
  mymodel.compile(OptimizerType::SGD, LossType::squared_error);
  mymodel.fit({input}, {output}, {call_back.callback()}, epochs, batch_size, {},
              0, false);

  std::vector<std::vector<std::vector<tf::tensor>>> predicted_output =
      call_back.get_parameter_on_batch_end(
          softmax.getLayerPtr(), Layer_Parameter::softmax_training_output);

  for (unsigned j = 0; j < batch_size; j++) {
    for (unsigned i = 0; j < dense_feature; j++) {
      unsigned index = i * j * dense_feature;
      EXPECT_NEAR(predicted_output[0][0][0].getData()[index],
                  output.getData()[index], 1e-6)
          << "at :" << index;
    }
  }
}