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

TEST_F(FrameworkTest, CategoricalClassification_Test_1) {
  tf::tensor x;
  tf::tensor weights_1, weights_2, weights_3;
  tf::tensor bias_1, bias_2, bias_3;
  tf::tensor input, output;

  unsigned batch_size = 128;
  unsigned input_feature = 8;
  unsigned input_sample_size = 1024;
  unsigned dense_unit_1 = 16;
  unsigned dense_unit_2 = 8;
  unsigned dense_unit_3 = 2;
  unsigned epoches = 200;

  x.tf_create(tf_float64, input_feature, batch_size);
  input.tf_create(tf_float64, input_feature, input_sample_size);
  output.tf_create(tf_float64, 2, input_sample_size);

  weights_1.tf_create(tf_float64, dense_unit_1, input_feature);
  weights_2.tf_create(tf_float64, dense_unit_2, dense_unit_1);
  weights_3.tf_create(tf_float64, dense_unit_3, dense_unit_2);

  bias_1.tf_create(tf_float64, dense_unit_1, 1);
  bias_2.tf_create(tf_float64, dense_unit_2, 1);
  bias_3.tf_create(tf_float64, dense_unit_3, 1);

  input.tensor_of(load_bin("test/data/Classification_Test_1_input.bin",
                           input_feature * input_sample_size)
                      .data());

  output.tensor_of(load_bin("test/data/Classification_Test_1_output.bin",
                            2 * input_sample_size)
                       .data());

  weights_1.tensor_of(
      load_bin("test/data/Classification_Test_1_dense1_weight.bin",
               dense_unit_1 * input_feature)
          .data());
  weights_2.tensor_of(
      load_bin("test/data/Classification_Test_1_dense2_weight.bin",
               dense_unit_2 * dense_unit_1)
          .data());
  weights_3.tensor_of(
      load_bin("test/data/Classification_Test_1_dense3_weight.bin",
               dense_unit_3 * dense_unit_2)
          .data());

  bias_1.tensor_of(
      load_bin("test/data/Classification_Test_1_dense1_bias.bin", dense_unit_1)
          .data());
  bias_2.tensor_of(
      load_bin("test/data/Classification_Test_1_dense2_bias.bin", dense_unit_2)
          .data());
  bias_3.tensor_of(
      load_bin("test/data/Classification_Test_1_dense3_bias.bin", dense_unit_3)
          .data());

  // initializing layers
  auto dense_1 = tf::layer::dense(dense_unit_1);
  auto relu_1 = tf::layer::relu();
  auto dense_2 = tf::layer::dense(dense_unit_2);
  auto relu_2 = tf::layer::relu();
  auto dense_3 = tf::layer::dense(dense_unit_3);
  auto softmax = tf::layer::softmax();

  // setting weights and biases for dense layers
  dense_1.set_weight(weights_1);
  dense_1.set_bias(bias_1);
  dense_2.set_weight(weights_2);
  dense_2.set_bias(bias_2);
  dense_3.set_weight(weights_3);
  dense_3.set_bias(bias_3);

  // setup model architecture
  auto dense_1_out = dense_1({x});
  auto relu_1_out = relu_1(dense_1_out);
  auto dense_2_out = dense_2(relu_1_out);
  auto relu_2_out = relu_2(dense_2_out);
  auto dense_3_out = dense_3(relu_2_out);
  auto softmax_out = softmax(dense_3_out);

  // putting things back togather
  tf::model mymodel({x}, softmax_out);
  mymodel.shuffle(false);
  mymodel.compile(OptimizerType::SGD, LossType::categorical_cross_entropy);

  tf::loss cce = mymodel.get_model_loss(softmax_out[0]);

  tf::callback::trace call_back;

  call_back.record_parameter_on_batch_end(
      dense_1.getLayerPtr(), Layer_Parameter::dense_training_output);

  call_back.record_parameter_on_batch_end(
      relu_1.getLayerPtr(), Layer_Parameter::relu_training_output);

  call_back.record_parameter_on_batch_end(
      dense_2.getLayerPtr(), Layer_Parameter::dense_training_output);

  call_back.record_parameter_on_batch_end(
      relu_2.getLayerPtr(), Layer_Parameter::relu_training_output);

  call_back.record_parameter_on_batch_end(
      dense_3.getLayerPtr(), Layer_Parameter::dense_training_output);

  call_back.record_parameter_on_batch_end(
      softmax.getLayerPtr(), Layer_Parameter::softmax_training_output);

  call_back.record_tensor_loss_on_batch_end(
      cce, Loss_Parameter::categorical_cross_entropy_error);

  auto history = mymodel.fit({input}, {output}, {call_back.callback()}, epoches,
                             batch_size);

  std::vector<std::vector<std::vector<tf::tensor>>> training_losses =
      call_back.get_tensor_loss_on_batch_end(
          cce, Loss_Parameter::categorical_cross_entropy_error);

  std::vector<std::vector<std::vector<tf::tensor>>> dense_1_training_output =
      call_back.get_parameter_on_batch_end(
          dense_1.getLayerPtr(), Layer_Parameter::dense_training_output);

  std::vector<std::vector<std::vector<tf::tensor>>> relu_1_training_output =
      call_back.get_parameter_on_batch_end(
          relu_1.getLayerPtr(), Layer_Parameter::relu_training_output);

  std::vector<std::vector<std::vector<tf::tensor>>> dense_2_training_output =
      call_back.get_parameter_on_batch_end(
          dense_2.getLayerPtr(), Layer_Parameter::dense_training_output);

  std::vector<std::vector<std::vector<tf::tensor>>> relu_2_training_output =
      call_back.get_parameter_on_batch_end(
          relu_2.getLayerPtr(), Layer_Parameter::relu_training_output);

  std::vector<std::vector<std::vector<tf::tensor>>> dense_3_training_output =
      call_back.get_parameter_on_batch_end(
          dense_3.getLayerPtr(), Layer_Parameter::dense_training_output);

  std::vector<std::vector<std::vector<tf::tensor>>> softmax_training_output =
      call_back.get_parameter_on_batch_end(
          softmax.getLayerPtr(), Layer_Parameter::softmax_training_output);

  //   std::cout << "\nDense 1\n";
  //   dense_1_training_output[0][0][0].print_data();
  //   std::cout << "\nrelu 1\n";
  //   relu_1_training_output[0][0][0].print_data();
  //   std::cout << "\nDense 2\n";
  //   dense_2_training_output[0][0][0].print_data();
  //   std::cout << "\nRelu 2\n";
  //   relu_2_training_output[0][0][0].print_data();
  //   std::cout << "\nDense 3\n";
  //   dense_3_training_output[0][0][0].print_data();
  //   std::cout << "\nsoftmax\n";
  //   softmax_training_output[0][0][0].print_data();
  //   std::cout << "\ncce error tensor\n";
  //   training_losses[0][0][0].print_data();
  for (auto loss : history["loss"])
    std::cout << loss << "\n";
}