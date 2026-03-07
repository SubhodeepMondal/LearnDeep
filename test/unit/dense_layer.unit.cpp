#include "LinearAlgebraFixtures.unit.hpp"
#include "dense_layer_data.hpp"
#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

TEST_F(FrameworkTest, DenseLayer_Test_3) {

  tf::tensor x;
  tf::tensor weight, bias;
  tf::tensor input, target_output, validation_data;

  x.tf_create(tf_float64, 43, 128);
  input.tf_create(tf_float64, 43, 128);
  weight.tf_create(tf_float64, 32, 43);
  bias.tf_create(tf_float64, 32, 1);
  target_output.tf_create(tf_float64, 32, 128);

  input.tensor_of(dense_layer_Test_3_input_data);
  weight.tensor_of(dense_layer_Test_3_weight_data);
  bias.tensor_of(dense_layer_Test_3_bias_data);
  target_output.tensor_of(dense_layer_Test_3_target_data);

  auto dense_1 = tf::layer::dense(32);
  auto dense_output = dense_1({x});
  dense_1.set_weight(weight);
  dense_1.set_bias(bias);

  /* --- call back setting --- */
  tf::callback call_back(false);
  call_back.record_parameter_on_epoch_begin(
      dense_1, Layer_Parameter::dense_training_input, false);

  call_back.record_parameter_on_epoch_end(
      dense_1, Layer_Parameter::dense_training_weight, false);

  call_back.record_parameter_on_epoch_end(
      dense_1, Layer_Parameter::dense_training_bias, false);

  call_back.record_parameter_on_epoch_end(
      dense_1, Layer_Parameter::dense_training_output, false);

  call_back.record_parameter_on_epoch_end(
      dense_1, Layer_Parameter::dense_grad_weight, false);

  call_back.record_parameter_on_epoch_end(
      dense_1, Layer_Parameter::dense_grad_bias, false);

  call_back.record_parameter_on_epoch_end(
      dense_1, Layer_Parameter::dense_updated_weight, false);

  call_back.record_parameter_on_epoch_end(
      dense_1, Layer_Parameter::dense_updated_bias, false);
  /* --- call back setting end --- */

  /* --- model creation --- */
  tf::model mymodel({x}, dense_output);
  mymodel.shuffle(false);
  mymodel.compile(OptimizerType::SGD, LossType::squared_error);
  /* --- end model creation --- */

  tf::loss loss_sgd = mymodel.get_model_loss(dense_output[0]);
  call_back.record_tensor_loss(
      loss_sgd, Loss_Parameter::squared_error_predicted_output, false);

  call_back.record_tensor_loss(
      loss_sgd, Loss_Parameter::squared_error_grad_predicted_output);

  /* --- model training --- */
  unsigned epoch = 1;
  unsigned batch_size = 128;
  mymodel.fit({input}, {target_output}, call_back, epoch, batch_size);

  /* --- Intercepting  training parameters --- */
  std::vector<std::vector<tf::tensor>> outputs =
      call_back.get_parameter_on_epoch_end(
          dense_1, Layer_Parameter::dense_training_output);

  std::vector<std::vector<tf::tensor>> training_weights =
      call_back.get_parameter_on_epoch_end(
          dense_1, Layer_Parameter::dense_training_weight);

  std::vector<std::vector<tf::tensor>> train_dense_1_grad_weights =
      call_back.get_parameter_on_epoch_end(dense_1,
                                           Layer_Parameter::dense_grad_weight);

  std::vector<std::vector<tf::tensor>> train_dense_1_grad_bias =
      call_back.get_parameter_on_epoch_end(dense_1,
                                           Layer_Parameter::dense_grad_bias);

  std::vector<std::vector<tf::tensor>> train_dense_1_updated_weights =
      call_back.get_parameter_on_epoch_end(
          dense_1, Layer_Parameter::dense_updated_weight);

  std::vector<std::vector<tf::tensor>> train_dense_1_updated_bias =
      call_back.get_parameter_on_epoch_end(dense_1,
                                           Layer_Parameter::dense_updated_bias);

  std::vector<std::vector<tf::tensor>> training_losses =
      call_back.get_tensor_loss(loss_sgd,
                                Loss_Parameter::squared_error_predicted_output);

  std::vector<std::vector<tf::tensor>> grad_training_losses =
      call_back.get_tensor_loss(
          loss_sgd, Loss_Parameter::squared_error_grad_predicted_output);

  for (auto output : outputs) {
    for (int j = 0; j < 128; j++)
      for (int i = 0; i < 32; i++)
        EXPECT_NEAR(output[0].getData()[i + j * 32],
                    dense_layer_Test_3_output_data[i + j * 32], 1e-6)
            << "at: " << i + j * 3;
  }

  for (unsigned ep = 0; ep < epoch; ep++)
    for (int i = 0; i < 32; i++)
      EXPECT_NEAR(training_losses[ep][0].getData()[i],
                  dense_layer_Test_3_loss_data[i], 1e-5)
          << "at ep : " << ep << "i: " << i;

  for (unsigned ep = 0; ep < epoch; ep++)
    for (int j = 0; j < 43; j++)
      for (int i = 0; i < 32; i++)
        EXPECT_NEAR(train_dense_1_grad_weights[ep][0].getData()[i + j * 32],
                    dense_layer_Test_3_grad_weight_data[i + j * 32], 1e-5)
            << "at ep : " << ep << " i: " << i << " j: " << j;

  for (unsigned ep = 0; ep < epoch; ep++)
    for (int i = 0; i < 32; i++)
      EXPECT_NEAR(train_dense_1_grad_bias[ep][0].getData()[i],
                  dense_layer_Test_3_grad_bias_data[i], 1e-5)
          << "at i: " << i;

  for (unsigned ep = 0; ep < epoch; ep++)
    for (int j = 0; j < 128; j++)
      for (int i = 0; i < 32; i++)
        EXPECT_NEAR(grad_training_losses[ep][0].getData()[i + j * 32],
                    dense_layer_Test_3_grad_loss_data[i + j * 32], 1e-5)
            << "at i: " << i << " j: " << j;

  for (unsigned ep = 0; ep < epoch; ep++)
    for (int j = 0; j < 43; j++)
      for (int i = 0; i < 32; i++)
        EXPECT_NEAR(train_dense_1_updated_weights[ep][0].getData()[i + j * 32],
                    dense_layer_Test_3_updated_weights_data[i + j * 32], 1e-5)
            << "at i: " << i << " j: " << j;

  for (unsigned ep = 0; ep < epoch; ep++)
    for (int i = 0; i < 32; i++)
      EXPECT_NEAR(train_dense_1_updated_bias[ep][0].getData()[i],
                  dense_layer_Test_3_updated_bias_data[i], 1e-5)
          << "at i: " << i;
}

TEST_F(FrameworkTest, DenseLayer_Test_4) {

  tf::tensor x;
  tf::tensor weight, bias;
  tf::tensor input, target_output, validation_data;

  unsigned sample_size = 128;
  unsigned batch_size = 32;
  unsigned no_of_input_feature = 17;
  unsigned no_of_dense_unit = 19;

  x.tf_create(tf_float64, no_of_input_feature, batch_size);
  input.tf_create(tf_float64, no_of_input_feature, sample_size);
  weight.tf_create(tf_float64, no_of_dense_unit, no_of_input_feature);
  bias.tf_create(tf_float64, no_of_dense_unit, 1);
  target_output.tf_create(tf_float64, no_of_dense_unit, sample_size);

  input.tensor_of(dense_layer_Test_4_input_data);
  weight.tensor_of(dense_layer_Test_4_weight_data);
  bias.tensor_of(dense_layer_Test_4_bias_data);
  target_output.tensor_of(dense_layer_Test_4_target_output_data);

  auto dense_1 = tf::layer::dense(no_of_dense_unit);
  auto dense_output = dense_1({x});
  dense_1.set_weight(weight);
  dense_1.set_bias(bias);

  /* --- call back setting --- */
  tf::callback call_back(false);
  call_back.record_parameter_on_epoch_begin(
      dense_1, Layer_Parameter::dense_training_input, false);

  call_back.record_parameter_on_epoch_end(
      dense_1, Layer_Parameter::dense_updated_weight, false);

  call_back.record_parameter_on_epoch_end(
      dense_1, Layer_Parameter::dense_updated_bias, false);

  call_back.record_parameter_on_epoch_end(
      dense_1, Layer_Parameter::dense_training_output, false);

  call_back.record_parameter_on_epoch_end(
      dense_1, Layer_Parameter::dense_grad_weight, false);

  call_back.record_parameter_on_epoch_end(
      dense_1, Layer_Parameter::dense_grad_bias, false);
  /* --- call back setting end --- */

  /* --- model creation --- */
  tf::model mymodel({x}, dense_output);
  mymodel.shuffle(false);
  mymodel.compile(OptimizerType::SGD, LossType::squared_error);
  /* --- end model creation --- */

  tf::loss loss_sgd = mymodel.get_model_loss(dense_output[0]);
  call_back.record_tensor_loss(
      loss_sgd, Loss_Parameter::squared_error_predicted_output, false);

  call_back.record_tensor_loss(
      loss_sgd, Loss_Parameter::squared_error_grad_predicted_output);

  call_back.record_tensor_loss(loss_sgd,
                               Loss_Parameter::squared_error_target_output);

  /* --- model training --- */
  unsigned epoch = 10;
  mymodel.fit({input}, {target_output}, call_back, epoch, batch_size);

  /* --- Intercepting  training parameters --- */
  std::vector<std::vector<tf::tensor>> dense_training_inputs =
      call_back.get_parameter_on_epoch_begin(
          dense_1, Layer_Parameter::dense_training_input);

  std::vector<std::vector<tf::tensor>> update_weights =
      call_back.get_parameter_on_epoch_end(
          dense_1, Layer_Parameter::dense_updated_weight);

  std::vector<std::vector<tf::tensor>> updated_bias =
      call_back.get_parameter_on_epoch_end(dense_1,
                                           Layer_Parameter::dense_updated_bias);

  std::vector<std::vector<tf::tensor>> outputs =
      call_back.get_parameter_on_epoch_end(
          dense_1, Layer_Parameter::dense_training_output);

  std::vector<std::vector<tf::tensor>> training_losses =
      call_back.get_tensor_loss(loss_sgd,
                                Loss_Parameter::squared_error_predicted_output);

  std::vector<std::vector<tf::tensor>> grad_training_losses =
      call_back.get_tensor_loss(
          loss_sgd, Loss_Parameter::squared_error_grad_predicted_output);

  std::vector<std::vector<tf::tensor>> target_training_output =
      call_back.get_tensor_loss(loss_sgd,
                                Loss_Parameter::squared_error_target_output);

  std::vector<std::vector<tf::tensor>> grad_weights =
      call_back.get_parameter_on_epoch_end(dense_1,
                                           Layer_Parameter::dense_grad_weight);

  std::vector<std::vector<tf::tensor>> grad_bias =
      call_back.get_parameter_on_epoch_end(dense_1,
                                           Layer_Parameter::dense_grad_bias);

  for (unsigned ep = 0; ep < epoch; ep++)
    for (unsigned j = 0; j < batch_size; j++)
      for (unsigned i = 0; i < no_of_input_feature; i++) {
        unsigned index = i + j * no_of_input_feature;
        EXPECT_NEAR(dense_training_inputs[ep][0].getData()[index],
                    dense_layer_Test_4_training_input_data[ep][index], 1e-6)
            << "at: epoch " << ep << ", index " << index;
      }

  for (unsigned ep = 0; ep < epoch; ep++)
    for (unsigned j = 0; j < batch_size; j++)
      for (unsigned i = 0; i < no_of_dense_unit; i++) {
        unsigned index = i + j * no_of_dense_unit;
        EXPECT_NEAR(target_training_output[ep][0].getData()[index],
                    dense_layer_Test_4_target_output_batch_data[ep][index],
                    1e-6)
            << "at: epoch " << ep << ", index " << index;
      }

  for (unsigned ep = 0; ep < epoch; ep++)
    for (unsigned i = 0; i < no_of_dense_unit; i++) {
      unsigned index = i;
      EXPECT_NEAR(training_losses[ep][0].getData()[index],
                  dense_layer_Test_4_loss_data[ep][index], 1e-6)
          << "at: epoch " << ep << ", index " << index;
    }

  for (unsigned ep = 0; ep < epoch; ep++)
    for (unsigned j = 0; j < batch_size; j++)
      for (unsigned i = 0; i < no_of_dense_unit; i++) {
        unsigned index = i + j * no_of_dense_unit;
        EXPECT_NEAR(grad_training_losses[ep][0].getData()[index],
                    dense_layer_Test_4_grad_loss_data[ep][index], 1e-6)
            << "at: epoch " << ep << ", index " << index;
      }

  for (unsigned ep = 0; ep < epoch; ep++)
    for (unsigned j = 0; j < no_of_input_feature; j++)
      for (unsigned i = 0; i < no_of_dense_unit; i++) {
        unsigned index = i + j * no_of_dense_unit;
        EXPECT_NEAR(update_weights[ep][0].getData()[index],
                    dense_layer_Test_4_updated_weight_data[ep][index], 1e-6)
            << "at: epoch " << ep << ", index " << index;
      }

  for (unsigned ep = 0; ep < epoch; ep++)
    for (unsigned i = 0; i < no_of_dense_unit; i++) {
      unsigned index = i;
      EXPECT_NEAR(updated_bias[ep][0].getData()[index],
                  dense_layer_Test_4_updated_bias_data[ep][index], 1e-6)
          << "at: epoch " << ep << ", index " << index;
    }

  for (unsigned ep = 0; ep < epoch; ep++)
    for (unsigned j = 0; j < batch_size; j++)
      for (unsigned i = 0; i < no_of_dense_unit; i++) {
        unsigned index = i + j * no_of_dense_unit;
        EXPECT_NEAR(outputs[ep][0].getData()[index],
                    dense_layer_Test_4_predicted_output_data[ep][index], 1e-6)
            << "at: epoch " << ep << ", index " << index;
      }

  for (unsigned ep = 0; ep < epoch; ep++)
    for (unsigned j = 0; j < no_of_input_feature; j++)
      for (unsigned i = 0; i < no_of_dense_unit; i++) {
        unsigned index = i + j * no_of_dense_unit;
        EXPECT_NEAR(grad_weights[ep][0].getData()[index],
                    dense_layer_Test_4_grad_weight_data[ep][index], 1e-6)
            << "at: epoch " << ep << ", index " << index;
      }

  for (unsigned ep = 0; ep < epoch; ep++)
    for (unsigned i = 0; i < no_of_dense_unit; i++) {
      unsigned index = i;
      EXPECT_NEAR(grad_bias[ep][0].getData()[index],
                  dense_layer_Test_4_grad_bias_data[ep][index], 1e-6)
          << "at: epoch " << ep << ", index " << index;
    }
}