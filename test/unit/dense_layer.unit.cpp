#include <memory>

// library headers
#include "LinearAlgebraFixtures.unit.hpp"
#include "dense_layer_data.hpp"
#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include <stdfloat>

TEST_F(FrameworkTest, DenseLayer_Test_1) {

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
  tf::callback::trace call_back;
  call_back.record_parameter_on_epoch_begin(
      dense_1.getLayerPtr(), Layer_Parameter::dense_training_input, false);

  call_back.record_parameter_on_epoch_end(
      dense_1.getLayerPtr(), Layer_Parameter::dense_training_weight, false);

  call_back.record_parameter_on_epoch_end(
      dense_1.getLayerPtr(), Layer_Parameter::dense_training_bias, false);

  call_back.record_parameter_on_epoch_end(
      dense_1.getLayerPtr(), Layer_Parameter::dense_training_output, false);

  call_back.record_parameter_on_epoch_end(
      dense_1.getLayerPtr(), Layer_Parameter::dense_grad_weight, false);

  call_back.record_parameter_on_epoch_end(
      dense_1.getLayerPtr(), Layer_Parameter::dense_grad_bias, false);

  call_back.record_parameter_on_epoch_end(
      dense_1.getLayerPtr(), Layer_Parameter::dense_updated_weight, false);

  call_back.record_parameter_on_epoch_end(
      dense_1.getLayerPtr(), Layer_Parameter::dense_updated_bias, false);
  /* --- call back setting end --- */

  /* --- model creation --- */
  tf::model mymodel({x}, dense_output);
  mymodel.shuffle(false);
  mymodel.compile(OptimizerType::SGD, LossType::squared_error);
  /* --- end model creation --- */

  tf::loss loss_sgd = mymodel.get_model_loss(dense_output[0]);
  call_back.record_tensor_loss_on_epoch_end(
      loss_sgd, Loss_Parameter::squared_error_predicted_output, false);

  call_back.record_tensor_loss_on_epoch_end(
      loss_sgd, Loss_Parameter::squared_error_grad_predicted_output);

  call_back.record_scalar_loss_on_epoch_end(loss_sgd, false);

  /* --- model training --- */
  unsigned epoch = 1;
  unsigned batch_size = 128;
  mymodel.fit({input}, {target_output}, {call_back.callback()}, epoch,
              batch_size);

  /* --- Intercepting  training parameters --- */
  std::vector<std::vector<tf::tensor>> outputs =
      call_back.get_parameter_on_epoch_end(
          dense_1.getLayerPtr(), Layer_Parameter::dense_training_output);

  std::vector<std::vector<tf::tensor>> training_weights =
      call_back.get_parameter_on_epoch_end(
          dense_1.getLayerPtr(), Layer_Parameter::dense_training_weight);

  std::vector<std::vector<tf::tensor>> train_dense_1_grad_weights =
      call_back.get_parameter_on_epoch_end(dense_1.getLayerPtr(),
                                           Layer_Parameter::dense_grad_weight);

  std::vector<std::vector<tf::tensor>> train_dense_1_grad_bias =
      call_back.get_parameter_on_epoch_end(dense_1.getLayerPtr(),
                                           Layer_Parameter::dense_grad_bias);

  std::vector<std::vector<tf::tensor>> train_dense_1_updated_weights =
      call_back.get_parameter_on_epoch_end(
          dense_1.getLayerPtr(), Layer_Parameter::dense_updated_weight);

  std::vector<std::vector<tf::tensor>> train_dense_1_updated_bias =
      call_back.get_parameter_on_epoch_end(dense_1.getLayerPtr(),
                                           Layer_Parameter::dense_updated_bias);

  std::vector<std::vector<tf::tensor>> training_losses =
      call_back.get_tensor_loss_on_epoch_end(
          loss_sgd, Loss_Parameter::squared_error_predicted_output);

  std::vector<std::vector<tf::tensor>> grad_training_losses =
      call_back.get_tensor_loss_on_epoch_end(
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

  std::vector<std::float64_t> scalar_losses =
      call_back.get_scaler_loss_on_epoch_end(loss_sgd);
}

TEST_F(FrameworkTest, DenseLayer_Test_2) {

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

  input.tensor_of(load_bin("test/data/DenseLayer_Test_2_input_data.bin",
                           no_of_input_feature * sample_size)
                      .data());
  weight.tensor_of(load_bin("test/data/DenseLayer_Test_2_weights_data.bin",
                            no_of_dense_unit * no_of_input_feature)
                       .data());
  bias.tensor_of(
      load_bin("test/data/DenseLayer_Test_2_bias_data.bin", no_of_dense_unit)
          .data());
  target_output.tensor_of(
      load_bin("test/data/DenseLayer_Test_2_output_data.bin",
               no_of_dense_unit * sample_size)
          .data());

  auto dense_1 = tf::layer::dense(no_of_dense_unit);
  auto dense_output = dense_1({x});
  dense_1.set_weight(weight);
  dense_1.set_bias(bias);

  /* --- call back setting --- */
  tf::callback::trace call_back;
  call_back.record_parameter_on_batch_begin(
      dense_1.getLayerPtr(), Layer_Parameter::dense_training_input, false);

  call_back.record_parameter_on_batch_end(
      dense_1.getLayerPtr(), Layer_Parameter::dense_updated_weight, false);

  call_back.record_parameter_on_batch_end(
      dense_1.getLayerPtr(), Layer_Parameter::dense_updated_bias, false);

  call_back.record_parameter_on_batch_end(
      dense_1.getLayerPtr(), Layer_Parameter::dense_training_output, false);

  call_back.record_parameter_on_batch_end(
      dense_1.getLayerPtr(), Layer_Parameter::dense_grad_weight, false);

  call_back.record_parameter_on_batch_end(
      dense_1.getLayerPtr(), Layer_Parameter::dense_grad_bias, false);

  /* --- call back setting end --- */

  /* --- model creation --- */
  tf::model mymodel({x}, dense_output);
  mymodel.shuffle(false);
  mymodel.compile(OptimizerType::SGD, LossType::squared_error);
  /* --- end model creation --- */

  tf::loss loss_sgd = mymodel.get_model_loss(dense_output[0]);
  call_back.record_tensor_loss_on_batch_end(
      loss_sgd, Loss_Parameter::squared_error_predicted_output, false);

  call_back.record_tensor_loss_on_batch_end(
      loss_sgd, Loss_Parameter::squared_error_grad_predicted_output);

  call_back.record_tensor_loss_on_batch_end(
      loss_sgd, Loss_Parameter::squared_error_target_output);

  call_back.record_scalar_loss_on_batch_end(loss_sgd, false);

  tf::callback::earlystopping early_stopping(loss_sgd, 5, 0.01, true);

  /* --- model training --- */
  unsigned epoch = 10;
  mymodel.fit({input}, {target_output},
              {call_back.callback(), early_stopping.callback()}, epoch,
              batch_size);

  /* --- Intercepting  training parameters --- */
  std::vector<std::vector<std::vector<tf::tensor>>> dense_training_inputs =
      call_back.get_parameter_on_batch_begin(
          dense_1.getLayerPtr(), Layer_Parameter::dense_training_input);

  std::vector<std::vector<std::vector<tf::tensor>>> update_weights =
      call_back.get_parameter_on_batch_end(
          dense_1.getLayerPtr(), Layer_Parameter::dense_updated_weight);

  std::vector<std::vector<std::vector<tf::tensor>>> updated_bias =
      call_back.get_parameter_on_batch_end(dense_1.getLayerPtr(),
                                           Layer_Parameter::dense_updated_bias);

  std::vector<std::vector<std::vector<tf::tensor>>> outputs =
      call_back.get_parameter_on_batch_end(
          dense_1.getLayerPtr(), Layer_Parameter::dense_training_output);

  std::vector<std::vector<std::vector<tf::tensor>>> training_losses =
      call_back.get_tensor_loss_on_batch_end(
          loss_sgd, Loss_Parameter::squared_error_predicted_output);

  std::vector<std::vector<std::vector<tf::tensor>>> grad_training_losses =
      call_back.get_tensor_loss_on_batch_end(
          loss_sgd, Loss_Parameter::squared_error_grad_predicted_output);

  std::vector<std::vector<std::vector<tf::tensor>>> target_training_output =
      call_back.get_tensor_loss_on_batch_end(
          loss_sgd, Loss_Parameter::squared_error_target_output);

  std::vector<std::vector<std::vector<tf::tensor>>> grad_weights =
      call_back.get_parameter_on_batch_end(dense_1.getLayerPtr(),
                                           Layer_Parameter::dense_grad_weight);

  std::vector<std::vector<std::vector<tf::tensor>>> grad_bias =
      call_back.get_parameter_on_batch_end(dense_1.getLayerPtr(),
                                           Layer_Parameter::dense_grad_bias);

  unsigned const no_of_batches = sample_size / batch_size;

  /* --- target input test --- */
  {
    std::vector<std::float64_t> dense_layer_Test_4_training_input_data =
        load_bin("test/data/DenseLayer_Test_2_input_batch_data.bin",
                 epoch * no_of_batches * batch_size * no_of_input_feature);
    for (unsigned ep = 0; ep < 10; ep++)
      for (unsigned bt = 0; bt < no_of_batches; bt++)
        for (unsigned j = 0; j < batch_size; j++)
          for (unsigned i = 0; i < no_of_input_feature; i++) {
            unsigned index = i + j * no_of_input_feature;
            unsigned data_index =
                index + bt * no_of_input_feature * batch_size +
                ep * no_of_input_feature * batch_size * no_of_batches;
            EXPECT_NEAR(dense_training_inputs[ep][bt][0].getData()[index],
                        dense_layer_Test_4_training_input_data[data_index],
                        1e-6)
                << "at: epoch " << ep << ", index " << index;
          }
  }

  /* --- target output test --- */
  {
    std::vector<std::float64_t> dense_layer_Test_4_target_output_batch_data =
        load_bin("test/data/DenseLayer_Test_2_target_output_batch_data.bin",
                 epoch * no_of_batches * batch_size * no_of_dense_unit);
    for (unsigned ep = 0; ep < 10; ep++)
      for (unsigned bt = 0; bt < no_of_batches; bt++)
        for (unsigned j = 0; j < batch_size; j++)
          for (unsigned i = 0; i < no_of_dense_unit; i++) {
            unsigned index = i + j * no_of_dense_unit;
            unsigned data_index =
                index + bt * no_of_dense_unit * batch_size +
                ep * no_of_dense_unit * batch_size * no_of_batches;
            EXPECT_NEAR(target_training_output[ep][bt][0].getData()[index],
                        dense_layer_Test_4_target_output_batch_data[data_index],
                        1e-6)
                << "at: epoch " << ep << ", index " << index;
          }
  }

  /* --- loss per batch test --- */
  {
    std::vector<std::float64_t> dense_layer_Test_2_loss_batch_data =
        load_bin("test/data/DenseLayer_Test_2_loss_batch_data.bin",
                 epoch * no_of_batches * no_of_dense_unit);
    for (unsigned ep = 0; ep < epoch; ep++)
      for (unsigned bt = 0; bt < no_of_batches; bt++)
        for (unsigned i = 0; i < no_of_dense_unit; i++) {
          unsigned index = i;
          unsigned data_index = index + bt * no_of_dense_unit +
                                ep * no_of_dense_unit * no_of_batches;
          EXPECT_NEAR(training_losses[ep][bt][0].getData()[index],
                      dense_layer_Test_2_loss_batch_data[data_index], 1e-6)
              << "at: epoch " << ep << ", at: batch " << bt << ", index "
              << index;
        }
  }

  /* --- loss gradeint per batch test --- */
  {
    std::vector<std::float64_t> dense_layer_Test_2_loss_grad_batch_data =
        load_bin("test/data/DenseLayer_Test_2_grad_loss_batch_data.bin",
                 epoch * no_of_batches * batch_size * no_of_dense_unit);
    for (unsigned ep = 0; ep < epoch; ep++)
      for (unsigned bt = 0; bt < no_of_batches; bt++)
        for (unsigned j = 0; j < batch_size; j++)
          for (unsigned i = 0; i < no_of_dense_unit; i++) {
            unsigned index = i + j * no_of_dense_unit;
            unsigned data_index =
                index + bt * no_of_dense_unit * batch_size +
                ep * no_of_dense_unit * batch_size * no_of_batches;
            EXPECT_NEAR(grad_training_losses[ep][bt][0].getData()[index],
                        dense_layer_Test_2_loss_grad_batch_data[data_index],
                        1e-6)
                << "at: epoch " << ep << ", batch: " << bt << ", index "
                << index;
          }
  }

  /* --- updated weights per batch test --- */
  {
    std::vector<std::float64_t> dense_layer_Test_4_updated_weight_data =
        load_bin("test/data/DenseLayer_Test_2_updated_weight_batch_data.bin",
                 epoch * no_of_batches * no_of_input_feature *
                     no_of_dense_unit);
    for (unsigned ep = 0; ep < epoch; ep++)
      for (unsigned bt = 0; bt < no_of_batches; bt++)
        for (unsigned j = 0; j < no_of_input_feature; j++)
          for (unsigned i = 0; i < no_of_dense_unit; i++) {
            unsigned index = i + j * no_of_dense_unit;
            unsigned data_index =
                index + bt * no_of_dense_unit * no_of_input_feature +
                ep * no_of_dense_unit * no_of_input_feature * no_of_batches;
            EXPECT_NEAR(update_weights[ep][bt][0].getData()[index],
                        dense_layer_Test_4_updated_weight_data[data_index],
                        1e-6)
                << "at: epoch " << ep << " , batch: " << bt << ", index "
                << index;
          }
  }

  /* --- updated bias per batch test --- */
  {
    std::vector<std::float64_t> dense_layer_Test_4_updated_bias_data =
        load_bin("test/data/DenseLayer_Test_2_updated_bias_batch_data.bin",
                 epoch * no_of_batches * no_of_dense_unit);
    for (unsigned ep = 0; ep < epoch; ep++)
      for (unsigned bt = 0; bt < no_of_batches; bt++)
        for (unsigned i = 0; i < no_of_dense_unit; i++) {
          unsigned index = i;
          unsigned data_index = index + bt * no_of_dense_unit +
                                ep * no_of_dense_unit * no_of_batches;
          EXPECT_NEAR(updated_bias[ep][bt][0].getData()[index],
                      dense_layer_Test_4_updated_bias_data[data_index], 1e-6)
              << "at: epoch " << ep << ", at: batch " << bt << ", index "
              << index;
        }
  }

  /* --- target output test --- */
  {
    std::vector<std::float64_t> dense_layer_Test_4_predicted_output_data =
        load_bin("test/data/DenseLayer_Test_2_output_batch_data.bin",
                 epoch * no_of_batches * batch_size * no_of_dense_unit);
    for (unsigned ep = 0; ep < 10; ep++)
      for (unsigned bt = 0; bt < no_of_batches; bt++)
        for (unsigned j = 0; j < batch_size; j++)
          for (unsigned i = 0; i < no_of_dense_unit; i++) {
            unsigned index = i + j * no_of_dense_unit;
            unsigned data_index =
                index + bt * no_of_dense_unit * batch_size +
                ep * no_of_dense_unit * batch_size * no_of_batches;
            EXPECT_NEAR(outputs[ep][bt][0].getData()[index],
                        dense_layer_Test_4_predicted_output_data[data_index],
                        1e-6)
                << "at: epoch " << ep << ", index " << index;
          }
  }

  /* --- updated weights per batch test --- */
  {
    std::vector<std::float64_t> dense_layer_Test_4_grad_weight_data = load_bin(
        "test/data/DenseLayer_Test_2_grad_weight_batch_data.bin",
        epoch * no_of_batches * no_of_input_feature * no_of_dense_unit);
    for (unsigned ep = 0; ep < epoch; ep++)
      for (unsigned bt = 0; bt < no_of_batches; bt++)
        for (unsigned j = 0; j < no_of_input_feature; j++)
          for (unsigned i = 0; i < no_of_dense_unit; i++) {
            unsigned index = i + j * no_of_dense_unit;
            unsigned data_index =
                index + bt * no_of_dense_unit * no_of_input_feature +
                ep * no_of_dense_unit * no_of_input_feature * no_of_batches;
            EXPECT_NEAR(grad_weights[ep][bt][0].getData()[index],
                        dense_layer_Test_4_grad_weight_data[data_index], 1e-6)
                << "at: epoch " << ep << ", index " << index;
          }
  }

  /* --- loss per batch test --- */
  {
    std::vector<std::float64_t> dense_layer_Test_4_grad_bias_data =
        load_bin("test/data/DenseLayer_Test_2_grad_bias_batch_data.bin",
                 epoch * no_of_batches * no_of_dense_unit);
    for (unsigned ep = 0; ep < epoch; ep++)
      for (unsigned bt = 0; bt < no_of_batches; bt++)
        for (unsigned i = 0; i < no_of_dense_unit; i++) {
          unsigned index = i;
          unsigned data_index = index + bt * no_of_dense_unit +
                                ep * no_of_dense_unit * no_of_batches;
          EXPECT_NEAR(grad_bias[ep][bt][0].getData()[index],
                      dense_layer_Test_4_grad_bias_data[data_index], 1e-6)
              << "at: epoch " << ep << ", batch:" << bt << ", index " << index;
        }
  }
}

TEST_F(FrameworkTest, DenseLayer_Test_3) {

  tf::tensor x;
  tf::tensor weight_1, weight_2, bias_1, bias_2;
  tf::tensor input, target_output, validation_data;

  unsigned sample_size = 512;
  unsigned batch_size = 256;
  unsigned no_of_input_feature = 27;
  unsigned no_of_dense_unit_1 = 31;
  unsigned no_of_dense_unit_2 = 7;

  x.tf_create(tf_float64, no_of_input_feature, batch_size);
  input.tf_create(tf_float64, no_of_input_feature, sample_size);
  weight_1.tf_create(tf_float64, no_of_dense_unit_1, no_of_input_feature);
  bias_1.tf_create(tf_float64, no_of_dense_unit_1, 1);
  weight_2.tf_create(tf_float64, no_of_dense_unit_2, no_of_dense_unit_1);
  bias_2.tf_create(tf_float64, no_of_dense_unit_2, 1);
  target_output.tf_create(tf_float64, no_of_dense_unit_2, sample_size);

  /* --- read all the data and save in buffer --- */
  input.tensor_of(load_bin("test/data/DenseLayer_Test_3_input_sample.bin",
                           sample_size * no_of_input_feature)
                      .data());
  weight_1.tensor_of(load_bin("test/data/DenseLayer_Test_3_weights_1.bin",
                              no_of_input_feature * no_of_dense_unit_1)
                         .data()); // random values generated with numpy
  bias_1.tensor_of(
      load_bin("test/data/DenseLayer_Test_3_bias_1.bin", no_of_dense_unit_1)
          .data()); // zeros
  weight_2.tensor_of(load_bin("test/data/DenseLayer_Test_3_weights_2.bin",
                              no_of_dense_unit_1 * no_of_dense_unit_2)
                         .data()); // random values generated with numpy
  bias_2.tensor_of(
      load_bin("test/data/DenseLayer_Test_3_bias_2.bin", no_of_dense_unit_2)
          .data()); // zeros
  target_output.tensor_of(
      load_bin("test/data/DenseLayer_Test_3_target_output.bin",
               sample_size * no_of_dense_unit_2)
          .data());

  auto dense_1 = tf::layer::dense(no_of_dense_unit_1);
  auto dense_2 = tf::layer::dense(no_of_dense_unit_2);
  auto dense_1_output = dense_1({x});
  auto dense_output = dense_2({dense_1_output});

  dense_1.set_weight(weight_1);
  dense_1.set_bias(bias_1);

  dense_2.set_weight(weight_2);
  dense_2.set_bias(bias_2);

  /* --- model creation --- */
  tf::model mymodel({x}, dense_output);
  mymodel.shuffle(false);
  mymodel.compile(OptimizerType::SGD, LossType::squared_error);
  /* --- end model creation --- */

  tf::loss loss_sgd = mymodel.get_model_loss(dense_output[0]);
  tf::callback::earlystopping early_stopping(loss_sgd, 5, 1e-3, true);

  /* --- training the model ----*/
  unsigned epoch = 100;
  auto hist = mymodel.fit({input}, {target_output}, {early_stopping.callback()},
                          epoch, batch_size);

  /* --- validating loss --- */
  std::vector<std::float64_t> loss_data =
      load_bin("test/data/DenseLayer_Test_3_epoch_loss.bin", epoch);
  unsigned i = 0;
  for (std::float64_t scalar_loss : hist["loss"]) {
    EXPECT_NEAR(scalar_loss, loss_data[i], 1e-4) << "at" << i;
    i++;
  }
}