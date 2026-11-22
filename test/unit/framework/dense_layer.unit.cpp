#include <memory>

// library headers
#include "../LinearAlgebraFixtures.unit.hpp"
#include <LearnDeep/api/tensor.h>
#include <callback/callback.hpp>
#include <gtest/gtest.h>

#include <iostream>
#include <stdfloat>
#include <string>
#include <string_view>
#include <vector>

TEST_F(FrameworkTest, DenseLayer_Test_1) {

  constexpr unsigned no_of_input_feature = 43;
  constexpr unsigned no_of_dense_unit = 32;
  constexpr unsigned sample_size = 128;

  auto input_data = load_bin("test/data/DenseLayer_Test_1_input_data.bin",
                             no_of_input_feature * sample_size);
  auto target_data = load_bin("test/data/DenseLayer_Test_1_target_data.bin",
                              no_of_dense_unit * sample_size);
  auto weight_data = load_bin("test/data/DenseLayer_Test_1_weights_data.bin",
                              no_of_dense_unit * no_of_input_feature);
  auto bias_data =
      load_bin("test/data/DenseLayer_Test_1_bias_data.bin", no_of_dense_unit);
  auto expected_output = load_bin("test/data/DenseLayer_Test_1_output_data.bin",
                                  no_of_dense_unit * sample_size);
  auto expected_loss = load_bin("test/data/DenseLayer_Test_1_loss_data.bin", 1);
  auto expected_grad_weight =
      load_bin("test/data/DenseLayer_Test_1_grad_weight_data.bin",
               no_of_dense_unit * no_of_input_feature);
  auto expected_grad_bias = load_bin(
      "test/data/DenseLayer_Test_1_grad_bias_data.bin", no_of_dense_unit);
  auto expected_grad_loss =
      load_bin("test/data/DenseLayer_Test_1_grad_loss_data.bin",
               no_of_dense_unit * sample_size);
  auto expected_updated_weights =
      load_bin("test/data/DenseLayer_Test_1_updated_weights_data.bin",
               no_of_dense_unit * no_of_input_feature);
  auto expected_updated_bias = load_bin(
      "test/data/DenseLayer_Test_1_updated_bias_data.bin", no_of_dense_unit);

  tf::tensor x;
  tf::tensor weight, bias;
  tf::tensor input, target_output, validation_data;

  x.tf_create(tf_float64, no_of_input_feature, sample_size);
  input.tf_create(tf_float64, no_of_input_feature, sample_size);
  weight.tf_create(tf_float64, no_of_dense_unit, no_of_input_feature);
  bias.tf_create(tf_float64, no_of_dense_unit, 1);
  target_output.tf_create(tf_float64, no_of_dense_unit, sample_size);

  input.tensor_of(input_data.data());
  weight.tensor_of(weight_data.data());
  bias.tensor_of(bias_data.data());
  target_output.tensor_of(target_data.data());

  auto dense_1 = tf::layer::dense(32);
  auto dense_output = dense_1({x});
  dense_1.set_weight(weight);
  dense_1.set_bias(bias);

  /* --- call back setting --- */
  tf::callback::trace call_back;
  call_back.record_parameter_on_batch_begin(
      dense_1, "training_input", false);

  call_back.record_parameter_on_batch_end(
      dense_1, "training_weight", false);

  call_back.record_parameter_on_batch_end(
      dense_1, "training_bias", false);

  call_back.record_parameter_on_batch_end(
      dense_1, "training_output", false);

  call_back.record_parameter_on_batch_end(
      dense_1, "grad_weight", false);

  call_back.record_parameter_on_batch_end(
      dense_1, "grad_bias", false);

  call_back.record_parameter_on_batch_end(
      dense_1, "updated_weight", false);

  call_back.record_parameter_on_batch_end(
      dense_1, "updated_bias", false);
  /* --- call back setting end --- */

  /* --- model creation --- */
  tf::model mymodel({x}, dense_output);
  mymodel.shuffle(false);
  mymodel.compile(OptimizerType::SGD, tf::loss::type::squared_error);
  /* --- end model creation --- */

  auto loss_sgd = mymodel.get_model_loss(dense_output[0]);
  call_back.record_tensor_loss_on_batch_end(
      loss_sgd, tf::loss::parameter::squared_error_predicted_output);

  call_back.record_tensor_loss_on_batch_end(
      loss_sgd, tf::loss::parameter::squared_error_grad_predicted_output);

  call_back.record_scalar_loss_on_batch_end(loss_sgd);

  /* --- model training --- */
  unsigned epoch = 1;
  unsigned batch_size = 128;
  mymodel.fit({input}, {target_output}, {call_back.callback()}, epoch,
              batch_size);

  /* --- Intercepting  training parameters --- */
  auto outputs = call_back.get_parameter_on_batch_end(
      dense_1, "training_output");

  auto training_weights = call_back.get_parameter_on_batch_end(
      dense_1, "training_weight");

  auto train_dense_1_grad_weights = call_back.get_parameter_on_batch_end(
      dense_1, "grad_weight");

  auto train_dense_1_grad_bias = call_back.get_parameter_on_batch_end(
      dense_1, "grad_bias");

  auto train_dense_1_updated_weights = call_back.get_parameter_on_batch_end(
      dense_1, "updated_weight");

  auto train_dense_1_updated_bias = call_back.get_parameter_on_batch_end(
      dense_1, "updated_bias");

  auto training_losses = call_back.get_tensor_loss_on_batch_end(
      loss_sgd, tf::loss::parameter::squared_error_predicted_output);

  auto grad_training_losses = call_back.get_tensor_loss_on_batch_end(
      loss_sgd, tf::loss::parameter::squared_error_grad_predicted_output);

  for (unsigned ep = 0; ep < epoch; ep++) {
    ASSERT_EQ(outputs[ep].size(), 1);
    for (int j = 0; j < 128; j++)
      for (int i = 0; i < 32; i++)
        EXPECT_NEAR(outputs[ep][0][0].getData()[i + j * 32],
                    expected_output[i + j * 32], 1e-6)
            << "at: " << i + j * 3;
  }

  for (unsigned ep = 0; ep < epoch; ep++) {
    ASSERT_EQ(training_losses[ep][0][0].getNoOfElem(), 1);
    EXPECT_NEAR(training_losses[ep][0][0].getData()[0], expected_loss[0], 1e-5)
        << "at ep: " << ep;
  }

  for (unsigned ep = 0; ep < epoch; ep++)
    for (int j = 0; j < 43; j++)
      for (int i = 0; i < 32; i++)
        EXPECT_NEAR(train_dense_1_grad_weights[ep][0][0].getData()[i + j * 32],
                    expected_grad_weight[i + j * 32], 1e-5)
            << "at ep : " << ep << " i: " << i << " j: " << j;

  for (unsigned ep = 0; ep < epoch; ep++)
    for (int i = 0; i < 32; i++)
      EXPECT_NEAR(train_dense_1_grad_bias[ep][0][0].getData()[i],
                  expected_grad_bias[i], 1e-5)
          << "at i: " << i;

  for (unsigned ep = 0; ep < epoch; ep++)
    for (int j = 0; j < 128; j++)
      for (int i = 0; i < 32; i++)
        EXPECT_NEAR(grad_training_losses[ep][0][0].getData()[i + j * 32],
                    expected_grad_loss[i + j * 32], 1e-5)
            << "at i: " << i << " j: " << j;

  for (unsigned ep = 0; ep < epoch; ep++)
    for (int j = 0; j < 43; j++)
      for (int i = 0; i < 32; i++)
        EXPECT_NEAR(
            train_dense_1_updated_weights[ep][0][0].getData()[i + j * 32],
            expected_updated_weights[i + j * 32], 1e-5)
            << "at i: " << i << " j: " << j;

  for (unsigned ep = 0; ep < epoch; ep++)
    for (int i = 0; i < 32; i++)
      EXPECT_NEAR(train_dense_1_updated_bias[ep][0][0].getData()[i],
                  expected_updated_bias[i], 1e-5)
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
      dense_1, "training_input", false);

  call_back.record_parameter_on_batch_end(
      dense_1, "updated_weight", false);

  call_back.record_parameter_on_batch_end(
      dense_1, "updated_bias", false);

  call_back.record_parameter_on_batch_end(
      dense_1, "training_output", false);

  call_back.record_parameter_on_batch_end(
      dense_1, "grad_weight", false);

  call_back.record_parameter_on_batch_end(
      dense_1, "grad_bias", false);

  /* --- call back setting end --- */

  /* --- model creation --- */
  tf::model mymodel({x}, dense_output);
  mymodel.shuffle(false);
  mymodel.compile(OptimizerType::SGD, tf::loss::type::squared_error);
  /* --- end model creation --- */

  auto loss_sgd = mymodel.get_model_loss(dense_output[0]);
  call_back.record_tensor_loss_on_batch_end(
      loss_sgd, tf::loss::parameter::squared_error_predicted_output);

  call_back.record_tensor_loss_on_batch_end(
      loss_sgd, tf::loss::parameter::squared_error_grad_predicted_output);

  call_back.record_tensor_loss_on_batch_end(
      loss_sgd, tf::loss::parameter::squared_error_target_output);

  call_back.record_scalar_loss_on_batch_end(loss_sgd);

  tf::callback::earlystopping early_stopping(loss_sgd, 10, 1.0, true);

  /* --- model training --- */
  unsigned epoch = 10;
  mymodel.fit({input}, {target_output},
              {call_back.callback(), early_stopping.callback()}, epoch,
              batch_size);

  /* --- Intercepting  training parameters --- */
  std::vector<std::vector<std::vector<tf::tensor>>> dense_training_inputs =
      call_back.get_parameter_on_batch_begin(
          dense_1, "training_input");

  std::vector<std::vector<std::vector<tf::tensor>>> update_weights =
      call_back.get_parameter_on_batch_end(
          dense_1, "updated_weight");

  std::vector<std::vector<std::vector<tf::tensor>>> updated_bias =
      call_back.get_parameter_on_batch_end(dense_1,
                                           "updated_bias");

  std::vector<std::vector<std::vector<tf::tensor>>> outputs =
      call_back.get_parameter_on_batch_end(
          dense_1, "training_output");

  std::vector<std::vector<std::vector<tf::tensor>>> training_losses =
      call_back.get_tensor_loss_on_batch_end(
          loss_sgd, tf::loss::parameter::squared_error_predicted_output);

  std::vector<std::vector<std::vector<tf::tensor>>> grad_training_losses =
      call_back.get_tensor_loss_on_batch_end(
          loss_sgd, tf::loss::parameter::squared_error_grad_predicted_output);

  std::vector<std::vector<std::vector<tf::tensor>>> target_training_output =
      call_back.get_tensor_loss_on_batch_end(
          loss_sgd, tf::loss::parameter::squared_error_target_output);

  std::vector<std::vector<std::vector<tf::tensor>>> grad_weights =
      call_back.get_parameter_on_batch_end(dense_1,
                                           "grad_weight");

  std::vector<std::vector<std::vector<tf::tensor>>> grad_bias =
      call_back.get_parameter_on_batch_end(dense_1,
                                           "grad_bias");

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
                 epoch * no_of_batches);
    for (unsigned ep = 0; ep < epoch; ep++)
      for (unsigned bt = 0; bt < no_of_batches; bt++) {
        ASSERT_EQ(training_losses[ep][bt][0].getNoOfElem(), 1);
        unsigned data_index = bt + ep * no_of_batches;
        EXPECT_NEAR(training_losses[ep][bt][0].getData()[0],
                    dense_layer_Test_2_loss_batch_data[data_index], 1e-6)
            << "at: epoch " << ep << ", batch " << bt;
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
  unsigned batch_size = 32;
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
  mymodel.compile(OptimizerType::SGD, tf::loss::type::squared_error);
  /* --- end model creation --- */

  auto loss_sgd = mymodel.get_model_loss(dense_output[0]);

  /* --- training the model ----*/
  unsigned epoches = 8;
  auto hist = mymodel.fit({input}, {target_output}, {}, epoches, batch_size);

  /* --- validating loss --- */
  std::vector<std::float64_t> expected_losses =
      load_bin("test/data/DenseLayer_Test_3_batch_loss.bin",
               epoches * (sample_size / batch_size));

  for (unsigned epoch = 0; epoch < hist["loss"].size(); epoch++) {
    for (unsigned batch = 0; batch < sample_size / batch_size; batch++) {
      unsigned index = epoch * (sample_size / batch_size) + batch;
      EXPECT_NEAR(hist["loss"][epoch][batch], expected_losses[index], 1e-4)
          << "at epoch " << epoch << ", batch " << batch
          << " loss: " << hist["loss"][epoch][batch] << "\n";
    }
  }

  auto model_prediction_output = mymodel.predict({input});

  std::vector<std::float64_t> prediction_history =
      load_bin("test/data/DenseLayer_Test_3_final_prediction.bin",
               no_of_dense_unit_2 * sample_size);
  //   output[0].print_data();

  for (unsigned i = 0; i < sample_size; i++) {
    for (unsigned j = 0; j < no_of_dense_unit_2; j++) {
      unsigned index = i * no_of_dense_unit_2 + j;
      EXPECT_NEAR(model_prediction_output[0].getData()[index],
                  prediction_history[index], 1e-4)
          << "at " << index << "\n";
    }
  }
}

TEST_F(FrameworkTest, DenseLayer_Test_4) {
  tf::tensor x;
  tf::tensor weight_1, bias_1, weight_2, bias_2;
  tf::tensor input, output;

  unsigned sample_size = 4;
  unsigned no_of_input_feature = 8;
  unsigned no_of_dense_unit_1 = 16;
  unsigned no_of_dense_unit_2 = 8;

  x.tf_create(tf_float64, no_of_input_feature, sample_size);
  input.tf_create(tf_float64, no_of_input_feature, sample_size);
  weight_1.tf_create(tf_float64, no_of_dense_unit_1, no_of_input_feature);
  bias_1.tf_create(tf_float64, no_of_dense_unit_1, 1);
  weight_2.tf_create(tf_float64, no_of_dense_unit_2, no_of_dense_unit_1);
  bias_2.tf_create(tf_float64, no_of_dense_unit_2, 1);
  output.tf_create(tf_float64, no_of_dense_unit_2, sample_size);

  /* --- read all the data and save in buffer --- */
  input.tensor_of(load_bin("test/data/DenseLayer_Test_4_input.bin",
                           sample_size * no_of_input_feature)
                      .data());
  weight_1.tensor_of(load_bin("test/data/DenseLayer_Test_4_weight_1.bin",
                              no_of_input_feature * no_of_dense_unit_1)
                         .data()); // random values generated with numpy
  bias_1.tensor_of(
      load_bin("test/data/DenseLayer_Test_4_bias_1.bin", no_of_dense_unit_1)
          .data()); // zeros
  weight_2.tensor_of(load_bin("test/data/DenseLayer_Test_4_weight_2.bin",
                              no_of_dense_unit_1 * no_of_dense_unit_2)
                         .data()); // random values generated with numpy
  bias_2.tensor_of(
      load_bin("test/data/DenseLayer_Test_4_bias_2.bin", no_of_dense_unit_2)
          .data()); // zeros
  output.tensor_of(load_bin("test/data/DenseLayer_Test_4_expected_output.bin",
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

  auto predicted_output = mymodel.predict({input});
  for (unsigned i = 0; i < sample_size * no_of_dense_unit_2; i++) {
    EXPECT_NEAR(predicted_output[0].getData()[i], output.getData()[i], 1e-6)
        << "at " << i;
  }
}

/**
  == == == == == == == == == == == == == == == == == == == == == == == == == ==
 Composite_Layer_Prediction_Test_5

  Model:
  Dense(19)->ReLU->Dense(11)->Softmax

  Prediction only : no training, no gradients.
  == == == == == == == == == == == == == == == == == == == == == == == == == ==
*/
TEST_F(FrameworkTest, Composite_Layer_Prediction_Test_5) {

  tf::tensor x;
  tf::tensor weight_1, bias_1, weight_2, bias_2;
  tf::tensor input, output;

  unsigned sample_size = 3050;
  unsigned no_of_input_feature = 17;
  unsigned no_of_dense_unit_1 = 19;
  unsigned no_of_dense_unit_2 = 11;

  x.tf_create(tf_float64, no_of_input_feature, sample_size);
  input.tf_create(tf_float64, no_of_input_feature, sample_size);
  weight_1.tf_create(tf_float64, no_of_dense_unit_1, no_of_input_feature);
  bias_1.tf_create(tf_float64, no_of_dense_unit_1, 1);
  weight_2.tf_create(tf_float64, no_of_dense_unit_2, no_of_dense_unit_1);
  bias_2.tf_create(tf_float64, no_of_dense_unit_2, 1);
  output.tf_create(tf_float64, no_of_dense_unit_2, sample_size);

  /* --- read all the data and save in buffer --- */
  input.tensor_of(
      load_bin("test/data/Composite_Layer_Prediction_Test_5_input.bin",
               sample_size * no_of_input_feature)
          .data());
  weight_1.tensor_of(
      load_bin("test/data/Composite_Layer_Prediction_Test_5_weight_1.bin",
               no_of_input_feature * no_of_dense_unit_1)
          .data()); // random values generated with numpy
  bias_1.tensor_of(
      load_bin("test/data/Composite_Layer_Prediction_Test_5_bias_1.bin",
               no_of_dense_unit_1)
          .data()); // zeros
  weight_2.tensor_of(
      load_bin("test/data/Composite_Layer_Prediction_Test_5_weight_2.bin",
               no_of_dense_unit_1 * no_of_dense_unit_2)
          .data()); // random values generated with numpy
  bias_2.tensor_of(
      load_bin("test/data/Composite_Layer_Prediction_Test_5_bias_2.bin",
               no_of_dense_unit_2)
          .data()); // zeros
  output.tensor_of(
      load_bin(
          "test/data/Composite_Layer_Prediction_Test_5_expected_output.bin",
          sample_size * no_of_dense_unit_2)
          .data());

  auto dense_1 = tf::layer::dense(no_of_dense_unit_1);
  auto relu = tf::layer::relu();
  auto dense_2 = tf::layer::dense(no_of_dense_unit_2);
  auto softmax = tf::layer::softmax();

  dense_1.set_weight(weight_1);
  dense_1.set_bias(bias_1);
  dense_2.set_weight(weight_2);
  dense_2.set_bias(bias_2);

  // model architecture
  auto dense_1_output = dense_1({x});
  auto relu_output = relu(dense_1_output);
  auto dense_2_output = dense_2(relu_output);
  auto softmax_output = softmax(dense_2_output);

  /* --- model creation --- */
  tf::model mymodel({x}, softmax_output);

  /* --- predictiona and assertion --- */
  auto predicted_output = mymodel.predict({input});
  for (unsigned i = 0; i < sample_size * no_of_dense_unit_2; i++) {
    EXPECT_NEAR(predicted_output[0].getData()[i], output.getData()[i], 1e-6)
        << "at " << i;
  }
}

TEST_F(FrameworkTest, Quadratic_Func_Fit_Test) {

  tf::tensor x;
  tf::tensor weight_1, weight_2, bias_1, bias_2;
  tf::tensor input, target_output;

  unsigned sample_size = 1024;
  unsigned batch_size = 32;
  unsigned epoch = 1000;
  unsigned no_of_batches = sample_size / batch_size;
  unsigned no_of_input_feature = 1;
  unsigned no_of_dense_unit_1 = 8;
  unsigned no_of_dense_unit_2 = 1;

  x.tf_create(tf_float64, no_of_input_feature, batch_size);
  input.tf_create(tf_float64, no_of_input_feature, sample_size);
  weight_1.tf_create(tf_float64, no_of_dense_unit_1, no_of_input_feature);
  bias_1.tf_create(tf_float64, no_of_dense_unit_1, 1);
  weight_2.tf_create(tf_float64, no_of_dense_unit_2, no_of_dense_unit_1);
  bias_2.tf_create(tf_float64, no_of_dense_unit_2, 1);
  target_output.tf_create(tf_float64, no_of_dense_unit_2, sample_size);

  input.tensor_of(
      load_bin("test/data/Quadratic_ReluDense_Test_1_input_data.bin",
               no_of_input_feature * sample_size)
          .data());
  target_output.tensor_of(
      load_bin("test/data/Quadratic_ReluDense_Test_1_target_output_data.bin",
               no_of_dense_unit_2 * sample_size)
          .data());
  weight_1.tensor_of(
      load_bin("test/data/Quadratic_ReluDense_Test_1_initial_weights_1.bin",
               no_of_dense_unit_1 * no_of_input_feature)
          .data());
  bias_1.tensor_of(
      load_bin("test/data/Quadratic_ReluDense_Test_1_initial_bias_1.bin",
               no_of_dense_unit_1)
          .data());
  weight_2.tensor_of(
      load_bin("test/data/Quadratic_ReluDense_Test_1_initial_weights_2.bin",
               no_of_dense_unit_2 * no_of_dense_unit_1)
          .data());
  bias_2.tensor_of(
      load_bin("test/data/Quadratic_ReluDense_Test_1_initial_bias_2.bin",
               no_of_dense_unit_2)
          .data());

  auto dense_1 = tf::layer::dense(no_of_dense_unit_1);
  auto relu_layer = tf::layer::relu();
  auto dense_2 = tf::layer::dense(no_of_dense_unit_2);

  auto dense_1_output = dense_1({x});
  auto relu_output = relu_layer(dense_1_output);
  auto predicted_output = dense_2(relu_output);

  dense_1.set_weight(weight_1);
  dense_1.set_bias(bias_1);
  dense_2.set_weight(weight_2);
  dense_2.set_bias(bias_2);

  tf::callback::trace call_back;
  call_back.record_parameter_on_batch_begin(
      dense_1, "training_input", false);
  call_back.record_parameter_on_batch_end(
      dense_2, "training_output", false);
  call_back.record_parameter_on_batch_end(
      dense_1, "updated_weight", false);
  call_back.record_parameter_on_batch_end(
      dense_1, "updated_bias", false);
  call_back.record_parameter_on_batch_end(
      dense_2, "updated_weight", false);
  call_back.record_parameter_on_batch_end(
      dense_2, "updated_bias", false);

  tf::model mymodel({x}, predicted_output);
  mymodel.shuffle(false);
  mymodel.compile(OptimizerType::SGD, tf::loss::type::squared_error);

  auto loss_sgd = mymodel.get_model_loss(predicted_output[0]);
  call_back.record_tensor_loss_on_batch_end(
      loss_sgd, tf::loss::parameter::squared_error_target_output);
  call_back.record_scalar_loss_on_batch_end(loss_sgd);
  call_back.record_scalar_loss_on_epoch_end(loss_sgd);

  tf::history_container hist = mymodel.fit(
      {input}, {target_output}, {call_back.callback()}, epoch, batch_size);

  const std::string loss_name = loss_sgd.get_loss_name();
  const auto &epoch_loss_history = hist.history["loss"][loss_name];

  const auto expected_epoch_losses = load_bin(
      "test/data/Quadratic_ReluDense_Test_1_epoch_loss_history.bin", epoch);
  ASSERT_EQ(epoch_loss_history.size(), epoch);
  for (unsigned ep = 0; ep < epoch; ep++) {
    EXPECT_NEAR(epoch_loss_history[ep], expected_epoch_losses[ep], 1e-6)
        << "history loss at epoch " << ep;
  }

  std::vector<std::vector<std::vector<tf::tensor>>> input_batches =
      call_back.get_parameter_on_batch_begin(
          dense_1, "training_input");
  std::vector<std::vector<std::vector<tf::tensor>>> output_batches =
      call_back.get_parameter_on_batch_end(
          dense_2, "training_output");
  std::vector<std::vector<std::vector<tf::tensor>>> target_output_batches =
      call_back.get_tensor_loss_on_batch_end(
          loss_sgd, tf::loss::parameter::squared_error_target_output);
  std::vector<std::vector<std::float64_t>> scalar_losses =
      call_back.get_scaler_loss_on_batch_end(loss_sgd);

  std::cout << "Quadratic_Func_Fit_Test mean squared error: "
            << scalar_losses.back().back() << "\n";

  ASSERT_EQ(input_batches.size(), epoch);
  ASSERT_EQ(output_batches.size(), epoch);
  ASSERT_EQ(target_output_batches.size(), epoch);
  ASSERT_EQ(scalar_losses.size(), epoch);

  auto expected_input_batches =
      load_bin("test/data/Quadratic_ReluDense_Test_1_input_batch_history.bin",
               epoch * no_of_batches * batch_size * no_of_input_feature);
  auto expected_target_output_batches = load_bin(
      "test/data/Quadratic_ReluDense_Test_1_target_output_batch_history.bin",
      epoch * no_of_batches * batch_size * no_of_dense_unit_2);
  auto expected_predicted_output_batches = load_bin(
      "test/data/Quadratic_ReluDense_Test_1_predicted_output_batch_history.bin",
      epoch * no_of_batches * batch_size * no_of_dense_unit_2);
  auto expected_loss_batches =
      load_bin("test/data/Quadratic_ReluDense_Test_1_loss_batch_history.bin",
               epoch * no_of_batches);

  for (unsigned ep = 0; ep < epoch; ep++) {
    ASSERT_EQ(input_batches[ep].size(), no_of_batches);
    ASSERT_EQ(output_batches[ep].size(), no_of_batches);
    ASSERT_EQ(target_output_batches[ep].size(), no_of_batches);
    ASSERT_EQ(scalar_losses[ep].size(), no_of_batches);

    for (unsigned bt = 0; bt < no_of_batches; bt++) {
      ASSERT_EQ(input_batches[ep][bt].size(), 1);
      ASSERT_EQ(output_batches[ep][bt].size(), 1);
      ASSERT_EQ(target_output_batches[ep][bt].size(), 1);

      unsigned batch_data_offset =
          (ep * no_of_batches + bt) * batch_size * no_of_input_feature;
      for (unsigned j = 0; j < batch_size; j++) {
        for (unsigned i = 0; i < no_of_input_feature; i++) {
          unsigned index = i + j * no_of_input_feature;
          EXPECT_NEAR(input_batches[ep][bt][0].getData()[index],
                      expected_input_batches[batch_data_offset + index], 1e-6)
              << "input at epoch " << ep << ", batch " << bt << ", index "
              << index;
        }
      }

      batch_data_offset =
          (ep * no_of_batches + bt) * batch_size * no_of_dense_unit_2;
      for (unsigned j = 0; j < batch_size; j++) {
        for (unsigned i = 0; i < no_of_dense_unit_2; i++) {
          unsigned index = i + j * no_of_dense_unit_2;
          EXPECT_NEAR(target_output_batches[ep][bt][0].getData()[index],
                      expected_target_output_batches[batch_data_offset + index],
                      1e-6)
              << "target output at epoch " << ep << ", batch " << bt
              << ", index " << index;
          EXPECT_NEAR(
              output_batches[ep][bt][0].getData()[index],
              expected_predicted_output_batches[batch_data_offset + index],
              1e-5)
              << "predicted output at epoch " << ep << ", batch " << bt
              << ", index " << index;
        }
      }

      unsigned loss_index = ep * no_of_batches + bt;
      EXPECT_NEAR(scalar_losses[ep][bt], expected_loss_batches[loss_index],
                  1e-5)
          << "loss at epoch " << ep << ", batch " << bt;
    }
  }
}

TEST_F(FrameworkTest, Cubic_Func_Fit_Test) {

  tf::tensor x;
  tf::tensor weight_1, weight_2, bias_1, bias_2;
  tf::tensor input, target_output;

  unsigned sample_size = 1024;
  unsigned batch_size = 256;
  unsigned epoch = 1000;
  unsigned no_of_batches = sample_size / batch_size;
  unsigned no_of_input_feature = 1;
  unsigned no_of_dense_unit_1 = 16;
  unsigned no_of_dense_unit_2 = 1;

  x.tf_create(tf_float64, no_of_input_feature, batch_size);
  input.tf_create(tf_float64, no_of_input_feature, sample_size);
  weight_1.tf_create(tf_float64, no_of_dense_unit_1, no_of_input_feature);
  bias_1.tf_create(tf_float64, no_of_dense_unit_1, 1);
  weight_2.tf_create(tf_float64, no_of_dense_unit_2, no_of_dense_unit_1);
  bias_2.tf_create(tf_float64, no_of_dense_unit_2, 1);
  target_output.tf_create(tf_float64, no_of_dense_unit_2, sample_size);

  input.tensor_of(load_bin("test/data/Cubic_ReluDense_Test_1_input_data.bin",
                           no_of_input_feature * sample_size)
                      .data());
  target_output.tensor_of(
      load_bin("test/data/Cubic_ReluDense_Test_1_output_data.bin",
               no_of_dense_unit_2 * sample_size)
          .data());
  weight_1.tensor_of(
      load_bin("test/data/Cubic_ReluDense_Test_1_initial_weights_1.bin",
               no_of_dense_unit_1 * no_of_input_feature)
          .data());
  bias_1.tensor_of(
      load_bin("test/data/Cubic_ReluDense_Test_1_initial_bias_1.bin",
               no_of_dense_unit_1)
          .data());
  weight_2.tensor_of(
      load_bin("test/data/Cubic_ReluDense_Test_1_initial_weights_2.bin",
               no_of_dense_unit_2 * no_of_dense_unit_1)
          .data());
  bias_2.tensor_of(
      load_bin("test/data/Cubic_ReluDense_Test_1_initial_bias_2.bin",
               no_of_dense_unit_2)
          .data());

  auto dense_1 = tf::layer::dense(no_of_dense_unit_1);
  auto relu_layer = tf::layer::relu();
  auto dense_2 = tf::layer::dense(no_of_dense_unit_2);

  auto dense_1_output = dense_1({x});
  auto relu_output = relu_layer(dense_1_output);
  auto predicted_output = dense_2(relu_output);

  dense_1.set_weight(weight_1);
  dense_1.set_bias(bias_1);
  dense_2.set_weight(weight_2);
  dense_2.set_bias(bias_2);

  tf::callback::trace call_back;
  call_back.record_parameter_on_batch_begin(
      dense_1, "training_input", false);
  call_back.record_parameter_on_batch_end(
      dense_2, "training_output", false);
  call_back.record_parameter_on_batch_end(
      dense_1, "updated_weight", false);
  call_back.record_parameter_on_batch_end(
      dense_1, "updated_bias", false);
  call_back.record_parameter_on_batch_end(
      dense_2, "updated_weight", false);
  call_back.record_parameter_on_batch_end(
      dense_2, "updated_bias", false);

  tf::model mymodel({x}, predicted_output);
  mymodel.shuffle(false);
  mymodel.compile(OptimizerType::SGD, tf::loss::type::squared_error);

  auto loss_sgd = mymodel.get_model_loss(predicted_output[0]);
  call_back.record_tensor_loss_on_batch_end(
      loss_sgd, tf::loss::parameter::squared_error_target_output);
  call_back.record_scalar_loss_on_batch_end(loss_sgd);

  tf::history_container hist = mymodel.fit(
      {input}, {target_output}, {call_back.callback()}, epoch, batch_size);

  const std::string loss_name = loss_sgd.get_loss_name();
  const auto &epoch_loss_history = hist.history["loss"][loss_name];
  const auto expected_epoch_losses = load_bin(
      "test/data/Cubic_ReluDense_Test_1_epoch_loss_history.bin", epoch);
  ASSERT_EQ(epoch_loss_history.size(), epoch);
  ASSERT_EQ(expected_epoch_losses.size(), epoch);
  for (unsigned ep = 0; ep < epoch; ep++) {
    EXPECT_NEAR(epoch_loss_history[ep], expected_epoch_losses[ep], 1e-6)
        << "epoch loss at epoch " << ep;
  }

  std::vector<std::vector<std::vector<tf::tensor>>> input_batches =
      call_back.get_parameter_on_batch_begin(
          dense_1, "training_input");
  std::vector<std::vector<std::vector<tf::tensor>>> output_batches =
      call_back.get_parameter_on_batch_end(
          dense_2, "training_output");
  std::vector<std::vector<std::vector<tf::tensor>>> target_output_batches =
      call_back.get_tensor_loss_on_batch_end(
          loss_sgd, tf::loss::parameter::squared_error_target_output);
  std::vector<std::vector<std::float64_t>> scalar_losses =
      call_back.get_scaler_loss_on_batch_end(loss_sgd);
  std::vector<std::vector<std::vector<tf::tensor>>> dense_1_updated_weights =
      call_back.get_parameter_on_batch_end(
          dense_1, "updated_weight");
  std::vector<std::vector<std::vector<tf::tensor>>> dense_1_updated_bias =
      call_back.get_parameter_on_batch_end(dense_1,
                                           "updated_bias");
  std::vector<std::vector<std::vector<tf::tensor>>> dense_2_updated_weights =
      call_back.get_parameter_on_batch_end(
          dense_2, "updated_weight");
  std::vector<std::vector<std::vector<tf::tensor>>> dense_2_updated_bias =
      call_back.get_parameter_on_batch_end(dense_2,
                                           "updated_bias");

  ASSERT_EQ(input_batches.size(), epoch);
  ASSERT_EQ(output_batches.size(), epoch);
  ASSERT_EQ(target_output_batches.size(), epoch);
  ASSERT_EQ(scalar_losses.size(), epoch);
  ASSERT_EQ(dense_1_updated_weights.size(), epoch);
  ASSERT_EQ(dense_1_updated_bias.size(), epoch);
  ASSERT_EQ(dense_2_updated_weights.size(), epoch);
  ASSERT_EQ(dense_2_updated_bias.size(), epoch);

  auto expected_input_data =
      load_bin("test/data/Cubic_ReluDense_Test_1_input_data.bin",
               sample_size * no_of_input_feature);
  auto expected_target_output_data =
      load_bin("test/data/Cubic_ReluDense_Test_1_output_data.bin",
               sample_size * no_of_dense_unit_2);
  auto expected_predicted_output_batches =
      load_bin("test/data/Cubic_ReluDense_Test_1_predicted_batch_output.bin",
               epoch * no_of_batches * batch_size * no_of_dense_unit_2);
  auto expected_loss_batches =
      load_bin("test/data/Cubic_ReluDense_Test_1_batch_loss_history.bin",
               epoch * no_of_batches);

  ASSERT_EQ(dense_1_updated_weights.back().size(), no_of_batches);
  ASSERT_EQ(dense_1_updated_bias.back().size(), no_of_batches);
  ASSERT_EQ(dense_2_updated_weights.back().size(), no_of_batches);
  ASSERT_EQ(dense_2_updated_bias.back().size(), no_of_batches);
  ASSERT_EQ(dense_1_updated_weights.back().back().size(), 1);
  ASSERT_EQ(dense_1_updated_bias.back().back().size(), 1);
  ASSERT_EQ(dense_2_updated_weights.back().back().size(), 1);
  ASSERT_EQ(dense_2_updated_bias.back().back().size(), 1);

  const std::float64_t *final_w1 =
      dense_1_updated_weights.back().back()[0].getData();
  const std::float64_t *final_b1 =
      dense_1_updated_bias.back().back()[0].getData();
  const std::float64_t *final_w2 =
      dense_2_updated_weights.back().back()[0].getData();
  const std::float64_t *final_b2 =
      dense_2_updated_bias.back().back()[0].getData();

  std::float64_t full_dataset_mse = 0.0;
  for (unsigned sample = 0; sample < sample_size; sample++) {
    std::float64_t predicted = final_b2[0];
    for (unsigned hidden = 0; hidden < no_of_dense_unit_1; hidden++) {
      std::float64_t dense_output =
          final_w1[hidden] * expected_input_data[sample] + final_b1[hidden];
      std::float64_t relu_output = dense_output > 0.0 ? dense_output : 0.0;
      predicted += final_w2[hidden] * relu_output;
    }
    std::float64_t diff = predicted - expected_target_output_data[sample];
    full_dataset_mse += diff * diff;
  }
  full_dataset_mse /= sample_size;

  std::cout << "Cubic_Func_Fit_Test full-dataset mean squared error: "
            << full_dataset_mse << "\n";

  for (unsigned ep = 0; ep < epoch; ep++) {
    ASSERT_EQ(input_batches[ep].size(), no_of_batches);
    ASSERT_EQ(output_batches[ep].size(), no_of_batches);
    ASSERT_EQ(target_output_batches[ep].size(), no_of_batches);
    ASSERT_EQ(scalar_losses[ep].size(), no_of_batches);

    for (unsigned bt = 0; bt < no_of_batches; bt++) {
      ASSERT_EQ(input_batches[ep][bt].size(), 1);
      ASSERT_EQ(output_batches[ep][bt].size(), 1);
      ASSERT_EQ(target_output_batches[ep][bt].size(), 1);

      unsigned sample_offset = bt * batch_size * no_of_input_feature;
      for (unsigned j = 0; j < batch_size; j++) {
        for (unsigned i = 0; i < no_of_input_feature; i++) {
          unsigned index = i + j * no_of_input_feature;
          EXPECT_NEAR(input_batches[ep][bt][0].getData()[index],
                      expected_input_data[sample_offset + index], 1e-6)
              << "input at epoch " << ep << ", batch " << bt << ", index "
              << index;
        }
      }

      sample_offset = bt * batch_size * no_of_dense_unit_2;
      unsigned batch_data_offset =
          (ep * no_of_batches + bt) * batch_size * no_of_dense_unit_2;
      for (unsigned j = 0; j < batch_size; j++) {
        for (unsigned i = 0; i < no_of_dense_unit_2; i++) {
          unsigned index = i + j * no_of_dense_unit_2;
          EXPECT_NEAR(target_output_batches[ep][bt][0].getData()[index],
                      expected_target_output_data[sample_offset + index], 1e-6)
              << "target output at epoch " << ep << ", batch " << bt
              << ", index " << index;
          EXPECT_NEAR(
              output_batches[ep][bt][0].getData()[index],
              expected_predicted_output_batches[batch_data_offset + index],
              1e-5)
              << "predicted output at epoch " << ep << ", batch " << bt
              << ", index " << index;
        }
      }

      unsigned loss_index = ep * no_of_batches + bt;
      EXPECT_NEAR(scalar_losses[ep][bt], expected_loss_batches[loss_index],
                  1e-5)
          << "loss at epoch " << ep << ", batch " << bt;
    }
  }
}

TEST_F(FrameworkTest, Sin_x_Func_Fit_Test) {

  tf::tensor x;
  tf::tensor weight_1, weight_2, weight_3, bias_1, bias_2, bias_3;
  tf::tensor input, target_output;

  unsigned sample_size = 1025;
  unsigned batch_size = 512;
  unsigned epoch = 1000;
  unsigned no_of_batches = sample_size / batch_size;
  unsigned no_of_input_feature = 1;
  unsigned no_of_dense_unit_1 = 16;
  unsigned no_of_dense_unit_2 = 16;
  unsigned no_of_dense_unit_3 = 1;

  x.tf_create(tf_float64, no_of_input_feature, batch_size);
  input.tf_create(tf_float64, no_of_input_feature, sample_size);
  weight_1.tf_create(tf_float64, no_of_dense_unit_1, no_of_input_feature);
  bias_1.tf_create(tf_float64, no_of_dense_unit_1, 1);
  weight_2.tf_create(tf_float64, no_of_dense_unit_2, no_of_dense_unit_1);
  bias_2.tf_create(tf_float64, no_of_dense_unit_2, 1);
  weight_3.tf_create(tf_float64, no_of_dense_unit_3, no_of_dense_unit_2);
  bias_3.tf_create(tf_float64, no_of_dense_unit_3, 1);
  target_output.tf_create(tf_float64, no_of_dense_unit_3, sample_size);

  input.tensor_of(load_bin("test/data/Sin_ReluDense_Test_1_input_data.bin",
                           no_of_input_feature * sample_size)
                      .data());
  target_output.tensor_of(
      load_bin("test/data/Sin_ReluDense_Test_1_target_output_data.bin",
               no_of_dense_unit_3 * sample_size)
          .data());
  weight_1.tensor_of(
      load_bin("test/data/Sin_ReluDense_Test_1_initial_weights_1.bin",
               no_of_dense_unit_1 * no_of_input_feature)
          .data());
  bias_1.tensor_of(load_bin("test/data/Sin_ReluDense_Test_1_initial_bias_1.bin",
                            no_of_dense_unit_1)
                       .data());
  weight_2.tensor_of(
      load_bin("test/data/Sin_ReluDense_Test_1_initial_weights_2.bin",
               no_of_dense_unit_2 * no_of_dense_unit_1)
          .data());
  bias_2.tensor_of(load_bin("test/data/Sin_ReluDense_Test_1_initial_bias_2.bin",
                            no_of_dense_unit_2)
                       .data());
  weight_3.tensor_of(
      load_bin("test/data/Sin_ReluDense_Test_1_initial_weights_3.bin",
               no_of_dense_unit_3 * no_of_dense_unit_2)
          .data());
  bias_3.tensor_of(load_bin("test/data/Sin_ReluDense_Test_1_initial_bias_3.bin",
                            no_of_dense_unit_3)
                       .data());

  auto dense_1 = tf::layer::dense(no_of_dense_unit_1);
  auto relu_layer_1 = tf::layer::relu();
  auto dense_2 = tf::layer::dense(no_of_dense_unit_2);
  auto relu_layer_2 = tf::layer::relu();
  auto dense_3 = tf::layer::dense(no_of_dense_unit_3);

  auto dense_1_output = dense_1({x});
  auto relu_1_output = relu_layer_1(dense_1_output);
  auto dense_2_output = dense_2(relu_1_output);
  auto relu_2_output = relu_layer_2(dense_2_output);
  auto predicted_output = dense_3(relu_2_output);

  dense_1.set_weight(weight_1);
  dense_1.set_bias(bias_1);
  dense_2.set_weight(weight_2);
  dense_2.set_bias(bias_2);
  dense_3.set_weight(weight_3);
  dense_3.set_bias(bias_3);

  tf::model mymodel({x}, predicted_output);
  mymodel.shuffle(false);
  mymodel.compile(OptimizerType::SGD, tf::loss::type::squared_error);

  auto loss_sgd = mymodel.get_model_loss(predicted_output[0]);
  tf::callback::trace call_back;
  call_back.record_scalar_loss_on_batch_end(loss_sgd);

  tf::history_container hist = mymodel.fit(
      {input}, {target_output}, {call_back.callback()}, epoch, batch_size);

  const std::string loss_name = loss_sgd.get_loss_name();
  const auto &epoch_loss_history = hist.history["loss"][loss_name];
  const auto expected_epoch_losses =
      load_bin("test/data/Sin_ReluDense_Test_1_epoch_loss_history.bin", epoch);
  ASSERT_EQ(epoch_loss_history.size(), epoch);
  ASSERT_EQ(expected_epoch_losses.size(), epoch);
  for (unsigned ep = 0; ep < epoch; ep++) {
    EXPECT_NEAR(epoch_loss_history[ep], expected_epoch_losses[ep], 1e-6)
        << "epoch loss at epoch " << ep;
  }

  std::vector<std::vector<std::float64_t>> scalar_losses =
      call_back.get_scaler_loss_on_batch_end(loss_sgd);

  ASSERT_EQ(scalar_losses.size(), epoch);
  ASSERT_EQ(scalar_losses.back().size(), no_of_batches);

  auto expected_last_epoch_mse =
      load_bin("test/data/Sin_ReluDense_Test_1_last_epoch_mse.bin", 1);

  std::float64_t last_epoch_mse = 0.0;
  for (std::float64_t scalar_loss : scalar_losses.back()) {
    last_epoch_mse += scalar_loss;
  }
  last_epoch_mse /= scalar_losses.back().size();

  std::cout << "Sin_x_Func_Fit_Test last epoch mean squared error: "
            << last_epoch_mse << "\n";

  EXPECT_NEAR(last_epoch_mse, expected_last_epoch_mse[0], 1e-5);
}

// --- Early stopping callback tests -----------------------------------------

namespace {
// Minimal public-API loss behavior that feeds a predetermined sequence of
// epoch-end losses into the callback, so the stopping logic can be checked
// without training. It registers itself exactly like an out-of-tree plugin.
inline constexpr std::string_view kSequenceLossType{"sequence_loss"};
inline constexpr std::string_view kSequenceLossValues{"sequence_loss_values"};

class SequenceLoss final : public tf::loss::behavior {
public:
  explicit SequenceLoss(const tf::loss::config &config) {
    const auto *values =
        config.getAs<std::vector<double>>(std::string(kSequenceLossValues));
    if (values)
      this->values = *values;
  }

  std::string_view type() const override { return kSequenceLossType; }
  void forward(std::vector<tf::tensor *>,
               unsigned) override {}
  void backward() override {}
  void setTargetOutput(const std::vector<tf::tensor> &) override {}
  std::float64_t calculate(std::vector<tf::tensor> &,
                           std::vector<tf::tensor> &) override {
    return 0.0;
  }
  std::float64_t getScalerLoss() override { return 0.0; }
  std::float64_t getEpochEndScalerLoss() override {
    if (this->values.empty())
      return 0.0;
    if (this->cursor >= this->values.size())
      return this->values.back();
    return this->values[this->cursor++];
  }
  std::vector<tf::tensor *> getParameter(std::string_view) override {
    return {};
  }
  std::unordered_set<std::string> parameterNames() const override {
    return {std::string(kSequenceLossValues)};
  }
  unsigned getNoOfInputs() const override { return 1; }

private:
  std::vector<double> values;
  std::size_t cursor{0};
};

REGISTER_LOSS(std::string(kSequenceLossType),
              [](const tf::loss::config &config)
                  -> std::unique_ptr<tf::loss::behavior> {
                return std::make_unique<SequenceLoss>(config);
              });

tf::loss::loss makeSequenceLoss(const std::vector<std::float64_t> &values) {
  tf::loss::config config;
  config.set(std::string(kSequenceLossValues),
             std::vector<double>(values.begin(), values.end()));
  return tf::loss::loss(kSequenceLossType, config);
}
} // namespace

TEST(EarlyStoppingTest, StopsAfterPatienceEpochsWithoutImprovement) {
  // Big improvement at epoch 2, then three consecutive stalls (< 0.2 change).
  auto loss = makeSequenceLoss({10.0, 8.0, 7.95, 7.93, 7.92});
  CallbackEarlyStopping early_stopping(loss.id(),
                                       /*patience=*/3, /*min_delta=*/0.2,
                                       /*minimize=*/true);

  EXPECT_FALSE(early_stopping.stopEpoch()); // no losses recorded yet

  early_stopping.callOnEpochEnd(); // 10.0 -> baseline
  EXPECT_FALSE(early_stopping.stopEpoch());

  early_stopping.callOnEpochEnd(); // 8.0 -> real improvement, counter reset
  EXPECT_FALSE(early_stopping.stopEpoch());

  early_stopping.callOnEpochEnd(); // 7.95 -> stall 1
  EXPECT_FALSE(early_stopping.stopEpoch());

  early_stopping.callOnEpochEnd(); // 7.93 -> stall 2
  EXPECT_FALSE(early_stopping.stopEpoch());

  early_stopping.callOnEpochEnd(); // 7.92 -> stall 3 -> stop
  EXPECT_TRUE(early_stopping.stopEpoch());
}

TEST(EarlyStoppingTest, ImprovementResetsThePatienceCounter) {
  auto loss = makeSequenceLoss({10.0, 9.95, 8.0, 7.99, 7.98});
  CallbackEarlyStopping early_stopping(loss.id(),
                                       /*patience=*/2, /*min_delta=*/0.1,
                                       /*minimize=*/true);

  early_stopping.callOnEpochEnd(); // baseline
  early_stopping.callOnEpochEnd(); // 9.95 -> small change, stall 1
  EXPECT_FALSE(early_stopping.stopEpoch());

  early_stopping.callOnEpochEnd(); // 8.0 -> improvement, counter reset
  EXPECT_FALSE(early_stopping.stopEpoch());

  early_stopping.callOnEpochEnd(); // stall 1 again
  EXPECT_FALSE(early_stopping.stopEpoch());

  early_stopping.callOnEpochEnd(); // stall 2 -> stop
  EXPECT_TRUE(early_stopping.stopEpoch());
}

TEST(EarlyStoppingTest, MaximizeModeUsesUpperBound) {
  auto loss = makeSequenceLoss({0.1, 0.5, 0.51, 0.505});
  CallbackEarlyStopping early_stopping(loss.id(),
                                       /*patience=*/2, /*min_delta=*/0.1,
                                       /*minimize=*/false);

  early_stopping.callOnEpochEnd(); // baseline
  early_stopping.callOnEpochEnd(); // 0.5 -> improvement, counter reset
  EXPECT_FALSE(early_stopping.stopEpoch());

  early_stopping.callOnEpochEnd(); // 0.51 -> stall 1
  EXPECT_FALSE(early_stopping.stopEpoch());

  early_stopping.callOnEpochEnd(); // 0.505 -> stall 2 -> stop
  EXPECT_TRUE(early_stopping.stopEpoch());
}

TEST_F(FrameworkTest, EarlyStopping_StopsTrainingViaFit) {
  const unsigned sample_size = 64;
  const unsigned batch_size = 16;
  const unsigned no_of_input_feature = 4;
  const unsigned no_of_dense_unit = 3;
  const unsigned requested_epochs = 20;

  tf::tensor x, input, target_output, weight, bias;
  x.tf_create(tf_float64, no_of_input_feature, batch_size);
  input.tf_create(tf_float64, no_of_input_feature, sample_size);
  target_output.tf_create(tf_float64, no_of_dense_unit, sample_size);
  weight.tf_create(tf_float64, no_of_dense_unit, no_of_input_feature);
  bias.tf_create(tf_float64, no_of_dense_unit, 1);

  input.tensor_of(0.0, 1.0);
  target_output.tensor_of(0.0, 1.0);
  weight.tensor_of(0.1, 0.2);
  bias.tensor_of(0.0, 0.1);

  auto dense = tf::layer::dense(no_of_dense_unit);
  auto dense_output = dense({x});
  dense.set_weight(weight);
  dense.set_bias(bias);

  tf::model model({x}, dense_output);
  model.shuffle(false);
  model.compile(OptimizerType::SGD, tf::loss::type::squared_error);
  auto loss = model.get_model_loss(dense_output[0]);

  // A very large min_delta makes every post-baseline epoch a "stall", so with
  // patience 2 the fit loop must break after 3 epochs regardless of the data.
  tf::callback::earlystopping early_stopping(loss, 2, 1e6, true);

  tf::history_container hist =
      model.fit({input}, {target_output}, {early_stopping.callback()},
                requested_epochs, batch_size);

  const auto &epoch_losses = hist.history["loss"][loss.get_loss_name()];
  ASSERT_FALSE(epoch_losses.empty());
  EXPECT_LT(epoch_losses.size(), requested_epochs)
      << "early stopping should end training before " << requested_epochs
      << " epochs";
}

// --- Callback registration verification tests -------------------------------

TEST_F(FrameworkTest, FitWithNullCallback_DoesNotCrash) {
  constexpr unsigned sample_size = 32;
  constexpr unsigned batch_size = 16;
  constexpr unsigned no_of_input_feature = 4;
  constexpr unsigned no_of_dense_unit = 3;

  tf::tensor x, input, target_output, weight, bias;
  x.tf_create(tf_float64, no_of_input_feature, batch_size);
  input.tf_create(tf_float64, no_of_input_feature, sample_size);
  target_output.tf_create(tf_float64, no_of_dense_unit, sample_size);
  weight.tf_create(tf_float64, no_of_dense_unit, no_of_input_feature);
  bias.tf_create(tf_float64, no_of_dense_unit, 1);

  input.tensor_of(0.0, 1.0);
  target_output.tensor_of(0.0, 1.0);
  weight.tensor_of(0.1, 0.2);
  bias.tensor_of(0.0, 0.1);

  auto dense = tf::layer::dense(no_of_dense_unit);
  auto dense_output = dense({x});
  dense.set_weight(weight);
  dense.set_bias(bias);

  tf::model model({x}, dense_output);
  model.shuffle(false);
  model.compile(OptimizerType::SGD, tf::loss::type::squared_error);

  // An explicit null entry used to dereference a nullptr in the callback
  // runners; fit must complete instead.
  tf::history_container hist =
      model.fit({input}, {target_output}, {nullptr}, 2, batch_size);

  EXPECT_FALSE(hist.history.empty());
}

TEST_F(FrameworkTest, CallbackTrace_RejectsForeignLossRegistration) {
  constexpr unsigned batch_size = 16;
  constexpr unsigned no_of_input_feature = 4;
  constexpr unsigned no_of_dense_unit = 3;

  tf::tensor x1, w1, b1, x2, w2, b2;
  x1.tf_create(tf_float64, no_of_input_feature, batch_size);
  w1.tf_create(tf_float64, no_of_dense_unit, no_of_input_feature);
  b1.tf_create(tf_float64, no_of_dense_unit, 1);
  w1.tensor_of(0.1, 0.2);
  b1.tensor_of(0.0, 0.1);

  auto dense1 = tf::layer::dense(no_of_dense_unit);
  auto dense1_output = dense1({x1});
  dense1.set_weight(w1);
  dense1.set_bias(b1);
  tf::model model1({x1}, dense1_output);
  model1.compile(OptimizerType::SGD, tf::loss::type::squared_error);
  auto own_loss = model1.get_model_loss(dense1_output[0]);

  x2.tf_create(tf_float64, no_of_input_feature, batch_size);
  w2.tf_create(tf_float64, no_of_dense_unit, no_of_input_feature);
  b2.tf_create(tf_float64, no_of_dense_unit, 1);
  w2.tensor_of(0.1, 0.2);
  b2.tensor_of(0.0, 0.1);

  auto dense2 = tf::layer::dense(no_of_dense_unit);
  auto dense2_output = dense2({x2});
  dense2.set_weight(w2);
  dense2.set_bias(b2);
  tf::model model2({x2}, dense2_output);
  model2.compile(OptimizerType::SGD, tf::loss::type::squared_error);
  auto foreign_loss = model2.get_model_loss(dense2_output[0]);

  tf::callback::trace valid;
  valid.record_scalar_loss_on_batch_end(own_loss);
  EXPECT_TRUE(valid.callback()->callOnTrainingBegin(model1));

  tf::callback::trace foreign_scalar_batch;
  foreign_scalar_batch.record_scalar_loss_on_batch_end(foreign_loss);
  EXPECT_FALSE(foreign_scalar_batch.callback()->callOnTrainingBegin(model1));

  tf::callback::trace foreign_scalar_epoch;
  foreign_scalar_epoch.record_scalar_loss_on_epoch_end(foreign_loss);
  EXPECT_FALSE(foreign_scalar_epoch.callback()->callOnTrainingBegin(model1));

  tf::callback::trace foreign_tensor_batch;
  foreign_tensor_batch.record_tensor_loss_on_batch_end(
      foreign_loss, tf::loss::parameter::squared_error_predicted_output);
  EXPECT_FALSE(foreign_tensor_batch.callback()->callOnTrainingBegin(model1));

  tf::callback::trace foreign_tensor_epoch;
  foreign_tensor_epoch.record_tensor_loss_on_epoch_end(
      foreign_loss, tf::loss::parameter::squared_error_predicted_output);
  EXPECT_FALSE(foreign_tensor_epoch.callback()->callOnTrainingBegin(model1));
}

TEST_F(FrameworkTest, CallbackTrace_RejectsForeignLayerRegistration) {
  constexpr unsigned batch_size = 16;
  constexpr unsigned no_of_input_feature = 4;
  constexpr unsigned no_of_dense_unit = 3;

  tf::tensor x1, w1, b1, x2, w2, b2;
  x1.tf_create(tf_float64, no_of_input_feature, batch_size);
  w1.tf_create(tf_float64, no_of_dense_unit, no_of_input_feature);
  b1.tf_create(tf_float64, no_of_dense_unit, 1);
  w1.tensor_of(0.1, 0.2);
  b1.tensor_of(0.0, 0.1);

  auto dense1 = tf::layer::dense(no_of_dense_unit);
  auto dense1_output = dense1({x1});
  dense1.set_weight(w1);
  dense1.set_bias(b1);
  tf::model model1({x1}, dense1_output);
  model1.compile(OptimizerType::SGD, tf::loss::type::squared_error);

  x2.tf_create(tf_float64, no_of_input_feature, batch_size);
  w2.tf_create(tf_float64, no_of_dense_unit, no_of_input_feature);
  b2.tf_create(tf_float64, no_of_dense_unit, 1);
  w2.tensor_of(0.1, 0.2);
  b2.tensor_of(0.0, 0.1);

  auto dense2 = tf::layer::dense(no_of_dense_unit);
  auto dense2_output = dense2({x2});
  dense2.set_weight(w2);
  dense2.set_bias(b2);
  tf::model model2({x2}, dense2_output);
  model2.compile(OptimizerType::SGD, tf::loss::type::squared_error);

  ASSERT_NE(dense1.id(), dense2.id());

  // A model-owned layer must be accepted.
  tf::callback::trace valid;
  valid.record_parameter_on_epoch_begin(dense1, "weight");
  EXPECT_TRUE(valid.callback()->callOnTrainingBegin(model1));

  // A layer belonging to a different model must be rejected at every hook.
  tf::callback::trace foreign_epoch_begin;
  foreign_epoch_begin.record_parameter_on_epoch_begin(dense2, "weight");
  EXPECT_FALSE(foreign_epoch_begin.callback()->callOnTrainingBegin(model1));

  tf::callback::trace foreign_epoch_end;
  foreign_epoch_end.record_parameter_on_epoch_end(dense2, "weight");
  EXPECT_FALSE(foreign_epoch_end.callback()->callOnTrainingBegin(model1));

  tf::callback::trace foreign_batch_begin;
  foreign_batch_begin.record_parameter_on_batch_begin(dense2, "weight");
  EXPECT_FALSE(foreign_batch_begin.callback()->callOnTrainingBegin(model1));

  tf::callback::trace foreign_batch_end;
  foreign_batch_end.record_parameter_on_batch_end(dense2, "weight");
  EXPECT_FALSE(foreign_batch_end.callback()->callOnTrainingBegin(model1));
}

TEST_F(FrameworkTest, LayerApiExposesInputTensorAliases) {
  tf::tensor x;
  x.tf_create(tf_float64, 3, 2);

  tf::layer::dense dense(4);
  tf::layer::relu relu;
  tf::layer::softmax softmax;

  dense({x});
  relu({x});
  softmax({x});

  for (const std::vector<const tf::tensor *> &inputs :
       {dense.get_input_tensors(), relu.get_input_tensors(),
        softmax.get_input_tensors()}) {
    ASSERT_EQ(inputs.size(), 1u);
    ASSERT_NE(inputs[0], nullptr);
    EXPECT_EQ(inputs[0]->getPtr(), x.getPtr());
  }
}
