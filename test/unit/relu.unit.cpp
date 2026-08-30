#include <LearnDeep/api/tensor.h>
#include <gtest/gtest.h>

#include "LinearAlgebraFixtures.unit.hpp"

TEST_F(MathTest, Relu_Test_1) {

  tf::tensor A, C;
  A.tf_create(tf_float64, 63, 32);

  A.tensor_of(
      load_bin("test/data/Relu_Test_1_input_sample.bin", 63 * 32).data());

  C = A.relu();

  auto output = load_bin("test/data/Relu_Test_1_output_sample.bin", 63 * 32);
  for (int i = 0; i < 63 * 32; i++) {
    EXPECT_NEAR(C.getData()[i], output[i], 1e-6) << "at: " << i;
  }
}

TEST_F(MathTest, Relu_Test_2) {

  tf::tensor A, C;
  A.tf_create(tf_float64, 51, 48, 36, 72);

  A.tensor_of(
      load_bin("test/data/Relu_Test_2_input_sample.bin", 51 * 48 * 36 * 72)
          .data());

  {
    tf::graph_context ctx;

    C = A.relu();

    ctx.run();

    auto output =
        load_bin("test/data/Relu_Test_2_output_sample.bin", 51 * 48 * 36 * 72);
    for (int i = 0; i < 51 * 48 * 36 * 72; i++) {
      EXPECT_NEAR(C.getData()[i], output[i], 1e-6) << "at: " << i;
    }
  }
}

TEST_F(MathTest, Relu_Test_3) {

  tf::tensor A, A_grad, B, B_grad, C, C_grad, D, D_grad, E, E_grad, F, G;
  A.tf_create(tf_float64, 18, 32);
  B.tf_create(tf_float64, 18, 32);
  D.tf_create(tf_float64, 34, 18);

  auto A_input = load_bin("test/data/Relu_Test_3_A_input.bin", 32 * 18);
  auto B_input = load_bin("test/data/Relu_Test_3_B_input.bin", 32 * 18);
  auto D_input = load_bin("test/data/Relu_Test_3_D_input.bin", 18 * 34);

  auto C_output = load_bin("test/data/Relu_Test_3_C_output.bin", 32 * 18);
  auto E_output = load_bin("test/data/Relu_Test_3_E_output.bin", 32 * 18);
  auto F_output = load_bin("test/data/Relu_Test_3_F_output.bin", 32 * 34);
  auto G_output = load_bin("test/data/Relu_Test_3_G_output.bin", 32 * 18);

  auto A_grad_output = load_bin("test/data/Relu_Test_3_A_grad.bin", 32 * 18);
  auto B_grad_output = load_bin("test/data/Relu_Test_3_B_grad.bin", 32 * 18);
  auto C_grad_output = load_bin("test/data/Relu_Test_3_C_grad.bin", 32 * 18);
  auto D_grad_output = load_bin("test/data/Relu_Test_3_D_grad.bin", 18 * 34);
  auto E_grad_output = load_bin("test/data/Relu_Test_3_E_grad.bin", 32 * 18);

  A.tensor_of(A_input.data());
  B.tensor_of(B_input.data());
  D.tensor_of(D_input.data());
  {
    tf::graph_context ctx;

    C = A.sub(B);
    E = C.relu();
    F = E.matmul(D);
    G = E.pow(3);

    ctx.run();

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(C.getData()[i], C_output[i], 1e-6) << "C at: " << i;

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(E.getData()[i], E_output[i], 1e-6) << "E at: " << i;

    for (int i = 0; i < 32 * 34; i++)
      EXPECT_NEAR(F.getData()[i], F_output[i], 1e-6) << "F at: " << i;

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(G.getData()[i], G_output[i], 1e-6) << "G at: " << i;

    ctx.initialize_gradient();
    ctx.compute_gradient();

    A_grad = ctx.get_gradient(A);
    B_grad = ctx.get_gradient(B);
    C_grad = ctx.get_gradient(C);
    D_grad = ctx.get_gradient(D);
    E_grad = ctx.get_gradient(E);

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(A_grad.getData()[i], A_grad_output[i], 1e-6)
          << "A_grad at: " << i;

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(B_grad.getData()[i], B_grad_output[i], 1e-6)
          << "B_grad at: " << i;

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(C_grad.getData()[i], C_grad_output[i], 1e-6)
          << "C_grad at: " << i;

    for (int i = 0; i < 18 * 34; i++)
      EXPECT_NEAR(D_grad.getData()[i], D_grad_output[i], 1e-6)
          << "D_grad at: " << i;

    for (int i = 0; i < 32 * 18; i++)
      EXPECT_NEAR(E_grad.getData()[i], E_grad_output[i], 1e-6)
          << "E_grad at: " << i;
  }
}

TEST_F(FrameworkTest, ReluLayer_Test_1) {

  tf::tensor x;
  tf::tensor weight, bias;
  tf::tensor input, target_output, validation_data;

  unsigned input_features = 43;
  unsigned batch_size = 512;
  unsigned no_of_dense_neuron = 32;

  x.tf_create(tf_float64, 43, 512);
  input.tf_create(tf_float64, 43, 512);
  weight.tf_create(tf_float64, 32, 43);
  bias.tf_create(tf_float64, 32, 1);
  target_output.tf_create(tf_float64, 32, 512);

  input.tensor_of(load_bin("test/data/ReluDense_Test_5_input_data.bin",
                           input_features * batch_size)
                      .data());
  weight.tensor_of(load_bin("test/data/ReluDense_Test_5_weight_data.bin",
                            input_features * no_of_dense_neuron)
                       .data());
  bias.tensor_of(
      load_bin("test/data/ReluDense_Test_5_bias_data.bin", no_of_dense_neuron)
          .data());
  target_output.tensor_of(
      load_bin("test/data/ReluDense_Test_5_target_output_data.bin",
               no_of_dense_neuron * batch_size)
          .data());

  auto dense_1 = tf::layer::dense(32);
  auto relu_layer = tf::layer::relu();

  auto dense_output = dense_1({x});
  auto relu_output = relu_layer(dense_output);

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
      relu_layer.getLayerPtr(), Layer_Parameter::relu_training_output, false);

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
  tf::model mymodel({x}, relu_output);
  mymodel.shuffle(false);
  mymodel.compile(OptimizerType::SGD, LossType::squared_error);
  /* --- end model creation --- */

  tf::loss loss_sgd = mymodel.get_model_loss(relu_output[0]);
  call_back.record_tensor_loss_on_epoch_end(
      loss_sgd, Loss_Parameter::squared_error_predicted_output, false);

  call_back.record_tensor_loss_on_epoch_end(
      loss_sgd, Loss_Parameter::squared_error_grad_predicted_output);

  call_back.record_scalar_loss_on_epoch_end(loss_sgd, false);

  /* --- model training --- */
  unsigned epoch = 1;
  mymodel.fit({input}, {target_output}, {call_back.callback()}, epoch,
              batch_size);

  /* --- Intercepting training parameters --- */
  std::vector<std::vector<tf::tensor>> dense_outputs =
      call_back.get_parameter_on_epoch_end(
          dense_1.getLayerPtr(), Layer_Parameter::dense_training_output);

  std::vector<std::vector<tf::tensor>> relu_outputs =
      call_back.get_parameter_on_epoch_end(
          relu_layer.getLayerPtr(), Layer_Parameter::relu_training_output);

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

  auto expected_dense_output =
      load_bin("test/data/ReluDense_Test_5_dense_output_data.bin",
               no_of_dense_neuron * batch_size);
  auto expected_relu_output =
      load_bin("test/data/ReluDense_Test_5_relu_output_data.bin",
               no_of_dense_neuron * batch_size);
  auto expected_loss =
      load_bin("test/data/ReluDense_Test_5_loss_data.bin", no_of_dense_neuron);
  auto expected_grad_relu_output =
      load_bin("test/data/ReluDense_Test_5_grad_relu_output_data.bin",
               no_of_dense_neuron * batch_size);
  auto expected_grad_weight =
      load_bin("test/data/ReluDense_Test_5_grad_weight_data.bin",
               no_of_dense_neuron * input_features);
  auto expected_grad_bias = load_bin(
      "test/data/ReluDense_Test_5_grad_bias_data.bin", no_of_dense_neuron);
  auto expected_updated_weight =
      load_bin("test/data/ReluDense_Test_5_updated_weight_data.bin",
               no_of_dense_neuron * input_features);
  auto expected_updated_bias = load_bin(
      "test/data/ReluDense_Test_5_updated_bias_data.bin", no_of_dense_neuron);

  ASSERT_EQ(dense_outputs.size(), epoch);
  ASSERT_EQ(relu_outputs.size(), epoch);
  ASSERT_EQ(train_dense_1_grad_weights.size(), epoch);
  ASSERT_EQ(train_dense_1_grad_bias.size(), epoch);
  ASSERT_EQ(train_dense_1_updated_weights.size(), epoch);
  ASSERT_EQ(train_dense_1_updated_bias.size(), epoch);
  ASSERT_EQ(training_losses.size(), epoch);
  ASSERT_EQ(grad_training_losses.size(), epoch);

  for (unsigned ep = 0; ep < epoch; ep++) {
    ASSERT_EQ(dense_outputs[ep].size(), 1);
    ASSERT_EQ(relu_outputs[ep].size(), 1);
    ASSERT_EQ(train_dense_1_grad_weights[ep].size(), 1);
    ASSERT_EQ(train_dense_1_grad_bias[ep].size(), 1);
    ASSERT_EQ(train_dense_1_updated_weights[ep].size(), 1);
    ASSERT_EQ(train_dense_1_updated_bias[ep].size(), 1);
    ASSERT_EQ(training_losses[ep].size(), 1);
    ASSERT_EQ(grad_training_losses[ep].size(), 1);

    for (unsigned j = 0; j < batch_size; j++) {
      for (unsigned i = 0; i < no_of_dense_neuron; i++) {
        unsigned index = i + j * no_of_dense_neuron;
        EXPECT_NEAR(dense_outputs[ep][0].getData()[index],
                    expected_dense_output[index], 1e-6)
            << "dense output at ep: " << ep << " i: " << i << " j: " << j;
        EXPECT_NEAR(relu_outputs[ep][0].getData()[index],
                    expected_relu_output[index], 1e-6)
            << "relu output at ep: " << ep << " i: " << i << " j: " << j;
        EXPECT_NEAR(grad_training_losses[ep][0].getData()[index],
                    expected_grad_relu_output[index], 1e-5)
            << "grad relu output at ep: " << ep << " i: " << i << " j: " << j;
      }
    }

    for (unsigned i = 0; i < no_of_dense_neuron; i++) {
      EXPECT_NEAR(training_losses[ep][0].getData()[i], expected_loss[i], 1e-5)
          << "loss at ep: " << ep << " i: " << i;
      EXPECT_NEAR(train_dense_1_grad_bias[ep][0].getData()[i],
                  expected_grad_bias[i], 1e-5)
          << "grad bias at ep: " << ep << " i: " << i;
      EXPECT_NEAR(train_dense_1_updated_bias[ep][0].getData()[i],
                  expected_updated_bias[i], 1e-5)
          << "updated bias at ep: " << ep << " i: " << i;
    }

    for (unsigned j = 0; j < input_features; j++) {
      for (unsigned i = 0; i < no_of_dense_neuron; i++) {
        unsigned index = i + j * no_of_dense_neuron;
        EXPECT_NEAR(train_dense_1_grad_weights[ep][0].getData()[index],
                    expected_grad_weight[index], 1e-5)
            << "grad weight at ep: " << ep << " i: " << i << " j: " << j;
        EXPECT_NEAR(train_dense_1_updated_weights[ep][0].getData()[index],
                    expected_updated_weight[index], 1e-5)
            << "updated weight at ep: " << ep << " i: " << i << " j: " << j;
      }
    }
  }

  std::vector<std::float64_t> scalar_losses =
      call_back.get_scaler_loss_on_epoch_end(loss_sgd);
  ASSERT_EQ(scalar_losses.size(), epoch);
}
