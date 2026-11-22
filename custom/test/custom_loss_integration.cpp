// Full integration test for the out-of-framework custom loss plugin.
//
// Boundary: this file includes only the public API header and links only the
// built framework libraries + the custom plugins. It reuses the NumPy fixtures
// from custom_scale_integration.cpp by configuring custom_scaled_mse with a
// scale of 1.0, so it must reproduce squared-error loss and gradients exactly.
#include <api/tensor.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Forces both plugins to be linked and loaded.
extern "C" void ld_custom_layers_load();
extern "C" void ld_custom_losses_load();

namespace {

int failures = 0;

void check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    ++failures;
  }
}

void check_near(double actual, double expected, double tolerance,
                const std::string &where) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    std::cerr << "FAILED: " << where << " actual=" << actual
              << " expected=" << expected << '\n';
    ++failures;
  }
}

std::vector<std::float64_t> load_bin(const std::string &name,
                                     std::size_t expected_size) {
  const std::string path =
      std::string(LD_CUSTOM_TEST_DATA_DIR) + "/" + name;
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Failed to open " + path);

  std::vector<std::float64_t> data(expected_size);
  file.read(reinterpret_cast<char *>(data.data()),
            static_cast<std::streamsize>(expected_size * sizeof(std::float64_t)));
  if (!file)
    throw std::runtime_error("Failed to read " + path);
  return data;
}

} // namespace

int main() {
  ld_custom_layers_load();
  ld_custom_losses_load();

  constexpr unsigned input_features = 17;
  constexpr unsigned batch_size = 128;
  constexpr unsigned epochs = 1;
  constexpr unsigned samples = input_features * batch_size;

  auto input_data = load_bin("CustomScale_Test_1_input_data.bin", samples);
  auto target_data = load_bin("CustomScale_Test_1_target_data.bin", samples);
  const auto expected_scale =
      load_bin("CustomScale_Test_1_scale_data.bin", 1).front();
  const auto expected_loss = load_bin("CustomScale_Test_1_loss_data.bin", 1);
  const auto expected_grad_loss =
      load_bin("CustomScale_Test_1_grad_loss_data.bin", samples);

  tf::tensor x, input, target_output;
  x.tf_create(tf_float64, input_features, batch_size);
  input.tf_create(tf_float64, input_features, batch_size);
  target_output.tf_create(tf_float64, input_features, batch_size);
  input.tensor_of(input_data.data());
  target_output.tensor_of(target_data.data());

  tf::layer::layer custom_layer("test_scale");
  check(custom_layer.valid(), "custom scale layer was registered");
  custom_layer.setParameter("scale", expected_scale);
  const auto custom_output = custom_layer({x});
  check(custom_output.size() == 1, "custom layer returns one output tensor");

  tf::model model({x}, custom_output);
  model.shuffle(false);
  model.compile(tf::optimizer::type::sgd, "custom_scaled_mse");

  auto loss = model.get_model_loss(custom_output[0]);
  check(loss.valid(), "custom loss was created by model.compile");
  check(loss.type() == "custom_scaled_mse", "compiled loss uses the custom key");
  loss.setParameter("scale", 1.0);

  tf::callback::trace call_back;
  call_back.record_tensor_loss_on_batch_end(loss, "custom_scaled_mse_loss");
  call_back.record_tensor_loss_on_batch_end(loss, "custom_scaled_mse_grad");
  call_back.record_scalar_loss_on_batch_end(loss);

  model.fit({input}, {target_output}, {call_back.callback()}, epochs,
            batch_size);

  const auto predicted = call_back.get_tensor_loss_on_batch_end(
      loss, "custom_scaled_mse_loss");
  const auto grad_predicted = call_back.get_tensor_loss_on_batch_end(
      loss, "custom_scaled_mse_grad");
  const auto losses = call_back.get_scaler_loss_on_batch_end(loss);

  check(predicted.size() == epochs, "one batch of loss tensors recorded");
  check(grad_predicted.size() == epochs, "one batch of loss gradients recorded");
  check(losses.size() == epochs, "one batch of scalar losses recorded");

  if (predicted.size() == epochs && !predicted.front().empty() &&
      !predicted.front().front().empty())
    check_near(predicted.front().front().front().getData()[0],
               expected_loss.front(), 1e-9, "custom loss tensor");
  else
    check(false, "custom loss tensor is available");

  if (grad_predicted.size() == epochs && !grad_predicted.front().empty() &&
      !grad_predicted.front().front().empty()) {
    const std::float64_t *values =
        grad_predicted.front().front().front().getData();
    for (unsigned k = 0; k < samples; ++k)
      check_near(values[k], expected_grad_loss[k], 1e-9,
                 "custom loss gradient");
  } else {
    check(false, "custom loss gradient tensor is available");
  }

  if (!losses.empty() && !losses.front().empty())
    check_near(losses.front().front(), expected_loss.front(), 1e-9,
               "custom scalar loss");
  else
    check(false, "custom scalar loss is available");

  if (failures != 0) {
    std::cerr << failures << " custom loss integration check(s) failed\n";
    return 1;
  }

  std::cout << "custom loss integration checks passed\n";
  return 0;
}
