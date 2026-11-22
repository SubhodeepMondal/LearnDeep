// Out-of-framework custom loss test. It includes only the public API header,
// links the built framework libraries and the custom plugin, and reaches the
// loss solely through its registered string key.
#include <api/tensor.h>

#include <cmath>
#include <iostream>
#include <stdfloat>
#include <string>
#include <string_view>
#include <vector>

// Forces the custom loss plugin to be linked and loaded.
extern "C" void ld_custom_losses_load();

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    ++failures;
  }
}

void check_near(double actual, double expected, double tolerance,
                std::string_view where) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    std::cerr << "FAILED: " << where << " actual=" << actual
              << " expected=" << expected << '\n';
    ++failures;
  }
}

} // namespace

int main() {
  ld_custom_losses_load();

  constexpr std::string_view kTypeName = "custom_scaled_mse";

  tf::loss::loss custom(kTypeName);
  check(custom.valid(), "out-of-tree loss was registered and created");
  check(custom.type() == kTypeName, "loss reports its registered key");
  check(custom.id() != tf::loss::invalid_loss_id,
        "loss has a stable, non-invalid id");
  check(custom.parameterNames().count("scale") == 1,
        "loss exposes its scale parameter");

  const tf::loss::loss missing("no_such_custom_loss");
  check(!missing.valid(), "unknown custom loss yields an inert handle");

  custom.setParameter("scale", 3.0);
  const std::vector<tf::tensor> scales = custom.getParameter("scale");
  check(scales.size() == 1, "custom loss reports its configured scale");
  if (scales.size() == 1 && scales.front().getPtr())
    check_near(scales.front().getData()[0], 3.0, 1e-12, "configured scale");

  std::vector<std::float64_t> predicted_values{1.0, 3.0};
  std::vector<std::float64_t> target_values{0.0, 1.0};
  tf::tensor predicted, target;
  predicted.tf_create(tf_float64, 1u, 2u);
  target.tf_create(tf_float64, 1u, 2u);
  predicted.tensor_of(predicted_values.data());
  target.tensor_of(target_values.data());

  std::vector<tf::tensor> predicted_batch;
  std::vector<tf::tensor> target_batch;
  predicted_batch.push_back(std::move(predicted));
  target_batch.push_back(std::move(target));

  // 3.0 * mean([(1-0)^2, (3-1)^2]) = 7.5
  check_near(custom.calculate(predicted_batch, target_batch), 7.5, 1e-12,
             "scaled mean squared error");

  std::vector<std::float64_t> input_values{0.0, 1.0, 2.0, 3.0};
  std::vector<std::float64_t> training_target_values{0.0, 0.5, 1.0, 1.5};
  tf::tensor x, training_target;
  x.tf_create(tf_float64, 1u, 4u);
  training_target.tf_create(tf_float64, 1u, 4u);
  x.tensor_of(input_values.data());
  training_target.tensor_of(training_target_values.data());

  tf::layer::dense dense(1);
  const std::vector<tf::tensor> output = dense({x});
  tf::model model({x}, output);
  model.shuffle(false);
  model.compile(OptimizerType::SGD, kTypeName);

  tf::loss::loss model_loss = model.get_model_loss(output[0]);
  check(model_loss.valid(), "compiled model exposes the custom loss");
  check(model_loss.type() == kTypeName, "model uses the custom loss key");
  model_loss.setParameter("scale", 2.0);

  tf::callback::trace trace;
  trace.record_parameter_on_batch_end(dense, "training_output", false);
  trace.record_scalar_loss_on_batch_end(model_loss);
  model.fit({x}, {training_target}, {trace.callback()}, 1, 4);

  const auto outputs =
      trace.get_parameter_on_batch_end(dense, "training_output");
  const auto losses = trace.get_scaler_loss_on_batch_end(model_loss);
  check(outputs.size() == 1, "one epoch of training outputs recorded");
  check(losses.size() == 1, "one epoch of custom loss recorded");

  if (outputs.size() == 1 && !outputs.front().empty() &&
      !outputs.front().front().empty() && losses.size() == 1 &&
      !losses.front().empty()) {
    const std::float64_t *values = outputs.front().front().front().getData();
    std::float64_t squared_error = 0.0;
    for (unsigned i = 0; i < 4; ++i) {
      const std::float64_t difference =
          values[i] - training_target_values[i];
      squared_error += difference * difference;
    }
    check_near(losses.front().front(), 2.0 * squared_error / 4.0, 1e-9,
               "training loss uses the configured scale");
  } else {
    check(false, "custom loss and layer outputs were recorded");
  }

  if (failures != 0) {
    std::cerr << failures << " custom loss check(s) failed\n";
    return 1;
  }

  std::cout << "custom loss checks passed\n";
  return 0;
}
