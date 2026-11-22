#include "scaled_squared_error.hpp"

#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

// Forces the custom loss plugin to be linked and loaded, even with --as-needed.
extern "C" void ld_custom_losses_load() {}

namespace {

bool configureScale(tf::loss::behavior &behavior, std::string_view parameter,
                    const tf::LossParameterValue &value) {
  if (parameter != kCustomScaleParameter)
    return false;

  // LossParameterValue stores `double`, not std::float64_t.
  const double *scale = std::get_if<double>(&value);
  auto *loss = dynamic_cast<ScaledSquaredError *>(&behavior);
  if (!scale || *scale <= 0.0 || !loss)
    return false;

  loss->setScale(*scale);
  return true;
}

} // namespace

REGISTER_LOSS(std::string(kCustomScaledMseType),
              [](const tf::loss::config &config)
                  -> std::unique_ptr<tf::loss::behavior> {
                return std::make_unique<ScaledSquaredError>(config);
              },
              configureScale);

ScaledSquaredError::ScaledSquaredError(const tf::loss::config &config) {
  const auto *inputs = config.getAs<std::vector<Tensor<std::float64_t> *>>(
      std::string(tf::loss::parameter::inputs));
  if (inputs && inputs->size() == 1 && (*inputs)[0]) {
    tf::tensor alias;
    alias.dt_type = tf_float64;
    alias.assign_ptr((*inputs)[0]); // non-owning alias
    this->training_input_aliases.push_back(alias);
  }

  const double *scale =
      config.getAs<double>(std::string(kCustomScaleParameter));
  this->setScale(scale ? *scale : 2.0);
}

void ScaledSquaredError::setScale(std::float64_t scale) {
  this->scale_ = scale;
  this->scale_tensor.tf_create(tf_float64, 1u);
  this->scale_tensor.tensor_of(scale, scale);
}

void ScaledSquaredError::forward(std::vector<tf::tensor *> inputs,
                                 const unsigned batch_size) {
  if (inputs.size() != 1 || !inputs[0])
    throw std::invalid_argument(
        "custom_scaled_mse expects exactly one forward input");

  // tf::tensor copies are non-owning aliases; reserve avoids reallocation
  // moving them and changing tensor_nodes ownership bookkeeping.
  this->training_input_aliases.clear();
  this->training_input_aliases.reserve(1);
  this->training_input_aliases.push_back(*inputs[0]);
  tf::tensor &training_input = this->training_input_aliases[0];

  std::vector<unsigned> dims;
  const unsigned feature_dimension_count =
      training_input.getNoOfDimensions() - 1;
  for (unsigned i = 0; i < feature_dimension_count; ++i)
    dims.push_back(training_input.getDimensions()[i]);

  this->loss_tensor_batch.tf_create(dims, training_input.dt_type);

  dims.push_back(batch_size);
  this->target_tensors.clear();
  this->target_tensors.reserve(1);
  this->target_tensors.emplace_back().tf_create(dims, training_input.dt_type);
  this->differences.tf_create(dims, training_input.dt_type);
  this->loss_tensor.tf_create({1}, training_input.dt_type);

  this->differences = training_input.sub(this->target_tensors[0]);
  this->loss_tensor_batch = this->differences.pow(2);

  this->loss_reduction_intermediates.clear();
  this->loss_reduction_intermediates.push_back(
      this->loss_tensor_batch.mean(feature_dimension_count));

  for (unsigned i = 0; i < feature_dimension_count; ++i) {
    tf::tensor reduced = this->loss_reduction_intermediates.back().mean(0);
    if (i + 1 == feature_dimension_count)
      this->loss_tensor = std::move(reduced);
    else
      this->loss_reduction_intermediates.push_back(std::move(reduced));
  }
  if (feature_dimension_count == 0)
    this->loss_tensor =
        std::move(this->loss_reduction_intermediates.back());

  this->scaled_loss_tensor = this->loss_tensor.scale(this->scale_);
}

void ScaledSquaredError::backward() {
  if (this->is_grad_recorded)
    return;
  this->is_grad_recorded = true;

  if (!this->training_input_aliases.empty())
    this->grad_training_input =
        tf::loss::gradient_of(this->training_input_aliases[0]);
}

void ScaledSquaredError::setTargetOutput(
    const std::vector<tf::tensor> &targets) {
  if (!this->target_tensors.empty() &&
      targets.size() == this->target_tensors.size())
    this->target_tensors[0].tensor_of(targets[0].getData());
}

std::float64_t ScaledSquaredError::getScalerLoss() {
  if (!this->scaled_loss_tensor.getPtr())
    return std::numeric_limits<std::float64_t>::max();
  return this->scaled_loss_tensor.getData()[0];
}

std::float64_t
ScaledSquaredError::calculate(std::vector<tf::tensor> &predicted_output,
                              std::vector<tf::tensor> &target_output) {
  if (predicted_output.size() != 1 ||
      predicted_output.size() != target_output.size())
    return std::numeric_limits<std::float64_t>::max();

  tf::tensor differences = predicted_output[0].sub(target_output[0], false);
  tf::tensor squared = differences.pow(2, false);

  const unsigned dimension_count = predicted_output[0].getNoOfDimensions();
  for (unsigned i = 0; i < dimension_count; ++i) {
    const unsigned axis = (i == 0) ? dimension_count - 1 : 0;
    squared = squared.mean(axis, false);
  }

  this->prev_loss_value = squared.getData()[0] * this->scale_;
  return this->prev_loss_value;
}

std::vector<tf::tensor *>
ScaledSquaredError::getParameter(std::string_view name) {
  if (name == kCustomScaleParameter)
    return {&this->scale_tensor};

  if (name == kCustomScaledMseLossParameter) {
    if (this->scaled_loss_tensor.getPtr())
      return {&this->scaled_loss_tensor};
    return {};
  }

  if (name == kCustomScaledMseGradParameter) {
    if (this->grad_training_input.getPtr())
      return {&this->grad_training_input};
    return {};
  }

  std::cerr << "Unknown parameter '" << name << "' for custom_scaled_mse.\n";
  return {};
}

std::unordered_set<std::string> ScaledSquaredError::parameterNames() const {
  return {std::string(kCustomScaleParameter),
          std::string(kCustomScaledMseLossParameter),
          std::string(kCustomScaledMseGradParameter)};
}
