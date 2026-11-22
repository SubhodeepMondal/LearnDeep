#include "scaled_squared_error.hpp"

#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>
#include <losses/loss_registry.hpp>

#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

// Forces the custom loss plugin to be linked and loaded, even with --as-needed.
extern "C" void ld_custom_losses_load() {}

namespace {

bool configureScale(const LossRegistry::LossHandle &loss,
                    const std::string &parameter,
                    const tf::LossParameterValue &value) {
  if (parameter != kCustomScaleParameter)
    return false;

  // LossParameterValue stores `double`, not std::float64_t.
  const double *scale = std::get_if<double>(&value);
  if (!scale || *scale <= 0.0)
    return false;

  if (auto scaled = std::dynamic_pointer_cast<ScaledSquaredError>(loss)) {
    scaled->setScale(*scale);
    return true;
  }
  return false;
}

} // namespace

REGISTER_LOSS(
    std::string(kCustomScaledMseType),
    [](const LossConfig &config) -> std::unique_ptr<Loss> {
      const double *scale =
          config.getAs<double>(std::string(kCustomScaleParameter));
      return std::make_unique<ScaledSquaredError>(
          lossInputs(config), lossName(config, "custom_scaled_mse"),
          scale ? *scale : 2.0);
    },
    configureScale);

ScaledSquaredError::ScaledSquaredError(
    std::vector<Tensor<std::float64_t> *> input_preds, std::string loss_name,
    std::float64_t scale)
    : Loss() {
  this->loss_name = loss_name + "_" + std::to_string(id());
  // A standalone generic handle may be created without inputs; model.compile
  // always supplies the single output tensor it is attached to.
  if (input_preds.size() > 1) {
    throw std::invalid_argument(
        "custom_scaled_mse expects at most one input tensor");
  }
  if (input_preds.size() == 1)
    this->recordInputs(input_preds);
  this->setScale(scale);
}

ScaledSquaredError::~ScaledSquaredError() {
  for (tf::tensor *tensor : this->target_outputs)
    delete tensor;
  delete this->differences;
  delete this->loss_tensor_batch;
  delete this->loss_tensor;
}

void ScaledSquaredError::setScale(std::float64_t scale) {
  this->scale_ = scale;
  this->scale_tensor_.tf_create(tf_float64, 1u);
  this->scale_tensor_.tensor_of(scale, scale);
}

void ScaledSquaredError::forward(std::vector<tf::tensor *> inputs,
                                 const unsigned batch_size) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "custom_scaled_mse expects exactly one forward input");
  }

  this->training_inputs.push_back(inputs[0]);

  std::vector<unsigned> dims;
  const unsigned feature_dimension_count =
      this->training_inputs[0]->getNoOfDimensions() - 1;
  for (unsigned i = 0; i < feature_dimension_count; ++i)
    dims.push_back(this->training_inputs[0]->getDimensions()[i]);

  this->loss_tensor_batch = new tf::tensor();
  this->loss_tensor_batch->tf_create(dims, this->training_inputs[0]->dt_type);

  dims.push_back(batch_size);
  this->target_outputs.push_back(new tf::tensor());
  this->target_outputs[0]->tf_create(dims, this->training_inputs[0]->dt_type);

  this->differences = new tf::tensor();
  this->differences->tf_create(dims, this->training_inputs[0]->dt_type);

  this->loss_tensor = new tf::tensor();
  this->loss_tensor->tf_create({1}, this->training_inputs[0]->dt_type);

  *(this->differences) =
      this->training_inputs[0]->sub(*(this->target_outputs[0]));
  *(this->loss_tensor_batch) = this->differences->pow(2);

  this->loss_reduction_intermediates.clear();
  this->loss_reduction_intermediates.push_back(
      this->loss_tensor_batch->mean(feature_dimension_count));

  for (unsigned i = 0; i < feature_dimension_count; ++i) {
    tf::tensor reduced = this->loss_reduction_intermediates.back().mean(0);
    if (i + 1 == feature_dimension_count) {
      *(this->loss_tensor) = std::move(reduced);
    } else {
      this->loss_reduction_intermediates.push_back(std::move(reduced));
    }
  }
  if (feature_dimension_count == 0)
    *(this->loss_tensor) =
        std::move(this->loss_reduction_intermediates.back());

  this->scaled_loss_tensor_ = this->loss_tensor->scale(this->scale_);
  this->scaled_loss_tensor_.dt_type = tf_float64;
}

void ScaledSquaredError::backward() {
  if (this->isGradRecorded)
    return;

  this->isGradRecorded = true;
  Graph *graph = GraphManager::instance().getCurrentGraph();
  if (graph)
    this->grad_training_inputs = tf::tensor(
        this->training_inputs[0]->dt_type,
        graph->getGradientTensor(this->training_inputs[0]->getPtr()));
}

void ScaledSquaredError::setPredictedOutput(
    const std::vector<tf::tensor *> &output_targets) {
  // Predictions are wired directly through forward() during graph construction.
}

void ScaledSquaredError::setTargetOutput(
    const std::vector<tf::tensor> output_targets) {
  if (!this->target_outputs.empty() &&
      output_targets.size() == this->target_outputs.size())
    this->target_outputs[0]->tensor_of(output_targets[0].getData());
}

std::float64_t const ScaledSquaredError::getScalerLoss() {
  if (this->scaled_loss_tensor_.getPtr())
    return this->scaled_loss_tensor_.getData()[0];
  return std::numeric_limits<std::float64_t>::max();
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
ScaledSquaredError::getLossParameter(std::string_view loss_parameter) {
  if (loss_parameter == kCustomScaleParameter)
    return {&this->scale_tensor_};

  if (loss_parameter == kCustomScaledMseLossParameter)
    return {&this->scaled_loss_tensor_};

  if (loss_parameter == kCustomScaledMseGradParameter)
    return {&this->grad_training_inputs};

  std::cerr << "Unknown parameter '" << loss_parameter
            << "' for custom_scaled_mse.\n";
  return {};
}

std::unordered_set<std::string> ScaledSquaredError::parameterNames() const {
  return {std::string(kCustomScaleParameter),
          std::string(kCustomScaledMseLossParameter),
          std::string(kCustomScaledMseGradParameter)};
}
