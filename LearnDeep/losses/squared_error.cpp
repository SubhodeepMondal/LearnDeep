// Library Header
#include <absl/log/log.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Library Header
#include "loss.hpp"
#include "loss_registry.hpp"
#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>

REGISTER_INTERNAL_LOSS(
    "squared_error",
    [](const LossConfig &config) -> std::unique_ptr<Loss> {
      return std::make_unique<SquaredError>(
          lossInputs(config), lossName(config, "mean_squared_loss"));
    });

std::string_view SquaredError::type() const {
  return tf::loss::type::squared_error;
}

SquaredError::SquaredError(std::vector<Tensor<std::float64_t> *> input_preds,
                           std::string loss_name) {
  this->recordInputs(input_preds);
  if (this->input_aliases.size() != 1) {
    LOG(ERROR) << "Fata! SquaredError: expects only 1 input tensor but given, "
               << input_preds.size() << ".\n";
    throw std::runtime_error(
        "Exiting program due to errorneous input for SquaredError");
  }

  if (loss_set.count(loss_name)) {
    this->loss_name = loss_name + "_" + std::to_string((long unsigned)this);
  } else {
    this->loss_name = loss_name;
    loss_set.insert(loss_name);
  }
}

void SquaredError::forward(std::vector<tf::tensor *> inputs,
                           const unsigned batch_size) {
  if (inputs.size() != 1)
    return;

  this->recordTrainingInputs(inputs);
  tf::tensor &training_input = this->training_input_aliases[0];

  std::vector<unsigned> dims;
  const unsigned feature_dimension_count =
      training_input.getNoOfDimensions() - 1;
  for (unsigned i = 0; i < feature_dimension_count; ++i)
    dims.push_back(training_input.getDimensions()[i]);

  this->loss_tensor_batch.tf_create(dims, training_input.dt_type);

  dims.push_back(batch_size);
  this->target_aliases.clear();
  this->target_aliases.emplace_back();
  this->target_aliases.back().tf_create(dims, training_input.dt_type);
  this->differences.tf_create(dims, training_input.dt_type);
  this->loss_tensor.tf_create({1}, training_input.dt_type);

  this->differences = training_input.sub(this->target_aliases[0]);
  this->loss_tensor_batch = this->differences.pow(2);

  // Reduce batch first, then all remaining feature dimensions.
  this->loss_reduction_intermediates.clear();
  this->loss_reduction_intermediates.push_back(
      this->loss_tensor_batch.mean(feature_dimension_count));

  for (unsigned i = 0; i < feature_dimension_count; ++i) {
    tf::tensor reduced = this->loss_reduction_intermediates.back().mean(0);
    if (i + 1 == feature_dimension_count) {
      this->loss_tensor = std::move(reduced);
    } else {
      this->loss_reduction_intermediates.push_back(std::move(reduced));
    }
  }
  if (feature_dimension_count == 0)
    this->loss_tensor =
        std::move(this->loss_reduction_intermediates.back());
}

void SquaredError::backward() {
  if (!this->isGradRecorded) {
    this->isGradRecorded = true;

    Graph *g = GraphManager::instance().getCurrentGraph();
    if (g)
      this->grad_training_inputs =
          tf::tensor(this->training_input_aliases[0].dt_type,
                     g->getGradientTensor(
                         this->training_input_aliases[0].getPtr()));
  }
}

std::float64_t
SquaredError::calculate(std::vector<tf::tensor> &predicted_output,
                        std::vector<tf::tensor> &target_outptut) {

  if (predicted_output.size() == 1) {

    tf::tensor temp_diff = predicted_output[0].sub(target_outptut[0], false);

    tf::tensor temp_loss_tensor = temp_diff.pow(2, false);

    for (unsigned i = 0; i < predicted_output[0].getNoOfDimensions(); i++) {

      unsigned axis =
          (i == 0) ? predicted_output[0].getNoOfDimensions() - 1 : 0;

      temp_loss_tensor = temp_loss_tensor.mean(axis, false);
    }
    this->prev_loss_value = temp_loss_tensor.getData()[0];
    return this->prev_loss_value;

  } else {

    std::cerr
        << "Mean Squared Error: requires only 1 tensor for prediction output\n";

    return std::numeric_limits<double>::max();
  }
}

void SquaredError::setPredictedOutput(
    const std::vector<tf::tensor *> &input_predicts) {
  // Predictions are recorded through recordInputs()/forward().
}

void SquaredError::setTargetOutput(
    const std::vector<tf::tensor> output_targets) {
  if (!this->target_aliases.empty() &&
      output_targets.size() == this->target_aliases.size())
    this->target_aliases[0].tensor_of(output_targets[0].getData());
}

std::float64_t const SquaredError::getScalerLoss() {
  if (!this->loss_tensor.getPtr())
    return std::numeric_limits<std::float64_t>::max();
  return this->loss_tensor.getData()[0];
}

std::vector<tf::tensor *>
SquaredError::getLossParameter(std::string_view loss_parameter) {
  std::vector<tf::tensor *> temp_tensor;
  using tf::loss::parameter::squared_error_grad_predicted_output;
  using tf::loss::parameter::squared_error_predicted_output;
  using tf::loss::parameter::squared_error_target_output;

  if (loss_parameter == squared_error_predicted_output) {
    if (this->loss_tensor.getPtr())
      temp_tensor.push_back(&this->loss_tensor);
  } else if (loss_parameter == squared_error_grad_predicted_output) {
    if (this->grad_training_inputs.getPtr())
      temp_tensor.push_back(&this->grad_training_inputs);
  } else if (loss_parameter == squared_error_target_output) {
    if (!target_aliases.empty() && target_aliases[0].getPtr())
      temp_tensor.push_back(&target_aliases[0]);
  } else {
    LOG(ERROR) << "Sever! the selected loss parameter '" << loss_parameter
               << "' is not available for squared error.\n";
  }
  return temp_tensor;
}

std::unordered_set<std::string> SquaredError::parameterNames() const {
  return {std::string(tf::loss::parameter::squared_error_predicted_output),
          std::string(tf::loss::parameter::squared_error_target_output),
          std::string(
              tf::loss::parameter::squared_error_grad_predicted_output)};
}
