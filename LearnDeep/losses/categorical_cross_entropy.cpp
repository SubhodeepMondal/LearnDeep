// Library Header
#include <absl/log/log.h>
#include <limits>
#include <stdexcept>
#include <vector>

// Library Header
#include "loss.hpp"
#include "loss_registry.hpp"
#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>

REGISTER_LOSS(
    "categorical_cross_entropy",
    [](const LossConfig &config) -> std::unique_ptr<Loss> {
      return std::make_unique<CategoricalCrossEntropy>(
          lossInputs(config), lossName(config, "categorical_cross_entropy"));
    });

std::string_view CategoricalCrossEntropy::type() const {
  return tf::loss::type::categorical_cross_entropy;
}

CategoricalCrossEntropy::CategoricalCrossEntropy(
    std::vector<Tensor<std::float64_t> *> input_preds, std::string loss_name) {
  this->recordInputs(input_preds);
  if (this->input_aliases.size() != 1) {
    LOG(ERROR) << "Fata! Categorical Cross Entropy: expects only 1 input "
                  "tensor but given, "
               << input_preds.size() << ".\n";
    throw std::runtime_error("Exiting program due to errorneous input for "
                             "Categorical Cross Entropy");
  }

  if (loss_set.count(loss_name)) {
    this->loss_name = loss_name + "_" + std::to_string((long unsigned)this);
  } else {
    this->loss_name = loss_name;
    loss_set.insert(loss_name);
  }
}

void CategoricalCrossEntropy::forward(std::vector<tf::tensor *> inputs,
                                      const unsigned batch_size) {
  if (inputs.size() != 1)
    return;

  this->recordTrainingInputs(inputs);
  tf::tensor &training_input = this->training_input_aliases[0];

  std::vector<unsigned> dims;
  for (unsigned i = 0; i < training_input.getNoOfDimensions() - 1; i++)
    dims.push_back(training_input.getDimensions()[i]);
  dims.push_back(batch_size);

  this->target_aliases.clear();
  this->target_aliases.emplace_back();
  this->target_aliases.back().tf_create(dims, training_input.dt_type);
  this->log_value.tf_create(dims, training_input.dt_type);
  this->log_difference.tf_create(dims, training_input.dt_type);
  this->loss_tensor_batch.tf_create(dims, training_input.dt_type);
  this->loss_tensor.tf_create(dims, training_input.dt_type);

  this->log_value = training_input.log();
  this->log_difference = this->log_value.mul(this->target_aliases[0]);
  this->loss_tensor_batch = this->log_difference.scale(-1.0f);
  this->loss_tensor = this->loss_tensor_batch.scale(
      1.0 / training_input
                 .getDimensions()[training_input.getNoOfDimensions() - 1]);
}

void CategoricalCrossEntropy::backward() {
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
CategoricalCrossEntropy::calculate(std::vector<tf::tensor> &predicted_output,
                                   std::vector<tf::tensor> &target_output) {

  if (predicted_output.size() == 1) {

    tf::tensor temp_log = predicted_output[0].log(false);

    tf::tensor temp_diff = temp_log.mul(target_output[0], false);

    tf::tensor temp_loss_tensor = temp_diff.scale(-1.0f, false);

    // Mean over batch
    temp_loss_tensor = temp_loss_tensor.scale(
        1.0 / predicted_output[0]
                  .getDimensions()[predicted_output[0].getNoOfDimensions() - 1],
        false);

    // Sum over classes/features
    std::float64_t loss_val = 0.0;

    for (unsigned i = 0; i < temp_loss_tensor.getNoOfElem(); i++) {

      loss_val += temp_loss_tensor.getData()[i];
    }

    this->prev_loss_value = loss_val;
    return this->prev_loss_value;

  } else {

    std::cerr << "Categorical Cross Entropy Error: requires only 1 tensor "
                 "for prediction output\n";

    return std::numeric_limits<double>::max();
  }
}

void CategoricalCrossEntropy::setPredictedOutput(
    const std::vector<tf::tensor *> &input_predicts) {
  // Predictions are recorded through recordInputs()/forward().
}

void CategoricalCrossEntropy::setTargetOutput(
    const std::vector<tf::tensor> output_targets) {
  if (!this->target_aliases.empty() &&
      output_targets.size() == this->target_aliases.size())
    this->target_aliases[0].tensor_of(output_targets[0].getData());
}

std::float64_t const CategoricalCrossEntropy::getScalerLoss() {
  if (!this->loss_tensor.getPtr())
    return std::numeric_limits<std::float64_t>::max();

  this->loss_value = 0.0;
  for (unsigned i = 0; i < this->loss_tensor.getNoOfElem(); i++)
    this->loss_value += this->loss_tensor.getData()[i];

  return loss_value;
}

std::vector<tf::tensor *>
CategoricalCrossEntropy::getLossParameter(std::string_view loss_parameter) {
  std::vector<tf::tensor *> temp_tensor;
  using tf::loss::parameter::categorical_cross_entropy_error;
  using tf::loss::parameter::categorical_cross_entropy_target_output;
  using tf::loss::parameter::mean_categorical_cross_entropy_error;

  if (loss_parameter == categorical_cross_entropy_error) {
    if (this->loss_tensor.getPtr())
      temp_tensor.push_back(&this->loss_tensor);
  } else if (loss_parameter == mean_categorical_cross_entropy_error) {
    if (this->grad_training_inputs.getPtr())
      temp_tensor.push_back(&this->grad_training_inputs);
  } else if (loss_parameter == categorical_cross_entropy_target_output) {
    if (!target_aliases.empty() && target_aliases[0].getPtr())
      temp_tensor.push_back(&target_aliases[0]);
  } else {
    LOG(ERROR) << "Sever! the selected loss parameter '" << loss_parameter
               << "' is not available for categorical cross entropy.\n";
  }
  return temp_tensor;
}

std::unordered_set<std::string>
CategoricalCrossEntropy::parameterNames() const {
  return {
      std::string(tf::loss::parameter::categorical_cross_entropy_error),
      std::string(tf::loss::parameter::mean_categorical_cross_entropy_error),
      std::string(
          tf::loss::parameter::categorical_cross_entropy_target_output)};
}
