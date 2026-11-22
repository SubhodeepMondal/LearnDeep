// Library Header
#include <absl/log/log.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Library Header
#include "loss.hpp"
#include "loss_enum.hpp"
#include "loss_registry.hpp"
#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>

REGISTER_LOSS(
    "squared_error",
    [](const LossConfig &config) -> std::unique_ptr<Loss> {
      return std::make_unique<SquaredError>(
          lossInputs(config), lossName(config, "mean_squared_loss"));
    });

std::string_view SquaredError::type() const {
  return tf::loss::type::squared_error;
}

SquaredError::~SquaredError() {}

SquaredError::SquaredError(std::vector<Tensor<std::float64_t> *> input_preds,
                           std::string loss_name) {
  if (input_preds.size() == 1) {
    this->input_predicts.push_back(input_preds[0]);
  } else {
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

  if (inputs.size() == 1) {

    this->training_inputs.push_back(inputs[0]);

    std::vector<unsigned> dims;

    for (unsigned i = 0; i < this->training_inputs[0]->getNoOfDimensions() - 1;
         i++)
      dims.push_back(this->training_inputs[0]->getDimensions()[i]);

    // Allocate batch-reduced tensor
    this->loss_tensor_batch = new tf::tensor();
    this->loss_tensor_batch->tf_create(dims, this->training_inputs[0]->dt_type);

    dims.push_back(batch_size);

    this->target_outputs.push_back(new tf::tensor());
    this->target_outputs[0]->tf_create(dims, this->training_inputs[0]->dt_type);

    this->differences = new tf::tensor();
    this->differences->tf_create(dims, this->training_inputs[0]->dt_type);

    // Allocate final scalar loss tensor
    this->loss_tensor = new tf::tensor();
    this->loss_tensor->tf_create({1}, this->training_inputs[0]->dt_type);

    *(this->differences) =
        this->training_inputs[0]->sub(*(this->target_outputs[0]));

    *(this->loss_tensor_batch) = this->differences->pow(2);

    // Reduce batch first, then all remaining feature dimensions.
    const unsigned feature_dimension_count =
        this->training_inputs[0]->getNoOfDimensions() - 1;
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
  }
}

void SquaredError::backward() {
  if (!this->isGradRecorded) {
    this->isGradRecorded = true;

    Graph *g = GraphManager::instance().getCurrentGraph();
    if (g)
      this->grad_training_inputs =
          tf::tensor(this->training_inputs[0]->dt_type,
                     g->getGradientTensor(this->training_inputs[0]->getPtr()));
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
  // this->input_predicts = input_predicts;
}

void SquaredError::setTargetOutput(
    const std::vector<tf::tensor> output_targets) {
  unsigned idx = 0;
  if (this->input_predicts.size() == output_targets.size()) {
    this->target_outputs[0]->tensor_of(output_targets[0].getData());
  } else {
    LOG(ERROR) << "Fatal! Espected No Of target outputs are: "
               << this->input_predicts.size()
               << ", but got:  " << output_targets.size() << "\n";
  }
}

std::float64_t const SquaredError::getScalerLoss() {
  return this->loss_tensor->getData()[0];
}

std::vector<tf::tensor *>
SquaredError::getLossParameter(Loss_Parameter loss_parameter) {
  std::vector<tf::tensor *> temp_tensor;
  switch (loss_parameter) {
  case Loss_Parameter::squared_error_predicted_output: {
    temp_tensor.push_back(this->loss_tensor);
    break;
  }
  case Loss_Parameter::squared_error_grad_predicted_output: {
    temp_tensor.push_back(&this->grad_training_inputs);
    break;
  }
  case Loss_Parameter::squared_error_target_output: {
    // tf::tensor temp_reference = tf::tensor(tf_float64, target_outputs[0]);
    temp_tensor.push_back(target_outputs[0]);
    break;
  }
  default:
    LOG(ERROR) << "Sever! the selected loss parameter is not available for "
                  "mean squared error layer.\n";
    break;
  }
  return temp_tensor;
}
