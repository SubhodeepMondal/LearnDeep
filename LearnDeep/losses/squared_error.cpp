// Library Header
#include "loss.hpp"
#include <absl/log/log.h>
#include <stdexcept>
#include <vector>

SquaredError::~SquaredError() {}

SquaredError::SquaredError(std::vector<Tensor<std::float64_t> *> input_preds) {
  if (input_preds.size() == 1) {
    this->input_predicts.push_back(input_preds[0]);
  } else {
    LOG(ERROR) << "Fata! SquaredError: expects only 1 input tensor but given, "
               << input_preds.size() << ".\n";
    throw std::runtime_error(
        "Exiting program due to errorneous input for SquaredError");
  }
}

void SquaredError::forward(std::vector<tf::tensor *> inputs,
                           const unsigned batch_size) {

  if (inputs.size() == 1) {
    this->training_inputs.push_back(inputs[0]);
    std::vector<unsigned> dims;
    for (unsigned i = 0; i < this->training_inputs[0]->getNoOfDimensions(); i++)
      dims.push_back(this->training_inputs[0]->getDimensions()[i]);
    target_outputs.push_back(new tf::tensor());
    target_outputs[0]->tf_create(dims, this->training_inputs[0]->dt_type);

    this->differences = new tf::tensor();
    this->differences->tf_create(dims, this->training_inputs[0]->dt_type);

    this->loss_tensor = new tf::tensor();
    this->loss_tensor->tf_create(dims, this->training_inputs[0]->dt_type);

    *(this->differences) =
        this->training_inputs[0]->sub(*(this->target_outputs[0]), false);
    *(this->loss_tensor) = this->differences->pow(2, false);
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

std::float64_t const SquaredError::getScalerLoss() { return loss_value; }

std::vector<tf::tensor *> SquaredError::getLossTensor() {
  std::vector<tf::tensor *> temp_tensor;
  temp_tensor.push_back(this->loss_tensor);
  return temp_tensor;
}