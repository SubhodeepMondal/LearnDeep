// Library Header
#include <absl/log/log.h>
#include <stdexcept>
#include <vector>

// Library Header
#include "loss.hpp"
#include "loss_enum.hpp"
#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>

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
    unsigned i;
    for (i = 0; i < this->training_inputs[0]->getNoOfDimensions() - 1; i++)
      dims.push_back(this->training_inputs[0]->getDimensions()[i]);

    this->loss_tensor_batch = new tf::tensor();
    this->loss_tensor_batch->tf_create(dims, this->training_inputs[0]->dt_type);
    dims.push_back(batch_size);
    target_outputs.push_back(new tf::tensor());
    target_outputs[0]->tf_create(dims, this->training_inputs[0]->dt_type);

    this->differences = new tf::tensor();
    this->differences->tf_create(dims, this->training_inputs[0]->dt_type);

    this->loss_tensor = new tf::tensor();
    this->loss_tensor->tf_create(dims, this->training_inputs[0]->dt_type);

    *(this->differences) =
        (this->training_inputs[0])->sub(*(this->target_outputs[0]));
    *(this->loss_tensor_batch) = this->differences->pow(2);
    *(this->loss_tensor) = (this->loss_tensor_batch)->scale(1.0 / batch_size);
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
  default:
    LOG(ERROR) << "Sever! the selected loss parameter is not available for "
                  "mean squared error layer.\n";
    break;
  }
  return temp_tensor;
}