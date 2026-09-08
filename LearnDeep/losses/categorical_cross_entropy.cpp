// Library Header
#include <absl/log/log.h>
#include <stdexcept>
#include <vector>

// Library Header
#include "loss.hpp"
#include "loss_enum.hpp"
#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>

CategoricalCrossEntropy::~CategoricalCrossEntropy() {}

CategoricalCrossEntropy::CategoricalCrossEntropy(
    std::vector<Tensor<std::float64_t> *> input_preds) {
  if (input_preds.size() == 1) {
    this->input_predicts.push_back(input_preds[0]);
  } else {
    LOG(ERROR) << "Fata! Categorical Cross Entropy: expects only 1 input "
                  "tensor but given, "
               << input_preds.size() << ".\n";
    throw std::runtime_error("Exiting program due to errorneous input for "
                             "Categorical Cross Entropy");
  }
}

void CategoricalCrossEntropy::forward(std::vector<tf::tensor *> inputs,
                                      const unsigned batch_size) {

  if (inputs.size() == 1) {
    this->training_inputs.push_back(inputs[0]);
    std::vector<unsigned> dims;
    unsigned i;
    for (i = 0; i < this->training_inputs[0]->getNoOfDimensions() - 1; i++)
      dims.push_back(this->training_inputs[0]->getDimensions()[i]);

    dims.push_back(batch_size);
    target_outputs.push_back(new tf::tensor());
    target_outputs[0]->tf_create(dims, this->training_inputs[0]->dt_type);

    this->log_value = new tf::tensor();
    this->log_value->tf_create(dims, this->training_inputs[0]->dt_type);

    this->log_difference = new tf::tensor();
    this->log_difference->tf_create(dims, this->training_inputs[0]->dt_type);

    this->loss_tensor_batch = new tf::tensor();
    this->loss_tensor_batch->tf_create(dims, this->training_inputs[0]->dt_type);

    this->loss_tensor = new tf::tensor();
    this->loss_tensor->tf_create(dims, this->training_inputs[0]->dt_type);

    *(this->log_value) = this->training_inputs[0]->log(); // log(y^)
    *(this->log_difference) =
        this->log_value->mul(*(this->target_outputs[0])); // y_i log(y^)
    *(this->loss_tensor_batch) =
        this->log_difference->scale(-1.0f); // -1 y_i log(y^)
    *(this->loss_tensor) =
        (this->loss_tensor_batch)
            ->mean(this->training_inputs[0]->getNoOfDimensions() -
                   1); //  -1 y_i log(y^) / batch_size;
  }
}

void CategoricalCrossEntropy::backward() {
  if (!this->isGradRecorded) {
    this->isGradRecorded = true;

    Graph *g = GraphManager::instance().getCurrentGraph();
    if (g)
      this->grad_training_inputs =
          tf::tensor(this->training_inputs[0]->dt_type,
                     g->getGradientTensor(this->training_inputs[0]->getPtr()));
  }
}

void CategoricalCrossEntropy::setPredictedOutput(
    const std::vector<tf::tensor *> &input_predicts) {
  // this->input_predicts = input_predicts;
}

void CategoricalCrossEntropy::setTargetOutput(
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

std::float64_t const CategoricalCrossEntropy::getScalerLoss() {
  tf::tensor temp_tensor;
  std::vector<unsigned> dims;

  for (unsigned i = 0; i < this->loss_tensor->getNoOfDimensions(); i++)
    dims.push_back(this->loss_tensor->getDimensions()[i]);
  temp_tensor.tf_create(dims, this->loss_tensor->dt_type);
  temp_tensor.tensor_of(this->loss_tensor->getData());

  if (temp_tensor.getNoOfDimensions()) {
    for (size_t i = 0; i < temp_tensor.getNoOfDimensions(); i++) {
      tf::tensor reduced_tensor = temp_tensor.mean(0, false);
      temp_tensor = std::move(reduced_tensor);
    }
    this->loss_value = temp_tensor.getData()[0];
  }
  return loss_value;
}

std::vector<tf::tensor *>
CategoricalCrossEntropy::getLossParameter(Loss_Parameter loss_parameter) {
  std::vector<tf::tensor *> temp_tensor;
  switch (loss_parameter) {
  case Loss_Parameter::categorical_cross_entropy_error: {
    temp_tensor.push_back(this->loss_tensor);
    break;
  }
  case Loss_Parameter::mean_categorical_cross_entropy_error: {
    temp_tensor.push_back(&this->grad_training_inputs);
    break;
  }
  case Loss_Parameter::categorical_cross_entropy_target_output: {
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
