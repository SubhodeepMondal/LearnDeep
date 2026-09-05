// Library Headers
#include "softmax_layer.hpp"
#include "layer_enum.hpp"
#include "layer_graph.hpp"
#include "layers.hpp"
#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>
#include <optimizers/optimizers.hpp>

// Third party
#include <absl/log/log.h>

Softmax::Softmax(unsigned axis) : axis(axis), layer_type(tf_softmax_layer) {
  this->isForwardGraphCreated = false;
  this->isBackwardGraphCreated = false;
  this->isGradRecorded = false;
}

// --- Destructor
Softmax::~Softmax() {
  this->layer_inputs.clear();
  this->layer_outputs.clear();
}

const std::vector<tf::tensor *> &
Softmax::operator()(std::vector<tf::tensor> input_tensors) {

  this->axis = axis;
  // do a lazy initialization
  if (input_tensors.size() == 1) {
    this->layer_inputs.push_back(input_tensors[0].getPtr());

    std::vector<unsigned> dims;
    for (unsigned i = 0; i < input_tensors[0].getNoOfDimensions(); i++)
      dims.push_back(input_tensors[0].getDimensions()[i]);

    this->output.tf_create(dims, tf_float64);
    this->layer_outputs.push_back(&output);

  } else {
    LOG(ERROR) << "Fatal! Relu: Layer expects only one tensor as input.\n";
  }
  return this->layer_outputs;
}

const std::vector<tf::tensor *> &
Softmax::forward(std::vector<const tf::tensor *> &input, unsigned batch_size) {
  if (input.size() == 1) {

    std::vector<unsigned> input_dims;
    for (unsigned i = 0;
         i < const_cast<tf::tensor *>(input[0])->getNoOfDimensions(); i++)
      input_dims.push_back(input[0]->getDimensions()[i]);

    this->training_inputs = input[0];

    this->training_output = training_inputs->softmax(this->axis);
    this->training_outputs.push_back(&training_output);

  } else {
    LOG(ERROR) << "Fatal! Relu: Layer expects only one input tensors but here "
                  "given no of tensors are "
               << input.size() << ".\n";
  }
  return this->training_outputs;
}

void Softmax::backward(Optimizer *optimiser) {}

LayerType Softmax::getLayerType() { return this->layer_type; }

std::vector<const Tensor<std::float64_t> *> &Softmax::getInputTensors() {
  return this->layer_inputs;
}

const std::vector<tf::tensor *> &Softmax::getOutputTensors() {
  return this->layer_outputs;
}

std::vector<const tf::tensor *> Softmax::getInputTrainingTensors() {
  return {this->training_inputs};
}

const std::vector<tf::tensor *> &Softmax::getOutputTrainingTensors() {
  return this->training_outputs;
}

std::vector<tf::tensor *>
Softmax::getLayerParameter(Layer_Parameter layer_parameter, bool print_flag) {
  std::vector<tf::tensor *> layer_parameter_tensor;
  switch (layer_parameter) {
  case Layer_Parameter::softmax_input:
    LOG(INFO) << "Layer: Softmax, input:\n";
    break;
  case Layer_Parameter::softmax_output:
    LOG(INFO) << "Layer: Softmax, output:\n";
    layer_parameter_tensor = this->layer_outputs;
    break;
  case Layer_Parameter::softmax_training_input:
    LOG(INFO) << "Layer: Selu, training input:\n";
    layer_parameter_tensor.push_back(
        const_cast<tf::tensor *>(this->training_inputs));
    break;
  case Layer_Parameter::softmax_training_output:
    LOG(INFO) << "Layer: Selu, training output:\n";
    for (tf::tensor *training_output_tensor : this->training_outputs)
      layer_parameter_tensor.push_back(training_output_tensor);
    break;
  default:
    LOG(ERROR) << "Sever! the selected layer parameter is not available for "
                  "relu layer.\n";
    break;
  }
  if (print_flag && layer_parameter_tensor.size())
    layer_parameter_tensor[0]->print_data();
  return layer_parameter_tensor;
}

void Softmax::initializeParameters() {};
