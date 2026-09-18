// Library Headers
#include "relu.hpp"
#include "layer_enum.hpp"
#include "layer_graph.hpp"
#include "layers.hpp"
#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>
#include <optimizers/optimizers.hpp>

// Third party
#include <absl/log/log.h>

Relu::Relu() {
  this->isForwardGraphCreated = false;
  this->isBackwardGraphCreated = false;
  this->isGradRecorded = false;
}

// --- Destructor
Relu::~Relu() {
  this->layer_inputs.clear();
  this->layer_outputs.clear();
}

const std::vector<tf::tensor *> &
Relu::operator()(std::vector<tf::tensor> input_tensors) {

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
Relu::forward(std::vector<const tf::tensor *> &input, unsigned batch_size) {
  if (input.size() == 1) {

    std::vector<unsigned> input_dims;
    for (unsigned i = 0;
         i < const_cast<tf::tensor *>(input[0])->getNoOfDimensions(); i++)
      input_dims.push_back(input[0]->getDimensions()[i]);

    this->training_inputs = input[0];

    this->training_output = training_inputs->relu();
    this->training_outputs.push_back(&training_output);

  } else {
    LOG(ERROR) << "Fatal! Relu: Layer expects only one input tensors but here "
                  "given no of tensors are "
               << input.size() << ".\n";
  }
  return this->training_outputs;
}

void Relu::backward(Optimizer *optimiser) {

  if (!this->isGradRecorded) {

    Graph *g = GraphManager::instance().getCurrentGraph();

    this->grad_input =
        tf::tensor(this->training_inputs->dt_type,
                   g->getGradientTensor(this->training_inputs->getPtr()));
    this->isGradRecorded = true;
  }
}

LayerType Relu::getLayerType() { return this->layer_type; }

std::vector<const Tensor<std::float64_t> *> &Relu::getInputTensors() {
  return this->layer_inputs;
}

const std::vector<tf::tensor *> &Relu::getOutputTensors() {
  return this->layer_outputs;
}

std::vector<const tf::tensor *> Relu::getInputTrainingTensors() {
  return {this->training_inputs};
}

const std::vector<tf::tensor *> &Relu::getOutputTrainingTensors() {
  return this->training_outputs;
}

std::vector<tf::tensor *>
Relu::getLayerParameter(Layer_Parameter layer_parameter, bool print_flag) {
  std::vector<tf::tensor *> layer_parameter_tensor;
  switch (layer_parameter) {
  case Layer_Parameter::relu_input:
    LOG(INFO) << "Layer: Relu, input:\n";
    break;
  case Layer_Parameter::relu_output:
    LOG(INFO) << "Layer: Relu, output:\n";
    layer_parameter_tensor = this->layer_outputs;
    break;
  case Layer_Parameter::relu_training_input:
    LOG(INFO) << "Layer: Relu, training input:\n";
    layer_parameter_tensor.push_back(
        const_cast<tf::tensor *>(this->training_inputs));
    break;
  case Layer_Parameter::relu_training_output:
    LOG(INFO) << "Layer: Relu, training output:\n";
    for (tf::tensor *training_output_tensor : this->training_outputs)
      layer_parameter_tensor.push_back(training_output_tensor);
    break;
  case Layer_Parameter::relu_grad_input:
    LOG(INFO) << "Layer: Relu, grad input:\n";
    layer_parameter_tensor.push_back(&this->grad_input);
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

void Relu::initializeParameters() {};
