// Library Headers
#include "softmax_layer.hpp"
#include "layer_enum.hpp"
#include "layer_registry.hpp"
#include "layers.hpp"
#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>
#include <optimizers/optimizers.hpp>

// Third party
#include <absl/log/log.h>

REGISTER_LAYER("softmax",
               [](const LayerConfig &config) -> std::unique_ptr<Layer> {
                 const unsigned *axis = config.getAs<unsigned>("axis");
                 return std::make_unique<Softmax>(axis ? *axis : 0u);
               });

Softmax::Softmax(unsigned axis) : axis(axis) {
  this->isForwardGraphCreated = false;
  this->isBackwardGraphCreated = false;
  this->isGradRecorded = false;
}

// --- Destructor
Softmax::~Softmax() { this->layer_outputs.clear(); }

const std::vector<tf::tensor *> &
Softmax::operator()(std::vector<tf::tensor> input_tensors) {

  // do a lazy initialization
  this->recordInputTensors(input_tensors);
  if (input_tensors.size() == 1) {
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
  (void)batch_size;
  this->recordTrainingInputs(input);
  if (input.size() == 1 && !this->training_input_aliases.empty()) {

    tf::tensor *training_inputs = &this->training_input_aliases.front();
    std::vector<unsigned> input_dims;
    for (unsigned i = 0; i < training_inputs->getNoOfDimensions(); i++)
      input_dims.push_back(training_inputs->getDimensions()[i]);

    this->training_output = training_inputs->softmax(this->axis);
    this->training_outputs.push_back(&training_output);

  } else {
    LOG(ERROR) << "Fatal! Relu: Layer expects only one input tensors but here "
                  "given no of tensors are "
               << input.size() << ".\n";
  }
  return this->training_outputs;
}

std::vector<tf::tensor> Softmax::calculate(std::vector<tf::tensor> inputs) {
  std::vector<tf::tensor> output;
  if (inputs.size() == 1) {
    output.emplace_back(inputs[0].softmax(this->axis, false));
  }
  return output;
}

void Softmax::backward(Optimizer *optimiser) {

  if (!this->isGradRecorded && !this->training_input_aliases.empty()) {

    tf::tensor *training_inputs = &this->training_input_aliases.front();
    Graph *g = GraphManager::instance().getCurrentGraph();
    this->grad_input =
        tf::tensor(training_inputs->dt_type,
                   g->getGradientTensor(training_inputs->getPtr()));
    this->isGradRecorded = true;
  }
}

std::string_view Softmax::type() const { return tf::layer::type::softmax; }

const std::vector<tf::tensor *> &Softmax::getOutputTensors() {
  return this->layer_outputs;
}

std::vector<const tf::tensor *> Softmax::getInputTrainingTensors() {
  return this->getTrainingInputAliases();
}

const std::vector<tf::tensor *> &Softmax::getOutputTrainingTensors() {
  return this->training_outputs;
}

std::vector<tf::tensor *>
Softmax::getLayerParameter(std::string_view layer_parameter, bool print_flag) {
  std::vector<tf::tensor *> layer_parameter_tensor;
  using tf::layer::parameter::grad_input;
  using tf::layer::parameter::input;
  using tf::layer::parameter::output;
  using tf::layer::parameter::training_input;
  using tf::layer::parameter::training_output;

  if (layer_parameter == input) {
    LOG(INFO) << "Layer: Softmax, input:\n";
  } else if (layer_parameter == output) {
    LOG(INFO) << "Layer: Softmax, output:\n";
    layer_parameter_tensor = this->layer_outputs;
  } else if (layer_parameter == training_input) {
    LOG(INFO) << "Layer: Softmax, training input:\n";
    for (const tf::tensor *training_input : this->getTrainingInputAliases())
      layer_parameter_tensor.push_back(const_cast<tf::tensor *>(training_input));
  } else if (layer_parameter == training_output) {
    LOG(INFO) << "Layer: Softmax, training output:\n";
    for (tf::tensor *training_output_tensor : this->training_outputs)
      layer_parameter_tensor.push_back(training_output_tensor);
  } else if (layer_parameter == grad_input) {
    LOG(INFO) << "Layer: Softmax, grad input:\n";
    layer_parameter_tensor.push_back(&this->grad_input);
  } else {
    LOG(ERROR) << "Sever! the selected layer parameter is not available for "
                  "softmax layer.\n";
  }

  if (print_flag && layer_parameter_tensor.size())
    layer_parameter_tensor[0]->print_data();
  return layer_parameter_tensor;
}

std::unordered_set<std::string> Softmax::parameterNames() const {
  return {std::string(tf::layer::parameter::input),
          std::string(tf::layer::parameter::output),
          std::string(tf::layer::parameter::training_input),
          std::string(tf::layer::parameter::training_output),
          std::string(tf::layer::parameter::grad_input)};
}

void Softmax::initializeParameters() {};
