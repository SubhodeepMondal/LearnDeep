// Library Headers
#include "dense.hpp"
#include "layer_graph.hpp"
#include "layers.hpp"
#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>

// --- Constructor
Dense::Dense(unsigned unit)
    : no_of_unit(unit), no_of_features(1), batch_size(1),
      forwardGraphCreated(false), backwardGraphCrated(false),
      layer_type(tf_dense) {

  global_layer_graph.addNode(this);
};

// ---Destructor
Dense::~Dense() {
  this->layer_inputs.clear();
  this->layer_outputs.clear();
}

std::vector<Tensor<std::float64_t> *>
Dense::operator()(const std::vector<Tensor<std::float64_t> *> &input_tensors) {

  // do a lazy initialization
  if (input_tensors.size() == 1) {
    this->layer_inputs = input_tensors;

    unsigned arr[2];
    arr[0] = this->no_of_unit;
    arr[1] = 1;

    this->weight = new Tensor<std::float64_t>(2, arr, tf_float64);
    this->bias = new Tensor<std::float64_t>(2, arr, tf_float64);
    this->matmul_result = new Tensor<std::float64_t>(2, arr, tf_float64);
    Tensor<std::float64_t> *output =
        new Tensor<std::float64_t>(2, arr, tf_float64);
    this->layer_outputs.push_back(output);
    this->ops.push_back(new Opsmatmul);
    this->ops.push_back(new Opsadd);

  } else {
    LOG(ERROR) << "Fatal! Dense: Layer expects only one tensor as input.\n";
  }
  return this->layer_outputs;
}

std::vector<Tensor<std::float64_t> *>
Dense::forward(std::vector<Tensor<std::float64_t> *> input,
               unsigned batch_size) {
  if (input.size() == 1) {
    if (input[0]->getNoOfDimensions() == 2) {
      this->no_of_features = input[0]->getDimensions()[0];
      this->batch_size = batch_size;

      this->training_input = input[0];

      unsigned arr[2];
      arr[0] = this->no_of_unit;
      arr[1] = this->no_of_features;
      this->weight->reshape(2, arr);
      this->training_weight = new Tensor<std::float64_t>(2, arr, tf_float64);

      arr[1] = this->batch_size;
      this->training_matmul_result =
          new Tensor<std::float64_t>(2, arr, tf_float64);
      this->training_bias = new Tensor<std::float64_t>(2, arr, tf_float64);
      this->training_output = new Tensor<std::float64_t>(2, arr, tf_float64);

      this->initializeWeight();
      this->initializeBias();

      Graph *g = GraphManager::instance().getCurrentGraph();
      if (g) {
        this->training_matmul_result = this->training_input->matmul(
            *(this->training_weight),
            std::span(ops).subspan(0)); // matmul_result = input.matmul(weights)

        this->training_output = this->training_matmul_result->add(
            *(this->training_bias),
            std::span(ops).subspan(1)); // output = matmul_result + bias
      } else {
        LOG(ERROR)
            << "Fatal! There no active graph session for forward calculation";
      }
    } else {
      LOG(ERROR)
          << "Fatal! Dense: Layer expects rank of 2, here input has rank of "
          << input[0]->getNoOfDimensions() << ".\n";
    }
  } else {
    LOG(ERROR) << "Fatal! Dense: Layer expects only one input tensors but here "
                  "given no of tensors are "
               << input.size() << ".\n";
  }
  return {this->training_output};
}

void Dense::backward() {}

LayerType Dense::getLayerType() { return this->layer_type; }

std::vector<Tensor<std::float64_t> *> Dense::getInputTensors() {
  return this->layer_inputs;
}

std::vector<Tensor<std::float64_t> *> Dense::getOutputTensors() {
  return this->layer_outputs;
}

std::vector<Tensor<std::float64_t> *> Dense::getInputTrainingTensors() {
  return {this->training_input};
}

std::vector<Tensor<std::float64_t> *> Dense::getOutputTrainingTensors() {
  return {this->training_output};
}

void Dense::setWeightInitializationMethod(
    InitializationMethod initilization_method) {
  this->weight_initialization_method = initilization_method;
}

void Dense::setBiasInitializationMethod(
    InitializationMethod initilization_method) {
  this->bias_initialization_method = initilization_method;
}

void Dense::setWeight(Tensor<std::float64_t> *weight_tensor) {
  this->weight_initialization_method = InitializationMethod::MANUAL;
  this->initialization_weight = *(weight_tensor);
}

void Dense::setBias(Tensor<std::float64_t> *bias_tensor) {
  this->bias_initialization_method = InitializationMethod::MANUAL;
  this->initialization_bias = *(bias_tensor);
}

std::vector<Tensor<std::float64_t> *>
Dense::getLayerParameter(Layer_Parameter layer_parameter, bool print_flag) {
  std::vector<Tensor<std::float64_t> *> layer_parameter_tensor;
  layer_parameter_tensor.clear();
  switch (layer_parameter) {
  case Layer_Parameter::dense_input:
    LOG(INFO) << "Layer: Dense, input:\n";
    layer_parameter_tensor = layer_inputs;
    break;
  case Layer_Parameter::dense_weight:
    LOG(INFO) << "Layer: Dense, weight:\n";
    layer_parameter_tensor.push_back(this->weight);
    break;
  case Layer_Parameter::dense_bias:
    LOG(INFO) << "Layer: Dense, bias:\n";
    layer_parameter_tensor.push_back(this->bias);
    break;
  case Layer_Parameter::dense_output:
    LOG(INFO) << "Layer: Dense, output:\n";
    layer_parameter_tensor = this->layer_outputs;
    break;
  case Layer_Parameter::dense_training_input:
    LOG(INFO) << "Layer: Dense, training input:\n";
    layer_parameter_tensor.push_back(this->training_input);
    break;
  case Layer_Parameter::dense_training_weight:
    LOG(INFO) << "Layer: Dense, training weight:\n";
    layer_parameter_tensor.push_back(this->training_weight);
    break;
  case Layer_Parameter::dense_training_bias:
    LOG(INFO) << "Layer: Dense, training bias:\n";
    layer_parameter_tensor.push_back(this->training_bias);
    break;
  case Layer_Parameter::dense_training_output:
    LOG(INFO) << "Layer: Dense, training output:\n";
    layer_parameter_tensor.push_back(this->training_output);
    break;
  case Layer_Parameter::dense_grad_weight:
    LOG(INFO) << "Layer: Dense, grad weight:\n";
    layer_parameter_tensor.push_back(this->grad_weight);
    break;
  case Layer_Parameter::dense_grad_bias:
    LOG(INFO) << "Layer: Dense, grad bias:\n";
    layer_parameter_tensor.push_back(this->grad_bias);
    break;

  default:
    LOG(ERROR) << "Sever! the selected layer parameter is not available for "
                  "dense layer.\n";
    break;
  }
  if (print_flag && layer_parameter_tensor.size())
    layer_parameter_tensor[0]->printData();
  return layer_parameter_tensor;
}

void Dense::initializeWeight() {
  switch (this->weight_initialization_method) {
  case InitializationMethod::MANUAL: {
    unsigned size = this->weight->getNoOfElem();
    this->weight->initData(this->initialization_weight.getData());
    this->training_weight->initData(this->weight->getData());
    break;
  }
  case InitializationMethod::ZEROS: {
    this->training_weight->initData(0.0);
    break;
  }
  case InitializationMethod::ONES: {
    this->training_weight->initData(1.0);
    break;
  }
  default:
    LOG(ERROR) << "Fatal! No initilizattion method for dense weight.\n";
    break;
  }
}

void Dense::initializeBias() {
  switch (this->bias_initialization_method) {
  case InitializationMethod::MANUAL: {
    unsigned size = this->bias->getNoOfElem();
    this->bias->initData(this->initialization_bias.getData());
    for (unsigned i = 0; i < this->batch_size; i++) {
      unsigned index = i * size;
      this->training_bias->initPartialData(index, size, this->bias->getData());
    }
    break;
  }
  case InitializationMethod::ZEROS: {
    this->bias->initData(0.0);
    this->training_bias->initData(0.0);
    break;
  }
  case InitializationMethod::ONES: {
    this->bias->initData(1.0);
    this->training_bias->initData(1.0);
    break;
  }
  default:
    LOG(ERROR) << "Fatal! No initilizattion method for dense bias.\n";
    break;
  }
}