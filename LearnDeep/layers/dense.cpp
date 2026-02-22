// Library Headers
#include "dense.hpp"
#include "layer_enum.hpp"
#include "layer_graph.hpp"
#include "layers.hpp"
#include <absl/log/log.h>
#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>
#include <optimizers/optimizers.hpp>

// --- Constructor
Dense::Dense(unsigned unit)
    : no_of_unit(unit), no_of_features(1), batch_size(1),
      forwardGraphCreated(false), backwardGraphCrated(false),
      layer_type(tf_dense), isGradRecorded(false){

                            };

// ---Destructor
Dense::~Dense() {
  this->layer_inputs.clear();
  this->layer_outputs.clear();
}

const std::vector<tf::tensor *> &
Dense::operator()(std::vector<tf::tensor> input_tensors) {

  // do a lazy initialization
  if (input_tensors.size() == 1) {
    this->layer_inputs.push_back(input_tensors[0].getPtr());

    std::vector<unsigned> arr(2);
    arr[0] = this->no_of_unit;
    arr[1] = 1;

    this->weight.tf_create(arr, tf_float64);
    this->bias.tf_create(arr, tf_float64);
    this->matmul_result.tf_create(arr, tf_float64);
    output.tf_create(arr, tf_float64);
    this->layer_outputs.push_back(&output);

  } else {
    LOG(ERROR) << "Fatal! Dense: Layer expects only one tensor as input.\n";
  }
  return this->layer_outputs;
}

const std::vector<tf::tensor *> &
Dense::forward(std::vector<const tf::tensor *> &input, unsigned batch_size) {
  if (input.size() == 1) {
    if (input[0]->getPtr()->getNoOfDimensions() == 2) {
      this->no_of_features = input[0]->getPtr()->getDimensions()[0];
      this->batch_size = batch_size;

      this->training_inputs = input[0];

      std::vector<unsigned> arr(2);
      arr[0] = this->no_of_unit;
      arr[1] = this->no_of_features;
      this->weight.reshape(arr);
      this->training_weight.tf_create(arr, tf_float64);
      this->grad_weight.tf_create(arr, tf_float64);

      arr[1] = this->batch_size;
      this->training_matmul_result.tf_create(arr, tf_float64);
      this->training_output.tf_create(arr, tf_float64);
      arr[1] = 1;
      this->training_bias.tf_create(arr, tf_float64);
      this->grad_bias.tf_create(arr, tf_float64);

      this->training_matmul_result = training_inputs->matmul(training_weight);
      this->training_output = training_matmul_result.add(training_bias);
      this->training_outputs.push_back(&training_output);

    } else {
      LOG(ERROR)
          << "Fatal! Dense: Layer expects rank of 2, here input has rank of "
          << input[0]->getPtr()->getNoOfDimensions() << ".\n";
    }
  } else {
    LOG(ERROR) << "Fatal! Dense: Layer expects only one input tensors but here "
                  "given no of tensors are "
               << input.size() << ".\n";
  }
  return this->training_outputs;
}

void Dense::backward(Optimizer *optimizer) {
  if (!this->isGradRecorded) {

    Graph *g = GraphManager::instance().getCurrentGraph();
    this->grad_weight =
        tf::tensor(this->grad_weight.dt_type,
                   g->getGradientTensor(this->training_weight.getPtr()));
    this->grad_bias =
        tf::tensor(this->grad_bias.dt_type,
                   g->getGradientTensor(this->training_bias.getPtr()));
    optimizer->createParameterUpdateGraph(
        this->training_weight, this->updated_weight, this->grad_weight);
    optimizer->createParameterUpdateGraph(this->training_bias,
                                          this->updated_bias, this->grad_bias);
    this->isGradRecorded = true;
  }
}

LayerType Dense::getLayerType() { return this->layer_type; }

std::vector<const Tensor<std::float64_t> *> &Dense::getInputTensors() {
  return this->layer_inputs;
}

const std::vector<tf::tensor *> &Dense::getOutputTensors() {
  return this->layer_outputs;
}

std::vector<const tf::tensor *> Dense::getInputTrainingTensors() {
  return {this->training_inputs};
}

const std::vector<tf::tensor *> &Dense::getOutputTrainingTensors() {
  return this->training_outputs;
}

void Dense::setWeightInitializationMethod(
    InitializationMethod initilization_method) {
  this->weight_initialization_method = initilization_method;
}

void Dense::setBiasInitializationMethod(
    InitializationMethod initilization_method) {
  this->bias_initialization_method = initilization_method;
}

void Dense::setWeight(const tf::tensor &weight_tensor) {
  this->weight_initialization_method = InitializationMethod::MANUAL;
  std::vector<unsigned> dims(2);
  dims[0] = weight_tensor.getDimensions()[0];
  dims[1] = weight_tensor.getDimensions()[1];

  this->initialization_weight.tf_create(dims, weight_tensor.dt_type);
  this->initialization_weight.tensor_of(weight_tensor.getData());
}

void Dense::setBias(const tf::tensor &bias_tensor) {
  this->bias_initialization_method = InitializationMethod::MANUAL;

  std::vector<unsigned> dims(2);
  dims[0] = bias_tensor.getDimensions()[0];
  dims[1] = bias_tensor.getDimensions()[1];

  this->initialization_bias.tf_create(dims, bias_tensor.dt_type);
  this->initialization_bias.tensor_of(bias_tensor.getData());
}

std::vector<tf::tensor *>
Dense::getLayerParameter(Layer_Parameter layer_parameter, bool print_flag) {
  std::vector<tf::tensor *> layer_parameter_tensor;
  switch (layer_parameter) {
  case Layer_Parameter::dense_input:
    LOG(INFO) << "Layer: Dense, input:\n";
    // layer_parameter_tensor = layer_inputs;
    break;
  case Layer_Parameter::dense_weight:
    LOG(INFO) << "Layer: Dense, weight:\n";
    layer_parameter_tensor.push_back(&this->weight);
    break;
  case Layer_Parameter::dense_bias:
    LOG(INFO) << "Layer: Dense, bias:\n";
    layer_parameter_tensor.push_back(&this->bias);
    break;
  case Layer_Parameter::dense_output:
    LOG(INFO) << "Layer: Dense, output:\n";
    layer_parameter_tensor = this->layer_outputs;
    break;
  case Layer_Parameter::dense_training_input:
    LOG(INFO) << "Layer: Dense, training input:\n";
    // layer_parameter_tensor.push_back(this->training_input);
    break;
  case Layer_Parameter::dense_training_weight:
    LOG(INFO) << "Layer: Dense, training weight:\n";
    layer_parameter_tensor.push_back(&this->training_weight);
    break;
  case Layer_Parameter::dense_training_bias:
    LOG(INFO) << "Layer: Dense, training bias:\n";
    layer_parameter_tensor.push_back(&this->training_bias);
    break;
  case Layer_Parameter::dense_training_output:
    LOG(INFO) << "Layer: Dense, training output:\n";
    for (tf::tensor *training_output_tensor : this->training_outputs)
      layer_parameter_tensor.push_back(training_output_tensor);
    break;
  case Layer_Parameter::dense_grad_weight:
    LOG(INFO) << "Layer: Dense, grad weight:\n";
    layer_parameter_tensor.push_back(&this->grad_weight);
    break;
  case Layer_Parameter::dense_grad_bias:
    LOG(INFO) << "Layer: Dense, grad bias:\n";
    layer_parameter_tensor.push_back(&this->grad_bias);
    break;
  case Layer_Parameter::dense_updated_weight:
    LOG(INFO) << "Layer: Dense, grad weight:\n";
    layer_parameter_tensor.push_back(&this->updated_weight);
    break;
  case Layer_Parameter::dense_updated_bias:
    LOG(INFO) << "Layer: Dense, grad bias:\n";
    layer_parameter_tensor.push_back(&this->updated_bias);
    break;

  default:
    LOG(ERROR) << "Sever! the selected layer parameter is not available for "
                  "dense layer.\n";
    break;
  }
  if (print_flag && layer_parameter_tensor.size())
    layer_parameter_tensor[0]->print_data();
  return layer_parameter_tensor;
}

void Dense::initializeWeight() {
  switch (this->weight_initialization_method) {
  case InitializationMethod::MANUAL: {
    this->weight.getPtr()->initData(this->initialization_weight.getData());
    this->training_weight.getPtr()->initData(this->weight.getData());
    break;
  }
  case InitializationMethod::ZEROS: {
    this->training_weight.getPtr()->initData(0.0);
    break;
  }
  case InitializationMethod::ONES: {
    this->training_weight.getPtr()->initData(1.0);
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
    this->bias.getPtr()->initData(this->initialization_bias.getData());
    this->training_bias.getPtr()->initData(this->bias.getData());
    this->bias_initialization_method = InitializationMethod::UPDATE_FROM_GRAD;
    break;
  }
  case InitializationMethod::ZEROS: {
    this->bias.getPtr()->initData(0.0);
    this->training_bias.getPtr()->initData(0.0);
    this->bias_initialization_method = InitializationMethod::UPDATE_FROM_GRAD;
    break;
  }
  case InitializationMethod::ONES: {
    this->bias.getPtr()->initData(1.0);
    this->training_bias.getPtr()->initData(1.0);
    this->bias_initialization_method = InitializationMethod::UPDATE_FROM_GRAD;
    break;
  }
  case InitializationMethod::UPDATE_FROM_GRAD: {
    this->bias.getPtr()->initData(this->updated_bias.getData());
    this->updated_bias.getPtr()->initData(this->bias.getData());
    break;
  }
  default:
    LOG(ERROR) << "Fatal! No initilizattion method for dense bias.\n";
    break;
  }
}

void Dense::initializeParameters() {
  this->initializeWeight();
  this->initializeBias();
};

// End of Dense