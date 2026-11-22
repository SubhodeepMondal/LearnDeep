// Library Headers
#include "dense.hpp"
#include "layer_enum.hpp"
#include "layer_registry.hpp"
#include "layers.hpp"
#include <absl/log/log.h>
#include <core/graph/graph_framework.hpp>
#include <core/graph/graph_manager.hpp>
#include <optimizers/optimizers.hpp>
#include <vector>

REGISTER_LAYER(
    "dense",
    [](const LayerConfig &config) -> std::unique_ptr<Layer> {
      const unsigned *units = config.getAs<unsigned>("units");
      if (!units)
        units = config.getAs<unsigned>("unit");

      std::unique_ptr<Dense> dense =
          std::make_unique<Dense>(units ? *units : 1u);

      if (const tf::tensor *weight = config.getAs<tf::tensor>("weight"))
        dense->setWeight(*weight);
      if (const tf::tensor *bias = config.getAs<tf::tensor>("bias"))
        dense->setBias(*bias);

      return dense;
    },
    [](const LayerRegistry::LayerHandle &layer, const std::string &name,
       const tf::LayerParameterValue &value) -> bool {
      const tf::tensor *tensor = std::get_if<tf::tensor>(&value);
      if (!tensor)
        return false;

      std::shared_ptr<Dense> dense = std::dynamic_pointer_cast<Dense>(layer);
      if (!dense)
        return false;

      if (name == tf::layer::parameter::weight) {
        dense->setWeight(*tensor);
        return true;
      }
      if (name == tf::layer::parameter::bias) {
        dense->setBias(*tensor);
        return true;
      }
      return false;
    });

// --- Constructor
Dense::Dense(unsigned unit)
    : no_of_unit(unit), no_of_features(1), batch_size(1),
      forwardGraphCreated(false), backwardGraphCrated(false),
      isGradRecorded(false),
      weight_initialization_method(InitializationMethod::ZEROS),
      bias_initialization_method(InitializationMethod::ZEROS){

      };

// ---Destructor
Dense::~Dense() { this->layer_outputs.clear(); }

const std::vector<tf::tensor *> &
Dense::operator()(std::vector<tf::tensor> input_tensors) {

  // do a lazy initialization
  this->recordInputTensors(input_tensors);
  if (input_tensors.size() == 1) {
    if (input_tensors[0].getNoOfDimensions() != 2) {
      LOG(ERROR)
          << "Fatal! Dense: Layer expects rank of 2, here input has rank of "
          << input_tensors[0].getNoOfDimensions() << ".\n";
      return this->layer_outputs;
    }

    this->no_of_features = input_tensors[0].getDimensions()[0];
    this->batch_size = input_tensors[0].getDimensions()[1];

    std::vector<unsigned> arr(2);
    arr[0] = this->no_of_unit;
    arr[1] = 1;

    this->weight.tf_create(arr, tf_float64);
    arr[1] = 1;
    this->bias.tf_create(arr, tf_float64);
    arr[1] = 1;
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
  this->recordTrainingInputs(input);
  if (input.size() == 1 && !this->training_input_aliases.empty()) {
    tf::tensor *training_inputs = &this->training_input_aliases.front();
    if (training_inputs->getPtr()->getNoOfDimensions() == 2) {
      this->no_of_features = training_inputs->getDimensions()[0];
      this->batch_size = batch_size;

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
          << training_inputs->getPtr()->getNoOfDimensions() << ".\n";
    }
  } else {
    LOG(ERROR) << "Fatal! Dense: Layer expects only one input tensors but here "
                  "given no of tensors are "
               << input.size() << ".\n";
  }
  return this->training_outputs;
}

std::vector<tf::tensor> Dense::calculate(std::vector<tf::tensor> inputs) {
  initializeParameters();
  std::vector<tf::tensor> output;
  if (inputs.size() == 1) {
    tf::tensor matmul_out = inputs[0].matmul(weight, false);
    output.emplace_back(matmul_out.add(bias, false));
  }
  return output;
}

void Dense::backward(Optimizer *optimizer) {
  if (!this->isGradRecorded && !this->training_input_aliases.empty()) {

    tf::tensor *training_inputs = &this->training_input_aliases.front();
    Graph *g = GraphManager::instance().getCurrentGraph();
    this->grad_weight =
        tf::tensor(this->grad_weight.dt_type,
                   g->getGradientTensor(this->training_weight.getPtr()));
    this->grad_bias =
        tf::tensor(this->grad_bias.dt_type,
                   g->getGradientTensor(this->training_bias.getPtr()));
    this->grad_input =
        tf::tensor(training_inputs->dt_type,
                   g->getGradientTensor(training_inputs->getPtr()));
    this->incoming_grad =
        tf::tensor(training_inputs->dt_type,
                   g->getGradientTensor(this->training_output.getPtr()));
    optimizer->createParameterUpdateGraph(
        this->training_weight, this->updated_weight, this->grad_weight);
    optimizer->createParameterUpdateGraph(this->training_bias,
                                          this->updated_bias, this->grad_bias);
    this->isGradRecorded = true;
  }
}

std::string_view Dense::type() const { return tf::layer::type::dense; }

const std::vector<tf::tensor *> &Dense::getOutputTensors() {
  return this->layer_outputs;
}

std::vector<const tf::tensor *> Dense::getInputTrainingTensors() {
  return this->getTrainingInputAliases();
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
Dense::getLayerParameter(std::string_view layer_parameter, bool print_flag) {
  std::vector<tf::tensor *> layer_parameter_tensor;
  using tf::layer::parameter::bias;
  using tf::layer::parameter::grad_bias;
  using tf::layer::parameter::grad_input;
  using tf::layer::parameter::grad_weight;
  using tf::layer::parameter::incoming_grad;
  using tf::layer::parameter::input;
  using tf::layer::parameter::matmul_result;
  using tf::layer::parameter::output;
  using tf::layer::parameter::training_bias;
  using tf::layer::parameter::training_input;
  using tf::layer::parameter::training_matmul_result;
  using tf::layer::parameter::training_output;
  using tf::layer::parameter::training_weight;
  using tf::layer::parameter::updated_bias;
  using tf::layer::parameter::updated_weight;
  using tf::layer::parameter::weight;

  if (layer_parameter == input) {
    LOG(INFO) << "Layer: Dense, input:\n";
  } else if (layer_parameter == weight) {
    LOG(INFO) << "Layer: Dense, weight:\n";
    layer_parameter_tensor.push_back(&this->weight);
  } else if (layer_parameter == bias) {
    LOG(INFO) << "Layer: Dense, bias:\n";
    layer_parameter_tensor.push_back(&this->bias);
  } else if (layer_parameter == matmul_result) {
    LOG(INFO) << "Layer: Dense, matmul result:\n";
    layer_parameter_tensor.push_back(&this->matmul_result);
  } else if (layer_parameter == output) {
    LOG(INFO) << "Layer: Dense, output:\n";
    layer_parameter_tensor = this->layer_outputs;
  } else if (layer_parameter == training_input) {
    LOG(INFO) << "Layer: Dense, training input:\n";
    for (const tf::tensor *training_input : this->getTrainingInputAliases())
      layer_parameter_tensor.push_back(const_cast<tf::tensor *>(training_input));
  } else if (layer_parameter == training_weight) {
    LOG(INFO) << "Layer: Dense, training weight:\n";
    layer_parameter_tensor.push_back(&this->training_weight);
  } else if (layer_parameter == training_bias) {
    LOG(INFO) << "Layer: Dense, training bias:\n";
    layer_parameter_tensor.push_back(&this->training_bias);
  } else if (layer_parameter == training_matmul_result) {
    LOG(INFO) << "Layer: Dense, training matmul result:\n";
    layer_parameter_tensor.push_back(&this->training_matmul_result);
  } else if (layer_parameter == training_output) {
    LOG(INFO) << "Layer: Dense, training output:\n";
    for (tf::tensor *training_output_tensor : this->training_outputs)
      layer_parameter_tensor.push_back(training_output_tensor);
  } else if (layer_parameter == grad_input) {
    LOG(INFO) << "Layer: Dense, grad input:\n";
    layer_parameter_tensor.push_back(&this->grad_input);
  } else if (layer_parameter == grad_weight) {
    LOG(INFO) << "Layer: Dense, grad weight:\n";
    layer_parameter_tensor.push_back(&this->grad_weight);
  } else if (layer_parameter == grad_bias) {
    LOG(INFO) << "Layer: Dense, grad bias:\n";
    layer_parameter_tensor.push_back(&this->grad_bias);
  } else if (layer_parameter == incoming_grad) {
    LOG(INFO) << "Layer: Dense, incoming grad:\n";
    layer_parameter_tensor.push_back(&this->incoming_grad);
  } else if (layer_parameter == updated_weight) {
    LOG(INFO) << "Layer: Dense, updated weight:\n";
    layer_parameter_tensor.push_back(&this->updated_weight);
  } else if (layer_parameter == updated_bias) {
    LOG(INFO) << "Layer: Dense, updated bias:\n";
    layer_parameter_tensor.push_back(&this->updated_bias);
  } else {
    LOG(ERROR) << "Sever! the selected layer parameter is not available for "
                  "dense layer.\n";
  }

  if (print_flag && layer_parameter_tensor.size())
    layer_parameter_tensor[0]->print_data();
  return layer_parameter_tensor;
}

std::unordered_set<std::string> Dense::parameterNames() const {
  return {std::string(tf::layer::parameter::input),
          std::string(tf::layer::parameter::weight),
          std::string(tf::layer::parameter::bias),
          std::string(tf::layer::parameter::matmul_result),
          std::string(tf::layer::parameter::output),
          std::string(tf::layer::parameter::training_input),
          std::string(tf::layer::parameter::training_weight),
          std::string(tf::layer::parameter::training_bias),
          std::string(tf::layer::parameter::training_matmul_result),
          std::string(tf::layer::parameter::training_output),
          std::string(tf::layer::parameter::grad_input),
          std::string(tf::layer::parameter::grad_weight),
          std::string(tf::layer::parameter::grad_bias),
          std::string(tf::layer::parameter::incoming_grad),
          std::string(tf::layer::parameter::updated_weight),
          std::string(tf::layer::parameter::updated_bias)};
}

void Dense::initializeWeight() {
  switch (this->weight_initialization_method) {
  case InitializationMethod::MANUAL: {
    this->weight.reshape({this->initialization_weight.getDimensions()[0],
                          this->initialization_weight.getDimensions()[1]});
    this->weight.getPtr()->initData(this->initialization_weight.getData());
    if (this->training_weight.getPtr())
      // this->training_weight.getPtr()->initData(this->weight.getData());
      this->training_weight.getPtr()->initData(this->weight.getData());
    this->weight_initialization_method = InitializationMethod::UPDATE_FROM_GRAD;
    break;
  }
  case InitializationMethod::ZEROS: {
    this->weight.getPtr()->initData(0.0);
    if (this->training_weight.getPtr())
      this->training_weight.getPtr()->initData(0.0);
    this->weight_initialization_method = InitializationMethod::UPDATE_FROM_GRAD;
    break;
  }
  case InitializationMethod::ONES: {
    this->weight.getPtr()->initData(1.0);
    if (this->training_weight.getPtr())
      this->training_weight.getPtr()->initData(1.0);
    this->weight_initialization_method = InitializationMethod::UPDATE_FROM_GRAD;
    break;
  }
  case InitializationMethod::UPDATE_FROM_GRAD: {
    if (this->updated_weight.getPtr()) {
      this->weight.reshape({this->updated_weight.getDimensions()[0],
                            this->updated_weight.getDimensions()[1]});
      this->weight.getPtr()->initData(this->updated_weight.getData());
      if (this->training_weight.getPtr())
        this->training_weight.getPtr()->initData(
            this->updated_weight.getData());
    }
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
    this->bias.reshape({this->initialization_bias.getDimensions()[0],
                        this->initialization_bias.getDimensions()[1]});
    this->bias.getPtr()->initData(this->initialization_bias.getData());
    if (this->training_bias.getPtr())
      this->training_bias.getPtr()->initData(this->bias.getData());
    this->bias_initialization_method = InitializationMethod::UPDATE_FROM_GRAD;
    break;
  }
  case InitializationMethod::ZEROS: {

    this->bias.getPtr()->initData(0.0);
    if (this->training_bias.getPtr())
      this->training_bias.getPtr()->initData(0.0);
    this->bias_initialization_method = InitializationMethod::UPDATE_FROM_GRAD;
    break;
  }
  case InitializationMethod::ONES: {
    this->bias.getPtr()->initData(1.0);
    if (this->training_bias.getPtr())
      this->training_bias.getPtr()->initData(1.0);
    this->bias_initialization_method = InitializationMethod::UPDATE_FROM_GRAD;
    break;
  }
  case InitializationMethod::UPDATE_FROM_GRAD: {
    if (this->updated_bias.getPtr()) {
      this->bias.reshape({this->updated_bias.getDimensions()[0],
                          this->updated_bias.getDimensions()[1]});
      this->bias.getPtr()->initData(this->updated_bias.getData());
      if (this->training_bias.getPtr())
        this->training_bias.getPtr()->initData(this->updated_bias.getData());
    }

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