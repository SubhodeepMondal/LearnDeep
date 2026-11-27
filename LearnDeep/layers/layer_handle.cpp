// Implementation of the public tf::layer::layer handle family.
//
// This lives in the layers library (not tensor_api) so that Model can use the
// public handle ABI without introducing a model <-> tensor_api link cycle.

#include <api/tensor.h>

#include <absl/log/log.h>

#include "layer_graph.hpp"
#include "layer_handle_helpers.hpp"
#include "layer_manager.hpp"
#include "layer_registry.hpp"
#include "layers.hpp"

#include <string>
#include <utility>

// --- Generic layer handle ---

tf::layer::layer::layer(std::shared_ptr<Layer> layer_ptr)
    : layer_ptr(std::move(layer_ptr)) {}

tf::layer::layer::layer() = default;

tf::layer::layer::layer(std::string_view type_name)
    : layer(type_name, LayerConfig{}) {}

tf::layer::layer::layer(std::string_view type_name, const LayerConfig &config) {
  this->layer_ptr =
      LayerManager::instance().create(std::string(type_name), config);
}

tf::layer::layer::~layer() = default;

std::vector<const tf::tensor *> tf::layer::layer::input_tensor_aliases() const {
  if (!this->layer_ptr)
    return {};
  return this->layer_ptr->getInputTensorAliases();
}

tf::layer::layer tf::layer::layer::from_id(layer_id id) {
  if (id == invalid_layer_id)
    return {};
  return layer(global_layer_graph_manager.lock(id));
}

tf::layer::layer_id
tf::layer::layer::producer_of(const Tensor<std::float64_t> *tensor) {
  return global_layer_graph_manager.producerOf(tensor);
}

std::vector<tf::layer::layer_id>
tf::layer::layer::consumers_of(const Tensor<std::float64_t> *tensor) {
  return global_layer_graph_manager.consumersOf(tensor);
}

std::vector<tf::tensor>
tf::layer::layer::operator()(const std::vector<tf::tensor> &inputs) {
  if (!this->layer_ptr)
    return {};

  this->layer_ptr->recordInputTensors(inputs);
  return ld::detail::copy_output_tensors((*this->layer_ptr)(inputs));
}

std::vector<tf::tensor>
tf::layer::layer::forward(const std::vector<tf::tensor> &inputs,
                          unsigned batch_size) {
  if (!this->layer_ptr)
    return {};

  this->layer_ptr->recordInputTensors(inputs);
  std::vector<const tf::tensor *> raw_inputs;
  raw_inputs.reserve(inputs.size());
  for (const tf::tensor &input : inputs)
    raw_inputs.push_back(&input);

  return ld::detail::copy_output_tensors(
      this->layer_ptr->forward(raw_inputs, batch_size));
}

std::vector<tf::tensor>
tf::layer::layer::calculate(std::vector<tf::tensor> inputs) {
  if (!this->layer_ptr)
    return {};

  this->layer_ptr->recordInputTensors(inputs);
  return this->layer_ptr->calculate(inputs);
}

std::vector<tf::tensor> tf::layer::layer::get_input_tensors() const {
  if (!this->layer_ptr)
    return {};

  const std::vector<const tf::tensor *> input_aliases =
      this->layer_ptr->getInputTensorAliases();
  std::vector<tf::tensor> aliases;
  // Reserve up front: tf::tensor's move constructor transfers ownership, so
  // vector reallocation would turn these non-owning aliases into owners.
  aliases.reserve(input_aliases.size());
  for (const tf::tensor *alias : input_aliases)
    if (alias)
      aliases.push_back(*alias);
  return aliases;
}

std::vector<tf::tensor> tf::layer::layer::get_output_tensors() const {
  if (!this->layer_ptr)
    return {};
  return ld::detail::copy_output_tensors(this->layer_ptr->getOutputTensors());
}

void tf::layer::layer::initialize_parameters() {
  if (this->layer_ptr)
    this->layer_ptr->initializeParameters();
}

void tf::layer::layer::backward(const tf::optimizer::optimizer &optimizer) {
  if (this->layer_ptr)
    this->layer_ptr->backward(optimizer);
}

void tf::layer::layer::reset_gradient_recording() {
  if (this->layer_ptr)
    this->layer_ptr->resetGradientRecording();
}

std::unordered_set<std::string> tf::layer::layer::parameterNames() const {
  if (!this->layer_ptr)
    return {};

  return this->layer_ptr->parameterNames();
}

std::vector<tf::tensor>
tf::layer::layer::getParameter(std::string_view name) const {
  if (!this->layer_ptr)
    return {};

  return ld::detail::copy_output_tensors(
      this->layer_ptr->getLayerParameter(name));
}

void tf::layer::layer::setParameter(std::string_view name,
                                    const LayerParameterValue &value) {
  if (!this->layer_ptr)
    return;

  // Dispatch on the layer's canonical type, not the registry key used to
  // construct the handle: aliases must still find their registered setter.
  LayerRegistry::instance().setParameter(std::string(this->layer_ptr->type()),
                                         this->layer_ptr, std::string(name),
                                         value);
}

std::string_view tf::layer::layer::type() const {
  return this->layer_ptr ? this->layer_ptr->type() : std::string_view{};
}

tf::layer::layer::operator bool() const {
  return static_cast<bool>(this->layer_ptr);
}

bool tf::layer::layer::valid() const { return static_cast<bool>(*this); }

tf::layer::layer_id tf::layer::layer::id() const {
  return this->layer_ptr ? this->layer_ptr->id() : invalid_layer_id;
}

// --- End of Generic layer handle ---

// --- Dense ---

tf::layer::dense::dense(unsigned unit)
    : layer(tf::layer::type::dense, ld::detail::config_with("units", unit)) {}

tf::layer::dense::~dense() = default;

std::vector<const tf::tensor *> tf::layer::dense::get_input_tensors() {
  return this->input_tensor_aliases();
}

std::vector<tf::tensor> tf::layer::dense::get_output_tensors() {
  return this->layer::get_output_tensors();
}

std::vector<tf::tensor> tf::layer::dense::get_output_training_tensor() {
  return this->getParameter(tf::layer::parameter::training_output);
}

std::vector<tf::tensor> tf::layer::dense::get_output_training_weight() {
  return this->getParameter(tf::layer::parameter::training_weight);
}

void tf::layer::dense::set_weight(const tf::tensor &weight_tensor) {
  if (!weight_tensor.getPtr()) {
    LOG(ERROR) << "Fatal! given tensor is not initialized with data\n";
    return;
  }

  this->setParameter(tf::layer::parameter::weight, weight_tensor);
}

void tf::layer::dense::set_bias(const tf::tensor &bias_tensor) {
  if (!bias_tensor.getPtr()) {
    LOG(ERROR) << "Fatal! given tensor is not initialized with data\n";
    return;
  }

  this->setParameter(tf::layer::parameter::bias, bias_tensor);
}

// --- End of Dense ---

// --- Relu ---

tf::layer::relu::relu() : layer(tf::layer::type::relu) {}

tf::layer::relu::~relu() = default;

std::vector<const tf::tensor *> tf::layer::relu::get_input_tensors() {
  return this->input_tensor_aliases();
}

std::vector<tf::tensor> tf::layer::relu::get_output_tensors() {
  return this->layer::get_output_tensors();
}

// --- End of Relu ---

// --- Softmax ---

tf::layer::softmax::softmax(unsigned axis)
    : layer(tf::layer::type::softmax, ld::detail::config_with("axis", axis)) {}

tf::layer::softmax::~softmax() = default;

std::vector<const tf::tensor *> tf::layer::softmax::get_input_tensors() {
  return this->input_tensor_aliases();
}

std::vector<tf::tensor> tf::layer::softmax::get_output_tensors() {
  return this->layer::get_output_tensors();
}

// --- End of Softmax ---
