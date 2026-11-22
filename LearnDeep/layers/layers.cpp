// C++ Headers
#include <atomic>

// Library Headers
#include "layers.hpp"

namespace {
std::atomic<tf::layer::layer_id> next_layer_id{1};
} // namespace

Layer::Layer()
    : id_(next_layer_id.fetch_add(1, std::memory_order_relaxed)) {}

void Layer::recordInputTensors(const std::vector<tf::tensor> &inputs) {
  // Copy construction of tf::tensor creates a non-owning alias. clear() plus
  // push_back() keeps the alias objects stable for the lifetime of this layer.
  this->input_tensor_aliases.clear();
  this->input_tensor_aliases.reserve(inputs.size());
  for (const tf::tensor &input : inputs)
    this->input_tensor_aliases.push_back(input);
}

void Layer::recordTrainingInputs(const std::vector<const tf::tensor *> &inputs) {
  this->training_input_aliases.clear();
  this->training_input_aliases.reserve(inputs.size());
  for (const tf::tensor *input : inputs)
    if (input)
      this->training_input_aliases.push_back(*input);
}

std::vector<const tf::tensor *> Layer::getInputTensorAliases() const {
  std::vector<const tf::tensor *> aliases;
  aliases.reserve(this->input_tensor_aliases.size());
  for (const tf::tensor &alias : this->input_tensor_aliases)
    aliases.push_back(&alias);
  return aliases;
}

std::vector<const tf::tensor *> Layer::getTrainingInputAliases() const {
  std::vector<const tf::tensor *> aliases;
  aliases.reserve(this->training_input_aliases.size());
  for (const tf::tensor &alias : this->training_input_aliases)
    aliases.push_back(&alias);
  return aliases;
}

std::vector<const Tensor<std::float64_t> *> Layer::getInputTensors() const {
  std::vector<const Tensor<std::float64_t> *> inputs;
  inputs.reserve(this->input_tensor_aliases.size());
  for (const tf::tensor &alias : this->input_tensor_aliases)
    inputs.push_back(alias.getPtr());
  return inputs;
}
