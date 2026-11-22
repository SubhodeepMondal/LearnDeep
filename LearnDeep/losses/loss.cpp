#include "loss.hpp"

#include <atomic>

namespace {
std::atomic<tf::loss::loss_id> next_loss_id{1};
} // namespace

Loss::Loss() : id_(next_loss_id.fetch_add(1, std::memory_order_relaxed)) {}

void Loss::recordInputs(const std::vector<Tensor<std::float64_t> *> &inputs) {
  // tf::tensor copies are non-owning aliases; the objects themselves live in
  // this vector for the whole lifetime of the loss.
  this->input_aliases.clear();
  this->input_aliases.reserve(inputs.size());
  for (Tensor<std::float64_t> *input : inputs) {
    if (!input)
      continue;
    tf::tensor alias;
    alias.dt_type = tf_float64;
    alias.assign_ptr(input); // non-owning: no tensor_nodes registration
    this->input_aliases.push_back(alias);
  }
}

void Loss::recordTrainingInputs(const std::vector<tf::tensor *> &inputs) {
  this->training_input_aliases.clear();
  this->training_input_aliases.reserve(inputs.size());
  for (const tf::tensor *input : inputs)
    if (input && input->getPtr())
      this->training_input_aliases.push_back(*input);
}
