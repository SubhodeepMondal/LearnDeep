// Library Headers
#include "dense.hpp"
#include "layer_graph.hpp"
#include <graph/graph_framework.hpp>
#include <graph/graph_manager.hpp>

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
Dense::forward(std::vector<Tensor<std::float64_t> *> input) {
  if (input.size() == 1) {
    if (input[0]->getNoOfDimensions() == 2) {
      this->no_of_features = input[0]->getDimensions()[0];
      this->batch_size = input[0]->getDimensions()[1];

      this->training_input = input[0];

      unsigned arr[2];
      arr[0] = this->no_of_unit;
      arr[1] = this->no_of_features;
      this->weight->reshape(2, arr);
      this->training_weight = new Tensor<std::float64_t>(2, arr, tf_float64);

      arr[1] = this->batch_size;
      this->training_bias = new Tensor<std::float64_t>(2, arr, tf_float64);

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