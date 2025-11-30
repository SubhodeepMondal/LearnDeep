#ifndef _TENSORFLOW_CORE_DENSE_LAYER_
#define _TENSORFLOW_CORE_DENSE_LAYER_

#include <core/Layers/Layers.hh>
#include <framework/MathLibrary.hh>
#include <graph/graph_framework.hpp>

class Dense : protected Layers {
  unsigned no_of_neuron, no_of_input, batch_size;
  Tensor<std::float64_t> *input;
  Tensor<std::float64_t> *weight;
  Tensor<std::float64_t> *bias;
  std::vector<Tensor<std::float64_t> *> inputs;
  std::vector<Tensor<std::float64_t> *> output;

  void initializeInputs();
  void initializeOutput();
  void compute();
  void computeGrad();

public:
  void Dense() {};
  std::vector<Tensor<std::float64_t> *> Dense(unsigned no_of_neuron,
                                              unsigned no_of_input);
  std::vector<Tensor<std::float64_t> *>
  Dense(std::vector<Tensor<std::float64_t> *> inputs);
};

#end // _TENSORFLOW_CORE_DENSE_LAYER_

// Layer * Dense_1 = Dense{}