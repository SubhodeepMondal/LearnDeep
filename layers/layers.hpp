#ifndef _TENSOR_LAYER_LAYERS_
#define _TENSOR_LAYER_LAYERS_

// C++ Headers
#include <stdfloat>
#include <string>
#include <vector>

// Library Headers
#include <framework/MathLibrary.h>

enum LayerType {
  tf_dense,
  tf_conv2d,
  tf_batchnormalization,
  tf_dropout

};

class Layer {

public:
  virtual std::vector<Tensor<std::float64_t> *>
  forward(std::vector<Tensor<std::float64_t> *> input) = 0;

  virtual void backward() = 0;

  virtual std::vector<Tensor<std::float64_t> *>
  operator()(const std::vector<Tensor<std::float64_t> *> &input_tensors) = 0;

  virtual std::vector<Tensor<std::float64_t> *> getInputTensors() = 0;

  virtual std::vector<Tensor<std::float64_t> *> getOutputTensors() = 0;

  virtual std::vector<Tensor<std::float64_t> *> getInputTrainingTensors() = 0;

  virtual std::vector<Tensor<std::float64_t> *> getOutputTrainingTensors() = 0;
};

#endif // _TENSOR_CORE_LAYER_