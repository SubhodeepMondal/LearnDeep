#ifndef _TENSOR_LAYER_LAYERS_
#define _TENSOR_LAYER_LAYERS_

// C++ Headers
#include <stdfloat>
#include <string>
#include <vector>

// Library Headers
#include "layer_enum.hpp"
#include <api/tensor.h>

class Layer {
protected:
public:
  std::vector<const Tensor<std::float64_t> *>
      layer_inputs; // not own by any layer
  std::vector<tf::tensor> layer_outputs;
  virtual std::vector<tf::tensor>
  forward(const std::vector<tf::tensor *> &input, unsigned batch_size) = 0;

  virtual void backward() = 0;

  virtual const std::vector<tf::tensor> &
  operator()(std::vector<tf::tensor> input_tensors) = 0;

  virtual std::vector<const Tensor<std::float64_t> *> getInputTensors() = 0;

  virtual std::vector<tf::tensor> getOutputTensors() = 0;

  virtual std::vector<const Tensor<std::float64_t> *>
  getInputTrainingTensors() = 0;

  virtual std::vector<tf::tensor> getOutputTrainingTensors() = 0;

  virtual LayerType getLayerType() = 0;

  virtual std::vector<tf::tensor>
  getLayerParameter(Layer_Parameter layer_parameter,
                    bool print_flag = false) = 0;
};

#endif // _TENSOR_CORE_LAYER_