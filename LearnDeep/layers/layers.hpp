#ifndef _TENSOR_LAYER_LAYERS_
#define _TENSOR_LAYER_LAYERS_

// C++ Headers
#include <stdfloat>
#include <string>
#include <vector>

// Library Headers
#include <core/framework/MathLibrary.h>

enum LayerType { tf_dense, tf_conv2d, tf_batchnormalization, tf_dropout };

enum class Layer_Parameter {
  dense_input,
  dense_weight,
  dense_bias,
  dense_matmul_result,
  dense_output,
  dense_training_input,
  dense_training_weight,
  dense_training_bias,
  dense_training_matmul_result,
  dense_training_output,
  dense_grad_weight,
  dense_grad_bias
};

enum class InitializationMethod {
  MANUAL = 0,
  ZEROS,
  ONES,
  RANDOM_UNIFORM,
  RANDOM_NORMAL,
  XAVIER_UNIFORM,
  XAVIER_NORMAL,
  HE_UNIFORM,
  HE_NORMAL,
  LECUN_UNIFORM,
  LECUN_NORMAL
};

enum class TargetTrainableParameter {
  dense_weight,
  dense_bias,
};

class Layer {

public:
  virtual std::vector<Tensor<std::float64_t> *>
  forward(std::vector<Tensor<std::float64_t> *> input, unsigned batch_size) = 0;

  virtual void backward() = 0;

  virtual std::vector<Tensor<std::float64_t> *>
  operator()(const std::vector<Tensor<std::float64_t> *> &input_tensors) = 0;

  virtual std::vector<Tensor<std::float64_t> *> getInputTensors() = 0;

  virtual std::vector<Tensor<std::float64_t> *> getOutputTensors() = 0;

  virtual std::vector<Tensor<std::float64_t> *> getInputTrainingTensors() = 0;

  virtual std::vector<Tensor<std::float64_t> *> getOutputTrainingTensors() = 0;

  virtual LayerType getLayerType() = 0;

  virtual std::vector<Tensor<std::float64_t> *>
  getLayerParameter(Layer_Parameter layer_parameter,
                    bool print_flag = false) = 0;
};

#endif // _TENSOR_CORE_LAYER_