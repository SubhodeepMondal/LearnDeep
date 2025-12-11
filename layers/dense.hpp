#ifndef _TENSORFLOW_CORE_DENSE_LAYER_
#define _TENSORFLOW_CORE_DENSE_LAYER_

// Library Headers
#include "layers.hpp"
#include <kernel/opskernel.h>

/** Tensor dimensions eager
 * input (features , 1)
 * weights (unit, features)
 * temp (unit, 1)
 * bias (unit, 1)
 * output (unit, 1)
 */

/** calculation
 * input.matmul(weights) + bias
 */

/** Tensor dimensions training
 * batch_size = n
 * input (features , batch_size)
 * weights (unit, features)
 * temp (unit, batch_size)
 * bias (unit, batch_size)
 * output (unit, batch_size)
 */

/** calculation
 * input.matmul(weights) + bias
 */
class Dense : public Layer {
  bool isForwardGraphCreated;
  bool isBackwardGraphCreated;

  LayerType layer_type;

  unsigned batch_size;
  unsigned no_of_unit;
  unsigned no_of_features;

  std::vector<Tensor<std::float64_t> *> layer_inputs;
  Tensor<std::float64_t> *weight;
  Tensor<std::float64_t> *bias;
  Tensor<std::float64_t> *matmul_result;
  std::vector<Tensor<std::float64_t> *> layer_outputs;

  Tensor<std::float64_t> *training_input;
  Tensor<std::float64_t> *training_weight;
  Tensor<std::float64_t> *training_bias;
  Tensor<std::float64_t> *training_matmul_result;
  Tensor<std::float64_t> *training_output;

  Tensor<std::float64_t> *grad_weight;
  Tensor<std::float64_t> *grad_bias;

  std::string layer_name;

  std::vector<Ops *> ops;

  bool forwardGraphCreated;
  bool backwardGraphCrated;
  bool isActiveGraphSession;

  void createForwardComputeGraph();

public:
  Dense(unsigned unit);

  ~Dense();

  std::vector<Tensor<std::float64_t> *> operator()(
      const std::vector<Tensor<std::float64_t> *> &input_tensors) override;

  std::vector<Tensor<std::float64_t> *>
  forward(std::vector<Tensor<std::float64_t> *> input) override;

  void backward() override;

  std::vector<Tensor<std::float64_t> *> getInputTensors() override;

  std::vector<Tensor<std::float64_t> *> getOutputTensors() override;

  virtual std::vector<Tensor<std::float64_t> *> getInputTrainingTensors();

  virtual std::vector<Tensor<std::float64_t> *> getOutputTrainingTensors();
};

#endif // _TENSORFLOW_CORE_DENSE_LAYER