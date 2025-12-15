#ifndef _TENSORFLOW_CORE_DENSE_LAYER_
#define _TENSORFLOW_CORE_DENSE_LAYER_

// Library Headers
#include "layers.hpp"
#include <core/kernel/opskernel.h>

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

  InitializationMethod weight_initialization_method;
  InitializationMethod bias_initialization_method;

  std::vector<Tensor<std::float64_t> *> layer_inputs; // parameter no 1
  Tensor<std::float64_t> *weight;                     // parameter no 2
  Tensor<std::float64_t> *bias;                       // parameter no 3
  Tensor<std::float64_t> *matmul_result;
  std::vector<Tensor<std::float64_t> *> layer_outputs; // parameter no 4
  Tensor<std::float64_t> *training_input;              // parameter no 5
  Tensor<std::float64_t> *training_weight;             // parameter no 6
  Tensor<std::float64_t> *training_bias;               // parameter no 7
  Tensor<std::float64_t> *training_matmul_result;      // parameter no 8
  Tensor<std::float64_t> *training_output;             // parameter no 9
  Tensor<std::float64_t> *grad_weight;                 // parameter no 10
  Tensor<std::float64_t> *grad_bias;                   // parameter no 11

  Tensor<std::float64_t> initialization_weight;
  Tensor<std::float64_t> initialization_bias;
  std::string layer_name;

  std::vector<Ops *> ops;

  bool forwardGraphCreated;
  bool backwardGraphCrated;
  bool isActiveGraphSession;

  void createForwardComputeGraph();

  void initializeWeight();

  void initializeBias();

public:
  Dense(unsigned unit);

  ~Dense();

  std::vector<Tensor<std::float64_t> *> operator()(
      const std::vector<Tensor<std::float64_t> *> &input_tensors) override;

  std::vector<Tensor<std::float64_t> *>
  forward(std::vector<Tensor<std::float64_t> *> input,
          unsigned batch_size) override;

  void backward() override;

  std::vector<Tensor<std::float64_t> *> getInputTensors() override;

  std::vector<Tensor<std::float64_t> *> getOutputTensors() override;

  std::vector<Tensor<std::float64_t> *> getInputTrainingTensors();

  std::vector<Tensor<std::float64_t> *> getOutputTrainingTensors();

  LayerType getLayerType();

  void setWeightInitializationMethod(InitializationMethod initilization_method);

  void setBiasInitializationMethod(InitializationMethod initilization_method);

  void setWeight(Tensor<std::float64_t> *weight_tensor);

  void setBias(Tensor<std::float64_t> *bias_tensor);

  std::vector<Tensor<std::float64_t> *>
  getLayerParameter(Layer_Parameter layer_parameter,
                    bool print_flag = false) override;
};
#endif // _TENSORFLOW_CORE_DENSE_LAYER