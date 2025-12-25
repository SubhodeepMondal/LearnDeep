#ifndef _TENSORFLOW_CORE_DENSE_LAYER_
#define _TENSORFLOW_CORE_DENSE_LAYER_

// C++ Headers

// Library Headers
#include "layers.hpp"
#include <api/tensor.h>

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

  tf::tensor weight;
  tf::tensor bias;
  tf::tensor matmul_result;
  tf::tensor output;
  const tf::tensor *training_inputs; // not own by dense layer
  tf::tensor training_weight;
  tf::tensor training_bias;
  tf::tensor training_matmul_result;
  tf::tensor training_output;
  std::vector<tf::tensor *> training_outputs;
  tf::tensor grad_weight;
  tf::tensor grad_bias;

  tf::tensor initialization_weight; // should own by dense layer
  tf::tensor initialization_bias;   // should own by dense layer
  std::string layer_name;

  bool forwardGraphCreated;
  bool backwardGraphCrated;
  bool isActiveGraphSession;

  void createForwardComputeGraph();

  void initializeWeight();

  void initializeBias();

public:
  Dense(unsigned unit);

  ~Dense();

  const std::vector<tf::tensor *> &
  operator()(std::vector<tf::tensor> input_tensors) override;

  const std::vector<tf::tensor *> &
  forward(std::vector<const tf::tensor *> &input, unsigned batch_size) override;

  void backward() override;

  std::vector<const Tensor<std::float64_t> *> &getInputTensors() override;

  const std::vector<tf::tensor *> &getOutputTensors() override;

  std::vector<const tf::tensor *> getInputTrainingTensors();

  const std::vector<tf::tensor *> &getOutputTrainingTensors();

  LayerType getLayerType();

  void setWeightInitializationMethod(InitializationMethod initilization_method);

  void setBiasInitializationMethod(InitializationMethod initilization_method);

  void setWeight(const tf::tensor &weight_tensor);

  void setBias(const tf::tensor &bias_tensor);

  std::vector<tf::tensor *> getLayerParameter(Layer_Parameter layer_parameter,
                                              bool print_flag = false) override;
};
#endif // _TENSORFLOW_CORE_DENSE_LAYER