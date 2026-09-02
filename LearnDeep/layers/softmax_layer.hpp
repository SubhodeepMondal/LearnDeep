#ifndef _TENSORFLOW_CORE_SOFTMAX_LAYER_
#define _TENSORFLOW_CORE_SOFTMAX_LAYER_

// Library Headers
#include "layers.hpp"
#include <api/tensor.h>
#include <vector>

class Softmax : public Layer {
  bool isForwardGraphCreated;
  bool isBackwardGraphCreated;
  bool isGradRecorded;

  LayerType layer_type;
  unsigned axis;

  const tf::tensor *training_inputs;
  tf::tensor output;
  tf::tensor training_output;
  std::vector<tf::tensor *> training_outputs;

  unsigned batch_size;

  bool forwardGraphCreated;
  bool backwardGraphCreated;
  bool isActiveGraphSession;

  void createForwardComputeGraph();
  void initializeWeight();
  void initializeBias();

public:
  Softmax(unsigned axis = 0);

  ~Softmax();

  const std::vector<tf::tensor *> &
  operator()(std::vector<tf::tensor> input_tensors) override;

  const std::vector<tf::tensor *> &
  forward(std::vector<const tf::tensor *> &input, unsigned batch_size) override;

  void backward(Optimizer *optimizer) override;

  std::vector<const Tensor<std::float64_t> *> &getInputTensors() override;

  const std::vector<tf::tensor *> &getOutputTensors() override;

  std::vector<const tf::tensor *> getInputTrainingTensors() override;

  const std::vector<tf::tensor *> &getOutputTrainingTensors() override;

  LayerType getLayerType();

  std::vector<tf::tensor *> getLayerParameter(Layer_Parameter layer_parameter,
                                              bool print_flag = false) override;

  void initializeParameters() override;
};

#endif