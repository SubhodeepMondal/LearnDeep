#ifndef _TENSOR_LAYER_LAYERS_
#define _TENSOR_LAYER_LAYERS_

// C++ Headers
#include <stdfloat>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

// Library Headers
#include <api/tensor.h>

class Layer {
private:
  tf::layer::layer_id id_;

protected:
  Layer();

  // Owned copies (non-owning aliases of the underlying Tensor) of the inputs
  // most recently seen by the API handle and by the layer itself. The API
  // handle deliberately has no input state of its own: pointers handed out to
  // callers and pointers stored by a layer across its forward() call must refer
  // to these vectors so they cannot dangle.
  std::vector<tf::tensor> input_tensor_aliases;
  std::vector<tf::tensor> training_input_aliases;

  void recordTrainingInputs(const std::vector<const tf::tensor *> &inputs);

public:
  virtual ~Layer() = default;

  tf::layer::layer_id id() const noexcept { return id_; }

  // Records non-owning aliases of the current inputs. Called by the API handle
  // and by internal operator() implementations.
  void recordInputTensors(const std::vector<tf::tensor> &inputs);
  std::vector<const tf::tensor *> getInputTensorAliases() const;
  std::vector<const tf::tensor *> getTrainingInputAliases() const;

  // Raw input pointers for graph construction. Derived from the owned alias
  // vector, so no separate unowned cache is needed.
  std::vector<const Tensor<std::float64_t> *> getInputTensors() const;

  std::vector<tf::tensor *> layer_outputs;
  virtual const std::vector<tf::tensor *> &
  forward(std::vector<const tf::tensor *> &input, unsigned batch_size) = 0;

  virtual std::vector<tf::tensor> calculate(std::vector<tf::tensor> inputs) = 0;

  virtual void backward(const tf::optimizer::optimizer &optimizer) = 0;

  // Clears one-shot gradient-recording state before a new training graph.
  virtual void resetGradientRecording() {}

  virtual const std::vector<tf::tensor *> &
  operator()(std::vector<tf::tensor> input_tensors) = 0;

  virtual const std::vector<tf::tensor *> &getOutputTensors() = 0;

  virtual std::vector<const tf::tensor *> getInputTrainingTensors() = 0;

  virtual const std::vector<tf::tensor *> &getOutputTrainingTensors() = 0;

  virtual std::string_view type() const = 0;

  virtual std::vector<tf::tensor *>
  getLayerParameter(std::string_view layer_parameter,
                    bool print_flag = false) = 0;

  virtual std::unordered_set<std::string> parameterNames() const { return {}; }

  virtual void initializeParameters() = 0;
};

#endif // _TENSOR_CORE_LAYER_