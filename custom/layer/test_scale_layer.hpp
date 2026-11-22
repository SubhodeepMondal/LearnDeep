#ifndef CUSTOM_LAYER_TEST_SCALE_LAYER_HPP
#define CUSTOM_LAYER_TEST_SCALE_LAYER_HPP

// Library Headers
#include <layers/layers.hpp>

// A layer defined outside the framework source tree. It is built as its own
// shared library by custom/CMakeLists.txt and self-registers via REGISTER_LAYER.
//
// The layer computes y = scale * x. `scale` is a fixed (non-trainable) scalar,
// but the forward pass still builds a graph node so gradients flow back to the
// input; this lets the custom layer take part in a full Model::fit pipeline.
class TestScaleLayer : public Layer {
  double scale_{1.0};
  tf::tensor scale_tensor_;

  tf::tensor output_;
  tf::tensor training_output_;
  tf::tensor grad_input_;
  std::vector<tf::tensor *> training_outputs_;

public:
  explicit TestScaleLayer(double scale = 1.0);
  ~TestScaleLayer() override = default;

  void setScale(double scale);
  double getScale() const noexcept { return scale_; }

  const std::vector<tf::tensor *> &
  forward(std::vector<const tf::tensor *> &input, unsigned batch_size) override;

  std::vector<tf::tensor> calculate(std::vector<tf::tensor> inputs) override;

  void backward(Optimizer *optimizer) override;

  const std::vector<tf::tensor *> &
  operator()(std::vector<tf::tensor> input_tensors) override;

  const std::vector<tf::tensor *> &getOutputTensors() override;

  std::vector<const tf::tensor *> getInputTrainingTensors() override;

  const std::vector<tf::tensor *> &getOutputTrainingTensors() override;

  std::string_view type() const override;

  std::vector<tf::tensor *>
  getLayerParameter(std::string_view layer_parameter,
                    bool print_flag = false) override;

  std::unordered_set<std::string> parameterNames() const override;

  void initializeParameters() override;
};

#endif // CUSTOM_LAYER_TEST_SCALE_LAYER_HPP
