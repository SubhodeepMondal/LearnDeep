#ifndef LD_CUSTOM_SCALED_SQUARED_ERROR_HPP
#define LD_CUSTOM_SCALED_SQUARED_ERROR_HPP

// Internal extension headers: like custom layers, out-of-tree losses are
// compiled against the framework's C++ ABI. The public tests do NOT include
// this header; they create the loss through the registry key string.
#include <losses/loss.hpp>

#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

inline constexpr std::string_view kCustomScaledMseType{"custom_scaled_mse"};
inline constexpr std::string_view kCustomScaleParameter{"scale"};
inline constexpr std::string_view kCustomScaledMseLossParameter{
    "custom_scaled_mse_loss"};
inline constexpr std::string_view kCustomScaledMseGradParameter{
    "custom_scaled_mse_grad"};

// Example out-of-tree loss: the usual mean squared error multiplied by a
// configurable positive scale. The graph forward pass scales the terminal loss
// tensor so reverse-mode autodiff scales the gradient as well.
class ScaledSquaredError : public Loss {
  std::vector<tf::tensor *> training_inputs;
  std::vector<tf::tensor *> target_outputs;
  tf::tensor grad_training_inputs;
  tf::tensor *differences{nullptr};
  tf::tensor *loss_tensor_batch{nullptr};
  tf::tensor *loss_tensor{nullptr};
  std::vector<tf::tensor> loss_reduction_intermediates;

  std::float64_t scale_{2.0};
  tf::tensor scale_tensor_;
  tf::tensor scaled_loss_tensor_;

public:
  ScaledSquaredError(std::vector<Tensor<std::float64_t> *> input_preds,
                     std::string loss_name = "custom_scaled_mse",
                     std::float64_t scale = 2.0);

  ~ScaledSquaredError() override;

  std::string_view type() const override { return kCustomScaledMseType; }

  void forward(std::vector<tf::tensor *> inputs, unsigned batch_size) override;

  void backward() override;

  void setTargetOutput(const std::vector<tf::tensor> output_predicts) override;

  void setPredictedOutput(
      const std::vector<tf::tensor *> &output_targets) override;

  std::float64_t const getScalerLoss() override;

  std::float64_t calculate(std::vector<tf::tensor> &predicted_output,
                           std::vector<tf::tensor> &target_output) override;

  std::vector<tf::tensor *>
  getLossParameter(std::string_view loss_parameter) override;

  std::unordered_set<std::string> parameterNames() const override;

  unsigned getNoOfInputs() override { return 1; }

  std::string getLossName() override { return this->loss_name; }

  void setScale(std::float64_t scale);
  std::float64_t scale() const { return scale_; }
};

#endif // LD_CUSTOM_SCALED_SQUARED_ERROR_HPP
