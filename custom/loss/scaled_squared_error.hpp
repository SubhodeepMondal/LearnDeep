#ifndef LD_CUSTOM_SCALED_SQUARED_ERROR_HPP
#define LD_CUSTOM_SCALED_SQUARED_ERROR_HPP

// Public-API-only custom loss. This plugin includes no framework-internal
// headers; REGISTER_LOSS adapts the behavior contract to the internal Loss ABI.
#include <api/tensor.h>

#include <limits>
#include <string_view>
#include <unordered_set>
#include <vector>

inline constexpr std::string_view kCustomScaledMseType{"custom_scaled_mse"};
inline constexpr std::string_view kCustomScaleParameter{"scale"};
inline constexpr std::string_view kCustomScaledMseLossParameter{
    "custom_scaled_mse_loss"};
inline constexpr std::string_view kCustomScaledMseGradParameter{
    "custom_scaled_mse_grad"};

// Mean squared error multiplied by a configurable scale. Scaling the terminal
// graph tensor scales the gradient recorded by reverse-mode autodiff as well.
class ScaledSquaredError final : public tf::loss::behavior {
public:
  explicit ScaledSquaredError(const tf::loss::config &config);

  std::string_view type() const override { return kCustomScaledMseType; }

  void forward(std::vector<tf::tensor *> inputs,
               unsigned batch_size) override;

  void backward() override;

  void setTargetOutput(const std::vector<tf::tensor> &targets) override;

  std::float64_t calculate(std::vector<tf::tensor> &predicted_output,
                           std::vector<tf::tensor> &target_output) override;

  std::float64_t getScalerLoss() override;

  std::float64_t getEpochEndScalerLoss() override {
    return this->prev_loss_value;
  }

  std::vector<tf::tensor *> getParameter(std::string_view name) override;

  std::unordered_set<std::string> parameterNames() const override;

  unsigned getNoOfInputs() const override { return 1; }

  void setScale(std::float64_t scale);
  std::float64_t scale() const { return this->scale_; }

private:
  std::vector<tf::tensor> training_input_aliases;
  std::vector<tf::tensor> target_tensors;

  tf::tensor grad_training_input;
  tf::tensor differences;
  tf::tensor loss_tensor_batch;
  tf::tensor loss_tensor;
  tf::tensor scaled_loss_tensor;
  tf::tensor scale_tensor;
  std::vector<tf::tensor> loss_reduction_intermediates;

  std::float64_t scale_{2.0};
  std::float64_t prev_loss_value{std::numeric_limits<std::float64_t>::max()};
  bool is_grad_recorded{false};
};

#endif // LD_CUSTOM_SCALED_SQUARED_ERROR_HPP
