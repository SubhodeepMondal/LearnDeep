#ifndef _TENSORFLOW_CORE_LOSS_
#define _TENSORFLOW_CORE_LOSS_

// C++ Headers
#include <unordered_map>
#include <vector>

// Library Headers
#include <api/tensor.h>

enum class LossParameter { output_predict, output_target, loss };

class Loss {
protected:
  std::vector<Tensor<std::float64_t> *> input_predicts; // not owning
  std::float64_t loss_value;

public:
  virtual void forward(std::vector<tf::tensor *> inputs,
                       const unsigned batch_size) = 0;

  virtual void
  setTargetOutput(const std::vector<tf::tensor> output_predicts) = 0;
  virtual void
  setPredictedOutput(const std::vector<tf::tensor *> &output_targets) = 0;
  virtual std::float64_t const getScalerLoss() = 0;

  virtual std::vector<tf::tensor *> getLossTensor() = 0;
};

class SquaredError : public Loss {

  std::vector<tf::tensor *> training_inputs;
  std::vector<tf::tensor *> target_outputs;
  tf::tensor *differences;
  tf::tensor *loss_tensor;
  tf::tensor *loss_gradient;

public:
  SquaredError() = default;

  SquaredError(std::vector<Tensor<std::float64_t> *> input_preds);

  ~SquaredError();

  void forward(std::vector<tf::tensor *> inputs,
               const unsigned batch_size) override;

  void setTargetOutput(const std::vector<tf::tensor> output_predicts) override;

  void
  setPredictedOutput(const std::vector<tf::tensor *> &output_targets) override;

  std::float64_t const getScalerLoss() override;

  std::vector<tf::tensor *> getLossTensor() override;
};

#endif // _TENSORFLOW_CORE_LOSS_