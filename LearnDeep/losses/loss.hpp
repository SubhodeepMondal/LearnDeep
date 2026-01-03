#ifndef _TENSORFLOW_CORE_LOSS_
#define _TENSORFLOW_CORE_LOSS_

// C++ Headers
#include <vector>

// Library Headers
#include <api/tensor.h>

enum class LossType {
  squared_error,
  mean_squared_error,
  absolute_error,
  mena_absolute_error
};

enum class LossParameter { output_predict, output_target, loss };

class Loss {
protected:
  std::vector<tf::tensor *> output_predicts; // not owning
  std::vector<tf::tensor *> output_targets;  // not owning
  std::vector<tf::tensor *> losses;          // owning

public:
  virtual void forward() = 0;

  virtual void
  setTargetOutput(const std::vector<tf::tensor *> &output_predicts) = 0;
  virtual void
  setPredictedOutput(const std::vector<tf::tensor *> &output_targets) = 0;
  virtual const std::vector<tf::tensor *> &getLoss() = 0;
  virtual std::vector<tf::tensor *>
  getLossParameter(LossParameter loss_parameter) = 0;
};

class SquaredError : protected Loss {

  std::vector<tf::tensor *> differences;

public:
  SquaredError() = default;

  ~SquaredError();

  void forward() override;

  void
  setTargetOutput(const std::vector<tf::tensor *> &output_predicts) override;
  void
  setPredictedOutput(const std::vector<tf::tensor *> &output_targets) override;
  const std::vector<tf::tensor *> &getLoss() override;
  std::vector<tf::tensor *>
  getLossParameter(LossParameter loss_parameter) override;
};

#endif // _TENSORFLOW_CORE_LOSS_