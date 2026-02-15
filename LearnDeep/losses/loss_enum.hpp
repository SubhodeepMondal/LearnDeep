#ifndef _TENSORFLOW_LOSS_ENUM_
#define _TENSORFLOW_LOSS_ENUM_

enum class Loss_Parameter {
  squared_error_predicted_output,
  squared_error_target_output,
  squared_error_grad_predicted_output
};

enum class LossType {
  squared_error,
  mean_squared_error,
  absolute_error,
  mena_absolute_error
};
#endif // _TENSORFLOW_LOSS_ENUM_