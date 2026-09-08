#ifndef _TENSORFLOW_LOSS_ENUM_
#define _TENSORFLOW_LOSS_ENUM_

enum class Loss_Parameter {
  squared_error_predicted_output,
  squared_error_target_output,
  squared_error_grad_predicted_output,
  categorical_cross_entropy_error,
  mean_categorical_cross_entropy_error,
  mean_absolute_categorical_cross_entropy_error,
  categorical_cross_entropy_target_output
};

enum class LossType {
  squared_error,
  mean_squared_error,
  absolute_error,
  mena_absolute_error,
  categorical_cross_entropy
};
#endif // _TENSORFLOW_LOSS_ENUM_