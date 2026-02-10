#ifndef _TENSORFLOW_LAYER_ENUM_
#define _TENSORFLOW_LAYER_ENUM_

enum LayerType { tf_dense, tf_conv2d, tf_batchnormalization, tf_dropout };

enum class Layer_Parameter {
  dense_input,
  dense_weight,
  dense_bias,
  dense_matmul_result,
  dense_output,
  dense_training_input,
  dense_training_weight,
  dense_training_bias,
  dense_training_matmul_result,
  dense_training_output,
  dense_grad_weight,
  dense_grad_bias
};

enum class InitializationMethod {
  MANUAL = 0,
  ZEROS,
  ONES,
  RANDOM_UNIFORM,
  RANDOM_NORMAL,
  XAVIER_UNIFORM,
  XAVIER_NORMAL,
  HE_UNIFORM,
  HE_NORMAL,
  LECUN_UNIFORM,
  LECUN_NORMAL,
  UPDATE_FROM_GRAD
};

enum class TargetTrainableParameter {
  dense_weight,
  dense_bias,
};
#endif // _TENSORFLOW_LAYER_ENUM_