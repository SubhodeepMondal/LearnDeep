#ifndef _TENSOR_CORE_LAYER_
#define _TENSOR_CORE_LAYER_
typedef enum LayerType {
  tf_dense,
  tf_conv2d,
  tf_batchnormalization,
  tf_dropout

} LayerType;

class Layer {
protected:
  std::string layer_name;
  LayerType layer_type;
  std::vector<Tensor *> layer_input;
  std::vector<Tensor *> layer_output;

public:
  std::compute() = 0;
  std::computeGradient() = 0;
}

#endif // _TENSOR_CORE_LAYER_