#ifndef LD_LAYER_API_H
#define LD_LAYER_API_H

// Public layer API. tensor.h includes this header, so users normally only need
// to include <api/tensor.h>. Including this header directly also works because
// it pulls in tensor.h before declaring the layer types.

#include <api/tensor.h>

#include <string>
#include <unordered_set>
#include <vector>

namespace tf {
namespace layer {

using layer_id = std::uint64_t;
inline constexpr layer_id invalid_layer_id = 0;

namespace type {
inline constexpr std::string_view dense{"dense"};
inline constexpr std::string_view relu{"relu"};
inline constexpr std::string_view softmax{"softmax"};
} // namespace type

namespace parameter {
inline constexpr std::string_view input{"input"};
inline constexpr std::string_view weight{"weight"};
inline constexpr std::string_view bias{"bias"};
inline constexpr std::string_view matmul_result{"matmul_result"};
inline constexpr std::string_view output{"output"};
inline constexpr std::string_view training_input{"training_input"};
inline constexpr std::string_view training_weight{"training_weight"};
inline constexpr std::string_view training_bias{"training_bias"};
inline constexpr std::string_view training_matmul_result{
    "training_matmul_result"};
inline constexpr std::string_view training_output{"training_output"};
inline constexpr std::string_view grad_input{"grad_input"};
inline constexpr std::string_view grad_weight{"grad_weight"};
inline constexpr std::string_view grad_bias{"grad_bias"};
inline constexpr std::string_view incoming_grad{"incoming_grad"};
inline constexpr std::string_view updated_weight{"updated_weight"};
inline constexpr std::string_view updated_bias{"updated_bias"};
} // namespace parameter

class layer {
private:
  std::shared_ptr<Layer> layer_ptr;

  explicit layer(std::shared_ptr<Layer> layer_ptr);

protected:
  // Pointers into the owning layer's alias storage. The handle itself keeps no
  // input state, so these cannot dangle and cannot go stale during training.
  std::vector<const tf::tensor *> input_tensor_aliases() const;

public:
  layer();
  layer(std::string_view type_name);
  layer(std::string_view type_name, const LayerConfig &config);
  virtual ~layer();

  // Resolve a live layer from the public id space. Invalid ids and dead layers
  // produce an inert handle.
  static layer from_id(layer_id id);

  // Graph topology queries exposed through the public API so that consumers do
  // not need the internal LayerGraph ABI.
  static layer_id producer_of(const Tensor<std::float64_t> *tensor);
  static std::vector<layer_id>
  consumers_of(const Tensor<std::float64_t> *tensor);

  std::vector<tf::tensor> operator()(const std::vector<tf::tensor> &inputs);

  std::vector<tf::tensor> forward(const std::vector<tf::tensor> &inputs,
                                  unsigned batch_size);

  std::vector<tf::tensor> calculate(std::vector<tf::tensor> inputs);

  // Non-owning aliases whose getPtr() values identify the layer's real graph
  // tensors. Used by Model to build its execution graph.
  std::vector<tf::tensor> get_input_tensors() const;
  std::vector<tf::tensor> get_output_tensors() const;

  void initialize_parameters();

  void backward(const tf::optimizer::optimizer &optimizer);

  void reset_gradient_recording();

  std::unordered_set<std::string> parameterNames() const;

  std::vector<tf::tensor> getParameter(std::string_view name) const;

  void setParameter(std::string_view name, const LayerParameterValue &value);

  std::string_view type() const;

  // True when registry construction succeeded. Invalid handles are safe no-ops.
  explicit operator bool() const;

  bool valid() const;

  layer_id id() const;
};

struct dense : public layer {
  dense(unsigned unit);

  ~dense() override;

  std::vector<const tf::tensor *> get_input_tensors();

  std::vector<tf::tensor> get_output_tensors();

  std::vector<tf::tensor> get_output_training_tensor();

  std::vector<tf::tensor> get_output_training_weight();

  void set_weight(const tf::tensor &weight_tensor);

  void set_bias(const tf::tensor &bias_tensor);
};

struct relu : public layer {
  relu();

  ~relu() override;

  std::vector<const tf::tensor *> get_input_tensors();

  std::vector<tf::tensor> get_output_tensors();
};

struct softmax : public layer {
  softmax(unsigned axis = 0);

  ~softmax() override;

  std::vector<const tf::tensor *> get_input_tensors();

  std::vector<tf::tensor> get_output_tensors();
};

} // namespace layer
} // namespace tf

namespace ld::detail {

// tf::tensor's copy constructor creates a non-owning alias to the same
// underlying Tensor, so pointer identity (and therefore graph edges) is kept.
inline std::vector<tf::tensor>
copy_output_tensors(const std::vector<tf::tensor *> &tensors) {
  std::vector<tf::tensor> copies;
  copies.reserve(tensors.size());
  for (const tf::tensor *tensor : tensors)
    if (tensor)
      copies.push_back(*tensor);
  return copies;
}

} // namespace ld::detail

#endif // LD_LAYER_API_H
