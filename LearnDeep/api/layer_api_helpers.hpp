#ifndef LD_LAYER_API_HELPERS_HPP
#define LD_LAYER_API_HELPERS_HPP

// Internal helpers shared by the public layer API implementation and the
// tensor API translation unit. They live in the api module because both
// consumers are API implementation files.

#include <api/tensor.h>
#include <layers/layer_registry.hpp>

#include <string>
#include <vector>

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

inline LayerConfig config_with(const std::string &key, unsigned value) {
  LayerConfig config;
  config.set(key, value);
  return config;
}

} // namespace ld::detail

#endif // LD_LAYER_API_HELPERS_HPP
