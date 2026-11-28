#ifndef _TENSORFLOW_LAYER_MANAGER_
#define _TENSORFLOW_LAYER_MANAGER_

// C++ Headers
#include <memory>
#include <string>

// Library Headers
#include <layers/layer_registry.hpp>

// Orchestrates layer creation: the registry builds an unregistered instance
// and the manager registers the resulting live instance with the graph.
class LayerManager {
public:
  using LayerHandle = LayerRegistry::LayerHandle;

  static LayerManager &instance();

  LayerHandle create(const std::string &name,
                     const LayerConfig &config = {}) const;

private:
  LayerManager() = default;
};

#endif // _TENSORFLOW_LAYER_MANAGER_
