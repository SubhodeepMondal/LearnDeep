// Library Headers
#include "layer_manager.hpp"
#include "layer_graph.hpp"
#include "layer_registry.hpp"
#include "layers.hpp"

LayerManager &LayerManager::instance() {
  static LayerManager manager;
  return manager;
}

LayerManager::LayerHandle
LayerManager::create(const std::string &name, const LayerConfig &config) const {
  // The registry only knows how to construct an instance. Registration is this
  // manager's responsibility, so a raw factory call leaves no graph node.
  std::unique_ptr<Layer> instance =
      LayerRegistry::instance().createInstance(name, config);
  if (!instance)
    return nullptr;

  LayerHandle layer(instance.release(), [](Layer *owned_layer) {
    if (!owned_layer)
      return;

    global_layer_graph_manager.unregisterLayer(owned_layer->id());
    delete owned_layer;
  });
  global_layer_graph_manager.registerLayer(layer);
  return layer;
}
