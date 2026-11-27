#ifndef _CORE_LAYER_GRAPH_
#define _CORE_LAYER_GRAPH_

// C++ Headers
#include <memory>
#include <stdfloat>
#include <unordered_map>
#include <vector>

// Library Headers
#include <api/tensor.h>

class Layer;

// Tracks live layer instances and their tensor relationships. This class owns
// instance management only; type registration and object construction live in
// LayerRegistry and LayerManager respectively.
class LayerGraphManager {
  std::unordered_map<tf::layer::layer_id, std::weak_ptr<Layer>> layers;

public:
  LayerGraphManager() = default;
  ~LayerGraphManager() = default;

  void registerLayer(const std::shared_ptr<Layer> &layer);
  void unregisterLayer(tf::layer::layer_id id);
  void prune();

  std::shared_ptr<Layer> lock(tf::layer::layer_id id);
  bool contains(tf::layer::layer_id id);

  std::vector<tf::layer::layer_id>
  consumersOf(const Tensor<std::float64_t> *tensor);

  tf::layer::layer_id producerOf(const Tensor<std::float64_t> *tensor);

  // Compatibility names for callers that still use the old LayerGraph API.
  void addNode(const std::shared_ptr<Layer> &layer) { registerLayer(layer); }
  void removeNode(tf::layer::layer_id id) { unregisterLayer(id); }

  std::vector<tf::layer::layer_id>
  getLayersOfIncomingTensor(const Tensor<std::float64_t> *tensor) {
    return consumersOf(tensor);
  }

  tf::layer::layer_id
  getLayerOfOutgoingTensor(const Tensor<std::float64_t> *tensor) {
    return producerOf(tensor);
  }
};

using LayerGraph = LayerGraphManager;

extern LayerGraphManager global_layer_graph_manager;
extern LayerGraphManager &global_layer_graph;

#endif // _CORE_LAYER_GRAPH_
