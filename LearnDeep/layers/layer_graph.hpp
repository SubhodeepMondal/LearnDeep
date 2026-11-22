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

class LayerGraph {
  std::unordered_map<tf::layer::layer_id, std::weak_ptr<Layer>> layers;

public:
  LayerGraph() = default;
  ~LayerGraph() = default;

  void addNode(const std::shared_ptr<Layer> &node);
  void removeNode(tf::layer::layer_id id);
  void prune();

  std::shared_ptr<Layer> lock(tf::layer::layer_id id);
  bool contains(tf::layer::layer_id id);

  std::vector<tf::layer::layer_id>
  getLayersOfIncomingTensor(const Tensor<std::float64_t> *tensor);

  tf::layer::layer_id
  getLayerOfOutgoingTensor(const Tensor<std::float64_t> *tensor);
};

extern LayerGraph global_layer_graph;

#endif // _CORE_LAYER_GRAPH_
