#ifndef _CORE_LAYER_GRAPH_
#define _CORE_LAYER_GRAPH_

// C++ Headers
#include <stdfloat>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Library Headers
#include <core/framework/MathLibrary.h>
#include <layers/layers.hpp>

enum LayerGraphFunctions { compute };

class LayerGraph {
  std::unordered_set<Layer *> layers;

public:
  LayerGraph();

  void addNode(Layer *node);
  void removeNode(Layer *node);

  std::vector<Layer *>
  getLayersOfIncomingTensor(const Tensor<std::float64_t> *tensor);

  Layer *getLayerOfOutgoingTensor(const Tensor<std::float64_t> *tensor);

  std::vector<Layer *> getAllLayers();
};

extern LayerGraph global_layer_graph;

#endif // _CORE_LAYER_GRAPH_