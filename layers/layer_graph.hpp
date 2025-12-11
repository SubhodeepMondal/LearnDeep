#ifndef _CORE_LAYER_GRAPH_
#define _CORE_LAYER_GRAPH_

// C++ Headers
#include <stdfloat>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Library Headers
#include <framework/MathLibrary.h>
#include <layers/layers.hpp>

enum LayerGraphFunctions { compute };

typedef struct LayerNode {
  Layer *layer;
  std::vector<LayerNode *> input_nodes;
  std::vector<LayerNode *> output_nodes;

  LayerNode(Layer *layer);
  void addIncomingNode(LayerNode *incoming_node);
  void addOutgoingNode(LayerNode *outgoing_node);
} LayerNode;

class LayerGraph {
  std::unordered_map<Layer *, LayerNode *> layer_graph;
  std::unordered_set<Layer *> layers;

  LayerNode *root_layer_node;

public:
  LayerGraph();

  std::vector<Tensor<std::float64_t> *>
  forward(std::unordered_map<Layer *, std::vector<Tensor<std::float64_t> *>>
              layer_input_map);

  void addNode(Layer *node);

  std::vector<Layer *>
  getLayersOfIncomingTensor(Tensor<std::float64_t> *tensor);

  Layer *getLayerOfOutgoingTensor(Tensor<std::float64_t> *tensor);
};

extern LayerGraph global_layer_graph;

#endif // _CORE_LAYER_GRAPH_