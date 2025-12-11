// C++ Headers
#include <algorithm>
#include <cstddef>
#include <unordered_set>

// Library Headers
#include "layer_graph.hpp"
#include "layers.hpp"

LayerGraph global_layer_graph;

LayerNode::LayerNode(Layer *layer) { this->layer = layer; }

void LayerNode::addIncomingNode(LayerNode *incoming_node) {
  input_nodes.push_back(incoming_node);
}

void LayerNode::addOutgoingNode(LayerNode *outgoing_node) {
  output_nodes.push_back(outgoing_node);
}

LayerGraph::LayerGraph() {
  root_layer_node = new LayerNode(NULL);
  layer_graph[0] = root_layer_node;
}

void LayerGraph::addNode(Layer *layer) {
  if (!this->layers.count(layer)) {
    this->layers.insert(layer);
    LayerNode *layer_node = new LayerNode(layer);
    this->layer_graph[layer] = layer_node;
  }
}

std::vector<Layer *>
LayerGraph::getLayersOfIncomingTensor(Tensor<std::float64_t> *tensor) {
  std::vector<Layer *> incoming_layer;
  for (Layer *layer : layers) {
    if (std::ranges::contains(layer->getInputTensors(), tensor)) {
      incoming_layer.push_back(layer);
    }
  }
  return incoming_layer;
}

Layer *LayerGraph::getLayerOfOutgoingTensor(Tensor<std::float64_t> *tensor) {
  Layer *incoming_layer = nullptr;
  for (Layer *layer : this->layers) {
    if (std::ranges::contains(layer->getOutputTensors(), tensor)) {
      incoming_layer = layer;
      break;
    }
  }
  return incoming_layer;
}

std::vector<Tensor<std::float64_t> *> LayerGraph::forward(
    std::unordered_map<Layer *, std::vector<Tensor<std::float64_t> *>>
        layer_input_map) {
  std::vector<Tensor<std::float64_t> *> incoming_node;
  return incoming_node;
}