// C++ Headers
#include <algorithm>
#include <cstddef>
#include <ranges>
#include <unordered_set>

// Library Headers
#include "layer_graph.hpp"
#include "layers.hpp"

LayerGraph global_layer_graph;

LayerGraph::LayerGraph() {}

LayerGraph::~LayerGraph() { this->layers.clear(); }

void LayerGraph::addNode(Layer *layer) { this->layers.insert(layer); }

std::vector<Layer *>
LayerGraph::getLayersOfIncomingTensor(const Tensor<std::float64_t> *tensor) {
  std::vector<Layer *> outgoing_layer;
  for (Layer *layer : layers) {
    if (std::ranges::contains(layer->getInputTensors(), tensor)) {
      outgoing_layer.push_back(layer);
      break;
    }
  }
  return outgoing_layer;
}
void LayerGraph::removeNode(Layer *layer) { layers.erase(layer); }

Layer *
LayerGraph::getLayerOfOutgoingTensor(const Tensor<std::float64_t> *tensor) {
  Layer *incoming_layer = nullptr;
  for (Layer *layer : this->layers) {
    std::vector<Tensor<std::float64_t> *> reference_output_tensors;

    for (tf::tensor *output_tensor : layer->getOutputTensors())
      reference_output_tensors.push_back(output_tensor->getPtr());

    if (std::ranges::contains(reference_output_tensors, tensor)) {
      incoming_layer = layer;
      break;
    }
  }
  return incoming_layer;
}

std::vector<Layer *> LayerGraph::getAllLayers() {
  std::vector<Layer *> output_layers;

  for (Layer *layer : this->layers)
    output_layers.push_back(layer);

  return output_layers;
}