// C++ Headers
#include <algorithm>
#include <ranges>

// Library Headers
#include "layer_graph.hpp"
#include "layers.hpp"

LayerGraph global_layer_graph;

void LayerGraph::addNode(const std::shared_ptr<Layer> &node) {
  if (!node)
    return;

  layers.insert_or_assign(node->id(), node);
}

void LayerGraph::removeNode(tf::layer::layer_id id) { layers.erase(id); }

void LayerGraph::prune() {
  for (auto it = layers.begin(); it != layers.end();) {
    if (it->second.expired())
      it = layers.erase(it);
    else
      ++it;
  }
}

std::shared_ptr<Layer> LayerGraph::lock(tf::layer::layer_id id) {
  auto entry = layers.find(id);
  if (entry == layers.end())
    return {};

  std::shared_ptr<Layer> layer = entry->second.lock();
  if (!layer)
    layers.erase(entry);
  return layer;
}

bool LayerGraph::contains(tf::layer::layer_id id) {
  return static_cast<bool>(this->lock(id));
}

std::vector<tf::layer::layer_id> LayerGraph::getLayersOfIncomingTensor(
    const Tensor<std::float64_t> *tensor) {
  this->prune();

  std::vector<tf::layer::layer_id> layers_with_incoming_tensor;
  for (const auto &[id, weak_layer] : layers) {
    std::shared_ptr<Layer> layer = weak_layer.lock();
    if (!layer)
      continue;

    if (std::ranges::contains(layer->getInputTensors(), tensor)) {
      layers_with_incoming_tensor.push_back(id);
      break;
    }
  }
  return layers_with_incoming_tensor;
}

tf::layer::layer_id LayerGraph::getLayerOfOutgoingTensor(
    const Tensor<std::float64_t> *tensor) {
  this->prune();

  for (const auto &[id, weak_layer] : layers) {
    std::shared_ptr<Layer> layer = weak_layer.lock();
    if (!layer)
      continue;

    std::vector<Tensor<std::float64_t> *> reference_output_tensors;
    for (tf::tensor *output_tensor : layer->getOutputTensors())
      reference_output_tensors.push_back(output_tensor->getPtr());

    if (std::ranges::contains(reference_output_tensors, tensor))
      return id;
  }
  return tf::layer::invalid_layer_id;
}

std::vector<tf::layer::layer_id> LayerGraph::getAllLayerIds() {
  this->prune();

  std::vector<tf::layer::layer_id> ids;
  ids.reserve(layers.size());
  for (const auto &[id, weak_layer] : layers)
    ids.push_back(id);
  return ids;
}
