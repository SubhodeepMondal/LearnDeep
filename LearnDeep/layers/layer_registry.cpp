// Library Headers
#include "layer_registry.hpp"
#include <layers/layer_graph.hpp>
#include <layers/layers.hpp>

// Third party
#include <absl/log/log.h>

LayerRegistry &LayerRegistry::instance() {
  static LayerRegistry registry;
  return registry;
}

bool LayerRegistry::registerLayer(const std::string &name, Factory factory,
                                  ParameterSetter setter) {
  if (name.empty() || !factory) {
    LOG(ERROR) << "Fatal! Layer registration requires a name and a factory.\n";
    return false;
  }

  if (layer_names_.count(name)) {
    LOG(ERROR) << "Fatal! Layer type '" << name << "' is already registered.\n";
    return false;
  }

  layer_names_.insert(name);
  factories_.insert_or_assign(name, std::move(factory));
  if (setter)
    parameter_setters_.insert_or_assign(name, std::move(setter));
  return true;
}

LayerRegistry::LayerHandle
LayerRegistry::create(const std::string &name, const LayerConfig &config) const {
  auto factory = factories_.find(name);
  if (factory == factories_.end()) {
    LOG(ERROR) << "Fatal! Unknown layer type '" << name << "'.\n";
    return nullptr;
  }

  std::unique_ptr<Layer> created = factory->second(config);
  if (!created) {
    LOG(ERROR) << "Fatal! Failed to create layer type '" << name << "'.\n";
    return nullptr;
  }

  LayerHandle layer(created.release(), [](Layer *owned_layer) {
    if (!owned_layer)
      return;

    global_layer_graph.removeNode(owned_layer->id());
    delete owned_layer;
  });
  global_layer_graph.addNode(layer);
  return layer;
}

bool LayerRegistry::setParameter(const std::string &name,
                                 const LayerHandle &layer,
                                 const std::string &parameter,
                                 const tf::LayerParameterValue &value) const {
  auto setter = parameter_setters_.find(name);
  if (!layer || setter == parameter_setters_.end() ||
      !setter->second(layer, parameter, value)) {
    LOG(ERROR) << "Fatal! Parameter '" << parameter
               << "' is not settable for layer type '" << name << "'.\n";
    return false;
  }

  return true;
}

bool LayerRegistry::contains(const std::string &name) const {
  return layer_names_.count(name) != 0;
}

const std::unordered_set<std::string> &LayerRegistry::names() const {
  return layer_names_;
}
