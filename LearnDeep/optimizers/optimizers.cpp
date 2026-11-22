#include "optimizers.hpp"

#include <atomic>

#include <core/graph/graph_framework.hpp>

namespace {
std::atomic<tf::optimizer::optimizer_id> next_optimizer_id{1};
} // namespace

Optimizer::Optimizer()
    : id_(next_optimizer_id.fetch_add(1, std::memory_order_relaxed)),
      optimizer_graph(new Graph()) {}

Optimizer::~Optimizer() {
  if (!optimizer_graph)
    return;

  // The tensors recorded by an optimizer graph are owned by tf::tensor
  // wrappers (and by layers), not by Graph. Mirror tf::graph_context and spare
  // every API-owned tensor before destroying the graph.
  optimizer_graph->setTensorToBeSpared(tf::tensor_to_be_spared);
  delete optimizer_graph;
  optimizer_graph = nullptr;
}

void Optimizer::executeOptimizer() { this->optimizer_graph->compute(); }
