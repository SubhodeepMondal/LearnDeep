#ifndef GRAPH_CONTEXT
#define GRAPH_CONTEXT

#include "graph_framework.hpp"
#include "graph_manager.hpp"

#include <framework/MathLibrary.h>

class GraphContext {
  Graph *graph;

public:
  GraphContext() {
    graph = new Graph();
    GraphManager::instance().pushGraph(graph);
  }

  ~GraphContext() {
    // graph->release_resources();
    graph = nullptr;
    GraphManager::instance().popGraph();
  }

  void run() { graph->compute(); }

  std::vector<void *> get_data_nodes() { return graph->getDataNodes(); }

  void graph_compute_gradeint() { graph->computeGradient(); }

  Tensor<std::float64_t> *
  graph_get_gradient(Tensor<std::float64_t> *input_tensor) {
    return graph->getGradientTensor(input_tensor);
  }

  void graph_initiize_gradient() { graph->createGradientGraph(); }
};

#endif // GRAPH_CONTEXT