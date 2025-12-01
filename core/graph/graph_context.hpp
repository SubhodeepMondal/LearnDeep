#ifndef GRAPH_CONTEXT
#define GRAPH_CONTEXT

#include "graph_framework.hpp"
#include "graph_manager.hpp"

class GraphContext {
  Graph *graph;

public:
  GraphContext() {
    graph = new Graph;
    GraphManager::instance().pushGraph(graph);
  }

  ~GraphContext() {
    graph = nullptr;
    GraphManager::instance().popGraph();
  }

  void run() { graph->compute(); }
};

#endif // GRAPH_CONTEXT