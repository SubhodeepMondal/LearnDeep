#ifndef GRAPH_MANAGER
#define GRAPH_MANAGER

#include "graph_framework.hpp"
#include <vector>

class GraphManager {
  thread_local static GraphManager *graph_manager_ptr;
  std::vector<Graph *> graphs;

public:
  static GraphManager &instance();

  Graph *getCurrentGraph();

  bool isValidGraphAvailable();

  void pushGraph(Graph *g);

  void popGraph();
};

#endif // GRAPH_MANAGER