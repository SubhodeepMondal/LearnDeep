#include "graph_manager.hpp"

#include "absl/log/log.h"

thread_local GraphManager *GraphManager::graph_manager_ptr = nullptr;

GraphManager &GraphManager::instance() {
  if (!graph_manager_ptr)
    graph_manager_ptr = new GraphManager();
  return *graph_manager_ptr;
}
Graph *GraphManager::getCurrentGraph() {
  if (graphs.size()) {
    return graphs[graphs.size() - 1];
  } else {
    LOG(INFO) << "Fatal! No active graph found!\n";
    return NULL;
  }
}

bool GraphManager::isValidGraphAvailable() {
  if (graphs.size())
    return true;
  else
    return false;
}

void GraphManager::popGraph() {
  if (this->graphs.size()) {
    LOG(INFO) << "Graph is cleared\n";
    graphs.clear();
  } else {
    LOG(INFO) << "No active graph to clear\n";
  }
}

void GraphManager::pushGraph(Graph *g) { this->graphs.push_back(g); }