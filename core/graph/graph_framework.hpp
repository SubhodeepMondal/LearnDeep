#ifndef GRAPH_FRAMEWORK_HPP
#define GRAPH_FRAMEWORK_HPP

#include <graph/graph_node.hpp>
#include <iostream>
#include <memory>
#include <stdfloat>
#include <unordered_map>
#include <vector>

class Graph {
  std::unordered_map<int, node *> graph;

  bool is_valid_graph;

public:
  Graph() : is_valid_graph(true) {}

  void addNode(Tensor<std::float64_t> *input_node);
  void addNode(Ops *ops);
  void addEdge(Tensor<std::float64_t> *src, Ops *dst);
  void addEdge(Ops *src, Tensor<std::float64_t> *dst);
};

#endif // GRAPH_FRAMEWORK_HPP