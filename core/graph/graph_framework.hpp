#ifndef GRAPH_FRAMEWORK_HPP
#define GRAPH_FRAMEWORK_HPP

#include <graph/graph_node.hpp>
#include <iostream>
#include <memory>
#include <queue>
#include <stdfloat>
#include <unordered_map>
#include <unordered_set>
#include <vector>

enum class Functions { compute, search, travarse, release_resource };

class Graph {
  std::unordered_set<Tensor<std::float64_t> *> data_nodes;
  std::unordered_map<unsigned long, node *> graph;

  bool is_valid_graph;

  node *root_node;
  unsigned long root_node_id;

  void dfs(node *start_node, std::unordered_set<node *> &visited,
           Functions func);

  void bfs(node *start_node, std::unordered_set<node *> &visited,
           Functions func);

public:
  Graph() : is_valid_graph(true) {
    root_node_id = 0;
    root_node = new node(0, type::root);
    graph[root_node_id] = root_node;
  }

  void addNode(Tensor<std::float64_t> *input_node);

  void addNode(Ops *ops);

  void addEdge(Tensor<std::float64_t> *src, Ops *dst);

  void addEdge(Ops *src, Tensor<std::float64_t> *dst);

  void compute();

  std::vector<void *> getDataNodes();

  void traverse();

  void release_resources();
};

#endif // GRAPH_FRAMEWORK_HPP