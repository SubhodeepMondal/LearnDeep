#ifndef GRAPH_FRAMEWORK_HPP
#define GRAPH_FRAMEWORK_HPP

#include <graph/graph_node.hpp>

#include <algorithm>
#include <iostream>
#include <queue>
#include <stdfloat>
#include <unordered_map>
#include <unordered_set>
#include <vector>

enum class Functions {
  compute,
  search,
  traverse,
  release_resource,
  reverse_mode_autodiff
};
std::string functionsToString(Functions func);

template <typename T> class Tensor;
class Ops;

class Graph {
  std::unordered_set<Tensor<std::float64_t> *> data_nodes;
  std::unordered_set<Ops *> ops_nodes;
  std::unordered_set<Tensor<std::float64_t> *> grad_data_nodes;
  std::unordered_set<Ops *> grad_ops_nodes;
  std::unordered_map<unsigned long, node *> graph;
  std::unordered_map<unsigned long, node *> auto_diff_graph;
  std::stack<node *> ops_stack;
  bool is_valid_graph;
  bool release_graph_resource;
  bool release_autograd_graph_resoruce;
  node *root_node;
  node *gradient_root_node;

  void dfs(node *start_node, std::unordered_set<node *> &visited,
           Functions func);

  void bfs(node *start_node, std::unordered_set<node *> &visited,
           Functions func);

public:
  Graph() {
    is_valid_graph = true;
    root_node = new node(0, type::root);
    gradient_root_node = new node(0, type::root);

    graph[0] = root_node;                    // 0 is root node id
    auto_diff_graph[0] = gradient_root_node; // 0 is gradient root node id
  }

  void addNode(Tensor<std::float64_t> *input_node);

  void addNode(Ops *ops);

  void addGradientNode(Tensor<std::float64_t> *input_node);

  void addGradientNode(Ops *ops);

  void addEdge(Tensor<std::float64_t> *src, Ops *dst);

  void addEdge(Ops *src, Tensor<std::float64_t> *dst);

  void addGradientEdge(Tensor<std::float64_t> *src, Ops *dst);

  void addGradientEdge(Ops *src, Tensor<std::float64_t> *dst);

  void createGradientGraph();

  std::vector<Tensor<std::float64_t> *> getGradient(Ops *ops);

  void compute();

  void computeGradient();

  std::vector<void *> getDataNodes();

  void getIncomingGradientForOpsNode(
      node *ops, std::vector<Tensor<std::float64_t> *> &gradient_tensors);

  std::vector<Tensor<std::float64_t> *>
  getGradientTensor(Tensor<std::float64_t> *input_tensor);

  void traverse();

  void traverseGradientGraph();

  void release_resources();
};

#endif // GRAPH_FRAMEWORK_HPP