#ifndef GRAPH_NODE_HPP
#define GRAPH_NODE_HPP

#include <stack>
#include <stdfloat>
#include <vector>

#include "absl/log/log.h"

template <typename T> class Tensor;
class Ops;
class Graph;

enum class type { data, compute, root };

typedef struct node {
public:
  unsigned long node_id; // Unique identifier for the node

  // node properties
  std::vector<node *> input_nodes;
  std::vector<node *> output_nodes;

  // node atributes
  Tensor<std::float64_t> *input_node;
  Ops *ops;

  type node_type;

  node(unsigned long id, type t, Tensor<std::float64_t> *input = nullptr,
       Ops *ops = nullptr)
      : node_id(id), node_type(t), input_node(input), ops(ops) {};

  void addGradient(Graph *autodiff_graph);

  void setInputNode(node *input_node); // Setting input

  void setOutputNode(node *output_node); // Setting output

  // Tensor<std::float64_t> *getIncomingGradientForOpsNode();

  void execute(); // Execution logic

  void print_data();

  void eraseNodeFromOutput(node *n);

  void release_resources();
} node;
#endif