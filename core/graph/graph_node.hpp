#ifndef GRAPH_NODE_HPP
#define GRAPH_NODE_HPP

#include <memory>
#include <stdfloat>
#include <unordered_set>
#include <vector>

template <typename T> class Tensor;
class Ops;

enum class type {
  data,
  compute,
};

class node {
private:
  type node_type;

  unsigned long node_id; // Unique identifier for the node

  // node properties
  unsigned output_node_count;
  unsigned input_node_count;
  std::vector<node *> input_nodes;
  std::vector<node *> output_nodes;

  // node atributes
  Tensor<std::float64_t> *input_node;
  Ops *ops;

public:
  // void initialize(); // Placeholder for initialization logic
  node(unsigned long id)
      : output_node_count(0), input_node_count(0), node_id(id) {}

  void
  addInputNode(Tensor<std::float64_t> *input); // Placeholder for adding input
  void addInputNode(Ops *ops);                 // Adding input node
  void setInputNode(node *input_node);         // Setting input
  void setOutputNode(node *output_node);       // Setting output
  void execute() {};                           // Execution logic
};
#endif