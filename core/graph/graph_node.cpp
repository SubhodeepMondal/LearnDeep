#include "graph_node.hpp"

void node::addInputNode(Tensor<std::float64_t>* input) {
  this->input_node = input;
  node_type = type::data; // This is a data node
}

void node::addInputNode(Ops* ops) {
  this->ops = ops;
  node_type = type::compute; // This is a compute node
}

void node::setInputNode(node* input_node) {
  if (input_node) {
    input_nodes.push_back(input_node);
    input_node_count++;
  }
}

void node::setOutputNode(node* output_node) {
  if (output_node) {
    output_nodes.push_back(output_node);
    output_node_count++;
  }
}