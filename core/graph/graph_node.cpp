#include "graph_node.hpp"

#include <framework/MathLibrary.h>
#include <iostream>
#include <kernel/opskernel.h>

void node::setInputNode(node *input_node) {
  if (input_node) {
    input_nodes.push_back(input_node);
  }
}

void node::setOutputNode(node *output_node) {
  if (output_node) {
    output_nodes.push_back(output_node);
  }
}

void node::eraseNodeFromOutput(node *n) {
  if (type::root == node_type) {
    auto it = std::remove(output_nodes.begin(), output_nodes.end(), n);
    if (it != output_nodes.end()) {
      output_nodes.erase(it, output_nodes.end());
    }
  }
}

void node::execute() {
  if (type::compute == node_type) {
    // Execution logic for compute nodes
    std::cout << "Executing node with ID: " << node_id << std::endl;
    // Add actual execution logic here
    ops->compute(); // Assuming Ops has an execute method
  } else {
    std::cout << "Node with ID: " << node_id << " is not a compute node."
              << std::endl;
  }
}

void node::print_data() {
  if (type::data == node_type) {
    std::cout << "Node ID: " << node_id << ", Node Type: Data" << std::endl;
    if (input_node) {
      input_node->printData(); // Assuming Tensor has a print_data method
    }
  } else if (type::compute == node_type) {
    std::cout << "Node ID: " << node_id << ", Node Type: Compute" << std::endl;
  } else {
    std::cout << "Node ID: " << node_id << ", Node Type: Root" << std::endl;
  }
}