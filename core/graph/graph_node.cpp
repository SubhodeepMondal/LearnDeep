#include "graph_node.hpp"

#include <absl/log/log.h>
#include <framework/MathLibrary.h>
#include <kernel/opskernel.h>

void node::addGradient(Graph *autodiff_graph) {
  if (type::compute == node_type) {
    // Execution logic for compute nodes
    LOG(INFO) << "Executing node with ID: " << node_id;
    // Add actual execution logic here
    ops->addGradGraph(autodiff_graph);
  } else {
    LOG(INFO) << "Node with ID: " << node_id << " is not a compute node.";
  }
}

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

Tensor<std::float64_t> *node::getIncomingGradientForOpsNode() {
  unsigned no_of_output = output_nodes[0]->output_nodes.size();
  if (output_nodes[0]->output_nodes.size()) {
    return output_nodes[0]->output_nodes[0]->ops->getGradientTensor();
  }
  return NULL;
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
    LOG(INFO) << "Executing node with ID: " << node_id;
    // Add actual execution logic here
    ops->compute();
  } else {
    LOG(INFO) << "Node with ID: " << node_id << " is not a compute node.";
  }
}

void node::print_data() {
  if (type::data == this->node_type) {
    LOG(INFO) << "Node ID: " << node_id << ", Node Type: Data";
    if (input_node) {
      input_node->printData(); // Assuming Tensor has a print_data method
    }
  } else if (type::compute == this->node_type) {
    LOG(INFO) << "Node ID: " << node_id << ", Node Type: Compute";
  } else {
    LOG(INFO) << "Node ID: " << node_id << ", Node Type: Root";
  }
}

void node::release_resources() {
  switch (this->node_type) {
  case type::data:
    delete input_node;
    break;
  case type::compute:
    delete ops;
    break;
  default:
    break;
  }
}