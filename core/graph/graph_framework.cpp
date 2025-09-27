#include <absl/log/log.h>
#include <cstddef>
#include <framework/MathLibrary.h>
#include <graph/graph_framework.hpp>
#include <kernel/opskernel.h>
#include <stack>

std::string functionsToString(Functions func) {
  switch (func) {
  case Functions::compute:
    return "compute";
    break;
  case Functions::search:
    return "search";
    break;
  case Functions::traverse:
    return "traverse";
    break;
  case Functions::release_resource:
    return "release resource";
    break;
  case Functions::reverse_mode_autodiff:
    return "reverse_mode_autodiff";
    break;
  default:
    return "Unknown";
    break;
  }
}

void Graph::addNode(Tensor<std::float64_t> *input_node) {
  if (data_nodes.count(input_node) == 0) {
    data_nodes.insert(input_node);
    // Create a new node for the input tensor and add it to the graph
    node *new_node = new node(reinterpret_cast<unsigned long>(input_node),
                              type::data, input_node);
    graph[reinterpret_cast<unsigned long>(input_node)] = new_node;
    root_node->setOutputNode(new_node);
  }
}
void Graph::addNode(Ops *ops) {
  if (ops_nodes.count(ops) == 0) {
    ops_nodes.insert(ops);
    node *new_node = new node(reinterpret_cast<unsigned long>(ops),
                              type::compute, nullptr, ops);
    graph[reinterpret_cast<unsigned long>(ops)] = new_node;
  }
}

void Graph::addEdge(Tensor<std::float64_t> *src, Ops *dst) {
  node *src_node = graph[reinterpret_cast<unsigned long>(src)];
  node *dst_node = graph[reinterpret_cast<unsigned long>(dst)];
  if (src_node && dst_node) {
    src_node->setOutputNode(dst_node);
    dst_node->setInputNode(src_node);
  }
}

void Graph::addEdge(Ops *src, Tensor<std::float64_t> *dst) {
  node *src_node = graph[reinterpret_cast<unsigned long>(src)];
  node *dst_node = graph[reinterpret_cast<unsigned long>(dst)];
  if (src_node && dst_node) {
    src_node->setOutputNode(dst_node);
    dst_node->setInputNode(src_node);
    root_node->eraseNodeFromOutput(dst_node); // Remove from root node's output
  }
}

void Graph::addGradientNode(Tensor<std::float64_t> *input_node) {
  if (grad_data_nodes.count(input_node) == 0) {
    grad_data_nodes.insert(input_node);
    // Create a new node for the input tensor and add it to the graph
    node *new_node = new node(reinterpret_cast<unsigned long>(input_node),
                              type::data, input_node);
    auto_diff_graph[reinterpret_cast<unsigned long>(input_node)] = new_node;
    gradient_root_node->setOutputNode(new_node);
  }
}

void Graph::addGradientNode(Ops *ops) {
  if (grad_ops_nodes.count(ops) == 0) {
    grad_ops_nodes.insert(ops);
    node *new_node = new node(reinterpret_cast<unsigned long>(ops),
                              type::compute, nullptr, ops);
    auto_diff_graph[reinterpret_cast<unsigned long>(ops)] = new_node;
  }
}

void Graph::addGradientEdge(Tensor<std::float64_t> *src, Ops *dst) {
  node *src_node = auto_diff_graph[reinterpret_cast<unsigned long>(src)];
  node *dst_node = auto_diff_graph[reinterpret_cast<unsigned long>(dst)];
  if (src_node && dst_node) {
    src_node->setOutputNode(dst_node);
    dst_node->setInputNode(src_node);
  }
}

void Graph::addGradientEdge(Ops *src, Tensor<std::float64_t> *dst) {
  node *src_node = auto_diff_graph[reinterpret_cast<unsigned long>(src)];
  node *dst_node = auto_diff_graph[reinterpret_cast<unsigned long>(dst)];
  if (src_node && dst_node) {
    src_node->setOutputNode(dst_node);
    dst_node->setInputNode(src_node);
    gradient_root_node->eraseNodeFromOutput(
        dst_node); // Remove from root node's output
  }
}

void Graph::bfs(node *start_node, std::unordered_set<node *> &visited,
                Functions func) {
  std::queue<node *> queue;
  queue.push(start_node);
  visited.insert(start_node);

  while (!queue.empty()) {
    node *current = queue.front();
    queue.pop();

    switch (func) {
    case Functions::compute:
      if (current->node_type == type::compute) {
        current->execute();
      }
      break;
    case Functions::traverse:
      current->print_data();
      break;
    default:
      break;
    }

    for (auto &input : current->output_nodes) {
      if (visited.count(input) == 0) {
        visited.insert(input);
        queue.push(input);
      }
    }
  }
}

void Graph::dfs(node *start_node, std::unordered_set<node *> &visited,
                Functions func) {
  if (visited.count(start_node) == 0) {
    visited.insert(start_node);
    switch (func) {
    case Functions::compute: {
      if (start_node->node_type == type::compute) {
        LOG(INFO) << "Computing node with ID: " << start_node->node_id;
        start_node->execute(); // Execute the node's logic
      } else {
        LOG(INFO) << "Skipping data node with ID: " << start_node->node_id;
      }
      break;
    }
    case Functions::traverse: {
      if (start_node->node_type == type::data) {
        LOG(INFO) << "Printing node with ID: " << start_node->node_id;
        start_node->print_data();
      } else {
        LOG(INFO) << "Skipping compute node with ID: " << start_node->node_id;
      }
      break;
    }
    case Functions::release_resource:
      start_node->release_resources();
      break;
    case Functions::reverse_mode_autodiff: {
      if (start_node->node_type == type::compute) {
        LOG(INFO) << "Adding node with ID: " << start_node->node_id;
        LOG(INFO) << " for gradient calculation\n";
        ops_stack.push(start_node);
      }
      break;
    }
    default:
      std::cerr << "Unknown Operation for Graph: " << functionsToString(func)
                << "\n";
      break;
    }

    for (auto &input : start_node->output_nodes) {
      dfs(input, visited, func);
    }
  }
}

void Graph::compute() {
  if (!is_valid_graph) {
    std::cerr << "Graph is not valid for computation.\n";
    return;
  }

  std::unordered_set<node *> visited;
  dfs(root_node, visited, Functions::compute);
}

void Graph::computeGradient() {
  if (!is_valid_graph) {
    std::cerr << "Graph is not valid for computation.\n";
    return;
  }

  std::unordered_set<node *> visited;
  dfs(gradient_root_node, visited, Functions::compute);
}

std::vector<void *> Graph::getDataNodes() {
  std::vector<void *> dataNodes;
  for (auto items : data_nodes) {
    dataNodes.push_back(static_cast<void *>(items));
  }
  for (auto grad_data_node : grad_data_nodes)
    if (!data_nodes.count(grad_data_node))
      dataNodes.push_back(static_cast<void *>(grad_data_node));
  return dataNodes;
}

void Graph::createGradientGraph() {
  if (!is_valid_graph) {
    std::cerr << "Graph is not valid for reverse mode autodiff.\n";
    return;
  }
  std::unordered_set<node *> visited;
  dfs(root_node, visited, Functions::reverse_mode_autodiff);

  while (!ops_stack.empty()) {
    ops_stack.top()->addGradient(this);
    ops_stack.pop();
  }
}

std::vector<Tensor<std::float64_t> *> Graph::getGradient(Ops *ops) {
  std::vector<Tensor<std::float64_t> *> gradient_tensors;
  if (graph[reinterpret_cast<unsigned long>(ops)]->node_type == type::compute) {
    this->getIncomingGradientForOpsNode(
        graph[reinterpret_cast<unsigned long>(ops)], gradient_tensors);
  } else {
    LOG(FATAL) << "Fatal! Not a compute node to get a gradient tensor.\n";
  }
  return gradient_tensors;
}

void Graph::getIncomingGradientForOpsNode(
    node *ops_node, std::vector<Tensor<std::float64_t> *> &gradient_tensors) {

  //   [ node:data ]
  //         |
  // [output_nodes:ops]

  node *output_node_for_ops =
      ops_node->output_nodes[0]; // As ops node has only one output node

  if (output_node_for_ops->output_nodes.size())
    for (node *output : output_node_for_ops->output_nodes)
      if (output->node_type == type::compute)
        gradient_tensors.push_back(
            output->ops->getGradientTensor(output_node_for_ops->input_node));
}

Tensor<std::float64_t> *
Graph::getGradientTensor(Tensor<std::float64_t> *input_tensor) {
  node *input_node = graph[reinterpret_cast<unsigned long>(input_tensor)];

  auto it = std::find(root_node->output_nodes.begin(),
                      root_node->output_nodes.end(), input_node);

  if (it != root_node->output_nodes.end()) {
    LOG(WARNING) << "This is end node gradient might be inconsistent.\n";
    node *ops_node = input_node->output_nodes[0];
    return ops_node->ops->getGradientTensor(input_tensor);
  } else if (!input_node->output_nodes.size()) {
    LOG(INFO) << "This is an end node, this doesn't have any gradient\n";
    return nullptr;
  } else {
    node *ops_node = input_node->output_nodes[0];
    return ops_node->ops->getGradientTensor(input_tensor);
  }
}

void Graph::traverse() {
  if (is_valid_graph) {
    std::unordered_set<node *> visited;
    bfs(root_node, visited, Functions::traverse);
  } else {
    std::cerr << "Graph is not valid!";
  }
}

void Graph::traverseGradientGraph() {

  if (is_valid_graph) {
    std::unordered_set<node *> visited;
    bfs(gradient_root_node, visited, Functions::traverse);
  } else {
    std::cerr << "Graph is not valid!";
  }
}

void Graph::release_resources() {
  if (!is_valid_graph) {
    LOG(ERROR) << "Graph is not valid!";
    return;
  }

  LOG(INFO) << "Total data node in graph: " << data_nodes.size() << "\n";
  for (Tensor<std::float64_t> *data_node : data_nodes)
    delete data_node;

  LOG(INFO) << "Total ops node in graph: " << ops_nodes.size() << "\n";
  for (Ops *ops_node : ops_nodes)
    delete ops_node;

  for (auto nodes : graph)
    delete nodes.second;

  // deleting gradient graph
  LOG(INFO) << "Total data node in gradient graph: " << grad_data_nodes.size()
            << "\n";
  for (Tensor<std::float64_t> *grad_data_node : grad_data_nodes)
    if (!data_nodes.count(grad_data_node))
      delete grad_data_node;

  LOG(INFO) << "Total ops node in gradient graph: " << grad_ops_nodes.size()
            << "\n";
  for (Ops *grad_ops_node : grad_ops_nodes)
    delete grad_ops_node;

  for (auto nodes : auto_diff_graph)
    delete nodes.second;
}