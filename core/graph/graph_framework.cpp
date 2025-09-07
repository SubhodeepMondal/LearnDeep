#include <absl/log/log.h>
#include <graph/graph_framework.hpp>
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
  node *new_node = new node(reinterpret_cast<unsigned long>(ops), type::compute,
                            nullptr, ops);
  graph[reinterpret_cast<unsigned long>(ops)] = new_node;
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
  node *new_node = new node(reinterpret_cast<unsigned long>(ops), type::compute,
                            nullptr, ops);
  auto_diff_graph[reinterpret_cast<unsigned long>(ops)] = new_node;
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
    // node *temp _node = ops_stack.top();
    ops_stack.top()->addGradient(this);
    ops_stack.pop();
  }
}

Tensor<std::float64_t> *Graph::getGradient(Ops *ops) {
  if (graph[reinterpret_cast<unsigned long>(ops)]->node_type == type::compute)
    return graph[reinterpret_cast<unsigned long>(ops)]
        ->getIncomingGradientForOpsNode();
  else
    LOG(ERROR) << "Fatal! Not a compute node to get a gradient tensor.\n";
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
    std::cerr << "Graph is not valid!";
    return;
  }
  std::unordered_set<node *> visited;
  dfs(root_node, visited, Functions::release_resource);

  for (auto nodes : graph)
    delete nodes.second;

  std::unordered_set<node *> grad_visited;
  dfs(gradient_root_node, grad_visited, Functions::release_resource);

  for (auto nodes : auto_diff_graph)
    delete nodes.second;
}