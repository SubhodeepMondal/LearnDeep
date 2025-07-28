#include <graph/graph_framework.hpp>

void Graph::addNode(Tensor<std::float64_t> *input_node) {
  if (data_nodes.count(input_node) == 0) {
    data_nodes.insert(input_node);
    // Create a new node for the input tensor and add it to the graph
    node *new_node = new node(reinterpret_cast<unsigned long>(input_node),
                              type::data, input_node);
    graph[reinterpret_cast<unsigned long>(input_node)] = new_node;
    // new_node->addInputNode(input_node);
    root_node->setOutputNode(new_node);
  }
}
void Graph::addNode(Ops *ops) {
  node *new_node = new node(reinterpret_cast<unsigned long>(ops), type::compute,
                            nullptr, ops);
  graph[reinterpret_cast<unsigned long>(ops)] = new_node;
  // new_node->addInputNode(ops);
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
    case Functions::travarse:
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
    case Functions::compute:
      if (start_node->node_type == type::compute) {
        std::cout << "Computing node with ID: " << start_node->node_id
                  << std::endl;
        start_node->execute(); // Execute the node's logic
      } else {
        std::cout << "Skipping data node with ID: " << start_node->node_id
                  << std::endl;
      }
      break;
    case Functions::travarse:
      if (start_node->node_type == type::data) {
        std::cout << "Printing node with ID: " << start_node->node_id
                  << std::endl;
        start_node->print_data();
      } else {
        std::cout << "Skipping compute node with ID: " << start_node->node_id
                  << std::endl;
      }
      break;

    default:
      break;
    }

    for (auto &input : start_node->output_nodes) {
      dfs(input, visited, func);
    }
  }
}

void Graph::compute() {
  if (!is_valid_graph) {
    std::cerr << "Graph is not valid for computation." << std::endl;
    return;
  }

  std::unordered_set<node *> visited;
  dfs(root_node, visited, Functions::compute);
}

void Graph::traverse() {
  if (!is_valid_graph) {
    std::cerr << "Graph is not valid!" << std::endl;
    return;
  }

  std::unordered_set<node *> visited;
  bfs(graph[root_node_id], visited, Functions::travarse);
}