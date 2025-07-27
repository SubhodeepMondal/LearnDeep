#include <graph/graph_framework.hpp>

void Graph::addNode(Tensor<std::float64_t> *input_node) {
  node *new_node = new node(reinterpret_cast<unsigned long>(input_node));
  graph[reinterpret_cast<unsigned long>(input_node)] = new_node;
  new_node->addInputNode(input_node);
}
void Graph::addNode(Ops *ops) {
  node *new_node = new node(reinterpret_cast<unsigned long>(ops));
  graph[reinterpret_cast<unsigned long>(ops)] = new_node;
  new_node->addInputNode(ops);
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
  }
}