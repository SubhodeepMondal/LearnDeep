#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <stdfloat>

template <typename T> class Tensor;

class Ops;

class Graph {

  typedef struct node {
    Ops *ops;

    node *next_node;
    node **output_nodes;
    node **input_nodes;
    unsigned input_node_count, output_node_count;
    unsigned input_node_index, output_node_index;

    bool ismemoryallocated;

    node() {
      next_node = NULL;
      output_nodes = input_nodes = NULL;
      input_node_count = input_node_index = output_node_count =
          output_node_index = 0;
      ismemoryallocated = false;
    }

    void addinputnode(node *input_ptr) {
      input_nodes[input_node_index++] = input_ptr;
    }

    void addoutputnode(node *output_ptr) {
      output_nodes[output_node_index++] = output_ptr;
    }

    void incinputnodecount() { input_node_count++; }
    void incoutputnodecount() { output_node_count++; }

    void allocatememoryforinputouputnode() {
      std::cout << "This node has " << input_node_count << " & "
                << output_node_count << " input and output nodes.\n";
      output_nodes = new node *[output_node_count];
      input_nodes = new node *[input_node_count];
    }

  } node;

  node *head, *current_node;
  bool isValidGraph;

public:
  Graph() : head(NULL), current_node(NULL), isValidGraph(true) {}

  void addcomputenode(Ops *ops);

  void optimize();

  void outputnode(Tensor<std::float64_t> *output);

  static void printnode(node *ptr);

  static void executionwrapper(node *ptr);

  void traversenode() { dfs(head, printnode); }

  void dfs(node *ptr, void (*func)(node *));

  void execute() { dfs(head, executionwrapper); }

  void setGraphInvalid() { this->isValidGraph = false; }
};

#endif // GRAPH_H