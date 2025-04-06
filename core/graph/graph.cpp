#include "graph.h"
#include "../framework/MathLibrary.h"
#include "../kernel/opskernel.h"
void Graph::printnode(node *ptr) {
  ptr->ops->printinputs();
  ptr->ops->printoutput();
}

void Graph::executionwrapper(node *ptr) { ptr->ops->compute(); }

void Graph::addcomputenode(Ops *ops) {
  node *ptr = new node;

  // ptr->operand_a = oper_a;
  // ptr->operand_b = oper_b;
  ptr->ops = ops;
  ptr->input_node_count = ops->getnoofinputs();

  if (head == NULL) {
    head = current_node = ptr;
  } else {
    current_node->next_node = ptr;
    current_node = ptr;
  }
}

void Graph::optimize() {
  node *ptr = head->next_node;
  node *prev_ptr = head;
  unsigned i;

  // std::cout << ptr << "\n";
  // std::cout << prev_ptr << "\n";

  // Pass 1: to get input and output count for each node
  if (!isValidGraph)
    return;

  while (ptr) {
    for (i = 0; i < ptr->input_node_count; i++) {
      // std::cout << ptr->ops->getinputs()[i] << "\n";
      prev_ptr = head;
      while (prev_ptr != ptr) {

        // std::cout << prev_ptr->ops->getoutput() << "\n";
        if (ptr->ops->getinputs()[i] == prev_ptr->ops->getoutput()) {
          // ptr->incinputnodecount();
          prev_ptr->incoutputnodecount();
        }
        prev_ptr = prev_ptr->next_node;
      }
      // std::cout << "\n";
    }
    ptr = ptr->next_node;
  }

  // pass 2: to allocate memory for each node
  ptr = head;
  while (ptr) {
    ptr->allocatememoryforinputouputnode();
    ptr = ptr->next_node;
  }

  // pass 3: to connect all the incoming and outgoing nodes
  ptr = head;
  while (ptr) {
    for (i = 0; i < ptr->input_node_count; i++) {
      prev_ptr = head;
      while (prev_ptr != ptr) {
        if (ptr->ops->getinputs()[i] == prev_ptr->ops->getoutput()) {
          ptr->addinputnode(prev_ptr);
          prev_ptr->addoutputnode(ptr);
        }
        prev_ptr = prev_ptr->next_node;
      }
    }
    ptr = ptr->next_node;
  }
}

void Graph::outputnode(Tensor<std::float64_t> *output) {
  // current_node->output = output;
  if (isValidGraph) {
    current_node->ops->initilizeoutput(output);
  } else
    std::cout << "Graph is invalid! check input sequence.\n";
}

void Graph::dfs(node *ptr, void (*func)(node *)) {
  if (isValidGraph) {
    func(ptr);

    for (unsigned i = 0; i < ptr->output_node_count; i++) {
      dfs(ptr->output_nodes[i], func);
    }
  } else
    std::cout << "Graph is invalid! check input sequence.\n";
}
