#ifndef GRAPH_NODE_HPP
#define GRAPH_NODE_HPP

#include <algorithm>
#include <memory>
#include <stdfloat>
#include <unordered_set>
#include <vector>


#include "absl/log/globals.h"
#include "absl/log/log.h"       

// struct AbslLogSilencer {
//   AbslLogSilencer() {
//     absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
//   }
// };

// static AbslLogSilencer g_absl_log_silencer;

template <typename T> class Tensor;
class Ops;


enum class type { data, compute, root };

typedef struct node {
public:
  unsigned long node_id; // Unique identifier for the node

  // node properties
  std::vector<node *> input_nodes;
  std::vector<node *> output_nodes;

  // node atributes
  Tensor<std::float64_t> *input_node;
  Ops *ops;

  type node_type;

  node(unsigned long id, type t, Tensor<std::float64_t> *input = nullptr,
       Ops *ops = nullptr)
      : node_id(id), node_type(t), input_node(input), ops(ops){};
  // void addRootNode();

  // void
  // addInputNode(Tensor<std::float64_t> *input); // Placeholder for adding
  // input void addInputNode(Ops *ops);                 // Adding input node
  void setInputNode(node *input_node);   // Setting input
  void setOutputNode(node *output_node); // Setting output
  void execute();                        // Execution logic
  void print_data();
  void eraseNodeFromOutput(node *n);
} node;
#endif