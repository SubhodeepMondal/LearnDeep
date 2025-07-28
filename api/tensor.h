#ifndef TENSOR_MAIN_API
#define TENSOR_MAIN_API

#include <algorithm>
#include <framework/MathLibrary.h>
#include <graph/graph_framework.hpp>
#include <iostream>
#include <iterator>
#include <vector>

namespace tf {
struct graph;

typedef struct graph_manager {
  graph *ptr;
  bool isValidGraph;
  std::vector<graph *> graph_list;

  void addGraph(graph *g) { graph_list.push_back(g); }

  bool isThereActiveSession();

  graph *findActivateSession();
} graph_manager;

extern graph_manager g_manager;

typedef struct tensor {
  void *ptr;
  DataType dt_type;

  void addDimensions(std::vector<unsigned> &dimensions, unsigned w) {
    dimensions.push_back(w);
  }

  template <typename... args>
  void addDimensions(std::vector<unsigned> &dimensions, unsigned w,
                     args... Args) {
    addDimensions(dimensions, w);
    addDimensions(dimensions, Args...);
  }

  template <typename... Args> void tf_create(DataType d_type, Args... args) {
    unsigned *arr;
    std::vector<unsigned> dimensions;

    addDimensions(dimensions, args...);

    arr = new unsigned[dimensions.size()];
    for (int i = 0; i < dimensions.size(); i++) {
      arr[i] = dimensions[i];
    }

    switch (d_type) {
    case tf_float64:
      ptr = new Tensor<std::float64_t>(dimensions.size(), dimensions.data(),
                                       d_type);
      break;
    default:
      ptr = nullptr;
    }
    dt_type = d_type;
    delete[] arr;
  }

  unsigned getNoOfDimensions() {
    return static_cast<Tensor<std::float64_t> *>(ptr)->getNoOfDimensions();
  }

  const unsigned *getDimensions() {
    return static_cast<Tensor<std::float64_t> *>(ptr)->getDimensions();
  }

  unsigned getNoOfElem() {
    return static_cast<Tensor<std::float64_t> *>(ptr)->getNoOfElem();
  }
  void tensor_of(double low_limit, double upper_limit);

  void tensor_of(std::float64_t *data);

  void print_data();

  void print_dimension();

  graph &add(tensor &input_b);

  void operator=(graph &g);

  // void mul(tensor &input_b);

  // void matmul(tensor &input_b);

  // void pow(tensor &input, unsigned power);

  // void scale(graph &g, tensor &output, tensor &input,
  //            std::float64_t scale_factor);

  // void reducesum(graph &g, tensor &output, tensor &input);
  // template <typename first_dim, typename... Args>
  // void reducesum(graph &g, tensor &output, tensor &input, first_dim n,
  //                Args... args);

  // template <typename... Args>
  // void reducesum(graph &g, tensor &output, tensor &input, Args... args);

  // void mean(graph &g, tensor &output, tensor &input, const unsigned n);

} tensor;

typedef struct graph {
  void *ptr;
  bool isSessionActive = bool(false);
  tensor input_a;
  tensor input_b;
  Ops *ops;
  void tf_create_graph();

  void graph_start_recording_session();

  void graph_end_recording_session();

  // void graph_optimize(graph &g);

  void graph_execute();

  void graph_travarse_data_node();

} graph;

struct arg_list {
  unsigned value;
  struct arg_list *next;
};

} // namespace tf

#endif // TENSOR_MAIN_API