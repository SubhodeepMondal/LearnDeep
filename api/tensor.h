#ifndef TENSOR_MAIN_API
#define TENSOR_MAIN_API

#include <framework/MathLibrary.h>
#include <graph/graph.h>
#include <iostream>

namespace tf {
typedef struct tensor {
  void *ptr;
  DataType dt_type;
} tensor;

typedef void *graph;

struct arg_list {
  unsigned value;
  struct arg_list *next;
};

extern dim_node *head;
extern dim_node *ptr;
extern dim_node *prev;

void addDimensions(unsigned *nDim, unsigned w);

template <typename... args>
void addDimensions(unsigned *nDim, unsigned w, args... Args);

template <typename... Args>
void tf_create(tensor &a, DataType d_type, unsigned num, Args... args) {
  unsigned nDim;
  unsigned *arr;

  nDim = 0;
  head = ptr = prev = NULL;

  dim_node *ptr_new = new dim_node;

  head = ptr = ptr_new;
  head->value = num;
  head->next = NULL;
  nDim++;

  addDimensions(&nDim, args...);

  arr = new unsigned[nDim];
  ptr = head;
  for (int i = 0; i < nDim; i++) {
    arr[i] = ptr->value;
    prev = ptr;
    ptr = ptr->next;
    delete prev;
  }

  switch (d_type) {
  case tf_float64:
    a.ptr = new Tensor<std::float64_t>(nDim, arr, d_type);
    break;
  default:
    a.ptr = nullptr;
  }
  a.dt_type = d_type;
  delete[] arr;
}

void tf_create_graph(graph &g);

void tensor_of(tensor &input, double low_limit, double upper_limit);

void tensor_of(tensor &input, std::float64_t *data);

void print_data(tensor &input);

void print_dimension(tensor &input);

void matmul(graph &g, tensor &output, tensor &input_a, tensor &input_b);

void mul(graph &g, tensor &output, tensor &input_a, tensor &input_b);

void add(graph &g, tensor &output, tensor &input_a, tensor &input_b);

void pow(graph &g, tensor &output, tensor &input, unsigned power);

void scale(graph &g, tensor &output, tensor &input,
           std::float64_t scale_factor);

void reducesum(graph &g, tensor &output, tensor &input);
template <typename first_dim, typename... Args>
void reducesum(graph &g, tensor &output, tensor &input, first_dim n,
               Args... args);

template <typename... Args>
void reducesum(graph &g, tensor &output, tensor &input, Args... args);

void mean(graph &g, tensor &output, tensor &input, const unsigned n);

void graph_optimize(graph &g);

void graph_execute(graph &g);

void graph_travarse_node(graph &g);

} // namespace tf

#endif // TENSOR_MAIN_API