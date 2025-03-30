#include "../core/framework/MathLibrary.h"
#include "../core/graph/graph.h"
#include <iostream>

namespace tf {

struct arg_list {
  unsigned value;
  struct arg_list *next;
};

struct arg_list *arg_head, *arg_ptr, *arg_ptr_prev;

dim_node *head;
dim_node *ptr;
dim_node *prev;

void addDimensions(unsigned *nDim, unsigned w) {
  dim_node *ptr_new = new dim_node;

  ptr->next = ptr_new;
  ptr_new->value = w;
  ptr_new->next = NULL;
  ptr = ptr->next;
  (*nDim)++;
}

template <typename... args>
void addDimensions(unsigned *nDim, unsigned w, args... Args) {
  addDimensions(nDim, w);
  addDimensions(nDim, Args...);
}

template <typename... Args>
tensor<double> tf_create(unsigned num, Args... args) {
  tensor<double> a;
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

  a = tensor<double>(nDim, arr);
  delete[] arr;
  return a;
}

void matmul(graph &g, tensor<double> &output, tensor<double> &input_a,
            tensor<double> &input_b) {
  output = input_a.matmul(g, input_b);
}

void mul(graph &g, tensor<double> &output, tensor<double> &input_a,
         tensor<double> &input_b) {
  output = input_a.mul(g, input_b);
}

void add(graph &g, tensor<double> &output, tensor<double> &input_a,
         tensor<double> &input_b) {
  output = input_a.add(g, input_b);
}

void pow(graph &g, tensor<double> &output, tensor<double> &input_a,
         unsigned exponent) {
  output = input_a.pow(g, exponent);
}

void scale(graph &g, tensor<double> &output, tensor<double> &input_a,
           const double scale_factor) {
  output = input_a.scale(g, scale_factor);
}

void reducesum(graph &g, tensor<double> &output, tensor<double> &input) {

  unsigned count = 0;
  unsigned *reduction_dims;
  unsigned *dims;

  arg_ptr = arg_ptr_prev = arg_head;

  while (arg_ptr) {
    count++;
    arg_ptr = arg_ptr->next;
  }
  std::cout << "count: " << count << "\n";
  reduction_dims = new unsigned[count];
  arg_ptr = arg_head;

  for (unsigned i = 0; i < count; i++) {
    reduction_dims[i] = arg_ptr->value;
    arg_ptr = arg_ptr->next;
    delete[] arg_ptr_prev;
    arg_ptr_prev = arg_ptr;
  }

  output = input.reducesum(g, count, reduction_dims);
}

template <typename first_dim, typename... Args>
void reducesum(graph &g, tensor<double> &output, tensor<double> &input,
               first_dim n, Args... args) {
  arg_ptr = new struct arg_list;
  arg_ptr->value = n;

  if (!arg_head) {

    arg_head = arg_ptr_prev = arg_ptr;
    arg_head->next = NULL;
  } else {
    arg_ptr->next = NULL;
    arg_ptr_prev->next = arg_ptr;
    arg_ptr_prev = arg_ptr;
  }
  reducesum(g, output, input, args...);
}

template <typename... Args>
void reducesum(graph &g, tensor<double> &output, tensor<double> &input,
               Args... args) {
  arg_head = NULL;
  reducesum(g, output, input, args...);
}

void mean(graph &g, tensor<double> &output, tensor<double> &input,
          const unsigned n) {
  output = input.mean(g, n);
}

} // namespace tf