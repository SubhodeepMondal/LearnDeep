#include "../core/framework/MathLibrary.h"
#include "../core/graph/graph.h"
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

void tf_create_graph(graph &g) { g = new Graph(); }

void tensor_of(tensor &input, double low_limit, double upper_limit) {

  switch (input.dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(input.ptr)->initRandData(low_limit,
                                                                   upper_limit);
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void print_data(tensor &input) {
  switch (input.dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(input.ptr)->printData();
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void print_dimension(tensor &input) {
  switch (input.dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(input.ptr)->printDimensions();
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void matmul(graph &g, tensor &output, tensor &input_a, tensor &input_b) {
  if ((input_a.dt_type == input_b.dt_type) &&
      (input_a.dt_type == output.dt_type)) {
    switch (output.dt_type) {
    case tf_float64:
      static_cast<Tensor<std::float64_t> *>(output.ptr)
          ->assign(static_cast<Tensor<std::float64_t> *>(input_a.ptr)
                       ->matmul(*(static_cast<Graph *>(g)),
                                *(static_cast<Tensor<std::float64_t> *>(
                                    input_b.ptr))));
      break;
    }
  }
}

void mul(graph &g, tensor &output, tensor &input_a, tensor &input_b) {
  if ((input_a.dt_type == input_b.dt_type) &&
      (input_a.dt_type == output.dt_type)) {
    switch (output.dt_type) {
    case tf_float64:
      static_cast<Tensor<std::float64_t> *>(output.ptr)
          ->assign(
              static_cast<Tensor<std::float64_t> *>(input_a.ptr)
                  ->mul(*(static_cast<Graph *>(g)),
                        *(static_cast<Tensor<std::float64_t> *>(input_b.ptr))));
    }
  }
}

void add(graph &g, tensor &output, tensor &input_a, tensor &input_b) {
  if ((input_a.dt_type == input_b.dt_type) &&
      (input_a.dt_type == output.dt_type)) {
    switch (output.dt_type) {
    case tf_float64:
      static_cast<Tensor<std::float64_t> *>(output.ptr)
          ->assign(
              static_cast<Tensor<std::float64_t> *>(input_a.ptr)
                  ->add(*(static_cast<Graph *>(g)),
                        *(static_cast<Tensor<std::float64_t> *>(input_b.ptr))));
    }
  }
}

void pow(graph &g, tensor &output, tensor &input, unsigned power) {
  if ((input.dt_type == output.dt_type)) {
    switch (output.dt_type) {
    case tf_float64:
      static_cast<Tensor<std::float64_t> *>(output.ptr)
          ->assign(static_cast<Tensor<std::float64_t> *>(input.ptr)->pow(
              *(static_cast<Graph *>(g)), power));
    }
  }
}

void scale(graph &g, tensor &output, tensor &input,
           std::float64_t scale_factor) {
  if ((input.dt_type == output.dt_type)) {
    switch (output.dt_type) {
    case tf_float64:
      static_cast<Tensor<std::float64_t> *>(output.ptr)
          ->assign(static_cast<Tensor<std::float64_t> *>(input.ptr)->scale(
              *(static_cast<Graph *>(g)), scale_factor));
    }
  }
}

void reducesum(graph &g, tensor &output, tensor &input) {

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

  if ((input.dt_type == output.dt_type)) {
    switch (output.dt_type) {
    case tf_float64:
      static_cast<Tensor<std::float64_t> *>(output.ptr)
          ->assign(static_cast<Tensor<std::float64_t> *>(input.ptr)->reducesum(
              *(static_cast<Graph *>(g)), count, reduction_dims));
    }
  }
}

template <typename first_dim, typename... Args>
void reducesum(graph &g, tensor &output, tensor &input, first_dim n,
               Args... args) {
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
void reducesum(graph &g, tensor &output, tensor &input, Args... args) {
  arg_head = NULL;
  reducesum(g, output, input, args...);
}

void mean(graph &g, tensor &output, tensor &input,
  const unsigned n) {
    if ((input.dt_type == output.dt_type)) {
      switch (output.dt_type) {
      case tf_float64:
        static_cast<Tensor<std::float64_t> *>(output.ptr)
            ->assign(static_cast<Tensor<std::float64_t> *>(input.ptr)->mean(
                *(static_cast<Graph *>(g)), n));
      }
    }
}

void graph_optimize(graph &g) { static_cast<Graph *>(g)->optimize(); }

void graph_execute(graph &g) { static_cast<Graph *>(g)->execute(); }

void graph_travarse_node(graph &g) { static_cast<Graph *>(g)->traversenode(); }

} // namespace tf