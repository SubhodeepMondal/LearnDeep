#include "tensor.h"

struct tf::arg_list *arg_head;
struct tf::arg_list *arg_ptr;
struct tf::arg_list *arg_ptr_prev;

dim_node *tf::head;
dim_node *tf::ptr;
dim_node *tf::prev;

void tf::addDimensions(unsigned *nDim, unsigned w) {
  dim_node *ptr_new = new dim_node;

  ptr->next = ptr_new;
  ptr_new->value = w;
  ptr_new->next = NULL;
  ptr = ptr->next;
  (*nDim)++;
}

template <typename... args>
void tf::addDimensions(unsigned *nDim, unsigned w, args... Args) {
  addDimensions(nDim, w);
  addDimensions(nDim, Args...);
}

void tf::tf_create_graph(graph &g) { g = new Graph(); }

void tf::tensor_of(tensor &input, double low_limit, double upper_limit) {

  switch (input.dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(input.ptr)->initRandData(low_limit,
                                                                   upper_limit);
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void tf::tensor_of(tensor &input, std::float64_t *data) {
  switch (input.dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(input.ptr)->initData(data);
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void tf::print_data(tensor &input) {
  switch (input.dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(input.ptr)->printData();
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void tf::print_dimension(tensor &input) {
  switch (input.dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(input.ptr)->printDimensions();
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void tf::matmul(graph &g, tensor &output, tensor &input_a, tensor &input_b) {
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

void tf::mul(graph &g, tensor &output, tensor &input_a, tensor &input_b) {
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

void tf::add(graph &g, tensor &output, tensor &input_a, tensor &input_b) {
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

void tf::pow(graph &g, tensor &output, tensor &input, unsigned power) {
  if ((input.dt_type == output.dt_type)) {
    switch (output.dt_type) {
    case tf_float64:
      static_cast<Tensor<std::float64_t> *>(output.ptr)
          ->assign(static_cast<Tensor<std::float64_t> *>(input.ptr)->pow(
              *(static_cast<Graph *>(g)), power));
      break;
    }
  }
}

void tf::reducesum(graph &g, tensor &output, tensor &input) {

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
void tf::reducesum(graph &g, tensor &output, tensor &input, first_dim n,
                   Args... args) {
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

template <typename... Args>
void tf::reducesum(graph &g, tensor &output, tensor &input, Args... args) {
  arg_head = NULL;
  reducesum(g, output, input, args...);
}

void tf::mean(graph &g, tensor &output, tensor &input,
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

void tf::scale(graph &g, tensor &output, tensor &input,
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

void tf::graph_optimize(graph &g) { static_cast<Graph *>(g)->optimize(); }

void tf::graph_execute(graph &g) { static_cast<Graph *>(g)->execute(); }

void tf::graph_travarse_node(graph &g) {
  static_cast<Graph *>(g)->traversenode();
}