#include "tensor.h"

struct tf::arg_list *arg_head;
struct tf::arg_list *arg_ptr;
struct tf::arg_list *arg_ptr_prev;

tf::graph_manager tf::g_manager;

bool tf::graph_manager::isThereActiveSession() {
  return std::any_of(graph_list.begin(), graph_list.end(),
                     [](graph *g) { return g->isSessionActive; });
}

tf::graph *tf::graph_manager::findActivateSession() {
  auto it = std::find_if(graph_list.begin(), graph_list.end(),
                         [](graph *g) { return g->isSessionActive; });
  return *it;
}

void tf::tensor::tensor_of(double low_limit, double upper_limit) {

  switch (dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(ptr)->initRandData(low_limit,
                                                             upper_limit);
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void tf::tensor::tensor_of(std::float64_t *data) {
  switch (dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(ptr)->initData(data);
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void tf::tensor::print_data() {
  switch (dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(ptr)->printData();
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void tf::tensor::print_dimension() {
  switch (dt_type) {
  case tf_float64:
    static_cast<Tensor<std::float64_t> *>(ptr)->printDimensions();
    break;
  default:
    std::cout << "Invalid data type!";
  }
}

void tf::tensor::operator=(graph &g) {
  if (g.input_a.dt_type == this->dt_type) {
    switch (this->dt_type) {
    case tf_float64:
      static_cast<Tensor<std::float64_t> *>(this->ptr)->assign(g.ops);
      static_cast<Graph *>(g.ptr)->addNode(
          static_cast<Tensor<std::float64_t> *>(this->ptr));
      static_cast<Graph *>(g.ptr)->addEdge(
          g.ops, static_cast<Tensor<std::float64_t> *>(this->ptr));

      break;
    default:
      std::cout << "Invalid data type!";
    }
  }
}

tf::graph &tf::tensor::add(tensor &input_b) {
  graph *g = nullptr;
  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      bool flag = true;

      if (g_manager.isThereActiveSession()) { // search for any activated graph
                                              // session
        g = g_manager.findActivateSession();
        g->input_a = *this;
        g->input_b = input_b;
        if (g) {
          std::cout << "Adding tensor::addition to the active graph session.\n";
          Ops *ops = static_cast<Tensor<std::float64_t> *>(this->ptr)->add(
              *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)), flag);

          g->ops = ops;

          static_cast<Graph *>(g->ptr)->addNode(
              static_cast<Tensor<std::float64_t> *>(this->ptr));
          static_cast<Graph *>(g->ptr)->addNode(
              static_cast<Tensor<std::float64_t> *>(input_b.ptr));
          static_cast<Graph *>(g->ptr)->addNode(ops);

          static_cast<Graph *>(g->ptr)->addEdge(
              static_cast<Tensor<std::float64_t> *>(this->ptr), ops);
          static_cast<Graph *>(g->ptr)->addEdge(
              static_cast<Tensor<std::float64_t> *>(input_b.ptr), ops);
        } else {
          std::cerr << "No active graph session found.\n";
        }
      } else {
        std::cerr << "No active graph session found.\n";
      }
      break;
    }
    }
  }
  return *g;
}

tf::graph &tf::tensor::mul(tensor &input_b) {
  graph *g = nullptr;
  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      bool flag = true;

      if (g_manager.isThereActiveSession()) { // search for any activated graph
                                              // session
        g = g_manager.findActivateSession();
        g->input_a = *this;
        g->input_b = input_b;
        if (g) {
          std::cout << "Adding tensor::addition to the active graph session.\n";
          Ops *ops = static_cast<Tensor<std::float64_t> *>(this->ptr)->mul(
              *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)), flag);

          g->ops = ops;

          static_cast<Graph *>(g->ptr)->addNode(
              static_cast<Tensor<std::float64_t> *>(this->ptr));
          static_cast<Graph *>(g->ptr)->addNode(
              static_cast<Tensor<std::float64_t> *>(input_b.ptr));
          static_cast<Graph *>(g->ptr)->addNode(ops);

          static_cast<Graph *>(g->ptr)->addEdge(
              static_cast<Tensor<std::float64_t> *>(this->ptr), ops);
          static_cast<Graph *>(g->ptr)->addEdge(
              static_cast<Tensor<std::float64_t> *>(input_b.ptr), ops);
        } else {
          std::cerr << "No active graph session found.\n";
        }
      } else {
        std::cerr << "No active graph session found.\n";
      }
      break;
    }
    }
  }
  return *g;
}

tf::graph &tf::tensor::matmul(tensor &input_b) {
  graph *g = nullptr;
  if (this->dt_type == input_b.dt_type) {
    switch (dt_type) {
    case tf_float64: {
      bool flag = true;

      if (g_manager.isThereActiveSession()) { // search for any activated graph
                                              // session
        g = g_manager.findActivateSession();
        g->input_a = *this;
        g->input_b = input_b;
        if (g) {
          std::cout << "Adding tensor::addition to the active graph session.\n";
          Ops *ops = static_cast<Tensor<std::float64_t> *>(this->ptr)->matmul(
              *(static_cast<Tensor<std::float64_t> *>(input_b.ptr)), flag);

          g->ops = ops;

          static_cast<Graph *>(g->ptr)->addNode(
              static_cast<Tensor<std::float64_t> *>(this->ptr));
          static_cast<Graph *>(g->ptr)->addNode(
              static_cast<Tensor<std::float64_t> *>(input_b.ptr));
          static_cast<Graph *>(g->ptr)->addNode(ops);

          static_cast<Graph *>(g->ptr)->addEdge(
              static_cast<Tensor<std::float64_t> *>(this->ptr), ops);
          static_cast<Graph *>(g->ptr)->addEdge(
              static_cast<Tensor<std::float64_t> *>(input_b.ptr), ops);
        } else {
          std::cerr << "No active graph session found.\n";
        }
      } else {
        std::cerr << "No active graph session found.\n";
      }
      break;
    }
    }
  }
  return *g;
}

tf::graph &tf::tensor::getReductionGraph(std::vector<unsigned> reduction_dims,
                                         bool &flag) {
  graph *g = nullptr;
  switch (dt_type) {
  case tf_float64:
    if (g_manager.isThereActiveSession()) { // search for any activated graph
      g = g_manager.findActivateSession();
      g->input_a = *this;
      if (g) {
        std::cout << "Adding tensor::reduction sum to the active graph "
                     "session.\n";
        Ops *ops = static_cast<Tensor<std::float64_t> *>(g->input_a.ptr)
                       ->reducesum(reduction_dims, flag);

        g->ops = ops;

        static_cast<Graph *>(g->ptr)->addNode(
            static_cast<Tensor<std::float64_t> *>(this->ptr));
        static_cast<Graph *>(g->ptr)->addNode(ops);

        static_cast<Graph *>(g->ptr)->addEdge(
            static_cast<Tensor<std::float64_t> *>(this->ptr), ops);
      } else {
        std::cerr << "No active graph session found.\n";
      }
    } else {
      std::cerr << "No active graph session found.\n";
    }
    break;

  default:
    break;
  }

  return *g;
}
// void tf::reducesum(graph &g, tensor &output, tensor &input) {

//   unsigned count = 0;
//   unsigned *reduction_dims;
//   unsigned *dims;

//   arg_ptr = arg_ptr_prev = arg_head;

//   while (arg_ptr) {
//     count++;
//     arg_ptr = arg_ptr->next;
//   }
//   std::cout << "count: " << count << "\n";
//   reduction_dims = new unsigned[count];
//   arg_ptr = arg_head;

//   for (unsigned i = 0; i < count; i++) {
//     reduction_dims[i] = arg_ptr->value;
//     arg_ptr = arg_ptr->next;
//     delete[] arg_ptr_prev;
//     arg_ptr_prev = arg_ptr;
//   }

//   if ((input.dt_type == output.dt_type)) {
//     switch (output.dt_type) {
//     case tf_float64:
//       static_cast<Tensor<std::float64_t> *>(output.ptr)
//           ->assign(static_cast<Tensor<std::float64_t>
//           *>(input.ptr)->reducesum(
//               *(static_cast<Graph *>(g)), count, reduction_dims));
//     }
//   }
// }

// template <typename first_dim, typename... Args>
// void tf::reducesum(graph &g, tensor &output, tensor &input, first_dim n,
//                    Args... args) {
//   unsigned count = 0;
//   unsigned *reduction_dims;
//   unsigned *dims;

//   arg_ptr = arg_ptr_prev = arg_head;

//   while (arg_ptr) {
//     count++;
//     arg_ptr = arg_ptr->next;
//   }
//   std::cout << "count: " << count << "\n";
//   reduction_dims = new unsigned[count];
//   arg_ptr = arg_head;

//   for (unsigned i = 0; i < count; i++) {
//     reduction_dims[i] = arg_ptr->value;
//     arg_ptr = arg_ptr->next;
//     delete[] arg_ptr_prev;
//     arg_ptr_prev = arg_ptr;
//   }

//   if ((input.dt_type == output.dt_type)) {
//     switch (output.dt_type) {
//     case tf_float64:
//       static_cast<Tensor<std::float64_t> *>(output.ptr)
//           ->assign(static_cast<Tensor<std::float64_t>
//           *>(input.ptr)->reducesum(
//               *(static_cast<Graph *>(g)), count, reduction_dims));
//     }
//   }
// }

// template <typename... Args>
// void tf::reducesum(graph &g, tensor &output, tensor &input, Args... args) {
//   arg_head = NULL;
//   reducesum(g, output, input, args...);
// }

// void tf::mean(graph &g, tensor &output, tensor &input, const unsigned n) {
//   if ((input.dt_type == output.dt_type)) {
//     switch (output.dt_type) {
//     case tf_float64:
//       static_cast<Tensor<std::float64_t> *>(output.ptr)
//           ->assign(static_cast<Tensor<std::float64_t> *>(input.ptr)->mean(
//               *(static_cast<Graph *>(g)), n));
//     }
//   }
// }

// void tf::scale(graph &g, tensor &output, tensor &input,
//                std::float64_t scale_factor) {
//   if ((input.dt_type == output.dt_type)) {
//     switch (output.dt_type) {
//     case tf_float64:
//       static_cast<Tensor<std::float64_t> *>(output.ptr)
//           ->assign(static_cast<Tensor<std::float64_t> *>(input.ptr)->scale(
//               *(static_cast<Graph *>(g)), scale_factor));
//     }
//   }
// }

void tf::graph::tf_create_graph() {
  ptr = new Graph();
  g_manager.addGraph(this);
}

void tf::graph::graph_start_recording_session() {
  if (g_manager.isThereActiveSession()) {
    std::cout << "A session is already active. Cannot start a new session.\n";
    return;
  }
  this->isSessionActive = true;
  std::cout << "Starting a new recording session.\n";
}

void tf::graph::graph_end_recording_session() {
  if (g_manager.isThereActiveSession()) {
    std::cout << "Ending the active session.\n";
    this->isSessionActive = false;
    g_manager.removeGraph(this);
  }
}

// void tf::graph::graph_optimize(graph &g) {}

void tf::graph::graph_execute() { static_cast<Graph *>(this->ptr)->compute(); }

void tf::graph::graph_travarse_data_node() {
  static_cast<Graph *>(this->ptr)->traverse();
}

void tf::graph::graph_clear() {
  if (ptr) {
    delete static_cast<Graph *>(ptr);
    ptr = nullptr;
  }
  g_manager.removeGraph(this);
  isSessionActive = false;
  input_a.ptr = nullptr;
  input_b.ptr = nullptr;
  ops = nullptr;
  std::cout << "Graph cleared and session ended.\n";
}