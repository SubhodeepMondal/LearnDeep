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

  void removeGraph(graph *g) {
    graph_list.erase(std::remove(graph_list.begin(), graph_list.end(), g),
                     graph_list.end());
  }

  bool isThereActiveSession();

  graph *findActivateSession();
} graph_manager;

extern graph_manager g_manager;

typedef struct tensor {
  void *ptr{nullptr};
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

  void operator=(graph &g);

  graph &add(graph &g, tensor &input_b);

  graph &mean(graph &g, unsigned dim);

  graph &mul(graph &g, tensor &input_b);

  graph &matmul(graph &g, tensor &input_b);

  graph &scale(graph &g, std::float64_t scaleFactor);

  graph &pow(graph &g, unsigned exponent);

  tensor matmul(tensor &input_b);

  tensor add(tensor &input_b);

  tensor getReduction(std::vector<unsigned> reduction_dims);

  template <typename... Args> tensor reducesum(Args... args) {
    std::vector<unsigned> dimensions;
    bool flag = true;

    // Add dimensions to the vector
    addDimensions(dimensions, args...);

    unsigned *reduction_dims = new unsigned[dimensions.size()];
    for (int i = 0; i < dimensions.size(); i++) {
      reduction_dims[i] = dimensions[i];
    }

    delete[] reduction_dims;

    return getReduction(dimensions);
  }

  tensor operator*(tensor &input_b);

  tensor scale(const std::float64_t scaleFactor);

  tensor pow(const unsigned exponent);

  tensor mean(const unsigned dim);

  // --- Default constructor
  tensor() = default;

  // --- Destructor
  ~tensor() { /*destory();*/ }

  // --- Copy constructor
  tensor(const tensor &other) {
    dt_type = other.dt_type;
    if (other.ptr) {
      auto *src = static_cast<Tensor<std::float64_t> *>(other.ptr);
      ptr = new Tensor<std::float64_t>(*src); // deep copy
    }
  }

  // --- Copy assignment
  tensor &operator=(const tensor &other) {
    if (this != &other) {
      // destory();
      dt_type = other.dt_type;
      if (other.ptr) {
        auto *src = static_cast<Tensor<std::float64_t> *>(other.ptr);
        ptr = new Tensor<std::float64_t>(*src); // deep copy
      } else {
        ptr = nullptr;
      }
    }
    return *this;
  }

  // --- Move constructor
  tensor(tensor &&other) noexcept {
    dt_type = other.dt_type;
    ptr = other.ptr;
    other.ptr = nullptr;
  }

  // --- Move assignment
  tensor &operator=(tensor &&other) noexcept {
    if (this != &other) {
      // destory();
      dt_type = other.dt_type;
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }

  // --- Destroy function
  void destory() {
    if (ptr) {
      static_cast<Tensor<std::float64_t> *>(ptr)->destroy();
      // delete static_cast<Tensor<std::float64_t> *>(ptr);
      ptr = nullptr;
    }
  }

  graph &getReductionGraph(graph &g, std::vector<unsigned> reduction_dims,
                           bool &flag);

  template <typename... Args> graph &reducesum(graph &g, Args... args) {
    std::vector<unsigned> dimensions;
    bool flag = true;

    // Add dimensions to the vector
    addDimensions(dimensions, args...);

    unsigned *reduction_dims = new unsigned[dimensions.size()];
    for (int i = 0; i < dimensions.size(); i++) {
      reduction_dims[i] = dimensions[i];
    }
    delete[] reduction_dims;

    return getReductionGraph(g, dimensions, flag);
  }

  std::float64_t *getPtr() {
    return static_cast<Tensor<std::float64_t> *>(this->ptr)->getData();
  }
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

  void graph_execute();

  void graph_travarse_data_node();

  void graph_clear();

} graph;

struct arg_list {
  unsigned value;
  struct arg_list *next;
};

} // namespace tf

#endif // TENSOR_MAIN_API