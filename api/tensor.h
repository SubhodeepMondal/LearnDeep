#ifndef TENSOR_MAIN_API
#define TENSOR_MAIN_API

#include <algorithm>
#include <cstddef>
#include <framework/MathLibrary.h>
#include <graph/graph_framework.hpp>
#include <iostream>
#include <iterator>
#include <vector>

namespace tf {
struct graph;

static std::vector<void *> tensor_nodes;

typedef struct tensor {
  void *ptr{nullptr};
  bool isNodeCleared = false;
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

    switch (d_type) {
    case tf_float64:
      this->ptr = new Tensor<std::float64_t>(dimensions.size(),
                                             dimensions.data(), d_type);
      break;
    default:
      ptr = nullptr;
    }
    dt_type = d_type;
    assign_ptr();
  }

  void assign_ptr();

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

  graph &pow(graph &g, unsigned exponent);

  graph &relu(graph &g);

  graph &sigmoid(graph &g);

  graph &scale(graph &g, std::float64_t scaleFactor);

  graph &sqrt(graph &g);

  graph &sub(graph &g, tensor &input_b);

  tensor add(tensor &input_b);

  tensor matmul(tensor &input_b);

  tensor pow(const unsigned exponent);

  tensor relu();

  tensor mean(const unsigned dim);

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

  tensor operator+(tensor &input_b);

  tensor operator*(tensor &input_b);

  tensor sigmoid();

  tensor scale(const std::float64_t scaleFactor);

  tensor sqrt();

  tensor sub(tensor &input_b);

  // --- Default constructor
  tensor();

  // --- Destructor
  ~tensor() {
    if (!isNodeCleared) {
      isNodeCleared = true;
      destory();
    }
  }

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
      dt_type = other.dt_type;
      if (other.ptr) {
        *static_cast<Tensor<std::float64_t> *>(this->ptr) =
            *static_cast<Tensor<std::float64_t> *>(other.ptr);
        // this->ptr = other.ptr;
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
      destory();
      dt_type = other.dt_type;
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }

  // --- Destroy function
  void destory();

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
  bool isGraphCleared = bool(false);
  tensor *input_a = nullptr;
  tensor *input_b = nullptr;
  tensor *output = nullptr;
  Ops *ops;
  void tf_create_graph();

  void graph_execute();

  void graph_travarse_data_node();

  void graph_clear();

} graph;

} // namespace tf

#endif // TENSOR_MAIN_API