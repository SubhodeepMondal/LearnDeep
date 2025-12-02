#ifndef TENSOR_MAIN_API
#define TENSOR_MAIN_API

#include <algorithm>
#include <cstddef>
#include <framework/MathLibrary.h>
#include <graph/graph_context.hpp>
#include <iostream>
#include <iterator>
#include <vector>

namespace tf {

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

  // ------- eager operations --------
  tensor operator+(tensor &input_b);

  tensor operator*(tensor &input_b);

  tensor add(tensor &input_b);

  tensor mean(const unsigned dim);

  tensor matmul(tensor &input_b);

  tensor mul(tensor &input_b);

  tensor pow(const unsigned exponent);

  tensor relu();

  tensor sigmoid();

  tensor scale(const std::float64_t scaleFactor);

  tensor sqrt();

  tensor sub(tensor &input_b);

  tensor transpose();

  tensor getReduction(std::vector<unsigned> reduction_dims);

  void gradient_required(bool is_grad_required);

  template <typename... Args> tensor reducesum(Args... args) {
    std::vector<unsigned> dimensions;
    bool flag = true;

    // -------- end of eager operations ---------

    // Add dimensions to the vector
    addDimensions(dimensions, args...);

    unsigned *reduction_dims = new unsigned[dimensions.size()];
    for (int i = 0; i < dimensions.size(); i++) {
      reduction_dims[i] = dimensions[i];
    }

    delete[] reduction_dims;

    return getReduction(dimensions);
  }

  // --- Default constructor
  tensor() { assign_ptr(); };

  // --- Destructor
  ~tensor() {
    if (!isNodeCleared) {
      destory();
      isNodeCleared = true;
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
      if (other.ptr) {
        if (this->ptr) {
          delete static_cast<Tensor<std::float64_t> *>(this->ptr);
          this->ptr = nullptr;
        }
        this->ptr = new Tensor<std::float64_t>(
            *static_cast<Tensor<std::float64_t> *>(other.ptr));
        this->dt_type = other.dt_type;
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
      if (this->ptr)
        delete static_cast<Tensor<std::float64_t> *>(this->ptr);
      this->dt_type = other.dt_type;
      this->ptr = other.ptr;
      other.ptr = nullptr;
      other.isNodeCleared = true;
      other.eraseRecord();
    }
    return *this;
  }

  // --- Destroy function
  void destory();
  void eraseRecord();

  std::float64_t *getPtr() {
    return static_cast<Tensor<std::float64_t> *>(this->ptr)->getData();
  }
} tensor;

static std::vector<tensor *> tensor_nodes;

typedef struct graph_context {
  void *graph_ctx;
  graph_context();

  ~graph_context();

  void run();

  tensor get_gradient(const tensor &a);

  void initialize_gradient();

  void compute_gradient();
} graph_context;

} // namespace tf

#endif // TENSOR_MAIN_API