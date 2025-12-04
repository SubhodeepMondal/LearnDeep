#ifndef TENSOR_MAIN_API
#define TENSOR_MAIN_API

// C++ Headers
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <vector>

// Library Headers
#include <framework/MathLibrary.h>
#include <graph/graph_context.hpp>
#include <kernel/opskernel.h>

namespace tf {

typedef struct tensor {
private:
  std::vector<Ops *> opsPtr;

public:
  bool activateGraphSession;
  void *ptr{nullptr};
  DataType dt_type;

  // --- Default constructor
  tensor();

  // --- Overloaded constructor
  tensor(DataType dt_type, Tensor<std::float64_t> *ptr);

  // --- Destructor
  ~tensor();

  // --- Copy constructor
  tensor(const tensor &other);

  // --- Copy assignment
  tensor &operator=(const tensor &other);

  // --- Move constructor
  tensor(tensor &&other) noexcept;

  // --- Move assignment
  tensor &operator=(tensor &&other) noexcept;

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

    this->dt_type = d_type;
    assign_pointer(dimensions);
  }

  void assign_ptr(std::vector<unsigned> dimensions);

  unsigned getNoOfDimensions();

  const unsigned *getDimensions();

  unsigned getNoOfElem();
  void tensor_of(double low_limit, double upper_limit);

  void tensor_of(std::float64_t *data);

  void print_data();

  void print_dimension();

  void assign_pointer(std::vector<unsigned> dimensions);

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

  std::float64_t *getPtr() {
    return static_cast<Tensor<std::float64_t> *>(this->ptr)->getData();
  }
} tensor;

static std::vector<tensor *> tensor_nodes;

typedef struct graph_context {
private:
  GraphContext *graph_ctx;

public:
  graph_context();

  ~graph_context();

  void run();

  tensor get_gradient(const tensor &a);

  void initialize_gradient();

  void compute_gradient();
} graph_context;

} // namespace tf

#endif // TENSOR_MAIN_API