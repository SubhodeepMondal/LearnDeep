#ifndef MATH_LIBRARY
#define MATH_LIBRARY

#include <algorithm>
#include <iostream>
#include <vector>

#include "NDynamicArray.h"
#include <LAS/CPULibrary.h>
#include <graph/graph_framework.hpp>
#include <kernel/opskernel.h>

template <typename T> class Tensor : public ndarray<T> {

public:
  Tensor() = default;

  // template <typename... args>
  // Tensor(unsigned num, args... Args) : ndarray<T>(num, Args...) {}
  Tensor(unsigned n, const unsigned *arr, DataType data_type)
      : ndarray<T>(n, arr, data_type) {}

  // copy constructor
  Tensor(const Tensor<T> &ndmath) : ndarray<T>(ndmath) {}

  // copy assignment (deep copy)
  Tensor<T> &operator=(const Tensor<T> &ndmath) {
    ndarray<T>::operator=(ndmath);
    return *this;
  }

  // move constructor (steal resources)
  Tensor(Tensor<T> &&ndmath) noexcept : ndarray<T>(std::move(ndmath)){};

  // move assignment (steal resources)
  Tensor<T> &operator=(Tensor<T> &&ndmath) noexcept {
    ndarray<T>::operator=(std::move(ndmath));
    return *this;
  }

  // destructor
  ~Tensor() {
    // // std::cout << "Destructor called for Tensor object: " << "\n";
    // this->~ndarray();
  }

  void assign(Ops *ops) { ops->initilizeoutput(this); }

  /// @brief This function calculates matrix multiplication A X B
  /// @tparam None
  /// @param input_B_matrix Tensor class, expects Tensor or similar y dimension
  /// that of current Tensor.
  /// @return returns Tensor class type.
  Tensor<T> *matmul(Tensor<T> &);

  /// @brief This function does element wise multiplication with pass costant
  /// value
  /// @tparam None
  /// @param const value std::float64_t
  /// @return returns Tensor class type.
  Tensor<T> scale(const std::float64_t);

  /// @brief This function calculates matrix addition A + B
  /// @tparam None
  /// @param input_B_matrix Tensor class, expects Tensor or similar rank and
  /// dimension that of current Tensor.
  /// @return returns Tensor class type.
  Tensor<T> *add(Tensor<T> &);

  /// @brief This function calculates matrix multiplication A X B
  /// @tparam None
  /// @param input_B_matrix Tensor class, expects Tensor or similar y dimension
  /// that of current Tensor.
  /// @return returns Tensor class type.
  Tensor<T> *operator*(Tensor<T> &);

  Tensor<T> *mul(Tensor<T> &);

  /// @brief This function calculates matrix addition A + B
  /// @tparam None
  /// @param input_B_matrix Tensor class, expects Tensor or similar rank and
  /// dimension that of current Tensor.
  /// @return returns Tensor class type.
  Tensor<T> operator+(const Tensor<T>);

  /// @brief This function calculates matrix addition A + B
  /// @tparam None
  /// @param input_B_matrix Tensor class, expects Tensor or similar rank and
  /// dimension that of current Tensor.
  /// @return returns Tensor class type.
  Tensor<T> operator-(const Tensor<T>);

  Tensor<T> vectoradd(const Tensor<T>);

  /// @brief This function transposes current Tensor
  /// @tparam None
  /// @return void, output (call by referance)
  void transpose();

  /// @brief This function reduction sum operation along the reduction
  /// dimension.
  /// @tparam None
  /// @param reduction_dimension Unsigned, Along which dimension the Tensor
  /// needs to be reduced
  /// @return returns Tensor class type.
  Tensor<T> *reducesum(std::vector<unsigned>);

  /// @brief This function calculates matrix addition A + B
  /// @tparam None
  /// @param input_B_matrix Tensor class, expects Tensor or similar rank and
  /// dimension that of current Tensor.
  /// @return returns Tensor class type.
  Tensor<T> pow(unsigned);

  Ops *add(Tensor<T> &input, bool &flag);

  Ops *mul(Tensor<T> &input, bool &flag);

  Ops *matmul(Tensor<T> &input, bool &flag);

  Ops *reducesum(std::vector<unsigned> n, bool &flag);
};

// template class Tensor<std::float64_t>;
#endif // MATH_LIBRARY