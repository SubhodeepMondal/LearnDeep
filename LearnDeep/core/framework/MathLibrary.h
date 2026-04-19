#ifndef MATH_LIBRARY
#define MATH_LIBRARY

// C++ Headers
#include <algorithm>
#include <iostream>
#include <span>
#include <vector>

// Library Headers
#include "NDynamicArray.h"
#include <core/kernel/opskernel.h>

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
  ~Tensor() {}

  void assign(Ops *ops) { ops->initializeoutput(this); }

  Tensor<T> operator+(const Tensor<T>);

  Tensor<T> operator-(const Tensor<T>);

  Tensor<T> *operator*(Tensor<T> &input);

  Tensor<T> *add(Tensor<T> &input, bool graph_flag = true);

  Tensor<T> *matmul(Tensor<T> &input, bool graph_flag = true);

  Tensor<T> *mul(Tensor<T> &input, bool graph_flag = true);

  Tensor<T> vectoradd(const Tensor<T>);

  Tensor<T> *reducesum(std::vector<unsigned> n, bool graph_flag = true);

  Tensor<T> *scale(const std::float64_t scaleFactor, bool graph_flag = true);

  Tensor<T> *sqrt(bool flag = true);

  Tensor<T> *sub(Tensor<T> &input, bool graph_flag = true);

  Tensor<T> *pow(unsigned exponent, bool graph_flag = true);

  Tensor<T> *relu(bool flag = true);

  Tensor<T> *sigmoid(bool flag = true);

  Tensor<T> *softmax(const unsigned axis, bool graph_flag = true);

  Tensor<T> *mean(const unsigned dim, bool graph_flag = true);

  Tensor<T> *transpose(bool flag = true);
};

// template class Tensor<std::float64_t>;
#endif // MATH_LIBRARY