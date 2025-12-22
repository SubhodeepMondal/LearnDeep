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

  Tensor<T> *add(Tensor<T> &input);

  Tensor<T> *matmul(Tensor<T> &input);

  Tensor<T> *operator*(Tensor<T> &);

  Tensor<T> *mul(Tensor<T> &input);

  Tensor<T> operator+(const Tensor<T>);

  Tensor<T> operator-(const Tensor<T>);

  Tensor<T> vectoradd(const Tensor<T>);

  Tensor<T> *reducesum(std::vector<unsigned> n);

  Tensor<T> *scale(const std::float64_t scaleFactor);

  Tensor<T> *sqrt();

  Tensor<T> *sub(Tensor<T> &input);

  Tensor<T> *pow(unsigned exponent);

  Tensor<T> *relu();

  Tensor<T> *sigmoid();

  Tensor<T> *softmax(const unsigned axis);

  Tensor<T> *mean(const unsigned dim);

  Tensor<T> *transpose();
};

// template class Tensor<std::float64_t>;
#endif // MATH_LIBRARY