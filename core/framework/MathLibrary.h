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

  void assign(Ops *ops) { ops->initializeoutput(this); }

  Tensor<T> *add(Tensor<T> &);

  Tensor<T> *matmul(Tensor<T> &);

  Tensor<T> *operator*(Tensor<T> &);

  Tensor<T> *mul(Tensor<T> &);

  Tensor<T> operator+(const Tensor<T>);

  Tensor<T> operator-(const Tensor<T>);

  Tensor<T> vectoradd(const Tensor<T>);

  Tensor<T> *reducesum(std::vector<unsigned>);

  Tensor<T> *scale(const std::float64_t);

  Tensor<T> *sqrt();

  Tensor<T> *sub(Tensor<T> &);

  Tensor<T> *pow(unsigned);

  Tensor<T> *relu();

  Tensor<T> *sigmoid();

  Tensor<T> *softmax(const unsigned axis);

  Tensor<T> *mean(const unsigned);

  Tensor<T> *transpose();
};

// template class Tensor<std::float64_t>;
#endif // MATH_LIBRARY