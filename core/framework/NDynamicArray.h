#ifndef NDYNAMIC_ARRAY
#define NDYNAMIC_ARRAY

#include <array>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <random>

#include "type.h"

template <typename T> class ndarray {
  std::string obj_name;
  DataType tensor_type;
  unsigned nDim, nElem;
  unsigned *dimension, *arr_dim;
  unsigned dim_iterator;
  bool isDiminitialized = false;
  bool isinitialized = false;
  bool is_grad_required = false;
  T *data;
  void *data_ptr;

public:
  // ----------------- Rule of 5 -------------------
  ndarray();

  ndarray(unsigned, const unsigned *, DataType);

  ndarray(unsigned, const unsigned *, DataType, bool);

  // copy constructor.
  ndarray(const ndarray<T> &ndarray);

  // Copy assignment (deep copy).
  ndarray<T> &operator=(const ndarray<T> &ndarray);

  // Move constructor (steal resources).
  ndarray(ndarray<T> &&ndarray) noexcept;

  // Move assignment (steal resources).
  ndarray<T> &operator=(ndarray<T> &&ndarray) noexcept;

  ~ndarray();

  /// @brief returns pointer to the first address of the tensor
  /// @tparam None
  /// @return returns pointer to the first address of the tensor.
  T *getData() const;

  DataType getDataType();

  void gradientRequired(bool is_gradient_required) {
    this->is_grad_required = is_gradient_required;
  }

  /// @brief returns unsigned array of dimension vector
  /// @tparam None
  /// @return returns unsigned array of dimension vector.
  const unsigned *getDimensions() const;

  /// @brief returns rank of the tensor.
  /// @tparam None
  /// @return returns unsigned array of dimension vector.
  unsigned getNoOfDimensions() const;

  /// @brief returns number of element for existing tensor
  /// @tparam None
  /// @return returns unsigned array of dimension vector.
  const unsigned getNoOfElem() const;

  DataType getType();

  void initData(T data);

  void initData(T *data);

  void initData(ndarray<T> incData);

  void initPartialData(unsigned index, unsigned n, T *data_source);

  void initRandData(double lower_limit, double upper_limit);

  void initPreinitializedData(T *Data);

  bool isGradRequired();

  void copyData(T *);

  void destroy();

  void printDimensions() const;

  void printData();

  void printLinearData();

  void printNoOfElements();

  void reshape(unsigned n, const unsigned *arr);

  void resetDimensions(unsigned, unsigned *);

  /// @brief sets name for the tensor
  /// @param str std::string type, assign this name to for the tensor
  void setObjName(std::string str);

  T operator()(unsigned);

  ndarray<T> &operator[](unsigned);
};

// #include "NDynamicArray.tpp"

#endif // NDYNAMIC_ARRAY