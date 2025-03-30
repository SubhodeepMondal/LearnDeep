#ifndef MATH_LIBRARY
#define MATH_LIBRARY

#include <iostream>

#include "../LAS/CPULibrary.h"
#include "../graph/graph.h"
#include "../kernel/opskernel.h"
#include "NDynamicArray.h"

template <typename T> class tensor : public ndarray<T> {
  struct arg_list {
    unsigned value;
    struct arg_list *next;
  };

  struct arg_list *head, *ptr, *ptr_prev;

  function_names fx_name;
  unsigned *arr_dims;

  /// @brief This function recursively iterates from (n-1)th dim to  2th dim and
  /// performs matrix operation for all higher dimensions.
  /// @tparam None
  /// @param index Unsigned, rank of the matrix n, value passed n-1,
  /// @param dimension_arr Unsigend*, used to store internal higher dimention
  /// co-ordinate, expects unsigned pointer of size n(i.e dimension),
  /// @param input tensor class type, expects operend B.
  /// @param output reference of tensor class type, expects output C.
  /// @param kernel function pointer, expects pointer to the function for the
  /// matrix operation.
  /// @param function_name String, expects the name of the matrix operation to
  /// be performed.
  /// @return void, output (call by referance)
  void recursive_iterator(unsigned, unsigned *, tensor<T>, tensor<T> &,
                          void (*)(double **, unsigned *), std::string,
                          unsigned *, double *, tensor<T>);

  /// @brief This function recursively iterates from (n-1)th dim to  3th
  /// dimension and performs reduction sum operation along the reduction
  /// dimension.
  /// @tparam None
  /// @param index Unsigned, rank of the matrix n, value passed n-1,
  /// @param dimension_arr Unsigend*, used to store internal higher dimention
  /// co-ordinate, expects unsigned pointer of size n(i.e dimension),
  /// @param input tensor class type, expects operend B.
  /// @param output reference of tensor class type, expects output C.
  /// @param reduction_dimension Unsigned, Along which dimension the tensor
  /// needs to be reduced,
  /// @param data_pointer data pointer, expects pointer to the array, which
  /// holds .
  /// @param output unsigned pointer* optional parameter.
  /// @param dl_arr double pointer*, optional parameter,
  /// @param misc_arr tensor obj, optional parameter.
  /// @return void, output (call by referance)
  void recursive_sum(unsigned, unsigned *, tensor<T>, tensor<T> &, unsigned,
                     T *);

  void reducesum(tensor<T> &);

  template <typename first_dim, typename... Args>
  void reducesum(tensor<T> &, first_dim, Args...);

public:
  tensor(const tensor<T> &ndmath) : ndarray<T>(ndmath) {}

  // template <typename... args>
  // tensor(unsigned num, args... Args) : ndarray<T>(num, Args...) {}
  tensor(unsigned n, const unsigned *arr) : ndarray<T>(n,arr){}

  tensor() {}

  ~tensor() {}

  tensor<T> &operator=(const tensor<T> &ndmath) {
    ndarray<T>::operator=(ndmath);
    return *this;
  }

  void operator=(graph g) { g.outputnode(this); }

  /// @brief This function calculates matrix multiplication A X B
  /// @tparam None
  /// @param input_B_matrix tensor class, expects tensor or similar y dimension
  /// that of current tensor.
  /// @return returns tensor class type.
  tensor<T> matmul(const tensor<double>);

  /// @brief This function does element wise multiplication with pass costant
  /// value
  /// @tparam None
  /// @param const value double
  /// @return returns tensor class type.
  tensor<T> scale(const double);

  /// @brief This function calculates matrix addition A + B
  /// @tparam None
  /// @param input_B_matrix tensor class, expects tensor or similar rank and
  /// dimension that of current tensor.
  /// @return returns tensor class type.
  tensor<T> add(const tensor<double>);

  /// @brief This function calculates matrix multiplication A X B
  /// @tparam None
  /// @param input_B_matrix tensor class, expects tensor or similar y dimension
  /// that of current tensor.
  /// @return returns tensor class type.
  tensor<T> operator*(const tensor<double>);

  /// @brief This function calculates matrix addition A + B
  /// @tparam None
  /// @param input_B_matrix tensor class, expects tensor or similar rank and
  /// dimension that of current tensor.
  /// @return returns tensor class type.
  tensor<T> operator+(const tensor<double>);

  /// @brief This function calculates matrix addition A + B
  /// @tparam None
  /// @param input_B_matrix tensor class, expects tensor or similar rank and
  /// dimension that of current tensor.
  /// @return returns tensor class type.
  tensor<T> operator-(const tensor<double>);

  tensor<T> vectoradd(const tensor<double>);

  /// @brief This function transposes current tensor
  /// @tparam None
  /// @return void, output (call by referance)
  void transpose();

  /// @brief This function reduction sum operation along the reduction
  /// dimension.
  /// @tparam None
  /// @param reduction_dimension Unsigned, Along which dimension the tensor
  /// needs to be reduced
  /// @return returns tensor class type.
  template <typename... Args> tensor<T> reducesum(Args... args);

  /// @brief This function calculates matrix addition A + B
  /// @tparam None
  /// @param input_B_matrix tensor class, expects tensor or similar rank and
  /// dimension that of current tensor.
  /// @return returns tensor class type.
  tensor<T> pow(unsigned);

  graph mul(graph &g, tensor<T> &input);

  graph add(graph &g, tensor<T> &input);

  graph matmul(graph &g, tensor<T> &input);

  graph pow(graph &g, unsigned power);

  graph reducesum(graph &, unsigned,  unsigned*);

  void reducesum(graph &g, Ops *ops);

  template <typename first_dim, typename... Args>
  void reducesum(graph &g, Ops *ops, first_dim n, Args... args);

  template <typename... Args> graph reducesum(graph &g, Args... args);

  graph scale(graph &g, const double scale_factor);

  graph mean(graph &g, const unsigned n);
};

#endif // MATH_LIBRARY