#ifndef MATH_LIBRARY
#define MATH_LIBRARY

#include <iostream>

#include "../LAS/CPULibrary.h"
#include "../graph/graph.h"
#include "../kernel/opskernel.h"
#include "NDynamicArray.h"

template <typename T> class Tensor : public ndarray<T> {
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
  /// @param input Tensor class type, expects operend B.
  /// @param output reference of Tensor class type, expects output C.
  /// @param kernel function pointer, expects pointer to the function for the
  /// matrix operation.
  /// @param function_name String, expects the name of the matrix operation to
  /// be performed.
  /// @return void, output (call by referance)
  void recursive_iterator(unsigned, unsigned *, Tensor<T>, Tensor<T> &,
                          void (*)(T**, unsigned *), std::string,
                          unsigned *, T*, Tensor<T>);

  /// @brief This function recursively iterates from (n-1)th dim to  3th
  /// dimension and performs reduction sum operation along the reduction
  /// dimension.
  /// @tparam None
  /// @param index Unsigned, rank of the matrix n, value passed n-1,
  /// @param dimension_arr Unsigend*, used to store internal higher dimention
  /// co-ordinate, expects unsigned pointer of size n(i.e dimension),
  /// @param input Tensor class type, expects operend B.
  /// @param output reference of Tensor class type, expects output C.
  /// @param reduction_dimension Unsigned, Along which dimension the Tensor
  /// needs to be reduced,
  /// @param data_pointer data pointer, expects pointer to the array, which
  /// holds .
  /// @param output unsigned pointer* optional parameter.
  /// @param dl_arr Tpointer*, optional parameter,
  /// @param misc_arr Tensor obj, optional parameter.
  /// @return void, output (call by referance)
  void recursive_sum(unsigned, unsigned *, Tensor<T>, Tensor<T> &, unsigned,
                     T *);

  void reducesum(Tensor<T> &);

  template <typename first_dim, typename... Args>
  void reducesum(Tensor<T> &, first_dim, Args...);

public:
  Tensor(const Tensor<T> &ndmath) : ndarray<T>(ndmath) {}

  // template <typename... args>
  // Tensor(unsigned num, args... Args) : ndarray<T>(num, Args...) {}
  Tensor(unsigned n, const unsigned *arr, DataType data_type) : ndarray<T>(n,arr,data_type){}

  Tensor() {}

  ~Tensor() {}

  Tensor<T> &operator=(const Tensor<T> &ndmath) {
    ndarray<T>::operator=(ndmath);
    return *this;
  }

  void assign(Graph g) { g.outputnode(this); }

  /// @brief This function calculates matrix multiplication A X B
  /// @tparam None
  /// @param input_B_matrix Tensor class, expects Tensor or similar y dimension
  /// that of current Tensor.
  /// @return returns Tensor class type.
  Tensor<T> matmul(const Tensor<T>);

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
  Tensor<T> add(const Tensor<T>);

  /// @brief This function calculates matrix multiplication A X B
  /// @tparam None
  /// @param input_B_matrix Tensor class, expects Tensor or similar y dimension
  /// that of current Tensor.
  /// @return returns Tensor class type.
  Tensor<T> operator*(const Tensor<T>);

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
  template <typename... Args> Tensor<T> reducesum(Args... args);

  /// @brief This function calculates matrix addition A + B
  /// @tparam None
  /// @param input_B_matrix Tensor class, expects Tensor or similar rank and
  /// dimension that of current Tensor.
  /// @return returns Tensor class type.
  Tensor<T> pow(unsigned);

  Graph mul(Graph &g, Tensor<T> &input);

  Graph add(Graph &g, Tensor<T> &input);

  Graph matmul(Graph &g, Tensor<T> &input);

  Graph pow(Graph &g, unsigned power);

  Graph reducesum(Graph &, unsigned,  unsigned*);

  void reducesum(Graph &g, Ops *ops);

  template <typename first_dim, typename... Args>
  void reducesum(Graph &g, Ops *ops, first_dim n, Args... args);

  template <typename... Args> Graph reducesum(Graph &g, Args... args);

  Graph scale(Graph &g, const std::float64_t scale_factor);

  Graph mean(Graph &g, const unsigned n);
};

// template class Tensor<std::float64_t>;
#endif // MATH_LIBRARY