#pragma ONCE

#include <map>
#include <iostream>
#include "CPULibrary.h"
#include "Graph.h"
#include "NDynamicArray.h"
#include "Opskernel.h"

// typedef struct struct_function_name
// {
//     enum function_names
//     {
//         matrix_multiplication,
//         matrix_scaler_multiplication,
//         matrix_element_wise_multiplication,
//         matrix_addition,
//         matrix_subtraction,
//         matrix_rollingsum,
//         matrix_power,
//         matrix_transpose,
//     };

//     std::map<std::string, function_names> function_name;

//     struct_function_name()
//     {
//         function_name["matrix_multiplication"] = matrix_multiplication;
//         function_name["matrix_scaler_multiplication"] =
//         matrix_scaler_multiplication;
//         function_name["matrix_element_wise_multiplication"] =
//         matrix_element_wise_multiplication; function_name["matrix_addition"]
//         = matrix_addition; function_name["matrix_subtraction"] =
//         matrix_subtraction; function_name["matrix_power"] = matrix_power;
//         function_name["matrix_rollingsum"] = matrix_rollingsum;
//         function_name["matrix_transpose"] = matrix_transpose;
//     }
// } struct_function_name;

template <typename T, int typeFlag>
class NDMath : public NDArray<T, typeFlag>
{
  struct arg_list
  {
    unsigned value;
    struct arg_list *next;
  };
  struct arg_list *head, *ptr, *ptr_prev;

  struct_function_name fx_name;
  unsigned *arr_dims;

  /// @brief This function recursively iterates from (n-1)th dim to  2th dim and
  /// performs matrix operation for all higher dimensions.
  /// @tparam None
  /// @param index Unsigned, rank of the matrix n, value passed n-1,
  /// @param dimension_arr Unsigend*, used to store internal higher dimention
  /// co-ordinate, expects unsigned pointer of size n(i.e dimension),
  /// @param input NDMath class type, expects operend B.
  /// @param output reference of NDMath class type, expects output C.
  /// @param kernel function pointer, expects pointer to the function for the
  /// matrix operation.
  /// @param function_name String, expects the name of the matrix operation to
  /// be performed.
  /// @return void, output (call by referance)
  void recursive_iterator(unsigned, unsigned *, NDMath<T, typeFlag>,
                          NDMath<T, typeFlag> &,
                          void (*)(double **, unsigned *), std::string,
                          unsigned *, double *, NDMath<T, typeFlag>);

  /// @brief This function recursively iterates from (n-1)th dim to  3th
  /// dimension and performs reduction sum operation along the reduction
  /// dimension.
  /// @tparam None
  /// @param index Unsigned, rank of the matrix n, value passed n-1,
  /// @param dimension_arr Unsigend*, used to store internal higher dimention
  /// co-ordinate, expects unsigned pointer of size n(i.e dimension),
  /// @param input NDMath class type, expects operend B.
  /// @param output reference of NDMath class type, expects output C.
  /// @param reduction_dimension Unsigned, Along which dimension the tensor
  /// needs to be reduced,
  /// @param data_pointer data pointer, expects pointer to the array, which
  /// holds .
  /// @param output unsigned pointer* optional parameter.
  /// @param dl_arr double pointer*, optional parameter,
  /// @param misc_arr NDMath obj, optional parameter.
  /// @return void, output (call by referance)
  void recursive_sum(unsigned, unsigned *, NDMath<T, typeFlag>,
                     NDMath<T, typeFlag> &, unsigned, T *);

  void reducesum(NDMath<T, typeFlag> &);

  template <typename first_dim, typename... Args>
  void reducesum(NDMath<T, typeFlag> &, first_dim, Args...);

public:
  NDMath(const NDMath<T, typeFlag> &ndmath) : NDArray<T, typeFlag>(ndmath) {}

  template <typename... args>
  NDMath(unsigned num, args... Args) : NDArray<T, typeFlag>(num, Args...) {}

  NDMath() {}

  ~NDMath() {}

  NDMath<T, typeFlag> &operator=(const NDMath<T, typeFlag> &ndmath)
  {
    NDArray<T, typeFlag>::operator=(ndmath);
    return *this;
  }

  void operator=(Graph<T, typeFlag> g) { g.outputnode(this); }

  /// @brief This function calculates matrix multiplication A X B
  /// @tparam None
  /// @param input_B_matrix NDMath class, expects tensor or similar y dimension
  /// that of current tensor.
  /// @return returns NDMath class type.
  NDMath<T, typeFlag> matmul(const NDMath<double, 0>);

  /// @brief This function does element wise multiplication with pass costant
  /// value
  /// @tparam None
  /// @param const value double
  /// @return returns NDMath class type.
  NDMath<T, typeFlag> scalermul(const double);

  /// @brief This function calculates matrix addition A + B
  /// @tparam None
  /// @param input_B_matrix NDMath class, expects tensor or similar rank and
  /// dimension that of current tensor.
  /// @return returns NDMath class type.
  NDMath<T, typeFlag> add(const NDMath<double, 0>);

  /// @brief This function calculates matrix multiplication A X B
  /// @tparam None
  /// @param input_B_matrix NDMath class, expects tensor or similar y dimension
  /// that of current tensor.
  /// @return returns NDMath class type.
  NDMath<T, typeFlag> operator*(const NDMath<double, 0>);

  /// @brief This function calculates matrix addition A + B
  /// @tparam None
  /// @param input_B_matrix NDMath class, expects tensor or similar rank and
  /// dimension that of current tensor.
  /// @return returns NDMath class type.
  NDMath<T, typeFlag> operator+(const NDMath<double, 0>);

  /// @brief This function calculates matrix addition A + B
  /// @tparam None
  /// @param input_B_matrix NDMath class, expects tensor or similar rank and
  /// dimension that of current tensor.
  /// @return returns NDMath class type.
  NDMath<T, typeFlag> operator-(const NDMath<double, 0>);

  NDMath<T, typeFlag> vectoradd(const NDMath<double, 0>);

  /// @brief This function transposes current tensor
  /// @tparam None
  /// @return void, output (call by referance)
  void transpose();

  /// @brief This function reduction sum operation along the reduction
  /// dimension.
  /// @tparam None
  /// @param reduction_dimension Unsigned, Along which dimension the tensor
  /// needs to be reduced
  /// @return returns NDMath class type.
  template <typename... Args>
  NDMath<T, typeFlag> reducesum(Args... args);

  /// @brief This function calculates matrix addition A + B
  /// @tparam None
  /// @param input_B_matrix NDMath class, expects tensor or similar rank and
  /// dimension that of current tensor.
  /// @return returns NDMath class type.
  NDMath<T, typeFlag> pow(unsigned);

  Graph<T, typeFlag> mul(NDMath<T, typeFlag> &input, Graph<T, typeFlag> &g)
  {

    unsigned i, no_of_dimensions;
    bool flag = true;

    no_of_dimensions = NDMath<T, typeFlag>::getNoOfDimensions();

    if (no_of_dimensions == input.getNoOfDimensions())
    {

      if (this->getDimensions()[0] == input.getDimensions()[1])
      {

        for (i = 2; i < no_of_dimensions; i++)
        {
          if (NDMath<T, typeFlag>::getDimensions()[i] !=
              input.getDimensions()[i])
          {
            flag = 0;
            break;
          }
        }
        if (flag)
        {

          Ops<T, typeFlag> *ops = new Opsmul<T, typeFlag>;
          NDMath<T, typeFlag> *inputs[2];
          inputs[0] = this;
          inputs[1] = &input;
          ops->initilizeinputs(inputs, 2);
          g.addcomputenode(ops);
        }
        else
        {
          std::cout << "Error!" << i
                    << "th Dimension does not match with second matrix.\n";
          g.setGraphInvalid();
        }
      }
      else
      {
        std::cout << "Error! First matrix's row length does not match with "
                     "second matrix column length.\n";
        g.setGraphInvalid();
      }
    }
    else
    {
      std::cout << "Dimension mismatch, First matrix doesn't have same no of "
                   "dimension of second matrix.\n";
      g.setGraphInvalid();
    }

    return g;
  }

  Graph<T, typeFlag> add(NDMath<T, typeFlag> &input, Graph<T, typeFlag> &g)
  {

    unsigned i, no_of_dimensions;
    bool flag = true;

    no_of_dimensions = NDMath<T, typeFlag>::getNoOfDimensions();

    if (no_of_dimensions == input.getNoOfDimensions())
    {
      for (i = 2; i < no_of_dimensions; i++)
      {
        if (NDMath<T, typeFlag>::getDimensions()[i] !=
            input.getDimensions()[i])
        {
          flag = 0;
          break;
        }
      }
      if (flag)
      {

        Ops<T, typeFlag> *ops = new Opsadd<T, typeFlag>;
        NDMath<T, typeFlag> *inputs[2];
        inputs[0] = this;
        inputs[1] = &input;
        ops->initilizeinputs(inputs, 2);
        g.addcomputenode(ops);
      }
      else
      {
        std::cout << "Error!" << i
                  << "th Dimension does not match with second matrix.\n";
        g.setGraphInvalid();
      }
    }
    else
    {
      std::cout << "Dimension mismatch, First matrix and second matrix has different rank.\n";
      g.setGraphInvalid();
    }
    return g;
  }

  Graph<T, typeFlag> matmul(NDMath<T, typeFlag> &input, Graph<T, typeFlag> &g)
  {

    unsigned i, no_of_dimensions;
    bool flag = true;

    no_of_dimensions = NDMath<T, typeFlag>::getNoOfDimensions();

    if (no_of_dimensions == input.getNoOfDimensions())
    {

      if (this->getDimensions()[0] == input.getDimensions()[1])
      {

        for (i = 2; i < no_of_dimensions; i++)
        {
          if (NDMath<T, typeFlag>::getDimensions()[i] !=
              input.getDimensions()[i])
          {
            flag = false;
            break;
          }
        }
        if (flag)
        {
          Ops<T, typeFlag> *ops = new Opsmatmul<T, typeFlag>;
          NDMath<T, typeFlag> *inputs[2];
          inputs[0] = this;
          inputs[1] = &input;
          ops->initilizeinputs(inputs, 2);
          g.addcomputenode(ops);
        }
        else
        {
          std::cout << "Error!" << i
                    << "th Dimension does not match with second matrix.\n";
          g.setGraphInvalid();
        }
      }
      else
      {
        std::cout << "Error! First matrix's row length does not match with "
                     "second matrix column length.\n";
        g.setGraphInvalid();
      }
    }
    else
    {
      std::cout << "Dimension mismatch, First matrix doesn't have same no of "
                   "dimension of second matrix.\n";
      g.setGraphInvalid();
    }

    return g;
  }

  Graph<T, typeFlag> pow(unsigned power, Graph<T, typeFlag> &g)
  {

    Ops<T, typeFlag> *ops = new Opspower<T, typeFlag>;
    NDMath<T, typeFlag> *inputs[1];
    inputs[0] = this;
    ops->initilizeinputs(inputs, power);
    g.addcomputenode(ops);
    return g;
  }

  void reducesum(Graph<T, typeFlag> &g, Ops<T, typeFlag> *ops)
  {

    unsigned count = 0;
    unsigned *reduction_dims;
    unsigned *dims;

    unsigned *arr_dims = new unsigned[this->getNoOfDimensions()];

    ptr = ptr_prev = head;

    std::cout << "in graph reducesum third func.\n";

    while (ptr)
    {
      count++;
      ptr = ptr->next;
    }

    reduction_dims = new unsigned[count];
    ptr = head;

    for (unsigned i = 0; i < count; i++)
    {
      reduction_dims[i] = this->getNoOfDimensions() - ptr->value - 1;
      ptr = ptr->next;
      delete[] ptr_prev;
      ptr_prev = ptr;
    }

    // shorting dimensions using bubble short
    for (unsigned j = 0; j < count; j++)
      for (unsigned i = 0; i < count - j - 1; i++)
        if (reduction_dims[i] < reduction_dims[i + 1])
        {
          unsigned temp = reduction_dims[i];
          reduction_dims[i] = reduction_dims[i + 1];
          reduction_dims[i + 1] = temp;
        }

    NDMath<T, typeFlag> *input[1];
    input[0] = this;
    ops->initilizeinputs(input, count, reduction_dims);
    g.addcomputenode(ops);

    delete[] reduction_dims;
  }

  template <typename first_dim, typename... Args>
  void reducesum(Graph<T, typeFlag> &g, Ops<T, typeFlag> *ops, first_dim n,
                 Args... args)
  {
    std::cout << "In graph reducesum sec func.\n";
    if (n < this->getNoOfDimensions())
    {

      ptr = new struct arg_list;
      ptr->value = n;

      if (!head)
      {

        head = ptr_prev = ptr;
        head->next = NULL;
      }
      else
      {
        ptr->next = NULL;
        ptr_prev->next = ptr;
        ptr_prev = ptr;
      }
      reducesum(g, ops, args...);
    }
    else
      std::cout
          << "Fatal error! reduction axis does not belong for the tensor\n";
  }

  template <typename... Args>
  Graph<T, typeFlag> reducesum(Graph<T, typeFlag> &g, Args... args)
  {
    head = NULL;
    Ops<T, typeFlag> *ops = new Opsreducesum<T, typeFlag>;
    std::cout << "in graph reducesum!.\n";
    reducesum(g, ops, args...);
    return g;
  }
};

#include "../lib/Math/MathLibrary.cpp"