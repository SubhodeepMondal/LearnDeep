#pragma ONCE
#include <map>
#include "NDynamicArray.h"
#include "CPULibrary.h"

typedef struct struct_function_name
{
    enum function_names
    {
        matrix_multiplication,
        matrix_scaler_multiplication,
        matrix_element_wise_multiplication,
        matrix_addition,
        matrix_subtraction,
        matrix_rollingsum,
        matrix_power,
        matrix_transpose,
    };

    std::map<std::string, function_names> function_name;

    struct_function_name()
    {
        function_name["matrix_multiplication"] = matrix_multiplication;
        function_name["matrix_scaler_multiplication"] = matrix_scaler_multiplication;
        function_name["matrix_element_wise_multiplication"] = matrix_element_wise_multiplication;
        function_name["matrix_addition"] = matrix_addition;
        function_name["matrix_subtraction"] = matrix_subtraction;
        function_name["matrix_power"] = matrix_power;
        function_name["matrix_rollingsum"] = matrix_rollingsum;
        function_name["matrix_transpose"] = matrix_transpose;
    }
} struct_function_name;

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

    /// @brief This function recursively iterates from (n-1)th dim to  2th dim and performs matrix operation for all higher dimensions.
    /// @tparam None
    /// @param index Unsigned, rank of the matrix n, value passed n-1,
    /// @param dimension_arr Unsigend*, used to store internal higher dimention co-ordinate, expects unsigned pointer of size n(i.e dimension),
    /// @param input NDMath class type, expects operend B.
    /// @param output reference of NDMath class type, expects output C.
    /// @param kernel function pointer, expects pointer to the function for the matrix operation.
    /// @param function_name String, expects the name of the matrix operation to be performed.
    /// @return void, output (call by referance)
    void recursive_iterator(unsigned, unsigned *, NDMath<T, typeFlag> , NDMath<T, typeFlag> &, void (*)(double **, unsigned *), std::string, unsigned *, double *, NDMath<T, typeFlag>);

    /// @brief This function recursively iterates from (n-1)th dim to  3th dimension and performs reduction sum operation along the reduction dimension.
    /// @tparam None
    /// @param index Unsigned, rank of the matrix n, value passed n-1,
    /// @param dimension_arr Unsigend*, used to store internal higher dimention co-ordinate, expects unsigned pointer of size n(i.e dimension),
    /// @param input NDMath class type, expects operend B.
    /// @param output reference of NDMath class type, expects output C.
    /// @param reduction_dimension Unsigned, Along which dimension the tensor needs to be reduced,
    /// @param data_pointer data pointer, expects pointer to the array, which holds .
    /// @param output unsigned pointer* optional parameter.
    /// @param dl_arr double pointer*, optional parameter,
    /// @param misc_arr NDMath obj, optional parameter.
    /// @return void, output (call by referance)
    void recursive_sum(unsigned, unsigned *, NDMath<T, typeFlag>, NDMath<T, typeFlag> &, unsigned, T *);

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

    /// @brief This function calculates matrix multiplication A X B
    /// @tparam None
    /// @param input_B_matrix NDMath class, expects tensor or similar y dimension that of current tensor.
    /// @return returns NDMath class type.
    NDMath<T, typeFlag> matrixMultiplication(const NDMath<double, 0>);

    /// @brief This function does element wise multiplication with pass costant value
    /// @tparam None
    /// @param const value double
    /// @return returns NDMath class type.
    NDMath<T, typeFlag> matrixMultiplication(const double);

    /// @brief This function calculates matrix addition A + B
    /// @tparam None
    /// @param input_B_matrix NDMath class, expects tensor or similar rank and dimension that of current tensor.
    /// @return returns NDMath class type.
    NDMath<T, typeFlag> matrixAddition(const NDMath<double, 0>);

    /// @brief This function calculates matrix multiplication A X B
    /// @tparam None
    /// @param input_B_matrix NDMath class, expects tensor or similar y dimension that of current tensor.
    /// @return returns NDMath class type.
    NDMath<T, typeFlag> operator*(const NDMath<double, 0>);

    /// @brief This function calculates matrix addition A + B
    /// @tparam None
    /// @param input_B_matrix NDMath class, expects tensor or similar rank and dimension that of current tensor.
    /// @return returns NDMath class type.
    NDMath<T, typeFlag> operator+(const NDMath<double, 0>);

    /// @brief This function calculates matrix addition A + B
    /// @tparam None
    /// @param input_B_matrix NDMath class, expects tensor or similar rank and dimension that of current tensor.
    /// @return returns NDMath class type.
    NDMath<T, typeFlag> operator-(const NDMath<double, 0>);

    NDMath<T, typeFlag> matrixVectorAddition(const NDMath<double, 0>);

    NDMath<T, typeFlag> matrixpow(const unsigned);

    /// @brief This function transposes current tensor
    /// @tparam None
    /// @return void, output (call by referance)
    void matrixTranspose();

    /// @brief This function reduction sum operation along the reduction dimension.
    /// @tparam None
    /// @param reduction_dimension Unsigned, Along which dimension the tensor needs to be reduced
    /// @return returns NDMath class type.
    template <typename... Args>
    NDMath<T, typeFlag> reducesum(Args... args);

    /// @brief This function calculates matrix addition A + B
    /// @tparam None
    /// @param input_B_matrix NDMath class, expects tensor or similar rank and dimension that of current tensor.
    /// @return returns NDMath class type.
    NDMath<T, typeFlag> power(unsigned);
};

#include "../lib/Math/MathLibrary.cpp"