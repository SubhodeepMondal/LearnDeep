#pragma ONCE
#include <map>
#include "NDynamicArray.h"
#include "CPULibrary.h"
#include "Opskernel.h"
#include "Graph.h"

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
//         function_name["matrix_scaler_multiplication"] = matrix_scaler_multiplication;
//         function_name["matrix_element_wise_multiplication"] = matrix_element_wise_multiplication;
//         function_name["matrix_addition"] = matrix_addition;
//         function_name["matrix_subtraction"] = matrix_subtraction;
//         function_name["matrix_power"] = matrix_power;
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

    /// @brief This function recursively iterates from (n-1)th dim to  2th dim and performs matrix operation for all higher dimensions.
    /// @tparam None
    /// @param index Unsigned, rank of the matrix n, value passed n-1,
    /// @param dimension_arr Unsigend*, used to store internal higher dimention co-ordinate, expects unsigned pointer of size n(i.e dimension),
    /// @param input NDMath class type, expects operend B.
    /// @param output reference of NDMath class type, expects output C.
    /// @param kernel function pointer, expects pointer to the function for the matrix operation.
    /// @param function_name String, expects the name of the matrix operation to be performed.
    /// @return void, output (call by referance)
    void recursive_iterator(unsigned, unsigned *, NDMath<T, typeFlag>, NDMath<T, typeFlag> &, void (*)(double **, unsigned *), std::string, unsigned *, double *, NDMath<T, typeFlag>);

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

    void operator=(Graph<T, typeFlag> g)
    {
        g.outputnode(this);
    }

    /// @brief This function calculates matrix multiplication A X B
    /// @tparam None
    /// @param input_B_matrix NDMath class, expects tensor or similar y dimension that of current tensor.
    /// @return returns NDMath class type.
    NDMath<T, typeFlag> matmul(const NDMath<double, 0>);

    /// @brief This function does element wise multiplication with pass costant value
    /// @tparam None
    /// @param const value double
    /// @return returns NDMath class type.
    NDMath<T, typeFlag> scalermul(const double);

    /// @brief This function calculates matrix addition A + B
    /// @tparam None
    /// @param input_B_matrix NDMath class, expects tensor or similar rank and dimension that of current tensor.
    /// @return returns NDMath class type.
    NDMath<T, typeFlag> add(const NDMath<double, 0>);

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

    NDMath<T, typeFlag> vectoradd(const NDMath<double, 0>);

    /// @brief This function transposes current tensor
    /// @tparam None
    /// @return void, output (call by referance)
    void transpose();

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
    NDMath<T, typeFlag> pow(unsigned);

    Graph<T, typeFlag> mul(NDMath<T, typeFlag> &input, Graph<T, typeFlag> &g)
    {

        unsigned i, j, no_of_dimensions, flag = 1;
        unsigned a_plan_dim, b_plan_dim, c_plan_dim;
        unsigned a_actual_index, b_actual_index, c_actual_index;
        unsigned dim_x, dim_y, dim_z;
        unsigned *output_dim;

        no_of_dimensions = NDMath<T, typeFlag>::getNoOfDimensions();

        dim_x = input.getDimensions()[0];
        dim_y = NDMath<T, typeFlag>::getDimensions()[1];
        dim_z = NDMath<T, typeFlag>::getDimensions()[0];

        if (no_of_dimensions == input.getNoOfDimensions())
        {
            output_dim = new unsigned[no_of_dimensions];

            output_dim[0] = dim_x;
            output_dim[1] = dim_y;

            if (this->getDimensions()[0] == input.getDimensions()[1])
            {

                for (i = 2; i < no_of_dimensions; i++)
                {
                    output_dim[i] = NDMath<T, typeFlag>::getDimensions()[i];
                    if (NDMath<T, typeFlag>::getDimensions()[i] != input.getDimensions()[i])
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
                    g.addcomputenode(this, &input, ops);
                }
                else
                    std::cout << "Error!" << i << "th Dimension does not match with second matrix.\n";
            }
            else
                std::cout << "Error! First matrix's row length does not match with second matrix column length.\n";
        }
        else
            std::cout << "Dimension mismatch, First matrix doesn't have same no of dimension of second matrix.\n";

        return g;
    }

    Graph<T, typeFlag> add(NDMath<T, typeFlag> &input, Graph<T, typeFlag> &g)
    {

        unsigned i, j, no_of_dimensions, flag = 1;
        unsigned a_plan_dim, b_plan_dim, c_plan_dim;
        unsigned a_actual_index, b_actual_index, c_actual_index;
        unsigned dim_x, dim_y, dim_z;
        unsigned *output_dim;

        no_of_dimensions = NDMath<T, typeFlag>::getNoOfDimensions();

        dim_x = input.getDimensions()[0];
        dim_y = NDMath<T, typeFlag>::getDimensions()[1];
        dim_z = NDMath<T, typeFlag>::getDimensions()[0];

        if (no_of_dimensions == input.getNoOfDimensions())
        {
            output_dim = new unsigned[no_of_dimensions];

            output_dim[0] = dim_x;
            output_dim[1] = dim_y;

            if (this->getDimensions()[0] == input.getDimensions()[1])
            {

                for (i = 2; i < no_of_dimensions; i++)
                {
                    output_dim[i] = NDMath<T, typeFlag>::getDimensions()[i];
                    if (NDMath<T, typeFlag>::getDimensions()[i] != input.getDimensions()[i])
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
                    g.addcomputenode(this, &input, ops);
                }
                else
                    std::cout << "Error!" << i << "th Dimension does not match with second matrix.\n";
            }
            else
                std::cout << "Error! First matrix's row length does not match with second matrix column length.\n";
        }
        else
            std::cout << "Dimension mismatch, First matrix doesn't have same no of dimension of second matrix.\n";

        return g;
    }
};

#include "../lib/Math/MathLibrary.cpp"