#pragma ONCE
#include<map>
template <typename T> class tensor;

typedef struct struct_function_name {
  enum function_names {
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

  struct_function_name() {
    function_name["matrix_multiplication"] = matrix_multiplication;
    function_name["matrix_scaler_multiplication"] =
        matrix_scaler_multiplication;
    function_name["matrix_element_wise_multiplication"] =
        matrix_element_wise_multiplication;
    function_name["matrix_addition"] = matrix_addition;
    function_name["matrix_subtraction"] = matrix_subtraction;
    function_name["matrix_power"] = matrix_power;
    function_name["matrix_rollingsum"] = matrix_rollingsum;
    function_name["matrix_transpose"] = matrix_transpose;
  }
} struct_function_name;