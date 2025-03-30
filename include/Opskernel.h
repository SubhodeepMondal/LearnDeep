#ifndef OPS_KERNEL
#define OPS_KERNEL

#include "../framework/MathLibrary.h"
#include <iostream>

template <typename T, int typeFlag> class Ops {

public:
  virtual void compute() = 0;
  virtual void initilizeoutput(NDMath<T, typeFlag> *) = 0;
  virtual void initilizeinputs(NDMath<T, typeFlag> **input_a,
                               unsigned no_of_inputs) {}
  virtual void initilizeinputs(NDMath<T, typeFlag> **input_a,
                               double scale_factor) {}
  virtual void initilizeinputs(NDMath<T, typeFlag> **input_a, unsigned n,
                               unsigned *arr) {}
  virtual unsigned getnoofinputs() { return 0; }
  virtual NDMath<T, typeFlag> *getoutput() { return NULL; }
  virtual NDMath<T, typeFlag> **getinputs() { return NULL; }
  virtual void printinputs() {}
  virtual void printoutput() {}
  virtual void autograd() {}

  struct_function_name fx_name;

  virtual void recursive_iterator(unsigned index, unsigned *dimension_arr,
                                  NDMath<T, typeFlag> input_a,
                                  NDMath<T, typeFlag> input_b,
                                  NDMath<T, typeFlag> &output,
                                  std::string function_name, unsigned *ui_arr,
                                  double *dl_arr,
                                  NDMath<T, typeFlag> misc_arr) {}

  static void recursive_sum(unsigned index, unsigned *dimension_arr,
                            NDMath<T, typeFlag> input_a,
                            NDMath<T, typeFlag> input_b,
                            NDMath<T, typeFlag> &output, unsigned reduction_dim,
                            T *temp_input) {

    if (index < 3) {
      unsigned i, j, k;
      unsigned x_axis, y_axis, z_axis, stride, n_dim_size;
      unsigned input_index, output_index;
      T *input_ptr, *output_ptr, *temp_inp;
      double *ptr[3];
      unsigned a[2];

      // T *input_b = input_a.getData();

      x_axis = input_b.getDimensions()[0];
      y_axis =
          (input_b.getNoOfDimensions() > 1) ? input_b.getDimensions()[1] : 1;
      z_axis =
          (input_b.getNoOfDimensions() > 2) ? input_b.getDimensions()[2] : 1;

      input_ptr = input_b.getData();
      output_ptr = output.getData();

      input_index = output_index = 0;

      if (input_b.getNoOfDimensions() > 3) {
        n_dim_size = x_axis * y_axis * z_axis;
        for (i = 3; i < input_b.getNoOfDimensions(); i++) {
          input_index += n_dim_size * dimension_arr[i];
          n_dim_size *= input_b.getDimensions()[i];
        }

        n_dim_size = 1;
        for (i = 0; i < input_b.getNoOfDimensions(); i++) {
          if (i != reduction_dim) {
            if (i < 3)
              output_index *= n_dim_size;
            else
              output_index += n_dim_size * dimension_arr[i];

            n_dim_size *= input_b.getDimensions()[i];
          }
        }
      }

      switch (reduction_dim) {
      case 0: {

        ptr[0] = ptr[2] = output_ptr + output_index;
        a[0] = x_axis;
        a[1] = z_axis;

        for (k = 0; k < x_axis; k++) {
          stride = 1;
          for (j = 0; j < z_axis; j++)
            for (i = 0; i < y_axis; i++)
              temp_input[i + j * y_axis] =
                  input_ptr[i * x_axis + j * x_axis * y_axis + stride * k +
                            input_index];

          ptr[1] = temp_input;
          cpu::__madd(ptr, a);
          // cpu::__madd(output_ptr + output_index, temp_input, output_ptr +
          // output_index, y_axis, z_axis);
        }
        break;
      }

      case 1: {

        ptr[0] = ptr[2] = output_ptr + output_index;
        a[0] = x_axis;
        a[1] = z_axis;
        for (k = 0; k < y_axis; k++) {
          stride = x_axis;
          for (j = 0; j < z_axis; j++)
            for (i = 0; i < x_axis; i++)
              temp_input[i + j * x_axis] =
                  input_ptr[i + j * x_axis * y_axis + stride * k + input_index];

          ptr[1] = temp_input;
          cpu::__madd(ptr, a);
          // cpu::__madd(output_ptr + output_index, temp_input, output_ptr +
          // output_index, x_axis, z_axis);
        }

        break;
      }
      case 2: {

        ptr[0] = ptr[2] = output_ptr + output_index;
        a[0] = x_axis;
        a[1] = y_axis;

        for (k = 0; k < z_axis; k++) {
          stride = x_axis * y_axis;
          temp_input = input_ptr + (stride * k + input_index);
          ptr[1] = temp_input;

          cpu::__madd(ptr, a);

          // for (int j = 0; j < y_axis; j++)
          //     for (int i = 0; i < x_axis; i++)
          //         std::cout << output_ptr[i + j * x_axis] << " ";

          // cpu::__madd(output_ptr + output_index, temp_inp, output_ptr +
          // output_index, x_axis, y_axis);
        }
        break;
      }

      default: {
        a[0] = x_axis;
        a[1] = y_axis;
        for (k = 0; k < z_axis; k++) {
          stride = x_axis * y_axis;

          ptr[0] = ptr[2] = output_ptr + (output_index + stride * k);

          temp_inp = input_ptr + (stride * k + input_index);
          ptr[1] = temp_inp;

          cpu::__madd(ptr, a);
          // cpu::__madd(output_ptr + (output_index + stride * k), temp_inp,
          // output_ptr + (output_index + stride * k), x_axis, y_axis);
        }
        break;
      }
      }
    } else {
      for (unsigned i = 0; i < input_b.getDimensions()[index]; i++) {
        dimension_arr[index] = i;
        recursive_sum(index - 1, dimension_arr, input_b, output, reduction_dim,
                      temp_input);
      }
    }
  }
};

template <typename T, int typeFlag> class Opsmul : public Ops<T, typeFlag> {
  unsigned no_of_inputs;
  struct_function_name fx_name;

  NDMath<T, typeFlag> **inputs;
  NDMath<T, typeFlag> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          NDMath<T, typeFlag> input_a,
                          NDMath<T, typeFlag> input_b,
                          NDMath<T, typeFlag> &output,
                          std::string function_name, unsigned *ui_arr,
                          double *dl_arr, NDMath<T, typeFlag> misc_arr);

public:
  Opsmul() {}
  void compute();
  void initilizeinputs(NDMath<T, typeFlag> **inputs, unsigned no_of_inputs);
  void initilizeoutput(NDMath<T, typeFlag> *output);

  NDMath<T, typeFlag> **getinputs() { return inputs; }

  NDMath<T, typeFlag> *getoutput() { return output; }

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs();

  void printoutput();
};

template <typename T, int typeFlag> class Opsadd : public Ops<T, typeFlag> {
  unsigned no_of_inputs;
  struct_function_name fx_name;

  NDMath<T, typeFlag> **inputs;
  NDMath<T, typeFlag> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          NDMath<T, typeFlag> input_a,
                          NDMath<T, typeFlag> input_b,
                          NDMath<T, typeFlag> &output,
                          std::string function_name, unsigned *ui_arr,
                          double *dl_arr, NDMath<T, typeFlag> misc_arr);

public:
  Opsadd() {}
  void compute();
  void initilizeinputs(NDMath<T, typeFlag> **inputs, unsigned no_of_inputs);
  void initilizeoutput(NDMath<T, typeFlag> *output);

  NDMath<T, typeFlag> **getinputs() { return inputs; }

  NDMath<T, typeFlag> *getoutput() { return output; }

  unsigned getnoofinputs() { return no_of_inputs; }

  // void initilizeinputs(NDMath<T, typeFlag> **inputs, unsigned no_of_inputs);

  // void initilizeoutput(NDMath<T, typeFlag> *output);

  void printinputs();
  void printoutput();
};

template <typename T, int typeFlag> class Opsmatmul : public Ops<T, typeFlag> {
  unsigned no_of_inputs;
  struct_function_name fx_name;

  NDMath<T, typeFlag> **inputs;
  NDMath<T, typeFlag> *output;
  void recursive_iterator(unsigned index, unsigned *, NDMath<T, typeFlag>,
                          NDMath<T, typeFlag>, NDMath<T, typeFlag> &,
                          std::string, unsigned *, double *,
                          NDMath<T, typeFlag>);

public:
  Opsmatmul() {}
  // void recursive_iterator(unsigned, unsigned *, NDMath<T, typeFlag>,
  //                         NDMath<T, typeFlag>, NDMath<T, typeFlag> &output,
  //                         std::string, unsigned *, double *,
  //                         NDMath<T, typeFlag>);
  void compute();
  // void initilizeinputs(NDMath<T, typeFlag>, unsigned);
  void initilizeoutput(NDMath<T, typeFlag> *output);

  NDMath<T, typeFlag> **getinputs() { return inputs; }

  NDMath<T, typeFlag> *getoutput() { return output; }

  void initilizeinputs(NDMath<T, typeFlag> **, unsigned);

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs();

  void printoutput();
};

template <typename T, int typeFlag> class Opspower : public Ops<T, typeFlag> {
  unsigned exponent;
  struct_function_name fx_name;

  NDMath<T, typeFlag> **inputs;
  NDMath<T, typeFlag> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          NDMath<T, typeFlag> input_a,
                          NDMath<T, typeFlag> input_b,
                          NDMath<T, typeFlag> &output,
                          std::string function_name, unsigned *ui_arr,
                          double *dl_arr, NDMath<T, typeFlag> misc_arr);

public:
  void compute();
  void initilizeinputs(NDMath<T, typeFlag> **inputs, unsigned exponent);

  void initilizeoutput(NDMath<T, typeFlag> *output);

  NDMath<T, typeFlag> **getinputs() { return inputs; }

  NDMath<T, typeFlag> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }

  void printinputs();

  void printoutput();
};

template <typename T, int typeFlag> class Opsreducesum : public Ops<T, typeFlag> {
  unsigned no_of_reduction_dim, *reduction_dims;
  struct_function_name fx_name;

  NDMath<T, typeFlag> **inputs;
  NDMath<T, typeFlag> *output;
  void recursive_sum(unsigned index, unsigned *dimension_arr,
                     NDMath<T, typeFlag> input, NDMath<T, typeFlag> &output,
                     unsigned reduction_dim, T *temp_input);

public:
  void compute();
  void initilizeinputs(NDMath<T, typeFlag> **inputs, unsigned n, unsigned *arr);
  void initilizeoutput(NDMath<T, typeFlag> *output);

  NDMath<T, typeFlag> **getinputs() { return inputs; }

  NDMath<T, typeFlag> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }

  void printinputs();

  void printoutput();
};

template <typename T, int typeFlag> class Opsscale : public Ops<T, typeFlag> {
  double scale_factor;
  struct_function_name fx_name;

  NDMath<T, typeFlag> **inputs;
  NDMath<T, typeFlag> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          NDMath<T, typeFlag> input_a,
                          NDMath<T, typeFlag> input_b,
                          NDMath<T, typeFlag> &output,
                          std::string function_name, unsigned *ui_arr,
                          double *dl_arr, NDMath<T, typeFlag> misc_arr);

public:
  void compute();
  void initilizeinputs(NDMath<T, typeFlag> **inputs, double scale_factor);
  void initilizeoutput(NDMath<T, typeFlag> *outputs);

  NDMath<T, typeFlag> **getinputs() { return inputs; }

  NDMath<T, typeFlag> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }
  void printinputs();
  void printoutput();
};

#endif // OPS_kernel