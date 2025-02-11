#pragma ONCE

template <typename T, int typeFlag> class NDMath;

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
                          double *dl_arr, NDMath<T, typeFlag> misc_arr) {
    if (index < 2) {
      unsigned i, inpA_x, inpA_y, inpB_x, inpB_y, out_x, out_y;
      unsigned a_plane_size, b_plane_size, c_plane_size, a_index, b_index,
          c_index;

      inpA_x =
          (input_a.getNoOfDimensions() > 0) ? input_a.getDimensions()[0] : 1;
      inpA_y =
          (input_a.getNoOfDimensions() > 1) ? input_a.getDimensions()[1] : 1;

      inpB_x =
          (input_b.getNoOfDimensions() > 0) ? input_b.getDimensions()[0] : 1;
      inpB_y =
          (input_b.getNoOfDimensions() > 1) ? input_b.getDimensions()[1] : 1;

      out_x = (output.getNoOfDimensions() > 0) ? output.getDimensions()[0] : 1;
      out_y = (output.getNoOfDimensions() > 1) ? output.getDimensions()[1] : 1;

      a_plane_size = inpA_x * inpA_y;
      b_plane_size = inpB_x * inpB_y;
      c_plane_size = out_x * out_y;

      a_index = b_index = c_index = 0;
      if (input_b.getNoOfDimensions() > 2)
        for (i = 2; i < input_b.getNoOfDimensions(); i++) {
          a_index += a_plane_size * dimension_arr[i];
          b_index += b_plane_size * dimension_arr[i];
          c_index += c_plane_size * dimension_arr[i];

          a_plane_size *= input_a.getDimensions()[i];
          b_plane_size *= input_b.getDimensions()[i];
          c_plane_size *= output.getDimensions()[i];
        }

      switch (fx_name.function_name[function_name]) {
      case this->fx_name.matrix_multiplication: {
        /* code */
        unsigned a[3];
        double *ptr[3];

        a[0] = inpB_x;
        a[1] = inpB_y;
        a[2] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;
        cpu::__mmul(ptr, a);
        // __kernel(ptr, a);

        break;
      }
      case this->fx_name.matrix_addition: {
        /* code */
        unsigned a[2];
        double *ptr[3];

        a[0] = inpA_x;
        a[1] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;
        cpu::__madd(ptr, a);
        // __kernel(ptr, a);

        break;
      }
      case this->fx_name.matrix_element_wise_multiplication: {
        /* code */
        unsigned a[2];
        double *ptr[3];

        a[0] = inpA_x;
        a[1] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;
        cpu::__melementwisemul(ptr, a);

        break;
      }
      case this->fx_name.matrix_power: {
        int j;
        // int exponent = ui_arr[0];
        unsigned a[2];
        double *ptr[3];

        a[0] = inpA_x;
        a[1] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;

        cpu::__melementwisemul(ptr, a);

        break;
      }

      default:
        break;
      }
    } else {
      // std::cout << "inside else\n";
      for (unsigned i = 0; i < input_a.getDimensions()[index]; i++) {
        dimension_arr[index] = i;
        recursive_iterator(index - 1, dimension_arr, input_a, input_b, output,
                           function_name, NULL, NULL, NULL);
      }
    }
  };

public:
  Opsmul() {}

  void compute() override {
    // std::cout << "inside compute\n";
    unsigned dim_x, dim_y;
    dim_x = inputs[0]->getDimensions()[0];
    dim_y = inputs[1]->getDimensions()[1];

    unsigned *arr = new unsigned[inputs[0]->getNoOfDimensions()];

    recursive_iterator(inputs[0]->getNoOfDimensions() - 1, arr, *inputs[0],
                       *inputs[1], *output,
                       "matrix_element_wise_multiplication", NULL, NULL, NULL);

    delete[] arr;
  }

  // template <typename Ta, typename Tb>
  void initilizeinputs(NDMath<T, typeFlag> **inputs,
                       unsigned no_of_inputs) override {
    unsigned i;
    this->no_of_inputs = no_of_inputs;

    this->inputs = new NDMath<T, typeFlag> *[this->no_of_inputs];

    for (i = 0; i < this->no_of_inputs; i++) {
      this->inputs[i] = inputs[i];
    }
  }

  void initilizeoutput(NDMath<T, typeFlag> *output) override {
    this->output = output;

    *(this->output) = *(inputs[0]);
  }

  NDMath<T, typeFlag> **getinputs() { return inputs; }

  NDMath<T, typeFlag> *getoutput() { return output; }

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs() override {
    unsigned i;
    for (i = 0; i < this->no_of_inputs; i++) {
      std::cout << "Input: " << i << "\n";
      inputs[i]->printData();
    }
  }

  void printoutput() override {
    // std::cout << output->getData() << "\n";
    std::cout << "output:\n";
    output->printData();
    std::cout << "\n";
  }
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
                          double *dl_arr, NDMath<T, typeFlag> misc_arr) {
    if (index < 2) {
      unsigned i, inpA_x, inpA_y, inpB_x, inpB_y, out_x, out_y;
      unsigned a_plane_size, b_plane_size, c_plane_size, a_index, b_index,
          c_index;

      inpA_x =
          (input_a.getNoOfDimensions() > 0) ? input_a.getDimensions()[0] : 1;
      inpA_y =
          (input_a.getNoOfDimensions() > 1) ? input_a.getDimensions()[1] : 1;

      inpB_x =
          (input_b.getNoOfDimensions() > 0) ? input_b.getDimensions()[0] : 1;
      inpB_y =
          (input_b.getNoOfDimensions() > 1) ? input_b.getDimensions()[1] : 1;

      out_x = (output.getNoOfDimensions() > 0) ? output.getDimensions()[0] : 1;
      out_y = (output.getNoOfDimensions() > 1) ? output.getDimensions()[1] : 1;

      a_plane_size = inpA_x * inpA_y;
      b_plane_size = inpB_x * inpB_y;
      c_plane_size = out_x * out_y;

      a_index = b_index = c_index = 0;
      if (input_b.getNoOfDimensions() > 2)
        for (i = 2; i < input_b.getNoOfDimensions(); i++) {
          a_index += a_plane_size * dimension_arr[i];
          b_index += b_plane_size * dimension_arr[i];
          c_index += c_plane_size * dimension_arr[i];

          a_plane_size *= input_a.getDimensions()[i];
          b_plane_size *= input_b.getDimensions()[i];
          c_plane_size *= output.getDimensions()[i];
        }

      switch (fx_name.function_name[function_name]) {
      case this->fx_name.matrix_multiplication: {
        /* code */
        unsigned a[3];
        double *ptr[3];

        a[0] = inpB_x;
        a[1] = inpB_y;
        a[2] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;
        cpu::__mmul(ptr, a);
        // __kernel(ptr, a);

        break;
      }
      case this->fx_name.matrix_addition: {
        /* code */
        unsigned a[2];
        double *ptr[3];

        a[0] = inpA_x;
        a[1] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;
        cpu::__madd(ptr, a);
        // __kernel(ptr, a);

        break;
      }
      case this->fx_name.matrix_element_wise_multiplication: {
        /* code */
        unsigned a[2];
        double *ptr[3];

        a[0] = inpA_x;
        a[1] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;
        cpu::__melementwisemul(ptr, a);

        break;
      }
      case this->fx_name.matrix_power: {
        int j;
        // int exponent = ui_arr[0];
        unsigned a[2];
        double *ptr[3];

        a[0] = inpA_x;
        a[1] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;

        cpu::__melementwisemul(ptr, a);

        break;
      }

      default:
        break;
      }
    } else {
      // std::cout << "inside else\n";
      for (unsigned i = 0; i < input_a.getDimensions()[index]; i++) {
        dimension_arr[index] = i;
        recursive_iterator(index - 1, dimension_arr, input_a, input_b, output,
                           function_name, NULL, NULL, NULL);
      }
    }
  };

public:
  Opsadd() {}

  void compute() override {
    // std::cout << "inside compute\n";
    unsigned dim_x, dim_y;
    dim_x = inputs[0]->getDimensions()[0];
    dim_y = inputs[1]->getDimensions()[1];

    unsigned *arr = new unsigned[inputs[0]->getNoOfDimensions()];

    recursive_iterator(inputs[0]->getNoOfDimensions() - 1, arr, *inputs[0],
                       *inputs[1], *output, "matrix_addition", NULL, NULL,
                       NULL);

    delete[] arr;
  }

  // template <typename Ta, typename Tb>
  void initilizeinputs(NDMath<T, typeFlag> **inputs,
                       unsigned no_of_inputs) override {
    unsigned i;
    this->no_of_inputs = no_of_inputs;

    this->inputs = new NDMath<T, typeFlag> *[this->no_of_inputs];

    for (i = 0; i < this->no_of_inputs; i++) {
      this->inputs[i] = inputs[i];
    }
  }

  void initilizeoutput(NDMath<T, typeFlag> *output) override {
    this->output = output;

    *(this->output) = *(inputs[0]);
  }

  NDMath<T, typeFlag> **getinputs() { return inputs; }

  NDMath<T, typeFlag> *getoutput() { return output; }

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs() override {
    unsigned i;
    for (i = 0; i < this->no_of_inputs; i++) {
      std::cout << "Input: " << i << "\n";
      inputs[i]->printData();
    }
  }

  void printoutput() override {
    // std::cout << output->getData() << "\n";
    std::cout << "output:\n";
    output->printData();
    std::cout << "\n";
  }
};

template <typename T, int typeFlag> class Opsmatmul : public Ops<T, typeFlag> {
  unsigned no_of_inputs;
  struct_function_name fx_name;

  NDMath<T, typeFlag> **inputs;
  NDMath<T, typeFlag> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          NDMath<T, typeFlag> input_a,
                          NDMath<T, typeFlag> input_b,
                          NDMath<T, typeFlag> &output,
                          std::string function_name, unsigned *ui_arr,
                          double *dl_arr, NDMath<T, typeFlag> misc_arr) {
    if (index < 2) {
      unsigned i, inpA_x, inpA_y, inpB_x, inpB_y, out_x, out_y;
      unsigned a_plane_size, b_plane_size, c_plane_size, a_index, b_index,
          c_index;

      inpA_x =
          (input_a.getNoOfDimensions() > 0) ? input_a.getDimensions()[0] : 1;
      inpA_y =
          (input_a.getNoOfDimensions() > 1) ? input_a.getDimensions()[1] : 1;

      inpB_x =
          (input_b.getNoOfDimensions() > 0) ? input_b.getDimensions()[0] : 1;
      inpB_y =
          (input_b.getNoOfDimensions() > 1) ? input_b.getDimensions()[1] : 1;

      out_x = (output.getNoOfDimensions() > 0) ? output.getDimensions()[0] : 1;
      out_y = (output.getNoOfDimensions() > 1) ? output.getDimensions()[1] : 1;

      a_plane_size = inpA_x * inpA_y;
      b_plane_size = inpB_x * inpB_y;
      c_plane_size = out_x * out_y;

      a_index = b_index = c_index = 0;
      if (input_b.getNoOfDimensions() > 2)
        for (i = 2; i < input_b.getNoOfDimensions(); i++) {
          a_index += a_plane_size * dimension_arr[i];
          b_index += b_plane_size * dimension_arr[i];
          c_index += c_plane_size * dimension_arr[i];

          a_plane_size *= input_a.getDimensions()[i];
          b_plane_size *= input_b.getDimensions()[i];
          c_plane_size *= output.getDimensions()[i];
        }

      switch (fx_name.function_name[function_name]) {
      case this->fx_name.matrix_multiplication: {
        /* code */
        unsigned a[3];
        double *ptr[3];

        a[0] = inpB_x;
        a[1] = inpB_y;
        a[2] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;
        cpu::__mmul(ptr, a);
        // __kernel(ptr, a);

        break;
      }
      case this->fx_name.matrix_addition: {
        /* code */
        unsigned a[2];
        double *ptr[3];

        a[0] = inpA_x;
        a[1] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;
        cpu::__madd(ptr, a);
        // __kernel(ptr, a);

        break;
      }
      case this->fx_name.matrix_element_wise_multiplication: {
        /* code */
        unsigned a[2];
        double *ptr[3];

        a[0] = inpA_x;
        a[1] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;
        cpu::__melementwisemul(ptr, a);

        break;
      }
      case this->fx_name.matrix_power: {
        int j;
        // int exponent = ui_arr[0];
        unsigned a[2];
        double *ptr[3];

        a[0] = inpA_x;
        a[1] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;

        cpu::__melementwisemul(ptr, a);

        break;
      }

      default:
        break;
      }
    } else {
      // std::cout << "inside else\n";
      for (unsigned i = 0; i < input_a.getDimensions()[index]; i++) {
        dimension_arr[index] = i;
        recursive_iterator(index - 1, dimension_arr, input_a, input_b, output,
                           function_name, NULL, NULL, NULL);
      }
    }
  };

public:
  Opsmatmul() {}

  void compute() override {
    // std::cout << "inside compute\n";
    unsigned dim_x, dim_y;
    dim_x = inputs[0]->getDimensions()[0];
    dim_y = inputs[1]->getDimensions()[1];

    unsigned *arr = new unsigned[inputs[0]->getNoOfDimensions()];

    recursive_iterator(inputs[0]->getNoOfDimensions() - 1, arr, *inputs[0],
                       *inputs[1], *output, "matrix_multiplication", NULL, NULL,
                       NULL);

    delete[] arr;
  }

  // template <typename Ta, typename Tb>
  void initilizeinputs(NDMath<T, typeFlag> **inputs,
                       unsigned no_of_inputs) override {
    unsigned i;
    this->no_of_inputs = no_of_inputs;

    this->inputs = new NDMath<T, typeFlag> *[this->no_of_inputs];

    for (i = 0; i < this->no_of_inputs; i++) {
      this->inputs[i] = inputs[i];
    }
  }

  void initilizeoutput(NDMath<T, typeFlag> *output) override {
    this->output = output;

    unsigned i, dim_x, dim_y, no_of_dimensions;
    unsigned *output_dim;

    no_of_dimensions = inputs[0]->getNoOfDimensions();

    dim_x = inputs[1]->getDimensions()[0];
    dim_y = inputs[0]->getDimensions()[1];

    output_dim = new unsigned[no_of_dimensions];

    output_dim[0] = dim_x;
    output_dim[1] = dim_y;

    for (i = 2; i < no_of_dimensions; i++)
      output_dim[i] = inputs[0]->getDimensions()[i];

    *(this->output) = NDMath<T, typeFlag>(no_of_dimensions, output_dim);

    delete[] output_dim;
  }

  NDMath<T, typeFlag> **getinputs() { return inputs; }

  NDMath<T, typeFlag> *getoutput() { return output; }

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs() override {
    unsigned i;
    for (i = 0; i < this->no_of_inputs; i++) {
      std::cout << "Input: " << i << "\n";
      inputs[i]->printData();
    }
  }

  void printoutput() override {
    // std::cout << output->getData() << "\n";
    std::cout << "output:\n";
    output->printData();
    std::cout << "\n";
  }
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
                          double *dl_arr, NDMath<T, typeFlag> misc_arr) {
    if (index < 2) {
      unsigned i, inpA_x, inpA_y, inpB_x, inpB_y, out_x, out_y;
      unsigned a_plane_size, b_plane_size, c_plane_size, a_index, b_index,
          c_index;

      inpA_x =
          (input_a.getNoOfDimensions() > 0) ? input_a.getDimensions()[0] : 1;
      inpA_y =
          (input_a.getNoOfDimensions() > 1) ? input_a.getDimensions()[1] : 1;

      inpB_x =
          (input_b.getNoOfDimensions() > 0) ? input_b.getDimensions()[0] : 1;
      inpB_y =
          (input_b.getNoOfDimensions() > 1) ? input_b.getDimensions()[1] : 1;

      out_x = (output.getNoOfDimensions() > 0) ? output.getDimensions()[0] : 1;
      out_y = (output.getNoOfDimensions() > 1) ? output.getDimensions()[1] : 1;

      a_plane_size = inpA_x * inpA_y;
      b_plane_size = inpB_x * inpB_y;
      c_plane_size = out_x * out_y;

      a_index = b_index = c_index = 0;
      if (input_b.getNoOfDimensions() > 2)
        for (i = 2; i < input_b.getNoOfDimensions(); i++) {
          a_index += a_plane_size * dimension_arr[i];
          b_index += b_plane_size * dimension_arr[i];
          c_index += c_plane_size * dimension_arr[i];

          a_plane_size *= input_a.getDimensions()[i];
          b_plane_size *= input_b.getDimensions()[i];
          c_plane_size *= output.getDimensions()[i];
        }

      switch (fx_name.function_name[function_name]) {
      case this->fx_name.matrix_multiplication: {
        /* code */
        unsigned a[3];
        double *ptr[3];

        a[0] = inpB_x;
        a[1] = inpB_y;
        a[2] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;
        cpu::__mmul(ptr, a);
        // __kernel(ptr, a);

        break;
      }
      case this->fx_name.matrix_addition: {
        /* code */
        unsigned a[2];
        double *ptr[3];

        a[0] = inpA_x;
        a[1] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;
        cpu::__madd(ptr, a);
        // __kernel(ptr, a);

        break;
      }
      case this->fx_name.matrix_element_wise_multiplication: {
        /* code */
        unsigned a[2];
        double *ptr[3];

        a[0] = inpA_x;
        a[1] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;
        cpu::__melementwisemul(ptr, a);

        break;
      }
      case this->fx_name.matrix_power: {
        int j;
        // int exponent = ui_arr[0];
        unsigned a[2];
        double *ptr[3];

        a[0] = inpA_x;
        a[1] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = input_b.getData() + b_index;
        ptr[2] = output.getData() + c_index;

        cpu::__melementwisemul(ptr, a);

        break;
      }

      default:
        break;
      }
    } else {
      // std::cout << "inside else\n";
      for (unsigned i = 0; i < input_a.getDimensions()[index]; i++) {
        dimension_arr[index] = i;
        recursive_iterator(index - 1, dimension_arr, input_a, input_b, output,
                           function_name, NULL, NULL, NULL);
      }
    }
  };

public:
  void compute() override {
    unsigned i, *arr;

    if (exponent == 0)
      output->initData(1);
    else if (exponent > 0) {
      output->initData(inputs[0]->getData());
      arr = new unsigned[inputs[0]->getNoOfDimensions()];

      // std::cout << output.getData() << "\n";
      for (i = 1; i < exponent; i++)
        recursive_iterator(inputs[0]->getNoOfDimensions() - 1, arr, *(output),
                           *(inputs[0]), *(output), "matrix_power", NULL, NULL,
                           NULL);
      delete[] arr;
    }
  }

  // template <typename Ta, typename Tb>
  void initilizeinputs(NDMath<T, typeFlag> **inputs,
                       unsigned exponent) override {
    unsigned i;
    this->exponent = exponent;

    this->inputs = new NDMath<T, typeFlag> *[1];

    for (i = 0; i < 1; i++) {
      this->inputs[i] = inputs[i];
    }
  }

  void initilizeoutput(NDMath<T, typeFlag> *output) override {
    this->output = output;

    *(this->output) = *(inputs[0]);
  }

  NDMath<T, typeFlag> **getinputs() { return inputs; }

  NDMath<T, typeFlag> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }

  void printinputs() override {
    unsigned i;
    for (i = 0; i < 1; i++) {
      std::cout << "Input: " << i << "\n";
      inputs[i]->printData();
    }
  }

  void printoutput() override {
    // std::cout << output->getData() << "\n";
    std::cout << "output:\n";
    output->printData();
    std::cout << "\n";
  }
};

template <typename T, int typeFlag>
class Opsreducesum : public Ops<T, typeFlag> {
  unsigned no_of_reduction_dim, *reduction_dims;
  struct_function_name fx_name;

  NDMath<T, typeFlag> **inputs;
  NDMath<T, typeFlag> *output;
  void recursive_sum(unsigned index, unsigned *dimension_arr,
                     NDMath<T, typeFlag> input, NDMath<T, typeFlag> &output,
                     unsigned reduction_dim, T *temp_input) {

    if (index < 3) {
      unsigned i, j, k;
      unsigned x_axis, y_axis, z_axis, stride, n_dim_size;
      unsigned input_index, output_index;
      T *input_ptr, *output_ptr, *temp_inp;
      double *ptr[3];
      unsigned a[2];

      // T *input = this->getData();

      x_axis = input.getDimensions()[0];
      y_axis = (input.getNoOfDimensions() > 1) ? input.getDimensions()[1] : 1;
      z_axis = (input.getNoOfDimensions() > 2) ? input.getDimensions()[2] : 1;

      input_ptr = input.getData();
      output_ptr = output.getData();

      input_index = output_index = 0;

      if (input.getNoOfDimensions() > 3) {
        n_dim_size = x_axis * y_axis * z_axis;
        for (i = 3; i < input.getNoOfDimensions(); i++) {
          input_index += n_dim_size * dimension_arr[i];
          n_dim_size *= input.getDimensions()[i];
        }

        n_dim_size = 1;
        for (i = 0; i < input.getNoOfDimensions(); i++) {
          if (i != reduction_dim) {
            if (i < 3)
              output_index *= n_dim_size;
            else
              output_index += n_dim_size * dimension_arr[i];

            n_dim_size *= input.getDimensions()[i];
          }
        }
      }

      switch (reduction_dim) {
      case 0: {

        ptr[0] = ptr[2] = output_ptr + output_index;
        a[0] = y_axis;
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
      for (unsigned i = 0; i < input.getDimensions()[index]; i++) {
        dimension_arr[index] = i;
        recursive_sum(index - 1, dimension_arr, input, output, reduction_dim,
                      temp_input);
      }
    }
  }

public:
  void compute() override {
    unsigned i, k, resulting_no_of_dims;

    unsigned *resulting_dims, *arr_dims;
    T *intermediate_input;
    NDMath<T, typeFlag> temp_output, temp_input;

    intermediate_input =
        new T[inputs[0]->getDimensions()[0] * inputs[0]->getDimensions()[1] *
              inputs[0]->getDimensions()[2]];

    resulting_dims = new unsigned(inputs[0]->getNoOfDimensions());
    arr_dims = new unsigned(inputs[0]->getNoOfDimensions());
    temp_output = *(inputs[0]);
    temp_input = *(inputs[0]);

    for (i = 0; i < no_of_reduction_dim; i++) {
      resulting_no_of_dims = temp_output.getNoOfDimensions()
                                 ? temp_output.getNoOfDimensions() - 1
                                 : 1;
      if (temp_output.getNoOfDimensions() > 1) {
        k = 0;
        for (unsigned j = 0; j < temp_output.getNoOfDimensions(); j++)
          if (j != reduction_dims[i])
            resulting_dims[k++] = temp_output.getDimensions()[j];
      } else {
        resulting_no_of_dims = 1;
        resulting_dims[0] = 1;
      }

      temp_output = NDMath<T, typeFlag>(resulting_no_of_dims, resulting_dims);
      temp_output.initData(0.0);

      recursive_sum(temp_input.getNoOfDimensions() - 1, arr_dims, temp_input,
                    temp_output, reduction_dims[i], intermediate_input);

      temp_input = temp_output;
    }
    output->initData(temp_output);
  }

  // template <typename Ta, typename Tb>
  void initilizeinputs(NDMath<T, typeFlag> **inputs, unsigned n,
                       unsigned *arr) override {
    unsigned i;
    this->no_of_reduction_dim = n;

    reduction_dims = new unsigned[n];
    for (i = 0; i < n; i++)
      reduction_dims[i] = arr[i];

    for (i = 0; i < n; i++)
      std::cout << reduction_dims[i] << ", ";
    std::cout << "\n";

    this->inputs = new NDMath<T, typeFlag> *[1];

    this->inputs[0] = inputs[0];
  }

  void initilizeoutput(NDMath<T, typeFlag> *output) override {
    unsigned no_of_resultent_dims, *resultent_dims;
    unsigned i, j;
    this->output = output;

    no_of_resultent_dims = inputs[0]->getNoOfDimensions() - no_of_reduction_dim;
    resultent_dims = new unsigned[no_of_reduction_dim];

    j = 0;

    for (unsigned i = 0; i < inputs[0]->getNoOfDimensions(); i++)
      if (i != reduction_dims[j])
        resultent_dims[j++] = inputs[0]->getDimensions()[i];

    *(this->output) = NDMath<T, typeFlag>(no_of_resultent_dims, resultent_dims);
  }

  NDMath<T, typeFlag> **getinputs() { return inputs; }

  NDMath<T, typeFlag> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }

  void printinputs() override {
    unsigned i;
    for (i = 0; i < 1; i++) {
      std::cout << "Input: " << i << "\n";
      inputs[i]->printData();
    }
  }

  void printoutput() override {
    // std::cout << output->getData() << "\n";
    std::cout << "output:\n";
    output->printData();
    std::cout << "\n";
  }
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
                          double *dl_arr, NDMath<T, typeFlag> misc_arr) {
    if (index < 2) {
      unsigned i, inpA_x, inpA_y, inpB_x, inpB_y, out_x, out_y;
      unsigned a_plane_size, b_plane_size, c_plane_size, a_index, b_index,
          c_index;

      inpA_x =
          (input_a.getNoOfDimensions() > 0) ? input_a.getDimensions()[0] : 1;
      inpA_y =
          (input_a.getNoOfDimensions() > 1) ? input_a.getDimensions()[1] : 1;

      out_x = (output.getNoOfDimensions() > 0) ? output.getDimensions()[0] : 1;
      out_y = (output.getNoOfDimensions() > 1) ? output.getDimensions()[1] : 1;

      a_plane_size = inpA_x * inpA_y;
      c_plane_size = out_x * out_y;

      a_index = b_index = c_index = 0;
      if (input_a.getNoOfDimensions() > 2)
        for (i = 2; i < input_a.getNoOfDimensions(); i++) {
          a_index += a_plane_size * dimension_arr[i];
          c_index += c_plane_size * dimension_arr[i];

          a_plane_size *= input_a.getDimensions()[i];
          c_plane_size *= output.getDimensions()[i];
        }

      switch (fx_name.function_name[function_name]) {
      case this->fx_name.matrix_scaler_multiplication: {
        unsigned a[2];
        double *ptr[3];

        a[0] = inpA_x;
        a[1] = inpA_y;

        ptr[0] = input_a.getData() + a_index;
        ptr[1] = dl_arr;
        ptr[2] = output.getData() + c_index;

        cpu::__mscalermul(ptr, a);

        break;
      }
      default:
        break;
      }
    } else {
      // std::cout << "inside else\n";
      for (unsigned i = 0; i < input_a.getDimensions()[index]; i++) {
        dimension_arr[index] = i;
        recursive_iterator(index - 1, dimension_arr, input_a, input_b, output,
                           function_name, ui_arr, dl_arr, misc_arr);
      }
    }
  };

public:
  void compute() override {
    unsigned *arr;

    arr = new unsigned[inputs[0]->getNoOfDimensions()];

    recursive_iterator(inputs[0]->getNoOfDimensions() - 1, arr, *(inputs[0]),
                       NULL, *(output), "matrix_scaler_multiplication", NULL,
                       &scale_factor, NULL);
    delete[] arr;
  }

  // template <typename Ta, typename Tb>
  void initilizeinputs(NDMath<T, typeFlag> **inputs,
                       double scale_factor) override {
    this->scale_factor = scale_factor;
    this->inputs = new NDMath<T, typeFlag> *[1];
    this->inputs[0] = inputs[0];
  }

  void initilizeoutput(NDMath<T, typeFlag> *outputs) override {
    this->output = outputs;
    *(this->output) = *(inputs[0]);
  }

  NDMath<T, typeFlag> **getinputs() { return inputs; }

  NDMath<T, typeFlag> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }

  void printinputs() override {
    unsigned i;
    for (i = 0; i < 1; i++) {
      std::cout << "Input: " << i << "\n";
      inputs[i]->printData();
    }
  }

  void printoutput() override {
    // std::cout << output->getData() << "\n";
    std::cout << "output:\n";
    output->printData();
    std::cout << "\n";
  }
};
