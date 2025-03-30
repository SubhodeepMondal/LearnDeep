#ifndef OPS_KERNEL
#define OPS_KERNEL

#include "../LAS/CPULibrary.h"
#include <iostream>

template <typename T> class tensor;

typedef enum function_names {
  matrix_multiplication,
  matrix_scaler_multiplication,
  matrix_element_wise_multiplication,
  matrix_addition,
  matrix_subtraction,
  matrix_rollingsum,
  matrix_power,
  matrix_transpose,
} function_names;

class Ops {

public:
  virtual void compute() = 0;
  virtual void initilizeoutput(tensor<double> *) = 0;
  virtual void initilizeinputs(tensor<double> **input_a,
                               unsigned no_of_inputs) {}
  virtual void initilizeinputs(tensor<double> **input_a, double scale_factor) {}
  virtual void initilizeinputs(tensor<double> **input_a, unsigned n,
                               unsigned *arr) {}
  virtual unsigned getnoofinputs() { return 0; }
  virtual tensor<double> *getoutput() { return NULL; }
  virtual tensor<double> **getinputs() { return NULL; }
  virtual void printinputs() {}
  virtual void printoutput() {}
  virtual void autograd() {}

  virtual void recursive_iterator(unsigned index, unsigned *dimension_arr,
                                  // tensor<double> input_a,
                                  tensor<double> input_b,
                                  tensor<double> &output,
                                  std::string function_name, unsigned *ui_arr,
                                  double *dl_arr, tensor<double> *misc_arr);

  static void recursive_sum(unsigned index, unsigned *dimension_arr,
                            tensor<double> input_b, tensor<double> &output,
                            unsigned reduction_dim, double *temp_input);
};

class Opsmul : public Ops {
  unsigned no_of_inputs;

  tensor<double> **inputs;
  tensor<double> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          tensor<double> input_a, tensor<double> input_b,
                          tensor<double> &output, std::string function_name,
                          unsigned *ui_arr, double *dl_arr,
                          tensor<double> *misc_arr);

public:
  Opsmul() {}
  void compute();
  void initilizeinputs(tensor<double> **inputs, unsigned no_of_inputs);
  void initilizeoutput(tensor<double> *output);

  tensor<double> **getinputs() { return inputs; }

  tensor<double> *getoutput() { return output; }

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs();

  void printoutput();
};

class Opsadd : public Ops {
  unsigned no_of_inputs;

  tensor<double> **inputs;
  tensor<double> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          tensor<double> input_a, tensor<double> input_b,
                          tensor<double> &output, std::string function_name,
                          unsigned *ui_arr, double *dl_arr,
                          tensor<double> *misc_arr);

public:
  Opsadd() {}
  void compute();
  void initilizeinputs(tensor<double> **inputs, unsigned no_of_inputs);
  void initilizeoutput(tensor<double> *output);

  tensor<double> **getinputs() { return inputs; }

  tensor<double> *getoutput() { return output; }

  unsigned getnoofinputs() { return no_of_inputs; }

  // void initilizeinputs(tensor<double> **inputs, unsigned no_of_inputs);

  // void initilizeoutput(tensor<double> *output);

  void printinputs();
  void printoutput();
};

class Opsmatmul : public Ops {
  unsigned no_of_inputs;

  tensor<double> **inputs;
  tensor<double> *output;
  void recursive_iterator(unsigned index, unsigned *, tensor<double>,
                          tensor<double>, tensor<double> &, std::string,
                          unsigned *, double *, tensor<double>*);

public:
  Opsmatmul() {}
  // void recursive_iterator(unsigned, unsigned *, tensor<double>,
  //                         tensor<double>, tensor<double> &output,
  //                         std::string, unsigned *, double *,
  //                         tensor<double>);
  void compute();
  // void initilizeinputs(tensor<double>, unsigned);
  void initilizeoutput(tensor<double> *output);

  tensor<double> **getinputs() { return inputs; }

  tensor<double> *getoutput() { return output; }

  void initilizeinputs(tensor<double> **, unsigned);

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs();

  void printoutput();
};

class Opspower : public Ops {
  unsigned exponent;

  tensor<double> **inputs;
  tensor<double> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          tensor<double> input_a, tensor<double> input_b,
                          tensor<double> &output, std::string function_name,
                          unsigned *ui_arr, double *dl_arr,
                          tensor<double> *misc_arr);

public:
  void compute();
  void initilizeinputs(tensor<double> **inputs, unsigned exponent);

  void initilizeoutput(tensor<double> *output);

  tensor<double> **getinputs() { return inputs; }

  tensor<double> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }

  void printinputs();

  void printoutput();
};

class Opsreducesum : public Ops {
  unsigned no_of_reduction_dim, *reduction_dims;

  tensor<double> **inputs;
  tensor<double> *output;
  void recursive_sum(unsigned index, unsigned *dimension_arr,
                     tensor<double> input, tensor<double> &output,
                     unsigned reduction_dim, double *temp_input);

public:
  void compute();
  void initilizeinputs(tensor<double> **inputs, unsigned n, unsigned *arr);
  void initilizeoutput(tensor<double> *output);

  tensor<double> **getinputs() { return inputs; }

  tensor<double> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }

  void printinputs();

  void printoutput();
};

class Opsscale : public Ops {
  double scale_factor;

  tensor<double> **inputs;
  tensor<double> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          tensor<double> input_a, tensor<double> *input_b,
                          tensor<double> &output, std::string function_name,
                          unsigned *ui_arr, double *dl_arr,
                          tensor<double> *misc_arr);

public:
  void compute();
  void initilizeinputs(tensor<double> **inputs, double scale_factor);
  void initilizeoutput(tensor<double> *outputs);

  tensor<double> **getinputs() { return inputs; }

  tensor<double> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }
  void printinputs();
  void printoutput();
};

#endif // OPS_kernel