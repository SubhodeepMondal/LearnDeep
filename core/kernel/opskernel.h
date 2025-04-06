#ifndef OPS_KERNEL
#define OPS_KERNEL

#include "../LAS/CPULibrary.h"
#include <cstdint>
#include <iostream>
#include <stdfloat>

template <typename T> class Tensor;

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
  virtual void initilizeoutput(Tensor<std::float64_t> *) = 0;
  virtual void initilizeinputs(Tensor<std::float64_t> **input_a,
                               unsigned no_of_inputs) {}
  virtual void initilizeinputs(Tensor<std::float64_t> **input_a,
                               std::float64_t scale_factor) {}
  virtual void initilizeinputs(Tensor<std::float64_t> **input_a, unsigned n,
                               unsigned *arr) {}
  virtual unsigned getnoofinputs() { return 0; }
  virtual Tensor<std::float64_t> *getoutput() { return NULL; }
  virtual Tensor<std::float64_t> **getinputs() { return NULL; }
  virtual void printinputs() {}
  virtual void printoutput() {}
  virtual void autograd() {}

  virtual void recursive_iterator(unsigned index, unsigned *dimension_arr,
                                  // Tensor<std::float64_t> input_a,
                                  Tensor<std::float64_t> input_b,
                                  Tensor<std::float64_t> &output,
                                  std::string function_name, unsigned *ui_arr,
                                  std::float64_t *dl_arr,
                                  Tensor<std::float64_t> *misc_arr);

  static void recursive_sum(unsigned index, unsigned *dimension_arr,
                            Tensor<std::float64_t> input_b,
                            Tensor<std::float64_t> &output,
                            unsigned reduction_dim, std::float64_t *temp_input);
};

class Opsmul : public Ops {
  unsigned no_of_inputs;

  Tensor<std::float64_t> **inputs;
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

public:
  Opsmul() {}
  void compute();
  void initilizeinputs(Tensor<std::float64_t> **inputs, unsigned no_of_inputs);

  void initilizeoutput(Tensor<std::float64_t> *output);

  Tensor<std::float64_t> **getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs();

  void printoutput();
};

class Opsadd : public Ops {
  unsigned no_of_inputs;

  Tensor<std::float64_t> **inputs;
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

public:
  Opsadd() {}
  void compute();
  void initilizeinputs(Tensor<std::float64_t> **inputs, unsigned no_of_inputs);
  void initilizeoutput(Tensor<std::float64_t> *output);

  Tensor<std::float64_t> **getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs();
  void printoutput();
};

class Opsmatmul : public Ops {
  unsigned no_of_inputs;

  Tensor<std::float64_t> **inputs;
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *, std::string, unsigned *,
                          std::float64_t *, Tensor<std::float64_t> *);

public:
  Opsmatmul() {}

  void compute();

  void initilizeoutput(Tensor<std::float64_t> *output);

  Tensor<std::float64_t> **getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  void initilizeinputs(Tensor<std::float64_t> **, unsigned);

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs();

  void printoutput();
};

class Opspower : public Ops {
  unsigned exponent;

  Tensor<std::float64_t> **inputs;
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

public:
  void compute();
  void initilizeinputs(Tensor<std::float64_t> **inputs, unsigned exponent);

  void initilizeoutput(Tensor<std::float64_t> *output);

  Tensor<std::float64_t> **getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }

  void printinputs();

  void printoutput();
};

class Opsreducesum : public Ops {
  unsigned no_of_reduction_dim, *reduction_dims;

  Tensor<std::float64_t> **inputs;
  Tensor<std::float64_t> *output;
  void recursive_sum(unsigned index, unsigned *dimension_arr,
                     Tensor<std::float64_t> input,
                     Tensor<std::float64_t> &output, unsigned reduction_dim,
                     std::float64_t *temp_input);

public:
  void compute();
  void initilizeinputs(Tensor<std::float64_t> **inputs, unsigned n,
                       unsigned *arr);
  void initilizeoutput(Tensor<std::float64_t> *output);

  Tensor<std::float64_t> **getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }

  void printinputs();

  void printoutput();
};

class Opsscale : public Ops {
  std::float64_t scale_factor;

  Tensor<std::float64_t> **inputs;
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

public:
  void compute();
  void initilizeinputs(Tensor<std::float64_t> **inputs,
                       std::float64_t scale_factor);
  void initilizeoutput(Tensor<std::float64_t> *outputs);

  Tensor<std::float64_t> **getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }
  void printinputs();
  void printoutput();
};

#endif // OPS_kernel