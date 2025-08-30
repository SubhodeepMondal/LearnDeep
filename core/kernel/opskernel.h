#ifndef OPS_KERNEL
#define OPS_KERNEL

#include <cstdint>
#include <iostream>
#include <stdfloat>
#include <vector>

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
  virtual ~Ops() = 0;
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
  virtual void destory() {}

  virtual void recursive_iterator(unsigned index, unsigned *dimension_arr,
                                  // Tensor<std::float64_t> input_a,
                                  Tensor<std::float64_t> input_b,
                                  Tensor<std::float64_t> &output,
                                  std::string function_name, unsigned *ui_arr,
                                  std::float64_t *dl_arr,
                                  Tensor<std::float64_t> *misc_arr);

  virtual void recursive_sum(unsigned index, unsigned *dimension_arr,
                             Tensor<std::float64_t> input_b,
                             Tensor<std::float64_t> &output,
                             unsigned reduction_dim,
                             std::float64_t *temp_input);
};

class Opsmul : public Ops {
  unsigned no_of_inputs;

  Tensor<std::float64_t> *inputs[2];
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opsmul() = default;
  ~Opsmul() {}
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

  Tensor<std::float64_t> *inputs[2];
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opsadd() = default;
  ~Opsadd() {}
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

  Tensor<std::float64_t> *inputs[2];
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *, std::string, unsigned *,
                          std::float64_t *, Tensor<std::float64_t> *);
  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opsmatmul() = default;

  ~Opsmatmul();

  void compute();

  void initilizeoutput(Tensor<std::float64_t> *output);

  Tensor<std::float64_t> **getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  void initilizeinputs(Tensor<std::float64_t> **, unsigned);

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs();

  void printoutput();

  void destory() override;
};

class Opspower : public Ops {
  unsigned exponent;

  Tensor<std::float64_t> *inputs[1];
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opspower() = default;
  ~Opspower() {}
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
  unsigned no_of_reduction_dim;
  std::vector<unsigned> reduction_dims;
  Tensor<std::float64_t> *inputs[1];
  Tensor<std::float64_t> *temp_output;
  Tensor<std::float64_t> *temp_input;
  Tensor<std::float64_t> *output;
  void recursive_sum(unsigned index, unsigned *dimension_arr,
                     unsigned reduction_dim, std::float64_t *temp_arr);

public:
  Opsreducesum() = default;
  ~Opsreducesum();
  void compute();
  void initilizeinputs(Tensor<std::float64_t> **inputs, unsigned n,
                       unsigned *arr);
  void initilizeoutput(Tensor<std::float64_t> *output);

  Tensor<std::float64_t> **getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }

  void printinputs();

  void printoutput();

  void kernel_dispatch(std::float64_t **, unsigned *);
};

class Opsscale : public Ops {
  std::float64_t scale_factor[1];

  Tensor<std::float64_t> *inputs[1];
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);
  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opsscale() = default;
  ~Opsscale() {}
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

class Opssqrt : public Ops {
  Tensor<std::float64_t> *inputs[1];
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opssqrt() = default;
  ~Opssqrt(){};
  void compute();
  void initilizeinputs(Tensor<std::float64_t> **inputs, unsigned no_of_inputs);
  void initilizeoutput(Tensor<std::float64_t> *output);
  Tensor<std::float64_t> **getinputs() { return inputs; }
  Tensor<std::float64_t> *getoutput() { return output; }
  unsigned getnoofinputs() { return 1; }
  void printinputs();
  void printoutput();
};

class Opssub : public Ops {
  unsigned no_of_inputs;

  Tensor<std::float64_t> *inputs[2];
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opssub() = default;
  ~Opssub() {}
  void compute();
  void initilizeinputs(Tensor<std::float64_t> **inputs, unsigned no_of_inputs);
  void initilizeoutput(Tensor<std::float64_t> *output);

  Tensor<std::float64_t> **getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs();
  void printoutput();
};

class Opsrelu : public Ops {
  Tensor<std::float64_t> *inputs[1];
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opsrelu() = default;
  ~Opsrelu() {}
  void compute();
  void initilizeinputs(Tensor<std::float64_t> **inputs, unsigned no_of_inputs);
  void initilizeoutput(Tensor<std::float64_t> *output);
  Tensor<std::float64_t> **getinputs() { return inputs; }
  Tensor<std::float64_t> *getoutput() { return output; }
  unsigned getnoofinputs() { return 1; }
  void printinputs();
  void printoutput();
};

class Opssigmoid : public Ops {
  Tensor<std::float64_t> *inputs[1];
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opssigmoid() = default;
  ~Opssigmoid() {}
  void compute();
  void initilizeinputs(Tensor<std::float64_t> **inputs, unsigned no_of_inputs);
  void initilizeoutput(Tensor<std::float64_t> *output);
  Tensor<std::float64_t> **getinputs() { return inputs; }
  Tensor<std::float64_t> *getoutput() { return output; }
  unsigned getnoofinputs() { return 1; }
  void printinputs();
  void printoutput();
};

class Opssoftmax : public Ops {
  Tensor<std::float64_t> *inputs[1];
  Tensor<std::float64_t> *output;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opssoftmax() = default;
  ~Opssoftmax() {}
  void compute();
  void initilizeinputs(Tensor<std::float64_t> **inputs, unsigned no_of_inputs);
  void initilizeoutput(Tensor<std::float64_t> *output);
  Tensor<std::float64_t> **getinputs() { return inputs; }
  Tensor<std::float64_t> *getoutput() { return output; }
  unsigned getnoofinputs() { return 1; }
  void printinputs();
  void printoutput();
};

#endif // OPS_kernel