#ifndef OPS_KERNEL
#define OPS_KERNEL

// CPP header
#include <cstdint>
#include <iostream>
#include <stdfloat>
#include <vector>

#include <graph/graph_framework.hpp>

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
  virtual void initializeoutput(Tensor<std::float64_t> *) = 0;
  virtual void initializeinputs(Tensor<std::float64_t> **input_a) = 0;
  virtual Tensor<std::float64_t> *getoutput() = 0;
  virtual std::vector<Tensor<std::float64_t> *> getinputs() = 0;

  virtual void initializeAxis(const unsigned axis) {}
  virtual void initializeExpoent(const unsigned exponent) {}
  virtual void initializeScale(const std::float64_t scale) {}
  virtual void initializeReductionDims(const unsigned n, const unsigned *arr) {}

  virtual void addGradGraph(Graph *gradient_graph) {};
  virtual Tensor<std::float64_t> *
  getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input) {
    return NULL;
  }
  virtual Tensor<std::float64_t> *
  getIncomingGradientTensor(Tensor<std::float64_t> *tensor) {
    return NULL;
  }
  virtual std::vector<Tensor<std::float64_t> *>
  getAllOutgoingGradientTensors() {
    std::vector<Tensor<std::float64_t> *> grads;
    return grads;
  }
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

  std::vector<Tensor<std::float64_t> *> inputs;
  Tensor<std::float64_t> *output;
  Tensor<std::float64_t> *incoming_gradient;
  Tensor<std::float64_t> *outgoing_gradients[2];
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opsmul() = default;
  ~Opsmul() {}
  void compute();

  void addGradGraph(Graph *gradient_graph);

  Tensor<std::float64_t> *
  getIncomingGradientTensor(Tensor<std::float64_t> *tensor) override {
    return incoming_gradient;
  }

  Tensor<std::float64_t> *
  getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input);

  std::vector<Tensor<std::float64_t> *> getAllOutgoingGradientTensors() {
    std::vector<Tensor<std::float64_t> *> grads;
    return grads;
  }

  void initializeinputs(Tensor<std::float64_t> **inputs);

  void initializeoutput(Tensor<std::float64_t> *output);

  std::vector<Tensor<std::float64_t> *> getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs();

  void printoutput();
};

class Opsadd : public Ops {
  unsigned no_of_inputs;

  std::vector<Tensor<std::float64_t> *> inputs;
  Tensor<std::float64_t> *output;
  Tensor<std::float64_t> *incoming_gradient;
  std::vector<Tensor<std::float64_t> *> outgoing_gradients;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opsadd() = default;
  ~Opsadd() {}
  void compute();
  void addGradGraph(Graph *gradient_graph);
  Tensor<std::float64_t> *
  getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input);

  Tensor<std::float64_t> *
  getIncomingGradientTensor(Tensor<std::float64_t> *tensor) override;
  std::vector<Tensor<std::float64_t> *> getAllOutgoingGradientTensors() {
    std::vector<Tensor<std::float64_t> *> grads;
    return grads;
  };
  void initializeinputs(Tensor<std::float64_t> **inputs);
  void initializeoutput(Tensor<std::float64_t> *output);

  std::vector<Tensor<std::float64_t> *> getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs();
  void printoutput();
};

class Opsmatmul : public Ops {
  unsigned no_of_inputs;

  std::vector<Tensor<std::float64_t> *> inputs;
  Tensor<std::float64_t> *output;
  Tensor<std::float64_t> *incoming_gradient;
  Tensor<std::float64_t> *outgoing_gradients[2];
  void recursive_iterator(unsigned index, unsigned *, std::string, unsigned *,
                          std::float64_t *, Tensor<std::float64_t> *);
  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opsmatmul() = default;

  ~Opsmatmul();

  void compute();

  void addGradGraph(Graph *gradient_graph);

  Tensor<std::float64_t> *
  getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input);

  std::vector<Tensor<std::float64_t> *> getAllOutgoingGradientTensors() {
    std::vector<Tensor<std::float64_t> *> grads;
    return grads;
  }

  void initializeoutput(Tensor<std::float64_t> *output);

  std::vector<Tensor<std::float64_t> *> getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  void initializeinputs(Tensor<std::float64_t> **);

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs();

  void printoutput();

  void destory() override;
};

class Opspower : public Ops {
  unsigned exponent;

  std::vector<Tensor<std::float64_t> *> inputs; // inputs for the op
  Tensor<std::float64_t> *output;               // output for the op
  Tensor<std::float64_t>
      *incoming_gradient; // z' incoming gradient from next ops
  Tensor<std::float64_t>
      *outgoing_gradients[1]; // d/dx * z' outgoing gradient for previous ops
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opspower() = default;

  ~Opspower();

  void addGradGraph(Graph *gradient_graph);

  void compute();

  void computeGrad();

  void initializeinputs(Tensor<std::float64_t> **inputs);

  void initializeExpoent(unsigned exponent) { this->exponent = exponent; }

  void initializeoutput(Tensor<std::float64_t> *output);

  std::vector<Tensor<std::float64_t> *> getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  unsigned getnoofinputs() { return 1; }

  Tensor<std::float64_t> *
  getIncomingGradientTensor(Tensor<std::float64_t> *tensor) override {
    return incoming_gradient;
  }

  Tensor<std::float64_t> *
  getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input);

  std::vector<Tensor<std::float64_t> *> getAllOutgoingGradientTensors();

  void printinputs();

  void printoutput();
};

class Opsreducesum : public Ops {
  unsigned no_of_reduction_dim;
  std::vector<unsigned> reduction_dims;
  std::vector<Tensor<std::float64_t> *> inputs;
  Tensor<std::float64_t> *temp_output;
  Tensor<std::float64_t> *temp_input;
  Tensor<std::float64_t> *output;
  Tensor<std::float64_t> *outgoing_gradient;
  void recursive_sum(unsigned index, unsigned *dimension_arr,
                     unsigned reduction_dim, std::float64_t *temp_arr);

public:
  Opsreducesum() = default;
  ~Opsreducesum();
  void compute();

  void addGradGraph(Graph *gradient_graph) {}
  Tensor<std::float64_t> *getOutgoingGradientTensor() {
    return outgoing_gradient;
  }

  std::vector<Tensor<std::float64_t> *> getAllOutgoingGradientTensors() {
    std::vector<Tensor<std::float64_t> *> grads;
    return grads;
  }
  void initializeinputs(Tensor<std::float64_t> **inputs);
  void initializeReductionDims(const unsigned n, const unsigned *arr);
  void initializeoutput(Tensor<std::float64_t> *output);
  std::vector<Tensor<std::float64_t> *> getinputs() { return inputs; }
  Tensor<std::float64_t> *getoutput() { return output; }
  unsigned getnoofinputs() { return 1; }
  void printinputs();
  void printoutput();
  void kernel_dispatch(std::float64_t **, unsigned *);
};

class Opsscale : public Ops {
  std::float64_t scale_factor[1];

  std::vector<Tensor<std::float64_t> *> inputs;
  Tensor<std::float64_t> *output;
  Tensor<std::float64_t> *incoming_gradient;
  Tensor<std::float64_t> *outgoing_gradients[1];
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);
  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opsscale() = default;
  ~Opsscale() {}

  void compute() override;

  void addGradGraph(Graph *gradient_graph) override;

  Tensor<std::float64_t> *
  getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input) override;

  Tensor<std::float64_t> *
  getIncomingGradientTensor(Tensor<std::float64_t> *gradient_input) override {
    return incoming_gradient;
  }

  std::vector<Tensor<std::float64_t> *>
  getAllOutgoingGradientTensors() override;
  void initializeinputs(Tensor<std::float64_t> **inputs) override;
  void initializeScale(const std::float64_t scale) override {
    this->scale_factor[0] = scale;
  }
  void initializeoutput(Tensor<std::float64_t> *outputs) override;

  std::vector<Tensor<std::float64_t> *> getinputs() override { return inputs; }

  Tensor<std::float64_t> *getoutput() override { return output; }

  void printinputs() override;

  void printoutput() override;
};

class Opssqrt : public Ops {
  std::vector<Tensor<std::float64_t> *> inputs;
  Tensor<std::float64_t> *output;
  Tensor<std::float64_t> *outgoing_gradient;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opssqrt() = default;
  ~Opssqrt(){};
  void compute();

  void addGradGraph(Graph *gradient_graph) {}
  Tensor<std::float64_t> *getOutgoingGradientTensor() {
    return outgoing_gradient;
  }

  std::vector<Tensor<std::float64_t> *> getAllOutgoingGradientTensors() {
    std::vector<Tensor<std::float64_t> *> grads;
    return grads;
  }
  void initializeinputs(Tensor<std::float64_t> **inputs);
  void initializeoutput(Tensor<std::float64_t> *output);
  std::vector<Tensor<std::float64_t> *> getinputs() { return inputs; }
  Tensor<std::float64_t> *getoutput() { return output; }
  unsigned getnoofinputs() { return 1; }
  void printinputs();
  void printoutput();
};

class Opssub : public Ops {
  unsigned no_of_inputs;

  std::vector<Tensor<std::float64_t> *> inputs;
  Tensor<std::float64_t> *output;
  Tensor<std::float64_t> *outgoing_gradient;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opssub() = default;
  ~Opssub() {}
  void compute();
  void addGradGraph(Graph *gradient_graph) {}
  Tensor<std::float64_t> *getOutgoingGradientTensor() {
    return outgoing_gradient;
  }

  std::vector<Tensor<std::float64_t> *> getAllOutgoingGradientTensors() {
    std::vector<Tensor<std::float64_t> *> grads;
    return grads;
  }
  void initializeinputs(Tensor<std::float64_t> **inputs);
  void initializeoutput(Tensor<std::float64_t> *output);

  std::vector<Tensor<std::float64_t> *> getinputs() { return inputs; }

  Tensor<std::float64_t> *getoutput() { return output; }

  unsigned getnoofinputs() { return no_of_inputs; }

  void printinputs();
  void printoutput();
};

class Opsrelu : public Ops {
  std::vector<Tensor<std::float64_t> *> inputs;
  Tensor<std::float64_t> *output;
  Tensor<std::float64_t> *outgoing_gradient;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opsrelu() = default;
  ~Opsrelu() {}
  void compute();
  void addGradGraph(Graph *gradient_graph) {}
  Tensor<std::float64_t> *getOutgoingGradientTensor() {
    return outgoing_gradient;
  }

  std::vector<Tensor<std::float64_t> *> getAllOutgoingGradientTensors() {
    std::vector<Tensor<std::float64_t> *> grads;
    return grads;
  }
  void initializeinputs(Tensor<std::float64_t> **inputs);
  void initializeoutput(Tensor<std::float64_t> *output);
  std::vector<Tensor<std::float64_t> *> getinputs() { return inputs; }
  Tensor<std::float64_t> *getoutput() { return output; }
  unsigned getnoofinputs() { return 1; }
  void printinputs();
  void printoutput();
};

class Opssigmoid : public Ops {
  std::vector<Tensor<std::float64_t> *> inputs;
  Tensor<std::float64_t> *output;
  Tensor<std::float64_t> *outgoing_gradient;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opssigmoid() = default;
  ~Opssigmoid() {}
  void compute();
  void addGradGraph(Graph *gradient_graph) {}
  Tensor<std::float64_t> *getOutgoingGradientTensor() {
    return outgoing_gradient;
  }

  std::vector<Tensor<std::float64_t> *> getAllOutgoingGradientTensors() {
    std::vector<Tensor<std::float64_t> *> grads;
    return grads;
  }
  void initializeinputs(Tensor<std::float64_t> **inputs);
  void initializeoutput(Tensor<std::float64_t> *output);
  std::vector<Tensor<std::float64_t> *> getinputs() { return inputs; }
  Tensor<std::float64_t> *getoutput() { return output; }
  unsigned getnoofinputs() { return 1; }
  void printinputs();
  void printoutput();
};

class Opssoftmax : public Ops {
  std::vector<Tensor<std::float64_t> *> inputs;
  int axis;
  Tensor<std::float64_t> *output;
  Tensor<std::float64_t> *outgoing_gradient;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opssoftmax() = default;
  ~Opssoftmax() {}
  void compute();
  void addGradGraph(Graph *gradient_graph) {}
  Tensor<std::float64_t> *getOutgoingGradientTensor() {
    return outgoing_gradient;
  }

  std::vector<Tensor<std::float64_t> *> getAllOutgoingGradientTensors() {
    std::vector<Tensor<std::float64_t> *> grads;
    return grads;
  }
  void initializeinputs(Tensor<std::float64_t> **inputs);
  void initializeAxis(const unsigned axis) { this->axis = axis; }
  void initializeoutput(Tensor<std::float64_t> *output);
  std::vector<Tensor<std::float64_t> *> getinputs() { return inputs; }
  Tensor<std::float64_t> *getoutput() { return output; }
  unsigned getnoofinputs() { return 1; }
  void printinputs();
  void printoutput();
};

class Opstranspose : public Ops {
  std::vector<Tensor<std::float64_t> *> inputs;
  unsigned no_of_inputs;
  Tensor<std::float64_t> *output;
  Tensor<std::float64_t> *outgoing_gradient;
  void recursive_iterator(unsigned index, unsigned *dimension_arr,
                          std::string function_name, unsigned *ui_arr,
                          std::float64_t *dl_arr,
                          Tensor<std::float64_t> *misc_arr);

  void kernel_dispatch(std::float64_t **, unsigned *);

public:
  Opstranspose() = default;
  ~Opstranspose() {}
  void compute();
  void addGradGraph(Graph *gradient_graph) {}
  Tensor<std::float64_t> *
  getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input);

  std::vector<Tensor<std::float64_t> *> getAllOutgoingGradientTensors() {
    std::vector<Tensor<std::float64_t> *> grads;
    return grads;
  }
  void initializeinputs(Tensor<std::float64_t> **inputs);
  void initializeoutput(Tensor<std::float64_t> *output);
  std::vector<Tensor<std::float64_t> *> getinputs() { return inputs; }
  Tensor<std::float64_t> *getoutput() { return output; }
  unsigned getnoofinputs() { return 1; }
  void printinputs();
  void printoutput();
};

#endif // OPS_kernel