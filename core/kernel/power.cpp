#ifdef CUDA_ENABLED
#include <LAS/gpu_interface.cuh>
#endif

// Thirdparty header
#include <absl/log/log.h>

// Tensor headers
#include <LAS/CPULibrary.h>
#include <LAS/avx2_micro_kernels.h>
#include <framework/MathLibrary.h>
#include <kernel/opskernel.h>

Opspower::~Opspower() {}

void Opspower::recursive_iterator(unsigned index, unsigned *dimension_arr,
                                  std::string function_name, unsigned *ui_arr,
                                  std::float64_t *dl_arr,
                                  Tensor<std::float64_t> *misc_arr) {
  if (index < 2) {
    unsigned i, inpA_x, inpA_y, inpB_x, inpB_y, out_x, out_y;
    unsigned a_plane_size, b_plane_size, c_plane_size, a_index, b_index,
        c_index;

    inpA_x = (inputs[0]->getNoOfDimensions() > 0)
                 ? inputs[0]->getDimensions()[0]
                 : 1;
    inpA_y = (inputs[0]->getNoOfDimensions() > 1)
                 ? inputs[0]->getDimensions()[1]
                 : 1;

    inpB_x = (inputs[0]->getNoOfDimensions() > 0)
                 ? inputs[0]->getDimensions()[0]
                 : 1;
    inpB_y = (inputs[0]->getNoOfDimensions() > 1)
                 ? inputs[0]->getDimensions()[1]
                 : 1;

    out_x = (output->getNoOfDimensions() > 0) ? output->getDimensions()[0] : 1;
    out_y = (output->getNoOfDimensions() > 1) ? output->getDimensions()[1] : 1;

    a_plane_size = inpA_x * inpA_y;
    b_plane_size = inpB_x * inpB_y;
    c_plane_size = out_x * out_y;

    a_index = b_index = c_index = 0;
    if (inputs[0]->getNoOfDimensions() > 2)
      for (i = 2; i < inputs[0]->getNoOfDimensions(); i++) {
        a_index += a_plane_size * dimension_arr[i];
        b_index += b_plane_size * dimension_arr[i];
        c_index += c_plane_size * dimension_arr[i];

        a_plane_size *= inputs[0]->getDimensions()[i];
        b_plane_size *= inputs[0]->getDimensions()[i];
        c_plane_size *= output->getDimensions()[i];
      }

    int j;
    unsigned a[2];
    std::float64_t *ptr[3];

    a[0] = inpA_x;
    a[1] = inpA_y;

    ptr[0] = inputs[0]->getData() + a_index;
    ptr[1] = output->getData() + c_index;
    ptr[2] = output->getData() + c_index;

    kernel_dispatch(ptr, a);

  } else {
    for (unsigned i = 0; i < inputs[0]->getDimensions()[index]; i++) {
      dimension_arr[index] = i;
      recursive_iterator(index - 1, dimension_arr, function_name, NULL, NULL,
                         NULL);
    }
  }
};

void Opspower::compute() {
  unsigned i, *arr;

  if (this->exponent == 0)
    output->initData(1);
  else if (this->exponent > 0) {
    output->initData(inputs[0]->getData());
    arr = new unsigned[inputs[0]->getNoOfDimensions()];
    for (i = 1; i < this->exponent; i++)
      recursive_iterator(inputs[0]->getNoOfDimensions() - 1, arr,
                         "matrix_power", NULL, NULL, NULL);
    delete[] arr;
  }
}

void Opspower::addGradGraph(Graph *gradient_graph) {
  // .......... reverse mode autodiff graph .........
  //
  //             [inputs[0]]
  //                 |
  //              [power]
  //                 |
  //        [temp_grad_tensor[0]]
  //                 |
  //              [scale]
  //                 |
  //        [temp_grad_tensor[1]]  *  [[incoming_gradients]...]
  //                           [[add]...]
  //                              |
  //                      [output_gradient]
  //
  // ........................ End .....................

  Tensor<std::float64_t> *tensor_ptr[2];
  Tensor<std::float64_t> **intermediate_gradients;
  std::vector<Tensor<std::float64_t> *> incoming_gradients =
      gradient_graph->getGradient(this);

  // graph setup for accumulating incoming gradients y' = sum ( z' )
  if (incoming_gradients.size()) {

    // graph setup for  x' = sum ( z' * d/dx )
    Tensor<std::float64_t> *intermediate_gradient_sum;
    intermediate_gradient_sum = new Tensor<std::float64_t>(*this->output);
    intermediate_gradient_sum->initData(0.0);
    int i = 0;
    for (Tensor<std::float64_t> *inc_grad_tensor : incoming_gradients) {

      // input initialization
      tensor_ptr[0] = intermediate_gradient_sum;
      tensor_ptr[1] = inc_grad_tensor;

      Ops *ops_add = new Opsadd;
      ops_add->initializeinputs(tensor_ptr);

      gradient_graph->addGradientNode(ops_add);
      gradient_graph->addGradientNode(tensor_ptr[0]);
      gradient_graph->addGradientNode(tensor_ptr[1]);
      gradient_graph->addGradientEdge(tensor_ptr[0], ops_add);
      gradient_graph->addGradientEdge(tensor_ptr[1], ops_add);

      // output initialization
      intermediate_gradient_sum = new Tensor<std::float64_t>(*this->output);
      intermediate_gradient_sum->initData(0.0);

      ops_add->initializeoutput(intermediate_gradient_sum);
      gradient_graph->addGradientNode(intermediate_gradient_sum);
      gradient_graph->addGradientEdge(ops_add, intermediate_gradient_sum);
    }
    this->incoming_gradient = intermediate_gradient_sum;
  } else {
    this->incoming_gradient = new Tensor<std::float64_t>(*this->output);
    this->incoming_gradient->initData(1.0);
  }

  unsigned i, j;

  Tensor<std::float64_t> *temp_grad_tensors[2];

  // initializing temp variables for grad calculation
  temp_grad_tensors[0] = new Tensor<std::float64_t>(*inputs[0]);
  temp_grad_tensors[1] = new Tensor<std::float64_t>(*inputs[0]);
  temp_grad_tensors[1]->initData(1.0);

  // graph setup for calculating derivation temp_grad_tensorsf power
  // operations graph setup for x ^ (n-1)
  Ops *ops_power = new Opspower;
  ops_power->initializeinputs(this->inputs.data());
  ops_power->initializeExpoent(this->exponent - 1);
  ops_power->initializeoutput(temp_grad_tensors[0]);

  gradient_graph->addGradientNode(this->inputs[0]);
  gradient_graph->addGradientNode(temp_grad_tensors[0]);
  gradient_graph->addGradientNode(ops_power);

  gradient_graph->addGradientEdge(this->inputs[0], ops_power);
  gradient_graph->addGradientEdge(ops_power, temp_grad_tensors[0]);

  // graph setup for  n * x
  Ops *ops_scale = new Opsscale;
  ops_scale->initializeinputs(&temp_grad_tensors[0]);
  ops_scale->initializeScale(this->exponent);
  ops_scale->initializeoutput(temp_grad_tensors[1]);

  gradient_graph->addGradientNode(temp_grad_tensors[0]);
  gradient_graph->addGradientNode(temp_grad_tensors[1]);
  gradient_graph->addGradientNode(ops_scale);

  gradient_graph->addGradientEdge(temp_grad_tensors[0], ops_scale);
  gradient_graph->addGradientEdge(ops_scale, temp_grad_tensors[1]);

  // graph setup for d/dx[i] * z'
  Ops *ops_mul = new Opsmul;
  tensor_ptr[0] = temp_grad_tensors[1];
  tensor_ptr[1] = this->incoming_gradient;

  // input initialization
  ops_mul->initializeinputs(tensor_ptr);
  gradient_graph->addGradientNode(ops_mul);
  gradient_graph->addGradientNode(tensor_ptr[0]);
  gradient_graph->addGradientNode(tensor_ptr[1]);
  gradient_graph->addGradientEdge(tensor_ptr[0], ops_mul);
  gradient_graph->addGradientEdge(tensor_ptr[1], ops_mul);

  // output initialization
  this->outgoing_gradients[i] = new Tensor<std::float64_t>(*this->inputs[0]);
  ops_mul->initializeoutput(this->outgoing_gradients[i]);
  gradient_graph->addGradientNode(this->outgoing_gradients[0]);
  gradient_graph->addGradientEdge(ops_mul, this->outgoing_gradients[0]);
  // End of d/dx[i] * z'
}

void Opspower::initializeinputs(Tensor<std::float64_t> **inputs) {
  this->inputs.push_back(inputs[0]);
}

void Opspower::initializeoutput(Tensor<std::float64_t> *output) {
  this->output = output;

  *(this->output) = *(inputs[0]);
}

void Opspower::printinputs() {
  unsigned i;
  for (i = 0; i < 1; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opspower::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

Tensor<std::float64_t> *
Opspower::getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input) {
  if (inputs[0] == gradient_input)
    return outgoing_gradients[0];
  else {
    LOG(FATAL) << "Requested gradint for the tensor doesn't exist.\n";
    return NULL;
  }
}

std::vector<Tensor<std::float64_t> *> Opspower::getAllOutgoingGradientTensors() {
  std::vector<Tensor<std ::float64_t> *> gradient_tensors;
  gradient_tensors.push_back(outgoing_gradients[0]);
  return gradient_tensors;
}

void Opspower::kernel_dispatch(std::float64_t **ptr, unsigned *arr) {
#ifdef CUDA_ENABLED
  double *d_arr[3];
  d_arr[0] = reinterpret_cast<double *>(ptr[0]);
  d_arr[1] = reinterpret_cast<double *>(ptr[1]);
  d_arr[2] = reinterpret_cast<double *>(ptr[2]);

  gpu::gpu_mat_hadamard_mul_f64(d_arr, arr);
#else
  if (__builtin_cpu_supports("avx2")) {
    avx2::avx2_mul_f64(ptr, arr);
  } else {
    cpu::__melementwisemul(ptr, arr);
  }
#endif
}