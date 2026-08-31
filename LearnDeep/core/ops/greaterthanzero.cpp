#ifdef ENABLE_CUDA
#include <core/LAS/gpu_interface.cuh>
#endif

// Thirdparty header
#include <absl/log/log.h>

// Tensor headers

#include "kernelmanager.h"
#include <core/LAS/CPULibrary.h>
#include <core/LAS/avx2_micro_kernels.h>
#include <core/framework/MathLibrary.h>
#include <core/ops/opskernel.h>

void Opsgreaterthanzero::compute() {
  std::float64_t *ptr[2];
  ptr[0] = this->inputs[0]->getData();
  ptr[1] = this->output->getData();

  this->kernel_dispatch(ptr, this->inputs[0]->getNoOfDimensions(),
                        this->inputs[0]->getDimensions());
}
/*
void Opsgreaterthanzero::addGradGraph(Graph *gradient_graph) {
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

  Tensor<std::float64_t> *temp_grad_tensors[2];

  // initializing temp variables for grad calculation
  temp_grad_tensors[0] = new Tensor<std::float64_t>(*inputs[0]);
  temp_grad_tensors[1] = new Tensor<std::float64_t>(*inputs[0]);
  temp_grad_tensors[1]->initData(1.0);

  // graph setup for calculating derivation temp_grad_tensorsf power
  // operations graph setup for x ^ (n-1)
  Ops *ops_power = new Opsgreaterthanzero;
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
  this->outgoing_gradients[0] = new Tensor<std::float64_t>(*this->inputs[0]);
  ops_mul->initializeoutput(this->outgoing_gradients[0]);
  gradient_graph->addGradientNode(this->outgoing_gradients[0]);
  gradient_graph->addGradientEdge(ops_mul, this->outgoing_gradients[0]);
  // End of d/dx[i] * z'
}
*/
void Opsgreaterthanzero::initializeinputs(Tensor<std::float64_t> **inputs) {
  this->inputs.push_back(inputs[0]);
}

void Opsgreaterthanzero::initializeoutput(Tensor<std::float64_t> *output) {
  this->output = output;

  *(this->output) = *(inputs[0]);
}

void Opsgreaterthanzero::printinputs() {
  unsigned i;
  for (i = 0; i < 1; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opsgreaterthanzero::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

Tensor<std::float64_t> *Opsgreaterthanzero::getOutgoingGradientTensor(
    Tensor<std::float64_t> *gradient_input) {
  if (inputs[0] == gradient_input)
    return outgoing_gradient;
  else {
    LOG(FATAL) << "Requested gradint for the tensor doesn't exist.\n";
    return NULL;
  }
}

void Opsgreaterthanzero::kernel_dispatch(std::float64_t **ptr,
                                         const unsigned nDim,
                                         unsigned const *dims) {

  KernelType kernel = get_global_kernel();
#ifdef ENABLE_CUDA
  bool gpu_available = true;
#else
  bool gpu_available = false;
#endif

  switch (kernel) {

  case KernelType::GPU:
    if (!this->warning_handled) {
      std::cerr << "GPU kernel requested but greater_than_zero has no GPU "
                   "implementation, falling back to AVX2 kernel\n";
      this->warning_handled = true;
    }
  case KernelType::AVX2:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_greater_than_zero_f64(ptr, const_cast<unsigned *>(dims), nDim);
    } else {
      throw std::runtime_error("AVX2 not supported on this CPU");
    }
    break;

  case KernelType::CPU_SCALAR:
    cpu::__mgreaterthanzero(ptr, dims, nDim);
    break;
  case KernelType::AUTO:
  default:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_greater_than_zero_f64(ptr, const_cast<unsigned *>(dims), nDim);
    } else {
      cpu::__mgreaterthanzero(ptr, dims, nDim);
    }
    break;
  }
}
