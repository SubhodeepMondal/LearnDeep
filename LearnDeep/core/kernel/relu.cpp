#ifdef ENABLE_CUDA
#include <core/LAS/gpu_interface.cuh>
#endif

// Thirdparty header
#include <absl/log/log.h>

// Library Headers
#include "kernelmanager.h"
#include "opskernel.h"
#include <core/LAS/CPULibrary.h>
#include <core/LAS/avx2_micro_kernels.h>
#include <core/framework/MathLibrary.h>

void Opsrelu::compute() {
  std::float64_t *ptr[2];
  ptr[0] = this->inputs[0]->getData();
  ptr[1] = this->output->getData();

  this->kernel_dispatch(ptr, this->inputs[0]->getNoOfDimensions(),
                        this->inputs[0]->getDimensions());
}

void Opsrelu::addGradGraph(Graph *gradient_graph) {
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

  Tensor<std::float64_t> *temp_grad_tensors[1];
  // Finding d/dx[i] for relu operation
  //  f(x[i]) = greater_then_zero(x[i]) * incoming_grad
  temp_grad_tensors[0] = new Tensor<std::float64_t>(*this->inputs[0]);
  // end of Finding d/dx[i]

  Ops *ops_greater_than_zero = new Opsgreaterthanzero();
  ops_greater_than_zero->initializeinputs(this->inputs.data());
  ops_greater_than_zero->initializeoutput(temp_grad_tensors[0]);

  gradient_graph->addGradientNode(this->inputs[0]);
  gradient_graph->addGradientNode(temp_grad_tensors[0]);
  gradient_graph->addGradientNode(ops_greater_than_zero);

  gradient_graph->addGradientEdge(this->inputs[0], ops_greater_than_zero);
  gradient_graph->addGradientEdge(ops_greater_than_zero, temp_grad_tensors[0]);

  // graph setup for d/dx[i] * z'
  Ops *ops_mul = new Opsmul;
  tensor_ptr[0] = temp_grad_tensors[0];
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

void Opsrelu::initializeinputs(Tensor<std::float64_t> **inputs) {
  this->inputs.push_back(inputs[0]);
}

void Opsrelu::initializeoutput(Tensor<std::float64_t> *outputs) {
  this->output = outputs;
  *(this->output) = *(inputs[0]);
}

void Opsrelu::printinputs() {
  unsigned i;
  for (i = 0; i < 1; i++) {
    std::cout << "Input: " << i << "\n";
    inputs[i]->printData();
  }
}

void Opsrelu::printoutput() {
  std::cout << "output:\n";
  output->printData();
  std::cout << "\n";
}

Tensor<std::float64_t> *
Opsrelu::getOutgoingGradientTensor(Tensor<std::float64_t> *gradient_input) {
  if (inputs[0] == gradient_input)
    return outgoing_gradients[0];
  else {
    LOG(FATAL) << "Requested gradint for the tensor doesn't exist.\n";
    return NULL;
  }
}

std::vector<Tensor<std::float64_t> *> Opsrelu::getAllOutgoingGradientTensors() {
  std::vector<Tensor<std ::float64_t> *> gradient_tensors;
  gradient_tensors.push_back(outgoing_gradients[0]);
  return gradient_tensors;
}

void Opsrelu::kernel_dispatch(std::float64_t **ptr, const unsigned nDim,
                              unsigned const *arr) {

  KernelType kernel = get_global_kernel();
#ifdef ENABLE_CUDA
  bool gpu_available = true;
#else
  bool gpu_available = false;
#endif

  switch (kernel) {

  case KernelType::GPU:
#ifdef ENABLE_CUDA
  {
    double *d_arr[2];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    gpu::gpu_mat_relu_f64(d_arr, nDim, arr);
  }
#else
    throw std::runtime_error("GPU kernel requested but CUDA not enabled");
#endif
  break;

  case KernelType::AVX2:
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_relu_f64(ptr, nDim, arr);
    } else {
      throw std::runtime_error("AVX2 not supported on this CPU");
    }
    break;

  case KernelType::CPU_SCALAR:
    cpu::__mrelu(ptr, arr);
    break;

  case KernelType::AUTO:
  default:
#ifdef ENABLE_CUDA
  {
    double *d_arr[2];
    d_arr[0] = reinterpret_cast<double *>(ptr[0]);
    d_arr[1] = reinterpret_cast<double *>(ptr[1]);
    gpu::gpu_mat_relu_f64(d_arr, nDim, arr);
  }
#else
    if (__builtin_cpu_supports("avx2")) {
      avx2::avx2_relu_f64(ptr, nDim, arr);
    } else {
      cpu::__mrelu(ptr, arr);
    }
#endif
  break;
  }
}
